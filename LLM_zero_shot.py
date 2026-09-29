# @TIME : 12/2/25 8:18 AM
# @AUTHOR : LZDH
"""LLM-based index selection for a SQL workload.

Selects indexes for each (workload, storage budget) pair with the chosen LLM strategy, evaluates the
selection on PostgreSQL (hypothetical cost and actual runtime), and appends the result to
``<save_dir>/result_naive_<storage>.csv``. Per-run logs go to ``<save_dir>/logs/<storage>/<workload_id>/``.
"""
import argparse
import glob
import json
import logging
import os
import random
import string

import numpy as np
import pandas as pd

import agents
from agents import (filt_schema, get_plan_long_memory, planning_agent_one_step, query_turbo_model,
                    reasoning_pipe, reasoning_pipe_one_path)
from eval_index import eval_index_hypo, eval_index_run, get_index_actual_storage
from index_graph import get_cand_nodes

logger = logging.getLogger("LLM_zero_shot")

# project modules whose loggers follow --log-level; everything else (openai, httpx, the
# index_selection_evaluation library) stays at WARNING
PROJECT_LOGGERS = ("LLM_zero_shot", "agents", "eval_index", "index_graph", "get_query_meta", "get_postgre", "get_index")
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

MODES = ("pipeline", "one_path", "one_step", "zero_shot", "one_shot")


def generate_turbo_prompt_pioneer(schema, workload, candidates=[], is_workload=False, storage=500):
    # w/wo index candidates and demo
    if is_workload:
        p = [{'role': "system", 'content': 'You are an online SQL index selection agent. You will be given a schema of '
                                           'a database, a workload of SQL queries querying this database and the list '
                                           'of index column candidates for the workload . You are required to propose '
                                           'an index selection '
                                           'ONLY based on the candidate columns so that the execution of the workload '
                                           'will be optimised. Each candidate in the given list is a Python list '
                                           'formatting: "[column_name, table_name, column_min_value, column_max_value'
                                           ', column_utility]", where column_utility represents the cost reduction of '
                                           'the workload if the column is selected as index alone. You should select '
                                           'no more than ' + str(storage) + ' candidates. '
                                           'You should return only the selected index, in the form of '
                                           '"Index selection: [Table_name1(column_name1, column_name2), '
                                           'Table_name2(column_name3)]". '
              }, ]
        p_0 = [{
            'role': "user",
            'content': "Schema: " + str(schema) + ". Workload: " + str(workload) + ". Index candidates: " + str(
                candidates) + "."
        }]
    else:
        p = [{'role': "system", 'content': 'You are an online SQL index selection agent. You will be given a schema of '
                                           'a database and the list of index column candidates from a workload of '
                                           'SQL queries '
                                           'querying this database. You are required to propose an index selection '
                                           'ONLY based on the candidate columns so that the execution of the workload '
                                           'will be optimised. Each candidate in the given list is a Python list '
                                           'formatting: "[table_name.column_name, column_type, column_rows_selected, '
                                           'column_utility, column_storage_cost, '
                                           ' column_min_value, column_max_value]", where column_utility represents the '
                                           'cost reduction of the workload if the column is selected as index alone, '
                                           'and column_storage_cost represents the STORAGE COST if the candidate is '
                                           'selected. '
                                           'Your candidates selected should have TOTAL STORAGE COST less than '
                                           + str(storage) + '. Try to fully utilize the budget given. You should '
                                           'return only the selected index, in the form of '
                                           '"Index selection: [Table_name1(column_name1, column_name2), '
                                           'Table_name2(column_name3)]". '
              }, ]
        p_0 = [{
            'role': "user",
            'content': "Schema: " + str(schema) + ". Index candidates: " + str(candidates) + "."
        }]
    return p + p_0


def generate_turbo_prompt_demo(schema, candidates, demos, storage=500, shots=1):
    p = [{'role': "system", 'content': 'You are an online SQL index selection agent. You will be given a schema of '
                                       'a database and the list of index column candidates from a workload of '
                                       'SQL queries querying this database. You are required to propose an index '
                                       'selection ONLY based on the candidate columns so that the execution of the '
                                       'workload will be optimised. Each candidate in the given list is a Python list '
                                       'formatting: "[table_name.column_name, column_type, column_rows_selected, '
                                       'column_utility, column_storage_cost, column_min_value, column_max_value]", '
                                       'where column_utility represents the cost reduction of the workload if the '
                                       'column is selected as index alone, and column_storage_cost represents the '
                                       'STORAGE COST if the candidate is selected. Your candidates selected should '
                                       'have TOTAL STORAGE COST less than ' + str(storage) +
                                       '. Try to fully utilize the budget given. You should return only the selected '
                                       'index, in the form of "Index selection: '
                                       '[Table_name1(column_name1, column_name2), Table_name2(column_name3)]". '
          }, ]
    p_0 = []
    for i in range(shots):
        schema_0, candidates_0, index_0 = demos[i]
        p_0 = p_0 + [{'role': "user",
                      'content': "Schema: " + str(schema_0) + ". Index candidates: " + str(candidates_0) + "."
                      },
                     {
                         'role': "assistant",
                         'content': str(index_0),
                     }, ]
    p_0 = p_0 + [{
        'role': "user",
        'content': "Schema: " + str(schema) + ". Index candidates: " + str(candidates) + "."
    }, ]
    return p + p_0


def create_dummy_codebook(original_info, codebook):
    if original_info not in codebook.values():
        dummy_val = '0'
        while dummy_val in codebook.keys():
            dummy_val = ''.join(random.choices(string.ascii_letters, k=len(original_info)))
        codebook[dummy_val] = original_info
    else:
        dummy_val = [k for k, v in codebook.items() if v == original_info][0]
    return dummy_val, codebook


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    data = parser.add_argument_group("workload and database")
    data.add_argument("--db-id", default="tpch",
                      help="benchmark name; selects data/schemas/<db-id>.json and data/dbstats/col_min_max_<db-id>.csv")
    data.add_argument("--working-db-id", default="indexselection_tpch___10",
                      help="PostgreSQL database the workload runs against")
    data.add_argument("--workload", nargs="+",
                      default=["../workloads/test/tpch/tpch_test_19q_modified.csv"],
                      help="workload CSV file(s) with a Query_text column; glob patterns allowed")
    data.add_argument("--storages", nargs="+", type=int, default=[1024],
                      help="storage budgets to run for each workload")

    out = parser.add_argument_group("inputs and outputs")
    out.add_argument("--save-dir", default="agents_ablation/tpch",
                     help="directory for result CSVs and per-run logs")
    out.add_argument("--baseline-path", default="agents_ablation/baseline_result_naive.csv",
                     help="CSV of cached no-index runtimes; the workload is evaluated without indexes if missing")
    out.add_argument("--skip-finished", action="store_true",
                     help="skip workloads already present in the result CSV")
    out.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                     help="verbosity of the console and run.log output; DEBUG adds raw agent outputs")

    sel = parser.add_argument_group("selection strategy")
    sel.add_argument("--mode", default="pipeline", choices=MODES,
                     help="pipeline: multi-agent planning loop; one_path: plan the whole action path up front; "
                          "one_step: single planning prompt; zero_shot / one_shot: single direct selection prompt")
    sel.add_argument("--model", default="gpt-4o", help="OpenAI chat model used by every agent")
    sel.add_argument("--iters", type=int, default=10, help="max planning iterations (pipeline mode)")
    sel.add_argument("--postprocess", action="store_true",
                     help="reject redundant indexes with a post-process agent (pipeline mode)")
    sel.add_argument("--dummy", action="store_true",
                     help="hide real table/column names from the LLM behind random dummy names")
    sel.add_argument("--filtered", action="store_true", help="restrict candidates to a pre-selected index list")
    sel.add_argument("--filter-path", default="../selected_index_list_top_6.json",
                     help="JSON with an 'index_selected' list (with --filtered)")
    sel.add_argument("--include-workload", action="store_true",
                     help="include the SQL workload in the prompt (zero_shot mode)")

    demo = parser.add_argument_group("one_shot mode")
    demo.add_argument("--demo-results", default="results/test/{db_id}/result_naive_{storage} copy.csv",
                      help="result CSV to sample the demonstration from; {db_id} and {storage} are filled in")
    demo.add_argument("--same-demo", action="store_true",
                      help="use the current workload's own candidates as the demonstration")

    mem = parser.add_argument_group("long-term memory")
    mem.add_argument("--long-mem", action="store_true",
                     help="retrieve a previously successful planning path as a hint (pipeline mode)")
    mem.add_argument("--mem-path", default="long_mem/train_workload_mem.json",
                     help="JSON of stored planning paths and their workload embeddings")
    mem.add_argument("--test-mem-index-path", default="long_mem/test_workload_mem_indexes.json",
                     help="JSON with the embedding of each test workload")
    return parser.parse_args()


def setup_logging(level):
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    root.addHandler(handler)
    for name in PROJECT_LOGGERS:
        logging.getLogger(name).setLevel(level)


def expand_workloads(patterns):
    files = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if not matched:
            raise FileNotFoundError(f"no workload file matches {pattern}")
        files.extend(matched)
    return files


def load_candidates(args, workload):
    """Column candidates as [tab.col, type, rows, utility, storage_cost]."""
    if args.filtered:
        with open(args.filter_path) as f:
            selected_list = json.load(f)['index_selected']
        candidates = get_cand_nodes(args.db_id, args.working_db_id, workload, selected_list)
        return [x for x in candidates if x[0] in selected_list]
    return get_cand_nodes(args.db_id, args.working_db_id, workload, [])


def encode_candidates(candidates, min_max_df, storage, dummy):
    """Rename candidates to tab(col) (or dummy names), append column min/max, and drop unusable ones.

    Returns the encoded candidates and the dummy-name -> real-index codebook (empty unless dummy).
    """
    dummy_tables = {'0': ''}
    dummy_columns = {'0': ''}
    dummy_indexes = {}
    for cand in candidates:
        tab, col = cand[0].split('.')[:2]
        if dummy:
            dummy_tab, dummy_tables = create_dummy_codebook(tab, dummy_tables)
            dummy_col, dummy_columns = create_dummy_codebook(col, dummy_columns)
            dummy_idx = dummy_tab + '(' + dummy_col + ')'
            dummy_indexes[dummy_idx] = tab + '(' + col + ')'
            cand[0] = dummy_idx
        else:
            cand[0] = tab + '(' + col + ')'
        stats = min_max_df[(min_max_df['table'] == tab) & (min_max_df['column'] == col)]
        if stats.empty:
            cand.extend(['NA', 'NA'])
        else:
            cand.extend([stats['min'].tolist()[0], stats['max'].tolist()[0]])
    candidates = [x for x in candidates if x[3] != 0 and x[4] <= storage]
    return candidates, dummy_indexes


def get_baseline_runtimes(args, w_id, workload, log_dir):
    """(hypothetical cost, actual runtime) of the workload without any index, cached in --baseline-path."""
    if args.baseline_path and os.path.isfile(args.baseline_path):
        df_org = pd.read_csv(args.baseline_path)
        rows = df_org[df_org['workload_id'] == w_id]
        if not rows.empty:
            return rows['hypo_runtime_org'].tolist()[0], rows['actual_runtime_org'].tolist()[0]
    logger.info("No cached baseline for %s, evaluating the workload without indexes", w_id)
    hypo_runtime_org = eval_index_hypo(args.working_db_id, workload, '')
    actual_runtime_org = eval_index_run(args.working_db_id, workload, '', log_dir, 'original')
    return hypo_runtime_org, actual_runtime_org


def load_long_memory(args, w_id, storage):
    with open(args.test_mem_index_path) as f:
        test_inds = json.load(f)
    workload_emb = [x for x in test_inds if x['workload_id'] == w_id][0]['emb_index']
    return get_plan_long_memory(storage, args.mem_path, np.array([workload_emb]).astype('float32'))


def build_one_shot_prompt(args, schema0, candidates, storage):
    if args.same_demo:
        candidates_0, index_0 = candidates, ''
    else:
        demo_results = pd.read_csv(args.demo_results.format(db_id=args.db_id, storage=storage))
        demo_id = random.choice(demo_results['workload_id'].tolist())
        demo = demo_results[demo_results['workload_id'] == demo_id]
        candidates_0 = demo['candidates_llm'].tolist()[0]
        index_0 = demo['selection_extend'].tolist()[0]
    return generate_turbo_prompt_demo(schema0, candidates, [[schema0, candidates_0, index_0]], storage)


def select_indexes(args, schema, schema0, workload, candidates, storage, dummy_indexes, plan_long_mem):
    """Run the selection strategy; returns the selection string "[tab(col), ...]" and the agent action log."""
    actions_log = []
    if args.mode == "pipeline":
        output, _, actions_log, _ = reasoning_pipe(args.working_db_id, schema0, workload, candidates, storage,
                                                   dummy_indexes, plan_long_mem, args.postprocess, args.dummy,
                                                   iters=args.iters)
    elif args.mode == "one_path":
        output, _, actions_log = reasoning_pipe_one_path(args.working_db_id, schema0, workload, candidates, storage)
    elif args.mode == "one_step":
        output = query_turbo_model(planning_agent_one_step(schema, workload, candidates, storage))
    elif args.mode == "one_shot":
        output = query_turbo_model(build_one_shot_prompt(args, schema0, candidates, storage))
    else:
        prompt = generate_turbo_prompt_pioneer(schema0, workload, candidates, args.include_workload, storage)
        output = query_turbo_model(prompt)
    return output.split('Index selection:')[-1], actions_log


def append_action_log(log_dir, actions_log):
    path = os.path.join(log_dir, 'action_log.json')
    file_data = []
    if os.path.isfile(path):
        with open(path) as f_log:
            file_data = json.load(f_log)
    file_data.append(actions_log[0])
    with open(path, 'w') as f_log:
        json.dump(file_data, f_log, indent=4)


def save_result(result_path, result):
    df = pd.DataFrame({k: [v] for k, v in result.items()})
    if os.path.isfile(result_path):
        df = pd.concat([pd.read_csv(result_path), df], ignore_index=True)
    df.to_csv(result_path, index=False)


def score(runtime_org, runtime, storage_cost):
    return 0 if storage_cost == 0 else (float(runtime_org) - float(runtime)) / storage_cost


def run_workload(args, workload_file, storage, schema, min_max_df):
    w_id = os.path.basename(workload_file).split('.')[0]
    result_path = os.path.join(args.save_dir, f"result_naive_{storage}.csv")
    if args.skip_finished and os.path.isfile(result_path) \
            and w_id in pd.read_csv(result_path)['workload_id'].tolist():
        logger.info("Skipping %s (storage %s): already in %s", w_id, storage, result_path)
        return

    # eval_index writes time_log.csv via log_dir + filename, so keep the trailing slash
    log_dir = os.path.join(args.save_dir, 'logs', str(storage), w_id) + os.sep
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, 'run.log'))
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(file_handler)
    try:
        logger.info("=== Workload %s | storage budget %s | mode %s ===", w_id, storage, args.mode)
        workload = pd.read_csv(workload_file)['Query_text'].tolist()

        candidates = load_candidates(args, workload)
        schema0 = filt_schema(schema, workload, candidates)
        candidates, dummy_indexes = encode_candidates(candidates, min_max_df, storage, args.dummy)
        logger.info("%d index candidates", len(candidates))
        logger.debug("Candidates: %s", candidates)

        hypo_runtime_org, actual_runtime_org = get_baseline_runtimes(args, w_id, workload, log_dir)
        logger.info("No-index runtime: hypo %s, actual %s", hypo_runtime_org, actual_runtime_org)

        plan_long_mem = load_long_memory(args, w_id, storage) if args.long_mem else None
        output, actions_log = select_indexes(args, schema, schema0, workload, candidates, storage,
                                             dummy_indexes, plan_long_mem)
        logger.info("LLM selection: %s", output)

        if '[]' in output:
            hypo_runtime, actual_runtime = hypo_runtime_org, actual_runtime_org
        else:
            if args.dummy:
                output = '[' + ', '.join([dummy_indexes[x] for x in output[1:-1].split(', ')]) + ']'
                logger.info("Decoded selection: %s", output)
            hypo_runtime = eval_index_hypo(args.working_db_id, workload, output)
            actual_runtime = eval_index_run(args.working_db_id, workload, output, log_dir)
        actual_storage = get_index_actual_storage(args.working_db_id, output)
        score_llm = score(actual_runtime_org, actual_runtime, actual_storage)
        logger.info("Result: hypo runtime %s, actual runtime %s, storage %s, score %s",
                    hypo_runtime, actual_runtime, actual_storage, score_llm)

        actions_log.append({'Actual Runtime LLM': actual_runtime})
        append_action_log(log_dir, actions_log)

        # placeholder for a baseline run alongside the LLM (no-op: no indexes, storage 1)
        output_extend, hypo_runtime_extend, actual_runtime_extend, actual_storage_extend = \
            '', 0, actual_runtime_org, 1
        save_result(result_path, {
            'workload_id': w_id,
            'queries': workload,
            'hypo_runtime_org': hypo_runtime_org,
            'actual_runtime_org': actual_runtime_org,
            'selection_extend': output_extend,
            'hypo_runtime_extend': hypo_runtime_extend,
            'actual_runtime_extend': actual_runtime_extend,
            'est_storage_cost_extend': actual_storage_extend,
            'candidates_llm': candidates,
            'selection_llm': output,
            'est_storage_cost_llm': actual_storage,
            'hypo_runtime_llm': hypo_runtime,
            'actual_runtime_llm': actual_runtime,
            'score_extend': score(actual_runtime_org, actual_runtime_extend, actual_storage_extend),
            'score_llm': score_llm,
        })
        logger.info("Result appended to %s", result_path)
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


def main():
    args = parse_args()
    setup_logging(args.log_level)
    agents.MODEL = args.model
    os.makedirs(args.save_dir, exist_ok=True)

    with open(f'data/schemas/{args.db_id}.json') as f_sch:
        schema = json.load(f_sch)
    min_max_df = pd.read_csv(f'data/dbstats/col_min_max_{args.db_id}.csv').fillna('NA')

    for workload_file in expand_workloads(args.workload):
        for storage in args.storages:
            run_workload(args, workload_file, storage, schema, min_max_df)


if __name__ == "__main__":
    main()
