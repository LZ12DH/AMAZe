# @TIME : 8/5/25 12:03 PM
# @AUTHOR : LZDH
import ast
import json
import logging
import os
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from eval_index import *

load_dotenv()

logger = logging.getLogger(__name__)

# OpenAI (public API, no custom base_url)
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
# chat model used by every agent; overridable by the driver script (--model)
MODEL = "gpt-4o"


def query_gpt_attempts(prompt, trys):
    try:
        output = query_turbo_model(prompt)
    except Exception as error:
        logger.warning('LLM call failed (attempt %s): %s', trys, error)
        trys += 1
        if trys <= 3:
            output = query_gpt_attempts(prompt, trys)
        else:
            output = {'content': 'NA'}
    return output


def query_turbo_model(prompt):
    chat_completion = client.chat.completions.create(
        messages=prompt,
        model=MODEL,
        temperature=0,
    )
    return chat_completion.choices[0].message.content


def filt_schema(schema, workload, candidates):
    filted_schema = []
    cand_cols = [x[0].split('.')[1] for x in candidates]
    for tab in schema:
        table_name = tab['table']
        if all([table_name not in query for query in workload]):
            continue
        else:
            filted_tab = {}
            filted_tab['table'] = table_name
            filted_tab['rows'] = tab['rows']
            filted_tab['columns'] = []
            for col in tab['columns']:
                col_name = col['name']
                for query in workload:
                    if col_name not in query or col_name not in cand_cols:
                        continue
                    else:
                        filted_tab['columns'].append(col)
                        break
            filted_schema.append(filted_tab)
    return filted_schema


def predict_index_storage(output, working_db_id):
    if output and 'I(C' not in output:
        pattern = r"\[(.*?)\]"
        matches = re.findall(pattern, output)
        if len(matches[0].split('), ')) != 1:
            inds_selected = [x + ')' if x[-1] != ')' else x for x in matches[0].split('), ')]
        else:
            inds_selected = [x for x in matches[0].split('), ')]
        indexes = []
        for ind in inds_selected:
            tab = ind.split('(')[0]
            cols = ind.split('(')[1][:-1].split(', ')
            inds = [tab + '.' + x for x in cols]
            indexes.append(inds)
    elif output and 'I(C' in output:
        pattern = r"\((.*?)\)"
        matches = re.findall(pattern, output)
        cols_in_ind = [x.replace('C ', '').split(',') for x in matches]
        indexes = []
        for i in cols_in_ind:
            tab = i[0].split('.')[0]
            cols = [x.split('.')[1] for x in i]
            ind = tab + '.'
            inds = []
            for c in cols:
                ind_i = ind + c
                inds.append(ind_i)
            indexes.append(inds)
    else:
        indexes = []
    est_storage_cost = predict_index_sizes(indexes, working_db_id)
    return est_storage_cost


def planning_agent(schema, candidates, current_index, current_storage, storage, prev_plan, long_mem=None, suggestion='None', ablation=None):
    """
    ablation: None | 'no_combination' | 'no_revision'
        None            - full planner, all four actions available (default behaviour).
        'no_combination' - ablation study 1: removes the "Combination Agent" option/prompt entirely.
        'no_revision'    - ablation study 2: removes the "Revision Agent" option/prompt entirely.
    """
    action_descriptions = {
        'Selection Agent': '"Selection Agent": call the selection agent to select more columns as index.\n',
        'Revision Agent': '"Revision Agent": call the revision agent to revise the current indexes '
                           'selected, removing potential duplicates, regressions and out-of budget.\n',
        'Combination Agent': '"Combination Agent": call the combination agent to merge single-column '
                              'indexes from the same table if more efficient.\n',
        'Stop': '"Stop": the current index selection is optimal and no more actions are required.\n',
    }
    action_order = ['Selection Agent', 'Revision Agent', 'Combination Agent', 'Stop']
    if ablation == 'no_combination':
        action_order.remove('Combination Agent')
    elif ablation == 'no_revision':
        action_order.remove('Revision Agent')
    actions_text = ''.join(action_descriptions[a] for a in action_order)
    avoid_clauses = ['"' + a + '"' for a in ('Revision Agent', 'Combination Agent') if a in action_order]
    avoid_text = ' or '.join(avoid_clauses)
    action_list_str = str(action_order)

    p = [{'role': "system", 'content': 'You are an online SQL index selection planner. You will be given a schema of '
                                       'a database, the list of index column candidates from a workload of '
                                       'SQL queries querying this database, the current list of selected index, the '
                                       'current storage used, the total storage budget, the previous plan steps '
                                       'before this one and a suggestion for the next planning step.'
                                       'Each candidate in the given list is a Python list '
                                       'formatting: "[table_name.column_name, column_type, column_rows_selected, '
                                       'column_utility, column_storage_cost, column_min_value, column_max_value]", '
                                       'where column_utility represents the cost reduction of the workload if the '
                                       'column is selected as index alone, and column_storage_cost represents the '
                                       'STORAGE COST if the candidate is selected. You are required to plan the next '
                                       'action ONLY from the following options:\n' + actions_text +
                                       ' Given the previous plan steps, avoid using more than three '
                                       + avoid_text + ' continuously. You should try to make full use of the storage budget. '
                                       'Output your ONE action choosing from: ' + action_list_str + '.'}, ]
    p_0 = [{
        'role': "user",
        'content': "Schema: " + str(schema) + ". Index candidates: " + str(candidates) + ". Current index selection: "
                   + str(current_index) + ". Current storage used: " + str(current_storage) + ". Total storage budget: "
                   + str(storage) + ". Previous plan steps: " + str(prev_plan) + ". Suggestion: " + str(suggestion) + "."
    }]
    if long_mem:
        p_1 = ' Previously successful planning path: ' + str(long_mem) + '.'
        p[0]['content'] = p[0]['content'] + p_1
    p = p + p_0
    return p


def planning_agent_one_step(schema, workload, candidates, storage):
    p = [{'role': "system", 'content': 'You are an online SQL index selection planner. You will be given a schema of '
                                       'a database, the list of SQL queries from a workload querying this database, the columns with their storages '
                                       'and the total storage budget. You are required to select columns '
                                       'from the SQL queries without explanation to form indexes optimising '
                                       'the workload execution, in the form of "Index selection: '
                                       '[Table_name1(column_name1, column_name2), Table_name2(column_name3)].'}, ]
    p_0 = [{
        'role': "user",
        'content': "Schema: " + str(schema) + ". Workload: " + str(workload) + ". Column storages: "
                   + str([[x[0], x[4]] for x in candidates]) + ". Total storage budget: " + str(storage) + "."
    }]
    p = p + p_0
    return p


def planning_agent_path(schema, candidates, storage):
    p = [{'role': "system", 'content': 'You are an online SQL index selection planner. You will be given a schema of '
                                       'a database, the list of index column candidates from a workload of '
                                       'SQL queries querying this database and the total storage budget.'
                                       'Each candidate in the given list is a Python list '
                                       'formatting: "[table_name.column_name, column_type, column_rows_selected, '
                                       'column_utility, column_storage_cost, column_min_value, column_max_value]", '
                                       'where column_utility represents the cost reduction of the workload if the '
                                       'column is selected as index alone, and column_storage_cost represents the '
                                       'STORAGE COST if the candidate is selected. You are required to plan the path of '
                                       'actions ONLY choosing from the following action options:\n'
                                       '"Selection Agent": call the selection agent to select more columns as index.\n'
                                       '"Revision Agent": call the revision agent to revise the current indexes '
                                       'selected, removing potential duplicates, regressions and out-of budget.\n'
                                       '"Combination Agent": call the combination agent to merge single-column '
                                       'indexes from the same table if more efficient.\n'
                                       '"Stop": the current index selection is optimal and no more actions are '
                                       'required.\n Avoid using more than three "Revision Agent" or "Combination Agent" continuously. You should try to make full use of the storage budget.' 'Output ONLY your list of actions in a Python list without explanation like: ["Selection Agent", "Revision Agent", "Combination Agent", "Stop"]. Always end with the "Stop" action.'}, ]
    p_0 = [{
        'role': "user",
        'content': "Schema: " + str(schema) + "Index candidates: "
                   + str(candidates) + ". Total storage budget: " + str(storage) + "."
    }]
    p = p + p_0
    return p


def planning_agent_no_cand(schema, workload, current_index, current_storage, storage, prev_plan, suggestion='None'):
    p = [{'role': "system", 'content': 'You are an online SQL index selection planner. You will be given a schema of '
                                       'a database, the SQL queries from a workload querying this database, the current list of selected index, the '
                                       'current storage used, the total storage budget, the previous plan steps '
                                       'before this one and a suggestion for the next planning step. You are required to plan the next '
                                       'action ONLY from the following options:\n'
                                       '"Selection Agent": call the selection agent to select more columns as index.\n'
                                       '"Revision Agent": call the revision agent to revise the current indexes '
                                       'selected, removing potential duplicates, regressions and out-of budget.\n'
                                       '"Combination Agent": call the combination agent to merge different single-column '
                                       'indexes from the same table to make it more efficient.\n'
                                       '"Stop": the current index selection is optimal and no more actions are '
                                       'required.\n Given the previous plan steps, avoid using too many '
                                       '"Revision Agent" or "Combination Agent" continuously. '
                                       'Output your ONE action choosing from: ["Selection Agent", '
                                       '"Revision Agent", "Combination Agent", "Stop"].'}, ]
    p_0 = [{
        'role': "user",
        'content': "Schema: " + str(schema) + ". Index candidates: " + str(workload) + ". Current index selection: "
                   + str(current_index) + ". Current storage used: " + str(current_storage) + ". Total storage budget: "
                   + str(storage) + ". Previous plan steps: " + str(prev_plan) + ". Suggestion: " + str(suggestion) + "."
    }]
    p = p + p_0
    return p


def post_process_agent(schema, indexes, new_index):
    remove_rule = "Remove Rule: Reject the new index on columns that may have too many distinct values."
    dedup_rule = "Deduplicate Rule: Reject the new index if it is already in the selected index list."
    rules = [remove_rule, dedup_rule]
    p = [{'role': "system", 'content': 'You are an online SQL index selection agent. You will be given a schema of '
                                       'a database, the list of indexes already selected for a workload given the form of '
                                       'of a list "[Table_name1(column_name1, column_name2), '
                                       'Table_name2(column_name3)]", and a new selected index [Table_name0(column_name0)]. '
                                       'You are required to refine the index '
                                       'selection based on the rules provided below: ' + str(rules) +
                                       '. You should return only "False" if the index is rejected and "True" if not rejected with no explanation.'
          }, ]
    p_0 = [{
        'role': "user",
        'content': "Schema: " + str(schema) + ". Index selected: " + str(indexes) + ". New index: " + str(new_index) + "."
    }]
    p = p + p_0
    return p


def selection_agent(schema, candidates, selection_mem):
    logger.debug('Selection Agent input, current selection: %s', selection_mem)
    p = [{'role': "system", 'content': 'You are an online SQL index selection agent. You will be given a schema of '
                                       'a database, the list of index column candidates from a workload of '
                                       'SQL queries querying this database, and the current list of index selected'
                                       '. You are required to select one column from the candidates to add into the '
                                       'index, so that the execution of the workload'
                                       ' will be more efficient. Each candidate in the given list is a Python list '
                                       'formatting: "[table_name(column_name), column_type, column_rows_selected, '
                                       'column_utility, column_storage_cost,  column_min_value, column_max_value]", '
                                       'where column_utility represents the cost reduction of the workload if the '
                                       'column is selected as index alone, and column_storage_cost represents the '
                                       'STORAGE COST if the candidate is selected. DO NOT select the column that '
                                       'is ALREADY SELECTED as a single column index. You should return only the selected '
                                       'column without explanation, in the form of "Index selection: '
                                       '[table_name(column_name)]". '
          }, ]
    p_0 = [{
        'role': "user",
        'content': "Schema: " + str(schema) + ". Index candidates: " + str(candidates) + ". Current index selection: "
                   + str(selection_mem) + "."
    }]
    p = p + p_0
    return p


def selection_agent_nocand(schema, workload, candidates, selection_mem, storage_left):
    p = [{'role': "system", 'content': 'You are an online SQL index selection agent. You will be given a schema of '
                                       'a database, the SQL queries from a workload'
                                       ' querying this database, the storage cost list of the columns, the storage budget left and the current list of index'
                                       '. You are required to select one column to add into the '
                                       'index, so that the execution of the workload'
                                       ' will be more efficient. You should return only the selected '
                                       'column without explanation, in the form of "Index selection: '
                                       '[table_name(column_name)]". '
          }, ]
    p_0 = [{
        'role': "user",
        'content': "Schema: " + str(schema) + ". Workload: " + str(workload) + ". Column storages: "
                   + str([[x[0], x[4]] for x in candidates]) + ". Storage budget left: "
                   + str(storage_left) + ". Current index selection: "
                   + str(selection_mem) + "."
    }]
    p = p + p_0
    return p


def revision_agent(schema, indexes, storage, storage_used, revision_mem, ablation=None):
    """
    ablation: None | 'reduced_revision_rules'
        None                     - full rule set (default behaviour).
        'reduced_revision_rules' - ablation study 3: keep both Revision Agent and Combination Agent in the
                                    planner, but strip the Revision Agent's own rules down to only the
                                    'Deduplicate Rule' and the 'Shrink Rule'.
    """
    remove_rule = {"Remove Rule": "Remove the index on columns that may have too many distinct values"}
    dedup_rule = {
        "Deduplicate Rule": "Simplify the index on columns from the same table that sharing too same information by only keeping the high utility columns."}
    selectivity_rule = {
        "High Selectivity Rule": "If the number of rows selected from a column is high, it is likely not an efficient index to recommend."}
    distinct_ratio_rule = {
        "Low Distinct Ratio Rule": "Columns having lots of repeated same values are likely not good indexes to recommend."}
    sort_rule = {
        "Sort Rule": "If a column appears frequently in SORT operations, is is likely an efficient index to recommend."}
    storage_limit_rule = {"Shrink Rule": f"If the current storage used exceeds the storage budget {storage}, shrink "
                                         f"the index by removing some less efficient columns."}
    evidence_rules = {
        'Evidence Rule': 'Apply rules only when their prerequisites are supported by the inputs. Do not infer distinct counts, frequency, correlation or query clauses from min/max values, similar row estimates or shared plan-node types. These restrictions take precedence over conflicting heuristic rules.',
        'Exact Duplicate Rule': 'Remove repeated occurrences of an identical table and ordered index-column sequence, retaining one. Prefix overlap or similar statistics do not establish duplication.',
        'Marginal Utility Rule': 'An explicit utility:<value> entry means estimated workload cost without that entire index minus cost with the current configuration. Positive values predict removal loss, zero predicts no change, and negative values predict improvement. Do not interpret other candidate scores as leave-one-out utilities.',
        'Single Removal Rule': 'Except for exact duplicate entries, remove at most one entire index per call. Current marginal utilities cannot justify multiple independent removals because indexes may substitute for each other. Further removals require fresh utilities.',
        'Nonpositive Utility Rule': 'When all selected indexes have explicit marginal utilities, prioritize removing one index with the most negative utility, or one with zero utility if none is negative. This is estimated aggregate cost evidence, not a runtime guarantee.',
        'Budget Retention Rule': 'When storage is within budget and all remaining indexes have positive explicit marginal utilities, keep the selection unchanged except for exact duplicates.',
        'Minimum Loss Shrink Rule': 'When over budget and all explicit marginal utilities are positive, propose removing the entire index with the smallest marginal utility. This is a greedy repair step, not an optimal solution or proof of budget compliance. Do not compute loss per byte without individual index sizes.',
        'Whole Index Rule': 'An index marginal utility supports evaluating removal of that entire index only. Do not use it to justify shortening or reordering a composite index.',
        'Measured Evidence Rule': 'Fresh explicit marginal utilities take precedence over heuristics based solely on value ranges, row estimates or plan-node labels. If no revision is justified and storage is within budget, return the selection unchanged.',
    }
    if ablation == 'reduced_revision_rules':
        rules = dedup_rule | storage_limit_rule
    else:
        rules = remove_rule | dedup_rule | selectivity_rule | distinct_ratio_rule | sort_rule | storage_limit_rule | evidence_rules
    p = [{'role': "system", 'content': 'You are an online SQL index revision agent. You will be given a schema of '
                                       'a database, the previously adopted revision rules and the list of indexes '
                                       'selected for a workload given the form of '
                                       '"[Table_name1(column_name1, column_name2), Table_name2(column_name3)]". '
                                       'You are required to refine the index '
                                       'selection based on the rules provided below: ' + str(rules) +
                                       '. You should return ONLY the keys of rules you used and the revised '
                                       'index without explanation. DO NOT select new index and STRICTLY in the form '
                                       'of "{"rules used": [rule keys], "revised index": [Table_name1(column_name1, column_name2), Table_name2(column_name3)]}". '
          }, ]
    p_0 = [{
        'role': "user",
        'content': "Schema: " + str(schema) + "Adopted rules: " + str(revision_mem) + ". Index selected: " + str(indexes) + ". Storage used: " + str(storage_used) + '.'
    }]
    p = p + p_0
    return p


def combination_agent(candidates, indexes, combination_mem):
    sort_rule = {
        "SORT order rule": "If multiple columns in the same ORDER BY clause are indexed, combine them by their sort order."}
    group_rule = {
        "GROUP order rule": "If multiple columns in the same GROUP BY clause are indexed, combine them by their group order."}
    indepedent_rule = {
        "Independent columns rule": "If all the columns only appear in the JOIN conditions, there is no need to combine them."}
    leftmost_rule = {
        "Leftmost Prefix rule": "If composite index (a, b) is constructed and b is also used along, create the single index b."}
    cross_op_rules_1 = {
        "Cross op rule 1": "Combine the indexed columns from the same table by the order of (in selection, in join)."}
    cross_op_rules_2 = {
        "Cross op rule 2": "Combine the indexed columns from the same table by the order of (in join, in selection)."}
    cross_op_rules_3 = {
        "Cross op rule 3": "Combine the indexed columns from the same table by the order of (in order by, in selection, in join)."}
    cross_op_rules_4 = {
        "Cross op rule 4": "Combine the indexed columns from the same table by the order of (in order by, in join, in selection)."}
    cross_op_rules_5 = {
        "Cross op rule 5": "Combine the indexed columns from the same table by the order of (in group by, in selection, in join)."}
    cross_op_rules_6 = {
        "Cross op rule 6": "Combine the indexed columns from the same table by the order of (in group by, in join, in selection)."}
    evidence_rules = {
        'Candidate Evidence Rule': 'Candidate fields are index identity, plan-node types, average plan-row estimates, an aggregated standalone utility-per-storage score, single-column storage, and optional min/max values. Node types are not SQL datatypes. Do not infer exact predicates, shared query membership, ORDER BY or GROUP BY key order, distinct counts or correlation from these fields. Apply operation-based rules only when their prerequisites are explicitly supported. These restrictions take precedence over conflicting heuristic rules.',
        'Same Table Rule': 'Merge only indexes from the same table using their already-selected columns. Do not introduce unrelated columns or repeat a column within a merged index.',
        'Ordered Identity Rule': 'Treat indexes as ordered column sequences. Indexes (a, b) and (b, a) are not interchangeable, and prefix overlap alone does not prove redundancy.',
        'Nonadditive Benefit Rule': 'Do not add constituent utility scores to predict composite utility. Do not present summed single-column sizes as measured composite size or assume fewer indexes necessarily use less storage.',
        'Key Order Evidence Rule': 'Do not claim that ordering columns by utility, row estimates or min/max range determines the optimal composite key order.',
        'Single Merge Rule': 'Propose at most one pairwise merge per call and preserve unrelated indexes. Prefer exploring small combinations before wider ones; this is a search heuristic, not proof of improvement.',
        'Abstention Rule': 'Return the current selection unchanged when supplied evidence does not justify a merge. Shared plan-node types or similar statistics alone are insufficient justification.',
    }
    rules = sort_rule | group_rule | indepedent_rule | leftmost_rule | cross_op_rules_1 | cross_op_rules_2 | cross_op_rules_3 | cross_op_rules_4 | cross_op_rules_5 | cross_op_rules_6 | evidence_rules
    p = [{'role': "system", 'content': 'You are an online SQL index combination agent. You will be given a set of index candidates of '
                                       'queries, the previously adopted combination rules and the list of indexes selected for this workload given the form of '
                                       '"[Table_name1(column_name1), Table_name1(column_name2), '
                                       'Table_name2(column_name3)]". You are required to refine the index '
                                       'selection by merging indexes on the same table into composed indexes if the '
                                       'merged index is more efficient, based on the rules provided below: ' + str(rules) +
                                       '. You should return ONLY the keys of rules you used and the refined index without explanation. DO NOT select '
                                       'new index or remove index. STRICTLY output in the form of "'
                                       '{"rules used": [rule keys], "combined index": [Table_name1(column_name1, column_name2), Table_name2(column_name3)]}". '
          }, ]
    p_0 = [{
        'role': "user",
        'content': "Index candidates: " + str(candidates) + "Adopted rules: " + str(combination_mem) + ". Index selected: " + str(indexes) + "."
    }]
    p = p + p_0
    return p


def reflection_agent(plan_path):
    rules = ["If the last three steps are all 'combination agent', output suggestion 'DO NOT output 'Combination Agent''", "If the last three steps are all 'revision agent', output suggestion 'DO NOT output 'Revision Agent''"]
    p = [{'role': "system", 'content': 'You are an online SQL index selection planning reflection agent. You will be given a '
                                       'planning path of the current index recommendation. You are required to review the planning '
                                       'path and offer suggestions for the next planning step. You should suggest based '
                                       'on the rules provided here: ' + str(rules) +
                                       '.  If no rules fit the current planning step, output "None". You should return ONLY the suggestion without explanation. '
          }, ]
    p_0 = [{
        'role': "user",
        'content': "Planning Path: " + str(plan_path) + "."
    }]
    p = p + p_0
    return p


def reflection_agent_refine(plan_path):
    p = [{'role': "system", 'content': 'You are an online SQL index selection plan reflection agent. You will be given a '
                                       'planning path of one index recommendation in the form of list of dictionary steps: '
                                       '[{"action": LLM_action, "output": LLM_output, "storage": storage_used}]. You are required to refine the planning '
                                       'path following the two rules: '
                                       '1. if more than one consecutive actions have the same "action", "output" and "storage", deduplicate the same actions; '
                                       '2. if an index is recursively selected by the selection action and removed by the revision action, remove the recursion. '
                                       'Output in revised planning steps in the same format: [{"action": LLM_action, "output": LLM_output, "storage": storage_used}]. Return ONLY this list of dictionaries'
          }, ]
    p_0 = [{
        'role': "user",
        'content': "Planning Path: " + str(plan_path) + "."
    }]
    p = p + p_0
    return p


def _decode_dummy_name(name, dummy_indexes):
    """Map a (possibly dummy-renamed) 'Table(col)' name back to its real form for actual SQL execution.
    A no-op when dummy_indexes is falsy or doesn't contain this name (i.e. dummy renaming isn't active)."""
    if dummy_indexes and name in dummy_indexes:
        return dummy_indexes[name]
    return name


def reevaluate_candidates_utility(working_db_id, workload, candidates, selection_mem, dummy_indexes=None):
    """
    Re-evaluate the column_utility (index 3) of every not-yet-selected candidate, assuming the indexes
    already in selection_mem are already created as hypothetical indexes - i.e. replace each remaining
    candidate's static, standalone utility with its MARGINAL utility given the current selection:
    cost(selected indexes only) - cost(selected indexes + this one candidate). Already-selected candidates
    are left untouched. Returns a new list; does not mutate the input candidates list.

    dummy_indexes: the same dummy-name -> real-name mapping reasoning_pipe() receives. When dummy renaming
    is active, `candidates`/`selection_mem` names are dummy names (not real table/column names) - they're
    decoded back to real names only for the eval_index_hypo() calls (which run real SQL), never in what's
    returned, so the LLM still only ever sees dummy names.
    """
    selected_names = [x[0] for x in selection_mem]
    selected_real_names = [_decode_dummy_name(n, dummy_indexes) for n in selected_names]
    selected_ind_output = '[' + ', '.join(selected_real_names) + ']' if selected_real_names else ''
    base_cost = eval_index_hypo(working_db_id, workload, selected_ind_output)
    new_candidates = []
    for cand in candidates:
        if cand[0] in selected_names:
            new_candidates.append(cand)
            continue
        combined_real_names = selected_real_names + [_decode_dummy_name(cand[0], dummy_indexes)]
        combined_ind_output = '[' + ', '.join(combined_real_names) + ']'
        cand_cost = eval_index_hypo(working_db_id, workload, combined_ind_output)
        new_cand = list(cand)
        new_cand[3] = base_cost - cand_cost
        new_candidates.append(new_cand)
    return new_candidates


def reevaluate_selected_utility(working_db_id, workload, selection_mem, dummy_indexes=None):
    """
    Re-evaluate the hypothetical utility of every currently-selected index (selection_mem), assuming all of
    them are already created together: each index's utility becomes its leave-one-out marginal contribution
    - cost(all currently-selected indexes EXCEPT this one) - cost(all currently-selected indexes) - i.e. how
    much removing just this index would hurt, given everything else stays. selection_mem entries only need a
    name in position 0 (their format varies - full candidate rows after a Selection Agent step, or plain
    [name, storage string] pairs after a Revision/Combination step) - this doesn't read or rely on any other
    field. Returns a new [name, 'utility:<value>'] list (names kept in whatever form selection_mem used, i.e.
    dummy names when dummy renaming is active); does not mutate the input.

    dummy_indexes: see reevaluate_candidates_utility - only used to resolve real names for eval_index_hypo().
    """
    selected_names = [x[0] for x in selection_mem]
    selected_real_names = [_decode_dummy_name(n, dummy_indexes) for n in selected_names]
    full_ind_output = '[' + ', '.join(selected_real_names) + ']' if selected_real_names else ''
    full_cost = eval_index_hypo(working_db_id, workload, full_ind_output)
    reevaluated = []
    for i, name in enumerate(selected_names):
        remaining_real_names = selected_real_names[:i] + selected_real_names[i + 1:]
        remaining_ind_output = '[' + ', '.join(remaining_real_names) + ']' if remaining_real_names else ''
        without_cost = eval_index_hypo(working_db_id, workload, remaining_ind_output)
        marginal_utility = without_cost - full_cost
        reevaluated.append([name, 'utility:' + str(marginal_utility)])
    return reevaluated


def reasoning_pipe(working_db_id, schema0, workload, candidates, storage, dummy_indexes, plan_long_mem=None, postprocess=False, dummy=False, iters=10, true_storage=False, no_cand=False, ablation=None):
    """
    ablation: None | 'no_combination' | 'no_revision' | 'reduced_revision_rules'
        'no_combination' / 'no_revision' - forwarded to planning_agent() to drop the corresponding action from
            its prompt (see planning_agent docstring). When ablation == 'no_revision', the hardcoded over-budget
            fallback that would otherwise force a 'Revision Agent' step is disabled too - planning simply stops
            instead.
        'reduced_revision_rules' - ablation study 3: planning_agent keeps both Revision Agent and Combination
            Agent as options, but revision_agent() itself is forwarded this value and strips its rule set down
            to only the Deduplicate Rule and the Shrink Rule (see revision_agent docstring).
    """
    ind_outputs = []
    selection_mem = []
    combination_mem = []
    revision_mem = []
    plan_short_mem = []
    actions_log = []
    storage_used = 0
    startover = False
    suggestion = 'None'
    prompts = []
    i = 0
    start_time = time.time()
    while i <= iters:
        i += 1
        if startover:
            logger.info('Revision emptied the selection, starting over')
            startover = False
            LLM_action = 'Selection Agent'
        elif storage_used >= storage:
            if ablation == 'no_revision':
                logger.info('Storage budget exceeded and Revision Agent disabled by ablation, stopping planning')
                break
            LLM_action = 'Revision Agent'
        else:
            if no_cand:
                p_act = planning_agent_no_cand(schema0, workload, selection_mem, storage_used, storage, plan_short_mem, suggestion)
            else:
                p_act = planning_agent(schema0, candidates, selection_mem, storage_used, storage, plan_short_mem, plan_long_mem, suggestion, ablation)
            LLM_action = query_turbo_model(p_act)
            prompts.append(p_act)
        try:
            if 'Selection Agent' in LLM_action:
                LLM_action = 'Selection Agent'
                logger.info('Step %d | action: %s', i, LLM_action)
                plan_short_mem.append(LLM_action)
                if no_cand:
                    storage_left = storage - storage_used
                    p_sel = selection_agent_nocand(schema0, workload, candidates, selection_mem, storage_left)
                else:
                    # re-evaluate remaining candidates' utility assuming the already-selected indexes are
                    # already created, so the Selection Agent sees marginal (not static/standalone) utility
                    candidates_for_selection = reevaluate_candidates_utility(working_db_id, workload, candidates, selection_mem, dummy_indexes)
                    p_sel = selection_agent(schema0, candidates_for_selection, selection_mem)
                # p_sel = selection_agent(schema0, candidates, selection_mem)
                prompts.append(p_sel)
                output_sel = query_turbo_model(p_sel)
                output_sel = output_sel.split('Index selection: ')[-1]
                if '[]' in output_sel:
                    continue
                elif postprocess:
                    p_post = post_process_agent(schema0, ind_outputs, output_sel)
                    output_postprocess = query_turbo_model(p_post)
                    if 'False' in output_postprocess:
                        logger.info('Post-process agent rejected redundant index: %s', output_sel)
                        continue
                # if no_cand:
                #     output_sel = '[' + output_sel + ']'
                logger.info('New index selected: %s', output_sel)
                pattern = r"\[(.*?)\]"
                matches = re.findall(pattern, output_sel)
                if len(matches[0].split('), ')) != 1:
                    inds_selected = [x + ')' if x[-1] != ')' else x for x in matches[0].split('), ')]
                else:
                    inds_selected = [x for x in matches[0].split('), ')]
                if not no_cand:
                    # new_ind_name = inds_selected[0].split('(')[0] + '.' + inds_selected[0].split('(')[1][:-1]
                    new_ind_name = inds_selected[0]
                    new_ind = [x for x in candidates if x[0] == new_ind_name]
                    selection_mem = selection_mem + new_ind
                else:
                    selection_mem = selection_mem + [inds_selected]
                if true_storage:
                    if dummy:
                        output_sel_o = '[' + dummy_indexes[output_sel[1:-1]] + ']'
                        storage_used = storage_used + get_index_actual_storage(working_db_id, output_sel_o)
                    else:
                        storage_used = storage_used + get_index_actual_storage(working_db_id, output_sel)
                else:
                    if dummy:
                        output_sel_o = '[' + dummy_indexes[output_sel[1:-1]] + ']'
                        storage_used = storage_used + predict_index_storage(output_sel_o, working_db_id)
                    else:
                        storage_used = storage_used + predict_index_storage(output_sel, working_db_id)
                ind_outputs.append(output_sel[1:-1])
                # candidates = [x for x in candidates if x not in new_ind]
                # update candidates
                # candidates = update_cand_nodes(working_db_id, workload, candidates, '[' + ', '.join(ind_outputs) + ']')
                candidates = [x for x in candidates if x[4] <= storage]
            elif 'Revision Agent' in LLM_action:
                LLM_action = 'Revision Agent'
                logger.info('Step %d | action: %s', i, LLM_action)
                plan_short_mem.append(LLM_action)
                # re-evaluate each selected index's hypothetical (leave-one-out marginal) utility given all
                # currently-selected indexes are already created, so the Revision Agent sees fresh signal
                selection_mem_for_revision = reevaluate_selected_utility(working_db_id, workload, selection_mem, dummy_indexes)
                p_rev = revision_agent(schema0, selection_mem_for_revision, storage, storage_used, revision_mem, ablation)
                llm_output_rev = query_turbo_model(p_rev)

                logger.debug('Revision Agent raw output: %s', llm_output_rev)
                llm_output_rev = ast.literal_eval(llm_output_rev)
                revision_mem.append(llm_output_rev['rules used'])
                revised_ind = str(llm_output_rev['revised index'])
                output_rev = revised_ind.split('Index selection: ')[-1]

                prompts.append(p_rev)
                output_rev = output_rev.replace("'", "")
                logger.info('Revised selection (rules %s): %s', llm_output_rev['rules used'], output_rev)
                if '[]' in output_rev:
                    startover = True
                    selection_mem = []
                    ind_outputs = []
                    continue
                pattern = r"\[(.*?)\]"
                matches = re.findall(pattern, output_rev)
                if len(matches[0].split('), ')) != 1:
                    inds_selected = [x + ')' if x[-1] != ')' else x for x in matches[0].split('), ')]
                else:
                    inds_selected = [x for x in matches[0].split('), ')]
                if dummy:
                    selection_mem = [[x, 'storage:' + str(get_index_actual_storage(working_db_id, '[' + dummy_indexes[x] + ']'))] for x in inds_selected]
                    output_sel_o = '[' + ', '.join([dummy_indexes[x] for x in inds_selected]) + ']'
                    if true_storage:
                        storage_used = get_index_actual_storage(working_db_id, output_sel_o)
                    else:
                        storage_used = predict_index_storage(output_sel_o, working_db_id)
                else:
                    selection_mem = [[x, 'storage:' + str(get_index_actual_storage(working_db_id, '[' + x + ']'))] for x in inds_selected]
                    if true_storage:
                        storage_used = get_index_actual_storage(working_db_id, output_rev)
                    else:
                        storage_used = predict_index_storage(output_rev, working_db_id)
                ind_outputs = inds_selected
            elif 'Combination Agent' in LLM_action:
                LLM_action = 'Combination Agent'
                logger.info('Step %d | action: %s', i, LLM_action)
                plan_short_mem.append(LLM_action)
                p_com = combination_agent(candidates, selection_mem, combination_mem)
                llm_output_com = query_turbo_model(p_com)

                logger.debug('Combination Agent raw output: %s', llm_output_com)
                llm_output_com = ast.literal_eval(llm_output_com)
                combination_mem.append(llm_output_com['rules used'])
                combined_ind = str(llm_output_com['combined index'])
                output_com = combined_ind.split('Index selection: ')[-1]
                prompts.append(p_com)
                output_com = output_com.replace("'", "")
                logger.info('Combined selection (rules %s): %s', llm_output_com['rules used'], output_com)
                pattern = r"\[(.*?)\]"
                matches = re.findall(pattern, output_com)
                if len(matches[0].split('), ')) != 1:
                    inds_selected = [x + ')' if x[-1] != ')' else x for x in matches[0].split('), ')]
                else:
                    inds_selected = [x for x in matches[0].split('), ')]
                if dummy:
                    selection_mem = [[x, 'storage:' + str(get_index_actual_storage(working_db_id, '[' + dummy_indexes[x] + ']'))] for x in inds_selected]
                    output_com_o = '[' + ', '.join([dummy_indexes[x] for x in inds_selected]) + ']'
                    if true_storage:
                        storage_used = get_index_actual_storage(working_db_id, output_com_o)
                    else:
                        storage_used = predict_index_storage(output_com_o, working_db_id)
                else:
                    selection_mem = [[x, 'storage:' + str(get_index_actual_storage(working_db_id, '[' + x + ']'))] for x in inds_selected]
                    if true_storage:
                        storage_used = get_index_actual_storage(working_db_id, output_com)
                    else:
                        storage_used = predict_index_storage(output_com, working_db_id)
                ind_outputs = inds_selected
            elif 'Stop' in LLM_action:
                logger.info('Step %d | action: %s', i, LLM_action)
                break
            p_ref = reflection_agent(plan_short_mem)
            suggestion = query_turbo_model(p_ref)
            logger.info('Reflection Agent suggestion: %s', suggestion)
            logger.info('Current indexes: %s | storage used %s / %s | elapsed %.1fs',
                        ind_outputs, storage_used, storage, time.time() - start_time)
            outputs = copy.deepcopy(ind_outputs)
            actions_log.append({'action': LLM_action, 'output': outputs, 'storage_used': storage_used})
        except Exception as e:
            logger.warning('Planning error at step %d: %s', i, e)
            continue
    ind_outputs = '[' + ', '.join(ind_outputs) + ']'
    ind_outputs = ind_outputs.replace("'", "")
    p_actions_log = reflection_agent_refine(actions_log)
    refined_actions_log = query_turbo_model(p_actions_log)
    refined_actions_log = ast.literal_eval(refined_actions_log)
    logger.info('Refined planning path: %s', refined_actions_log)
    for j in range(len(refined_actions_log)):
        refined_actions_log[j]['storage budget'] = storage
        del refined_actions_log[j]['output']
    return ind_outputs, storage_used, actions_log, refined_actions_log


def get_plan_long_memory(storage, mem_dir, input_emb, k=1):
    index = faiss.IndexFlatL2(1024)   # build the index
    index.reset()
    with open(mem_dir, 'r') as f_mem:
        long_memories = json.load(f_mem)
    indexes = []
    long_memories_box = [x for x in long_memories if x['storage_index'] == storage]
    for ind in long_memories_box:
        emb_ind = ind['emb_index']
        indexes.append(emb_ind)                  # add vectors to the index
    indexes = np.array(indexes).astype('float32')
    index.add(indexes)
    logger.debug('Long-term memory entries indexed: %d', index.ntotal)
    D, I = index.search(input_emb, k)
    mem_inds = I[0]
    mem_embs = [indexes[x].tolist() for x in mem_inds]
    memories = [x for x in long_memories_box if x['emb_index'] in mem_embs]
    return [x['planning_path'] for x in memories]


def reasoning_pipe_one_path(working_db_id, schema0, workload, candidates, storage):
    ind_outputs = []
    actions_log = []
    storage_used = 0
    selection_mem = []
    combination_mem = []
    revision_mem = []
    startover = False
    start_time = time.time()
    p_act = planning_agent_path(schema0, candidates, storage)
    LLM_action_path = query_turbo_model(p_act)
    LLM_action_path = ast.literal_eval(LLM_action_path)
    logger.info('Planned action path: %s', LLM_action_path)
    for step, act in enumerate(LLM_action_path, 1):
        try:
            if 'Selection Agent' in act:
                act = 'Selection Agent'
                logger.info('Step %d | action: %s', step, act)
                p_sel = selection_agent(schema0, candidates, selection_mem)
                output_sel = query_turbo_model(p_sel)
                output_sel = output_sel.split('Index selection: ')[-1]
                if '[]' in output_sel:
                    continue
                logger.info('New index selected: %s', output_sel)
                pattern = r"\[(.*?)\]"
                matches = re.findall(pattern, output_sel)
                if len(matches[0].split('), ')) != 1:
                    inds_selected = [x + ')' if x[-1] != ')' else x for x in matches[0].split('), ')]
                else:
                    inds_selected = [x for x in matches[0].split('), ')]
                selection_mem = selection_mem + [inds_selected]
                storage_used = storage_used + predict_index_storage(output_sel, working_db_id)
                ind_outputs.append(output_sel[1:-1])
                candidates = [x for x in candidates if x[4] <= storage]
            elif 'Revision Agent' in act:
                act = 'Revision Agent'
                logger.info('Step %d | action: %s', step, act)
                p_rev = revision_agent(schema0, selection_mem, storage, storage_used, revision_mem)
                llm_output_rev = query_turbo_model(p_rev)

                logger.debug('Revision Agent raw output: %s', llm_output_rev)
                llm_output_rev = ast.literal_eval(llm_output_rev)
                revision_mem.append(llm_output_rev['rules used'])
                revised_ind = str(llm_output_rev['revised index'])
                output_rev = revised_ind.split('Index selection: ')[-1]
                output_rev = output_rev.replace("'", "")
                logger.info('Revised selection (rules %s): %s', llm_output_rev['rules used'], output_rev)
                pattern = r"\[(.*?)\]"
                matches = re.findall(pattern, output_rev)
                if len(matches[0].split('), ')) != 1:
                    inds_selected = [x + ')' if x[-1] != ')' else x for x in matches[0].split('), ')]
                else:
                    inds_selected = [x for x in matches[0].split('), ')]
                storage_used = predict_index_storage(output_rev, working_db_id)
                ind_outputs = inds_selected
            elif 'Combination Agent' in act:
                act = 'Combination Agent'
                logger.info('Step %d | action: %s', step, act)
                p_com = combination_agent(candidates, selection_mem, combination_mem)
                llm_output_com = query_turbo_model(p_com)
                logger.debug('Combination Agent raw output: %s', llm_output_com)
                llm_output_com = ast.literal_eval(llm_output_com)
                combination_mem.append(llm_output_com['rules used'])
                combined_ind = str(llm_output_com['combined index'])
                output_com = combined_ind.split('Index selection: ')[-1]
                output_com = output_com.replace("'", "")
                logger.info('Combined selection (rules %s): %s', llm_output_com['rules used'], output_com)
                pattern = r"\[(.*?)\]"
                matches = re.findall(pattern, output_com)
                if len(matches[0].split('), ')) != 1:
                    inds_selected = [x + ')' if x[-1] != ')' else x for x in matches[0].split('), ')]
                else:
                    inds_selected = [x for x in matches[0].split('), ')]
                storage_used = predict_index_storage(output_com, working_db_id)
                ind_outputs = inds_selected
            elif 'Stop' in act:
                logger.info('Step %d | action: %s', step, act)
                break
            logger.info('Current indexes: %s | storage used %s / %s | elapsed %.1fs',
                        ind_outputs, storage_used, storage, time.time() - start_time)
            outputs = copy.deepcopy(ind_outputs)
            actions_log.append({'action': act, 'output': outputs, 'storage_used': storage_used})
        except Exception as e:
            logger.warning('Planning error at step %d: %s', step, e)
            continue
    ind_outputs = '[' + ', '.join(ind_outputs) + ']'
    ind_outputs = ind_outputs.replace("'", "")
    return ind_outputs, storage_used, actions_log

