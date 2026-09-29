# @TIME : 12/3/25 4:24 PM
# @AUTHOR : LZDH
import logging
import os.path
import re
import time

import pandas as pd

from get_postgre import *
from get_query_meta import *
import ast
import multiprocessing
import numpy as np
import psycopg2

logger = logging.getLogger(__name__)


# Tracks how many times eval_index_hypo() is called (initial per-column candidate evals in
# index_graph.get_cand_nodes, the no-index baseline eval, the final selected-index eval, and any other
# caller) so the driver scripts can report it as a results statistic. Exposed via getter/reset functions
# rather than a bare module attribute so `from eval_index import *` callers always see the live count.
_eval_index_hypo_call_count = 0


def get_eval_index_hypo_call_count():
    return _eval_index_hypo_call_count


def reset_eval_index_hypo_call_count():
    global _eval_index_hypo_call_count
    _eval_index_hypo_call_count = 0


def eval_query_time(db_id, input, queue):
    # Connect to the PostgreSQL database
    pg_conn = psycopg2.connect(
        host="localhost",
        port="5432",
        database=db_id,
        user="postgres",
        password="your_password"
    )
    # disable_query_cache(pg_conn)
    pg_cursor = pg_conn.cursor()
    clr_q = "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE pid <> pg_backend_pid() AND datname = '" + db_id + "';"
    pg_cursor.execute(clr_q)
    start_time = time.time()
    pg_cursor.execute(input)
    # results = pg_cursor.fetchall()
    pg_conn.commit()
    end_time = time.time()
    execution_time = end_time - start_time
    pg_cursor.close()
    pg_conn.close()
    # return execution_time
    queue.put(execution_time)


def eval_index_hypo(db_id, workload, ind_output):
    global _eval_index_hypo_call_count
    _eval_index_hypo_call_count += 1
    if ind_output and 'I(C' not in ind_output:
        pattern = r"\[(.*?)\]"
        matches = re.findall(pattern, ind_output)
        if len(matches[0].split('), ')) != 1:
            inds_selected = [x + ')' if x[-1] != ')' else x for x in matches[0].split('), ')]
        else:
            inds_selected = [x for x in matches[0].split('), ')]
    elif ind_output and 'I(C' in ind_output:
        pattern = r"\((.*?)\)"
        matches = re.findall(pattern, ind_output)
        cols_in_ind = [x.replace('C ', '').split(',') for x in matches]
        inds_selected = []
        for i in cols_in_ind:
            tab = i[0].split('.')[0]
            cols = [x.split('.')[1] for x in i]
            ind = tab + '(' + ', '.join(cols) + ')'
            inds_selected.append(ind)
    else:
        inds_selected = []
    pg_conn = psycopg2.connect(
        host="localhost",
        port="5432",
        database=db_id,
        user="postgres",
        password="your_password"
    )
    pg_cursor = pg_conn.cursor()
    clr_q = "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE pid <> pg_backend_pid() AND datname = '" + db_id + "';"
    pg_cursor.execute(clr_q)
    reset = 'SELECT hypopg_reset();'
    pg_cursor.execute(reset)
    if inds_selected:
        for ind in inds_selected:
            tab = ind.split('(')[0]
            columns = '(' + ind.split('(')[1]
            create_index = "SELECT hypopg_create_index('CREATE INDEX ON " + tab + columns + "');"
            pg_cursor.execute(create_index)
            pg_conn.commit()
    total_est_time = 0
    for query in workload:
        try:
            pg_cursor.execute('EXPLAIN (FORMAT JSON) ' + query)
            results = pg_cursor.fetchall()
            plan = results[0][0][0]['Plan']
            est_time = float(plan['Total Cost'])
        except Exception as error1:
            logger.warning('EXPLAIN failed, counting cost 0: %s', error1)
            est_time = 0
        total_est_time = total_est_time + est_time
    pg_cursor.execute(reset)
    pg_conn.commit()
    pg_cursor.close()
    pg_conn.close()
    return total_est_time


def eval_index_run(db_id, workload, ind_output, log_dir, type='llm', iters=6):
    if ind_output and 'I(C' not in ind_output:
        pattern = r"\[(.*?)\]"
        matches = re.findall(pattern, ind_output)
        if len(matches[0].split('), ')) != 1:
            inds_selected = [x + ')' if x[-1] != ')' else x for x in matches[0].split('), ')]
        else:
            inds_selected = [x for x in matches[0].split('), ')]
    elif ind_output and 'I(C' in ind_output:
        pattern = r"\((.*?)\)"
        matches = re.findall(pattern, ind_output)
        cols_in_ind = [x.replace('C ', '').split(',') for x in matches]
        inds_selected = []
        for i in cols_in_ind:
            tab = i[0].split('.')[0]
            cols = [x.split('.')[1] for x in i]
            ind = tab + '(' + ', '.join(cols) + ')'
            inds_selected.append(ind)
    else:
        inds_selected = []
    pg_conn = psycopg2.connect(
        host="localhost",
        port="5432",
        database=db_id,
        user="postgres",
        password="your_password"
    )
    pg_cursor = pg_conn.cursor()
    clr_q = "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE pid <> pg_backend_pid() AND datname = '" + db_id + "';"
    pg_cursor.execute(clr_q)
    ind_names = []
    if inds_selected:
        tables = []
        for ind in inds_selected:
            try:
                tab = ind.split('(')[0]
                tables.append(tab)
                columns = '(' + ind.split('(')[1]
                ind_name = 'idx_' + tab + '_' + '_'.join([x for x in columns[1:-1].split(', ')])
                reset = 'DROP INDEX if exists ' + ind_name + ';'
                pg_cursor.execute(reset)
                create_index = "CREATE INDEX if not exists " + ind_name + " ON " + tab + columns + ";"
                pg_cursor.execute(create_index)
                pg_conn.commit()
                ind_names.append(ind_name)
            except Exception as e_ind:
                logger.warning('Creating index %s failed: %s', ind, e_ind)
                # break
                continue
    pg_cursor.close()
    pg_conn.close()
    total_time = 0
    queries = []
    times = []
    for query in workload:
        queries.append(query)
        try:
            act_times = []
            for i in range(iters):
                queue = multiprocessing.Queue()
                p = multiprocessing.Process(target=eval_query_time, args=(db_id, query, queue))
                p.start()
                # Set a timeout value
                timeout = 240
                # Wait for the function to finish or timeout
                p.join(timeout)
                # Check if the process is still running
                if p.is_alive():
                    logger.warning('Query timed out after %ss, recording 500s per run', timeout)
                    act_times = [500, 500, 500, 500, 500, 500]
                    p.terminate()
                    p.join()
                    break
                # Rest of your code here
                else:
                    execution_time = queue.get()
                # run_time = eval_query_time(db_id, query)
                act_times.append(execution_time)
            logger.debug('Query runtimes (min/max trimmed): %s', sorted(act_times)[1:-1])
            act_time = np.average(sorted(act_times)[1:-1])
        except Exception as error1:
            logger.warning('Timing query failed, counting runtime 0: %s', error1)
            act_time = 0
        total_time = total_time + act_time
        times.append(act_time)
    log_f = log_dir + 'time_log.csv'
    if os.path.isfile(log_f):
        df = pd.read_csv(log_f)
        df1 = pd.DataFrame({type: times})
        df = pd.concat([df, df1], axis=1)
    else:
        df = pd.DataFrame({'Query_text': queries, type: times})
    df.to_csv(log_f)
    pg_conn = psycopg2.connect(
        host="localhost",
        port="5432",
        database=db_id,
        user="postgres",
        password="your_password"
    )
    pg_cursor = pg_conn.cursor()
    clr_q = "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE pid <> pg_backend_pid() AND datname = '" + db_id + "';"
    pg_cursor.execute(clr_q)
    for i in ind_names:
        try:
            reset = 'DROP INDEX ' + i + ';'
            pg_cursor.execute(reset)
            pg_conn.commit()
        except:
            continue
    pg_cursor.close()
    pg_conn.close()
    return total_time


def get_index_actual_storage(db_id, ind_output):
    if ind_output and 'I(C' not in ind_output:
        pattern = r"\[(.*?)\]"
        matches = re.findall(pattern, ind_output)
        if len(matches[0].split('), ')) != 1:
            inds_selected = [x + ')' if x[-1] != ')' else x for x in matches[0].split('), ')]
        else:
            inds_selected = [x for x in matches[0].split('), ')]
    elif ind_output and 'I(C' in ind_output:
        pattern = r"\((.*?)\)"
        matches = re.findall(pattern, ind_output)
        cols_in_ind = [x.replace('C ', '').split(',') for x in matches]
        inds_selected = []
        for i in cols_in_ind:
            tab = i[0].split('.')[0]
            cols = [x.split('.')[1] for x in i]
            ind = tab + '(' + ', '.join(cols) + ')'
            inds_selected.append(ind)
    else:
        inds_selected = []
    pg_conn = psycopg2.connect(
            host="localhost",
            port="5432",
            database=db_id,
            user="postgres",
            password="your_password"
        )
    pg_cursor = pg_conn.cursor()
    clr_q = "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE pid <> pg_backend_pid() AND datname = '" + db_id + "';"
    pg_cursor.execute(clr_q)
    tables = []
    ind_names = []
    for ind in inds_selected:
        try:
            tab = ind.split('(')[0]
            tables.append(tab)
            columns = '(' + ind.split('(')[1]
            ind_name = 'idx_' + tab + '_' + '_'.join([x for x in columns[1:-1].split(', ')])
            reset = 'DROP INDEX if exists ' + ind_name + ';'
            pg_cursor.execute(reset)
            create_index = "CREATE INDEX if not exists " + ind_name + " ON " + tab + columns + ";"
            pg_cursor.execute(create_index)
            pg_conn.commit()
            ind_names.append(ind_name)
        except Exception as e_ind:
            logger.warning('Creating index %s failed: %s', ind, e_ind)
            continue
    storage_mb = 0
    tables = list(set(tables))
    for t in tables:
        q_t = f"SELECT pg_size_pretty(pg_indexes_size('{t}'));"
        pg_cursor.execute(q_t)
        data = pg_cursor.fetchall()
        storage_raw_t = data[0][0]
        if 'kB' in storage_raw_t:
            storage_raw_t = float(storage_raw_t.split()[0]) / 1024
        elif 'MB' in storage_raw_t:
            storage_raw_t = float(storage_raw_t.split()[0])
        elif 'GB' in storage_raw_t:
            storage_raw_t = float(storage_raw_t.split()[0]) * 1024
        elif 'TB' in storage_raw_t:
            storage_raw_t = float(storage_raw_t.split()[0]) * 1024 * 1024
        storage_mb = storage_mb + storage_raw_t
        pg_conn.commit()
    for i in ind_names:
        try:
            reset = 'DROP INDEX ' + i + ';'
            pg_cursor.execute(reset)
            pg_conn.commit()
        except:
            continue
    pg_cursor.close()
    pg_conn.close()
    return storage_mb
