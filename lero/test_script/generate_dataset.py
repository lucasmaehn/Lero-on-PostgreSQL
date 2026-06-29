from os.path import join
import os
from config import *
from utils import *

import argparse

LERO_CARD_FILEPATH = join(LERO_SERVER_PATH, LERO_DUMP_CARD_FILE)
ITER=5


def get_plan(q, run_args):
    _, plan_json = run_query("EXPLAIN (COSTS FALSE, FORMAT JSON, SUMMARY) " + q, run_args)
    plan_json = plan_json[0][0]
    return plan_json[0]['Plan']

def run_plan(q, run_args):
    t, plan_json = run_query("EXPLAIN (ANALYZE, TIMING, VERBOSE, COSTS, SUMMARY, FORMAT JSON) " + q, run_args)
    plan_json = plan_json[0][0][0]
    plan_json["pytime"] = t
    print("query execution took", t)
    return plan_json



def get_card_test_args(card_file_name):
    run_args = ["SET enable_lero TO True"]
    run_args.append("SET lero_joinest_fname TO '" + card_file_name + "'")
    return run_args


def generate_dataset(query_filepath, output_filepath):
    with open(query_filepath) as f:
        for q in f.readlines():
            qsplit = q.split(SEP)
            explain_query(qsplit[1],["SET enable_lero TO True"])
            results = generate_plans(qsplit[1], qsplit[0])
            os.makedirs(f"{output_filepath}/{qsplit[0]}", exist_ok=True)
            for plan, execs in results.items():
                plan_filepath = f"{output_filepath}/{qsplit[0]}/{encode_str(plan)}"
                with open(plan_filepath, "w") as writer:
                    for plan_exec in execs:
                        writer.write(json.dumps(plan_exec) + "\n")


def generate_plans(query, name):
    results = {}
    with open(LERO_CARD_FILEPATH, 'r') as f:
        cards = [line.strip().split(";")[0] for line in f.readlines()]
        i = 0
        for card in cards:
            card_str = "\n".join(card.strip().split(" "))
            # ensure that the cardinality file will not be changed during planning
            card_file_name = "lero_" + name + "_" + str(i) + ".txt"
            card_file_path = join(PG_DB_PATH, card_file_name)
            with open(card_file_path, "w") as card_file:
                card_file.write(card_str)

            run_args = get_card_test_args(card_file_name)
            plan = get_plan(query, run_args)

            execs = []
            for _ in range(ITER):
                execs.append(run_plan(query, run_args))

            results[json.dumps(plan)] = execs
            i+=1

    pq_plan = get_plan(query,[])
    
    for x in results[json.dumps(pq_plan)]:
        x["__is_default__"] = True

    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Dataset Generator")
    parser.add_argument("--dataset",
                        metavar="PATH",
                        help="path to the dataset")
    parser.add_argument("--output_file",
                        metavar="PATH",
                        help="path to the dataset")

    parser.add_argument("--iterations", type=int, default=10)

    args = parser.parse_args()

    ITER=args.iterations

    generate_dataset(args.dataset, args.output_file)
