import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
from lm_eval.evaluator import Evaluator

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task_name",
        type=str,
        help="Task to evaluate on, can be a single task",
    )
    parser.add_argument(
        "--load_generations_path",
        type=str,
        help="Path of file with previously generated solutions, if provided generation is skipped and only evaluation is done",
    )
    parser.add_argument(
        "--metric_output_path",
        type=str,
        help="Path to save the results",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args.allow_code_execution = True
    args.postprocessed_output_path = None
    args.check_references = False

    results = {}
    evaluator = Evaluator(args)
    results["results"] = evaluator.evaluate(args.task_name)
    results["load_generations_path"] = args.load_generations_path

    dumped = json.dumps(results)
    print(dumped)
    with open(args.metric_output_path, "a+") as f:
        f.write(dumped+"\n")

