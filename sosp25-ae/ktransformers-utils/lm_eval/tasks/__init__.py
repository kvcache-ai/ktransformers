import inspect
from pprint import pprint
from . import (mbpp, human_eval, strategy_qa, gsm8k)

TASK_REGISTRY = {
    "mbpp": mbpp.MBPP,
    "strategy_qa": strategy_qa.StrategyQA,
    "human_eval": human_eval.HumanEval,
    'gsm8k': gsm8k.GSM8K,
}

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name, args=None):
    try:
        kwargs = {}
        if "postprocessed_output_path" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
            kwargs["postprocessed_output_path"] = args.postprocessed_output_path
        return TASK_REGISTRY[task_name](**kwargs)
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
