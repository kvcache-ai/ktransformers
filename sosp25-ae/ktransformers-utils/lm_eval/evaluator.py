import inspect
import json
import os

from lm_eval import tasks

_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The "code_eval"/"apps_metric" you are about to use, execute untrusted 
model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).
Once you have read this disclaimer and taken appropriate precautions, set the argument 
"allow_code_execution" to True.
################################################################################\
"""


class Evaluator:
    def __init__(self, args):
        self.args = args

        # setup arguments
        self.metric_output_path = args.metric_output_path

        # code evaluation permission
        self.allow_code_execution = args.allow_code_execution

    def get_generate_text(self, task_name):
        task = tasks.get_task(task_name, self.args)
        dataset = task.get_dataset()
        # if args.limit is None, use all samples
        n_tasks = len(dataset)
        references = [task.get_reference(dataset[i]) for i in range(n_tasks)]

        if self.args.check_references:
            if "get_solution" in inspect.signature(task.get_reference).parameters:
                solutions = [[task.get_reference(dataset[i], get_solution=True)] for i in range(n_tasks)]
            else:
                solutions = [[ref] for ref in references]
            return solutions, references
        if self.args.load_generations_path:
            generations = []
            with open(self.args.load_generations_path) as fp:
                for line in fp:
                    json_obj = json.loads(line)
                    generations.append(json_obj)
                print(
                    f"generations loaded, {n_tasks} selected from {len(generations)}."
                )
        # generations = generations[:n_tasks]
        generations = [[_['completion']] for _ in generations]
        references = [references[i % len(references)] for i in range(len(generations))]

        return generations, references

    def evaluate(self, task_name):
        task = tasks.get_task(task_name, self.args)
        if task.requires_execution and not self.allow_code_execution:
            raise ValueError(_WARNING)

        generations, references = self.get_generate_text(task_name)

            # make sure tokenizer plays nice with multiprocessing
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if self.allow_code_execution and task.requires_execution:
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        print("Evaluating generations...")
        results = task.process_results(generations, references)
        return results
