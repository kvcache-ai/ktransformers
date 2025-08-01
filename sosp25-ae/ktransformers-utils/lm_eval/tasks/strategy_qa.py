import re
from evaluate import load
from lm_eval.base import Task
import pandas as pd
import os

class StrategyQA(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = os.path.dirname(__file__) + "/../../data/strategy_qa"

    def __init__(self, postprocessed_output_path):
        self.postprocessed_output_path = postprocessed_output_path
        super().__init__(
            requires_execution=True,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        dataset = self.dataset["test"]
        # the wrong split of commonsense_qa can be loaded with old datasets cache
        assert (
            len(dataset) == 2286
        ), "please ensure you have the latest version of commonsense_qa dataset, try deleting its old cache"
        return dataset

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return "".join(doc["label"])

    def postprocess_generation(self, generation):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        answer_key = "None"
        answer_begin_hints = ["answer is", "answer to", "answer choice is", "i would choose", "answer would be", "answer seems to be", "the correct answer is", "answer to the question is"]
        answer_end_hints = ["is correct", "is the best choice", "is the correct answer", "is the correct choice", "answer choice is correct"]
        completion = generation[0].lower().strip()
        if "\n\n" in completion:
            completion = completion.split("\n\n")[0]
            
        matched_begin_hints = [_ for _ in answer_begin_hints if _ in completion]
        matched_end_hints = [_ for _ in answer_end_hints if _ in completion]
        if len(matched_begin_hints) > 0:
            completion = completion.split(matched_begin_hints[-1])[-1]
        elif len(matched_end_hints) > 0:
            completion = completion.split(matched_end_hints[-1])[0]
        pattern = r'( yes | no )'
        completion = " " + completion.replace(".", " ").replace(",", " ").replace(";", " ") + " "
        matches = re.findall(pattern, completion)

        if matches:
            return matches[-1].strip()
        else:
            # print("=" * 25 + "No clear yes or no results" + "=" * 25 )
            # if "\n\n" in generation[0].strip():
            #     print(generation[0].strip().split("\n\n")[0])
            # else:
            #     print(generation[0].strip())
            # print("=" * 50 + "=" * len("No clear yes or no results") + "\n")
            return "no"


    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        generations = [self.postprocess_generation(_) for _ in generations]
        if self.postprocessed_output_path:
            postprocessed_output = pd.DataFrame()
            postprocessed_output['results'] = generations
            postprocessed_output.to_json(self.postprocessed_output_path, orient='records', lines=True)
        cnt = 0
        for i in range(len(generations)):
            if generations[i] == "None":
                cnt += 1
        acc_metric = load("exact_match")
        results = acc_metric.compute(
            references=references,
            predictions=generations,
        ) 
        results["match_template"] = 1 - cnt / len(generations)
        return results['exact_match']
