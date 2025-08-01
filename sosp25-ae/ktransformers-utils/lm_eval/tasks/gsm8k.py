import re
from evaluate import load
from lm_eval.base import Task
import multiprocessing
from math import isclose
from typing import Union
from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
import os

class GSM8K(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = os.path.dirname(__file__) + "/../../data/gsm8k"

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
            len(dataset) == 1319
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
        completion = generation[0].strip()
        if "\n\nQuestion:" in completion:
            completion = completion.split("\n\nQuestion:")[0]
        return extract_answer(completion)


    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        generations = [self.postprocess_generation(_) for _ in generations]
        results = [math_equal(references[_], generations[_]) for _ in range(len(references))]
        return len([_ for _ in results if _]) / len(results)


def extract_answer(pred_str):
    if ('he answer is' in pred_str):
        pred = pred_str.split('he answer is')[-1].strip()
        if not is_digit(pred):
            pred_str = pred
            pattern = '-?\d*\.?\d+'
            pred = re.findall(pattern, pred_str.replace(",", ""))
            if(len(pred) >= 1):
                pred = pred[-1]
            else: 
                pred = ''
    else: # use the last number
        pattern = '-?\d*\.?\d+'
        pred = re.findall(pattern, pred_str.replace(",", ""))
        if(len(pred) >= 1):
            pred = pred[-1]
        else: 
            pred = ''
    
    # multiple line
    pred = pred.split("\n")[0]
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_string(pred)
    return pred

def strip_string(string):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    string = string.replace("\\!", "")
    string = string.replace("\\ ", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    string = string.replace("\\text", "")
    string = string.replace("x\\in", "")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    string = string.replace("\\cdot", "")

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and 
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace("\"", "")
    
    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string

def is_digit(s):
    try:
        float(str(s).replace(",", ""))
        return True
    except ValueError:
        return False

def math_equal(prediction: Union[bool, float, str],
                reference: Union[float, str],
                include_percentage: bool = True,
                is_close: bool = True,
                timeout: bool = False,
                ) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    try: # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            prediction = float(str(prediction).replace(",", ""))
            reference = float(str(reference).replace(",", ""))
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if isclose(item, prediction, rel_tol=1e-4):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    ## deal with [], (), {}
    pred_str, ref_str = prediction, reference
    if (prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")) or \
        (prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ['{', "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str == ref_str:
        return True

    ## [a, b] vs. [c, d], return a==c and b==d
    if (prediction.startswith("[") and prediction.endswith("]")) and (reference.startswith("[") and reference.endswith("]")) or \
        (prediction.startswith("(") and prediction.endswith(")")) and (reference.startswith("(") and reference.endswith(")")):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all([math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close) for i in range(len(pred_parts))]):
                return True

    # symbolic equal with sympy
    if timeout:
        if call_with_timeout(symbolic_equal_process, prediction, reference):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


def math_equal_process(param):
    return math_equal(param[-2], param[-1])


def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr]:
            try:
                return f(s)
            except:
                pass
        return s
    a = _parse(a)
    b = _parse(b)

    try:
        if simplify(a-b) == 0:
            return True
    except:
        pass

    try:
        if isclose(N(a), N(b), rel_tol=1e-3):
            return True
    except:
        pass
    return False


def symbolic_equal_process(a, b, output_queue):  
    result = symbolic_equal(a, b)
    output_queue.put(result)  


def call_with_timeout(func, *args, timeout=1, **kwargs):  
    output_queue = multiprocessing.Queue()  
    process_args = args + (output_queue,)  
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)  
    process.start()  
    process.join(timeout)  
  
    if process.is_alive():  
        process.terminate()  
        process.join()  
        return False  
  
    return output_queue.get()