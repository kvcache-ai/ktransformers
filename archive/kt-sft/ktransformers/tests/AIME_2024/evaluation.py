# reference: https://github.com/declare-lab/instruct-eval/blob/main/human_eval/main.py#L35
def filter_answer(completion: str) -> str:
    # the answer is the last part of the completion, it's a int64 number
    # get the last line
    completion = completion.strip().split("\n")[-1]
    # handle the $\\boxed{...}$ format
    if "$\\boxed{" in completion:
        return completion.split("}")[0].split("{")[-1]
    return completion.split()[-1]

