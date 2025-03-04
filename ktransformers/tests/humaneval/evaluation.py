# reference: https://github.com/declare-lab/instruct-eval/blob/main/human_eval/main.py#L35
def filter_code(completion: str) -> str:
    # The program tends to overwrite, we only take the first function
    completion = completion.lstrip("\n")
    # we also remove ```python\n and ```
    completion = completion.replace("```python\n", "").replace("```", "")
    if 'if __name__ == "__main__":' in completion:
        completion = completion.split('if __name__ == "__main__":')[0]
    if "# Example usage" in completion:
        completion = completion.split("# Example usage")[0]
    return completion.split("\n\n")[0]


def fix_indents(text: str) -> str:
    return text.replace("\t", "    ")
