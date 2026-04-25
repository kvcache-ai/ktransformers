def instruct_prompt(prompt: str) -> str:
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nSolve the following math problem without any tests or explanation only one answer surrounede by '$\\boxed{{}}$'\n{prompt}\n\n### Response:"""
