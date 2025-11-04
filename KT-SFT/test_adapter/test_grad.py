import torch, glob

records = sorted(glob.glob("/home/lpl/KT-SFT/tmp/train_logs/step_*.pt"))
example = torch.load(records[1])

# print("step:", example["step"])
# print("inputs keys:", list(example["inputs"].keys()))
# print("loss:", example["loss"])


# print("param 'base_model.model.model.orig_module.layers.1.mlp.orig_module.gate.weight' 形状:",
#       example["params"]["base_model.model.model.orig_module.layers.1.mlp.orig_module.gate.weight"].shape)
# print("grad 'base_model.model.model.orig_module.layers.1.mlp.orig_module.gate.weight':", example["grads"]["base_model.model.model.orig_module.layers.1.mlp.orig_module.gate.weight"])

print(example)
