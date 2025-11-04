from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, inject_adapter_in_model, TaskType
from transformers import Trainer, TrainingArguments
from datasets import Dataset, load_dataset
import transformers
from transformers.trainer import TRAINING_ARGS_NAME
import os
import torch
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
from torchviz import make_dot

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained('/home/yj/ktransformers/DeepSeek-V2-Lite-Chat', trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained('/data/model/Qwen2.5-7B-Instruct', trust_remote_code=True)
save_path = '/home/yj/ktransformers/tmp/Qwen_Lora_model'
data_file = '/home/yj/ktransformers/test_adapter/sft_translation.json'

dataset = Dataset.from_json(data_file)

def preprocess_function(examples):
    inputs = examples["input"]
    targets = examples["output"]
    
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(targets, padding="max_length", truncation=True, max_length=512)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def print_model_with_params(model, prefix="", max_layers=3, max_params=5):
    print(f"\n{prefix}模型结构:")
    print(model)  # 原始结构打印
    
    print(f"\n{prefix}参数示例:")
    total_params = 0
    for name, param in model.named_parameters():
        if total_params >= max_layers:  # 控制打印层数
            break
        # 过滤非LoRA相关参数（可根据需要调整）
        if "lora" not in name and "embed" not in name and "proj" not in name:
            continue
        print(f"层名: {name}")
        print(f"形状: {param.shape}")
        print(f"数据类型: {param.dtype}")
        print(f"参数示例值 (前{max_params}个): {param.data.flatten()[:max_params].cpu().numpy()}\n")
        total_params += 1

processed_dataset = dataset.map(preprocess_function, batched=True)
split_dataset = processed_dataset.train_test_split(test_size=0.1)

train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8)

model = AutoModelForCausalLM.from_pretrained(
    '/home/yj/ktransformers/DeepSeek-V2-Lite-Chat', 
    trust_remote_code=True,
    torch_dtype=torch.float16)
# model = AutoModelForCausalLM.from_pretrained('/data/model/Qwen2.5-7B-Instruct', trust_remote_code=True)

print_model_with_params(model, prefix="原始模型")

# 配置 LoRA
lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            # "q_proj"
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            # "o_proj"
        ],
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

model = get_peft_model(model, lora_config)
# model = inject_adapter_in_model(lora_config, model)

for name, parms in model.named_parameters():	
        print('-->name:', name)
        print('-->para:', parms)
        print('-->grad_requirs:',parms.requires_grad)
        print('-->grad_fn:',parms.grad_fn)
        print('-->grad_value:',parms.grad)
        print("===")

# print(model)

model.train()

# for name, parms in model.named_parameters():	
#         print('-->name:', name)
#         print('-->para:', parms)
#         print('-->grad_requirs:',parms.requires_grad)
#         print('-->grad_fn:',parms.grad_fn)
#         print('-->grad_value:',parms.grad)
#         print("===")

model.to(device='cuda')
x = torch.tensor([[1,2,3]], dtype=torch.int32).to("cuda")
output = model(x)
loss = output.logits.mean()
print(f"output:{output}")
print(f"loss:{loss}")

# output = model(input_ids=torch.tensor([[1,2,3]], dtype=torch.int32))
# loss = output.logits.mean()
# # print_grad_fn(loss.grad_fn)
# # 生成计算图
dot = make_dot(loss, params=dict(model.named_parameters()))
dot.render("PEFT_compute_one_layer_model_graph", format="svg")  # 保存为SVG格式的文件

# 暂时先不训练
# model = model.to('cuda')
# model.config.use_cache = False

# # 定义训练参数
# training_args = TrainingArguments(
#     output_dir='./results',         # 模型保存和日志输出的目录路径
#     num_train_epochs=3,             # 训练的总轮数（epochs）
#     per_device_train_batch_size=1, # 每个设备（如GPU或CPU）上的训练批次大小，16表示每次输入模型的数据数量
#     learning_rate=5e-5,             # 学习率
#     logging_steps=10,               # 每隔多少步（steps）进行一次日志记录
#     save_steps=100,                 # 每隔多少步保存模型
#     save_total_limit=2,             # 保留最近的两个模型
#     fp16=True,                   
# )

class KTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        # 改写trainer的save_model，在checkpoint的时候只存lora权重
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))

trainer = KTrainer(
    model=model,
    train_dataset=train_dataset,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=16,
        num_train_epochs=10,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=10,
        save_steps=200,
        output_dir=save_path
    ),
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)

trainer.train()
# model.save_pretrained(save_path)

# print_model_with_params(model, prefix="LoRA微调模型")

# model.print_trainable_parameters() 

# model = model.merge_and_unload()

# print_model_with_params(model, prefix="合并后模型")

for name, parms in model.named_parameters():	
        print('-->name:', name)
        print('-->para:', parms)
        print('-->grad_requirs:',parms.requires_grad)
        print('-->grad_fn:',parms.grad_fn)
        print('-->grad_value:',parms.grad)
        print("===")