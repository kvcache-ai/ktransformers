from torch.profiler import profile, record_function, ProfilerActivity
import os
from transformers import TrainerCallback

class ProfilerCallback(TrainerCallback):
    def __init__(self, profiler):
        self.profiler = profiler

    def on_step_end(self, args, state, control, **kwargs):
        self.profiler.step()

def _short(t):
    return tuple(t.shape) if isinstance(t, torch.Tensor) else type(t)

def install_shape_probes(model):
    if os.environ.get("KT_DEBUG_MOE","0") != "1":
        print("[KT_DEBUG_MOE] off"); return

    try:
        acc = trainer.accelerator
        cfg = getattr(acc, "dataloader_config", None)
        if cfg is not None:
            print("[ACCEL DL CONFIG]",
                  "split_batches=", getattr(cfg,"split_batches",None),
                  "dispatch_batches=", getattr(cfg,"dispatch_batches",None),
                  "even_batches=", getattr(cfg,"even_batches",None),
                  "use_seedable_sampler=", getattr(cfg,"use_seedable_sampler",None),
                  "non_blocking=", getattr(cfg,"non_blocking",None))
    except Exception as e:
        print("[ACCEL DL CONFIG] <err>", e)

    try:
        emb = model.base_model.model.model.embed_tokens
        def _emb_pre(mod, inp):
            x = inp[0]
            if not hasattr(mod, "_dbg_once"):
                print(f"[DBG] embed input_ids shape = {tuple(x.shape)}  (expect B,S)")
                mod._dbg_once = True
        emb.register_forward_pre_hook(_emb_pre)
    except Exception as e:
        print("[DBG] embed hook failed:", e)

    try:
        first_layer = model.base_model.model.model.layers[0]
        _orig_fwd = first_layer.forward
        def _wrap_fwd(self, *args, **kwargs):
            hs = args[0] if args else kwargs.get("hidden_states")
            if not hasattr(self, "_dbg_once_in"):
                print(f"[DBG] L0.in hidden_states = {_short(hs)}  (expect B,S,H)")
                self._dbg_once_in = True
            out = _orig_fwd(*args, **kwargs)
            hs_out = out[0] if isinstance(out, (tuple, list)) else out
            if not hasattr(self, "_dbg_once_out"):
                print(f"[DBG] L0.out hidden_states = {_short(hs_out)}")
                self._dbg_once_out = True
            return out
        first_layer.forward = MethodType(_wrap_fwd, first_layer)
    except Exception as e:
        print("[DBG] L0 wrap failed:", e)

    try:
        moe_layer = None
        for i, lyr in enumerate(model.base_model.model.model.layers):
            if hasattr(lyr, "mlp"):
                moe_layer = lyr.mlp
                moe_idx = i
                break
        if moe_layer is not None:
            _moe_orig = moe_layer.forward
            def _moe_wrap(self, *args, **kwargs):
                x = args[0] if args else kwargs.get("hidden_states")
                if not hasattr(self, "_dbg_once"):
                    print(f"[DBG] MLP(in) @layer{moe_idx} hidden_states = {_short(x)}")
                    if isinstance(x, torch.Tensor) and x.dim() == 3:
                        B,S,H = x.shape
                        print(f"[DBG] tokens before flatten = B*S = {B}*{S} = {B*S}")
                    self._dbg_once = True
                return _moe_orig(*args, **kwargs)
            moe_layer.forward = MethodType(_moe_wrap, moe_layer)
        else:
            print("[DBG] no moe_layer found")
    except Exception as e:
        print("[DBG] moe wrap failed:", e)

    try:
        from ktransformers.operators.experts import KTransformersExperts
        def _experts_pre(mod, args):
            if hasattr(mod, "_dbg_once"): return
            try:
                input_tensor, expert_ids, weights = args[:3]
                print(f"[DBG] experts.in input_tensor={tuple(input_tensor.shape)} "
                      f"expert_ids={tuple(expert_ids.shape)} weights={tuple(weights.shape)}")
                if input_tensor.dim()==2:
                    N = input_tensor.shape[0]
                    print(f"[DBG] N(input rows)={N}")
                if expert_ids.dim()==2:
                    T,K = expert_ids.shape
                    print(f"[DBG] tokens(T)={T}, K={K}, T*K={T*K}")
                mod._dbg_once = True
            except Exception as e:
                print("[DBG] experts hook parse err:", e)
        count=0
        for name,m in model.named_modules():
            if isinstance(m, KTransformersExperts):
                m.register_forward_pre_hook(_experts_pre); count+=1
        print(f"[KT_DEBUG_MOE] installed experts hook on {count} modules.")
    except Exception as e:
        print("[DBG] experts hook failed:", e)

def inspect_device(model, write_file):
    for name, module in model.named_modules(): 
        with open(write_file, 'a') as file:
            file.write(f"Layer: {name}\n")
        for param_name, param in module.named_parameters(recurse=False): 
            with open(write_file, 'a') as file:
                file.write(f"  Parameter '{param_name}' device: {param.device}\n")
        for buffer_name, buffer in module.named_buffers(recurse=False): 
            with open(write_file, 'a') as file:
                file.write(f"  Buffer '{buffer_name}' device: {buffer.device}\n")

def print_model_params(model):
    # for layer_idx in range(len(model.model.orig_module.layers)):
    for layer_idx in range(0, 3):
        layer = model.model.orig_module.layers[layer_idx]
        
        print(f"\n================ Layer {layer_idx} Attention ================")
        
        q_proj = layer.self_attn.orig_module.q_proj.orig_module
        print(f"\nq_proj.generate_linear.weight (shape: {q_proj.generate_linear.weight.shape})")
        print(q_proj.generate_linear.weight.cpu())
        
        # kv_a_proj = layer.self_attn.orig_module.kv_a_proj_with_mqa.orig_module
        # print(f"\nkv_a_proj.weight (shape: {kv_a_proj.weight.shape})")
        # print(kv_a_proj.weight.data[:3, :5].detach().cpu().numpy())
        
        # o_proj = layer.self_attn.orig_module.o_proj.orig_module
        # print(f"\no_proj.weight (shape: {o_proj.weight.shape})")
        # print(o_proj.weight.data[:3, :5].detach().cpu().numpy())
        
        # print(f"\n================ Layer {layer_idx} MLP/MoE ================")
        
        # if layer_idx == 0:
        #     mlp = layer.mlp
        #     for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
        #         module = getattr(mlp, proj_type).orig_module
        #         print(f"\n{proj_type}.weight (shape: {module.weight.shape})")
        #         print(module.weight.data[:3, :5].detach().cpu().numpy())
        # else:
        #     moe = layer.mlp.orig_module
        #     print("\n[Shared Experts]")
        #     for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
        #         module = getattr(moe.shared_experts, proj_type).orig_module
        #         print(f"\nshared_{proj_type}.weight (shape: {module.weight.shape})")
        #         print(module.weight.data[:3, :5].detach().cpu().numpy())
            
        #     print("\n[Experts]")
        #     for expert_idx in range(3):
        #         expert = moe.experts.orig_module[expert_idx]
        #         print(f"\nExpert {expert_idx}:")
        #         for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
        #             module = getattr(expert, proj_type)
        #             print(f"{proj_type}.weight (shape: {module.weight.shape})")
        #             print(module.weight.data[:3, :5].detach().cpu().numpy())

def print_lora_params(model):
    # for layer_idx in range(len(model.model.orig_module.layers)):
    for layer_idx in range(0, 3):
        layer = model.base_model.model.model.orig_module.layers[layer_idx]
        # layer = model.model.orig_module.layers[layer_idx]
        
        q_proj_module = layer.self_attn.orig_module.q_proj.orig_module
        
        linear_weight = q_proj_module.generate_linear.weight
        lora_A_weight = q_proj_module.lora_A["default"].weight
        lora_B_weight = q_proj_module.lora_B["default"].weight
        
        print(f"\n=================== Layer {layer_idx} ===================")
        
        print("\nOriginal Linear (first row slice):")
        print(linear_weight.cpu())
        
        print("\nLora_A (first row slice):")
        print(lora_A_weight.cpu())
        
        print("\nLora_B (first row slice):")
        print(lora_B_weight.cpu())

def print_grad_fn(grad_fn, indent=0):
    """递归打印计算图节点"""
    if grad_fn is None:
        return
    print(' ' * indent, f"Node: {str(grad_fn).split('(')[0]}")
    print(' ' * indent, f"  Metadata: {grad_fn.metadata}")
    for child in getattr(grad_fn, 'next_functions', []):
        if child[0] is not None:
            print_grad_fn(child[0], indent + 2)

def forward_hook(module, inputs, output):
    if isinstance(output, (tuple, list)):
        for i, o in enumerate(output):
            if o is None:
                print(f"{module.__class__.__name__} output index {i} is None")
            else:
                print(f"{module.__class__.__name__} output index {i}: requires_grad={o.requires_grad}, grad_fn={o.grad_fn}")
    elif output is None:
        print(f"{module.__class__.__name__} returned None")
    else:
        print(f"{module.__class__.__name__}: requires_grad={output.requires_grad}, grad_fn={output.grad_fn}")

def check_moe_gradients(model):
    moe_layer = model.base_model.model.model.orig_module.layers[1].mlp.orig_module
    for name, param in moe_layer.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = torch.norm(param.grad)
            print(f"MoE参数 {name} 梯度范数: {grad_norm}")
        else:
            print(f"MoE参数 {name} 无梯度")

def disable_all_dropout(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Dropout):
                child.p = 0
                child.inplace = False
            disable_all_dropout(child)

def verify_lora_layers(model):
    for layer_path in target_layers:
        module = model.get_submodule(layer_path)
        orig_module = module.orig_module
        
        W = orig_module.weight.data  # [576, 2048] -> [2048, 576]
        lora_A = module.lora_A['default'].weight.data  # [8, 2048]
        lora_B = module.lora_B['default'].weight.data  # [576, 8]
        alpha_over_r = 32/8  # alpha=32, r=8
        
        input_tensor = layer_data[layer_path]['input']  # [1, 512, 2048]
        
        try:
            original_output = torch.matmul(input_tensor, W)  # [1,512,2048] @ [2048,576] => [1,512,576]
        except:
            original_output = torch.matmul(input_tensor, W.T)  # [1,512,2048] @ [2048,576] => [1,512,576]
        
        lora_effect = torch.matmul(
            torch.matmul(input_tensor, lora_A.T),  # [1,512,2048] @ [2048,8] => [1,512,8]
            lora_B.T  # [1,512,8] @ [8,576] => [1,512,576]
        ) * alpha_over_r
        
        manual_output = original_output + lora_effect  # [1,512,576]
        
        model_output = layer_data[layer_path]['output']

        print(f"manual_output:{manual_output}")
        print(f"model_output:{model_output}")
        
        if torch.allclose(manual_output, model_output, atol=1e-5):
            print(f"{layer_path} 验证通过")
        else:
            print(f"{layer_path} 验证失败！最大误差：{torch.max(torch.abs(manual_output - model_output))}")

def print_moe_stats(moe_layer: KExpertsTorch):
    print(f"Total Params: {moe_layer.total_params/1e6:.2f}M")
    
    total_time = sum(moe_layer.times)
    gflops = (moe_layer.total_flops / 1e9) / total_time if total_time !=0 else 0
    
    print(f"Total Calls: {moe_layer.call_count}")
    # print(f"Avg GFLOPS per Call: {gflops/moe_layer.call_count:.2f}")
    print(f"Overall GFLOPS: {gflops:.2f}")
    
    if moe_layer.call_count > 0:
        last_flops = moe_layer.flops_per_call[-1]
        last_time = moe_layer.times[-1]
        print(f"\nLast Call - FLOPs: {last_flops/1e9:.2f}G  Time: {last_time*1000:.2f}ms  "
              f"GFLOPS: {(last_flops/1e9)/last_time:.2f}")
        
def recursive_traverse(model, parent_name=''):
    """
    递归遍历模型，查找MoE层并调用print_moe_stats。
    """
    for name, module in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        
        if isinstance(module, KTransformersExperts):
            print(f"Found MoE layer: {full_name}")
            print_moe_stats(module.generate_experts)
        
        recursive_traverse(module, full_name)

def log_step_state(
    step: int,
    inputs: dict,
    loss: torch.Tensor,
    model: nn.Module,
    log_dir: str = "train_logs",
):
    """
    把当前 step 的输入 / loss / grad / param 保存到 log_dir/step_{step}.pt
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logged_inputs = {
        k: v.detach().cpu()
        for k, v in inputs.items()
        if isinstance(v, torch.Tensor)
    }

    loss_val = loss.detach().cpu()

    params, grads = {}, {}
    for name, p in model.named_parameters():
        params[name] = p.detach().cpu()
        grads[name] = p.grad.detach().cpu() if p.grad is not None else None

    torch.save(
        {
            "step": step,
            "inputs": logged_inputs,
            "loss": loss_val,
            "params": params,
            "grads": grads,
        },
        f"{log_dir}/step_{step:08d}.pt",
    )

def collect_gradients(model, input_ids):
    torch.manual_seed(42)
    
    output = model(input_ids=input_ids)
    
    logits = output.logits
    loss = logits.mean()
    
    model.zero_grad()
    loss.backward()
    
    grads = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grads.append(f"{name}: {param.grad.norm().item():.6f}")
    
    return grads

def report_meta_tensors(model):
    import torch, inspect
    meta_modules = []
    for mod_name, mod in model.named_modules():
        metas = []
        for n, p in list(mod.named_parameters(recurse=False)):
            if getattr(p, "is_meta", False) and p.is_meta:
                metas.append(("param", n, tuple(p.shape)))
        for n, b in list(mod.named_buffers(recurse=False)):
            if getattr(b, "is_meta", False) and b.is_meta:
                metas.append(("buffer", n, tuple(b.shape)))
        if metas:
            print(f"[META] {mod_name} ({type(mod).__name__}): {metas}")
            meta_modules.append((mod_name, type(mod).__name__, metas))
    return meta_modules

# def lora_and_load_adapter(model, tokenizer, sft_data_path, save_adapter_path, is_profiler=False):
    # show some lora test
    
    '''
    # multi-gpu dataloader test
    # _ = report_meta_tensors(model)
    
    # print("=== SAMPLE INSPECT ===")
    # for i in range(2):
    #     summary = {}
    #     for k,v in ex.items():
    #         if isinstance(v, list):
    #             if len(v)>0 and isinstance(v[0], list):
    #                 summary[k] = f"list-of-lists len={len(v)} x len0={len(v[0])}"
    #             else:
    #                 summary[k] = f"list len={len(v)}"
    #         elif torch.is_tensor(v):
    #             summary[k] = f"tensor shape={tuple(v.shape)}"
    #         else:
    #             summary[k] = str(type(v))
    #     print(f"[SAMPLE {i}]", summary)
    
    # trainer.accelerator = Accelerator(device_placement=False)
    # first_batch = next(iter(trainer.get_train_dataloader()))
    # print("Batch keys:", list(first_batch.keys()))
    
    # acc = KAccelerator(device_placement=False)
    # acc.state.device_ids = [0]
    # acc.state.num_processes = 1
    # acc.state.num_gpus = 1
    # trainer.accelerator = acc

    # print("Accelerator device_ids:", trainer.accelerator.state.device_ids)
    # print(f"type(trainer.model):{type(trainer.model)}")
    # print(f"type(trainer.accelerator):{type(trainer.accelerator)}")
    
    
    # print("-------------------------START TRAINING!!!-------------------------")

    # cfg = getattr(trainer.accelerator, "dataloader_config", None)
    # print(
    #     "[ACCEL DL CONFIG]",
    #     "split_batches=", getattr(cfg, "split_batches", None),
    #     "dispatch_batches=", getattr(cfg, "dispatch_batches", None),
    #     "even_batches=", getattr(cfg, "even_batches", None),
    #     "use_seedable_sampler=", getattr(cfg, "use_seedable_sampler", None),
    #     "non_blocking=", getattr(cfg, "non_blocking", None),
    # )
    # print("--------------------NEW DEBUG--------------------")
    # install_shape_probes(trainer.model) # print some debug info about multi-gpu placement.

    # input_ids = torch.randint(0, 1000, (32, 128), device="cuda:0")
    # gradients = collect_gradients(model, input_ids)
    '''
    
    # with open(f"/home/lpl/KT-SFT/tmp/KSFTExpertsCPU_grads.txt", "w") as f:
    #     f.write("\n".join(gradients))
    # print(xx)
    
    # total_length = 0
    # valid_count = 0
    # for batch in tqdm(train_dataloader):
    #     input_ids = batch['input_ids']
    #     # print(f"Token count per sample: {[len(ids) for ids in input_ids]}")
    #     for ids in input_ids:
    #         if not torch.equal(ids, torch.tensor([100001])):
    #             total_length += len(ids)
    #     valid_count += 1
    #     # print(f"Input tensor: {input_ids}")
    #     # print(f"total_length:{total_length}")
    #     # break

    # if valid_count > 0:
    #     average_length = total_length / valid_count
    # else:

    # print(xx)
    
    # from ktransformers.sft.flops_utils.custom_profile import custom_profile

    # for module in model.modules():
    #     if not hasattr(module, 'total_ops'):
    #         module.register_buffer('total_ops', torch.zeros(1, dtype=torch.float64))
    #     if not hasattr(module, 'total_params'):
    #         module.register_buffer('total_params', torch.zeros(1, dtype=torch.float64))
            
    # # print(f"input:{input}")
    # for inputs in tqdm(train_dataloader):
    #     # input_ids = batch['input_ids']
    #     # del inputs['instruction']
    #     # del inputs['input']
    #     # del inputs['output']
    #     # output = model(**inputs)
    #     model.eval()
    #     content = inputs['instruction'][0] + inputs['input'][0]
    #     # flops,params = custom_profile(model, inputs=inputs, content=content, tokenizer=tokenizer, custom_ops={YourModule: count_your_model})
    #     # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    #     # print('Params = ' + str(params / 1000 ** 2) + 'M')

    #     messages = [{"role": "user", "content": content}]
    #     input_tensor = tokenizer.apply_chat_template(
    #         messages, add_generation_prompt=True, return_tensors="pt"
    #     )
    #     with torch.no_grad():
    #         # model(*inputs)
    #         # model.model to deal with the PeftModelForCaualLM temp
    #         prefill_and_generate(
    #             model.model, tokenizer, input_tensor.cuda(), max_new_tokens=1000, use_cuda_graph=False, mode = 'normal', force_think = False, chunk_prefill_size = 8192,
    #         )
    #     recursive_traverse(model)
    
    # output = model(input_ids=torch.tensor([[1,2,3]], dtype=torch.int32, device="cuda:0"))
    # loss = output.logits.mean()
        
    # dot = make_dot(loss, params=dict(model.named_parameters()))
    # dot.render("KT_compute_cpuinfer_moe_model_graph", format="svg")

    # with open("tmp/output_loss_KCPU.txt", "w") as file:
    #     file.write("Output (logits):\n")
    #     file.write("\n\nLoss:\n")
    
    # disable_all_dropout(model)

    # def print_dropout_status(module, prefix=""):
    #     for name, child in module.named_children():
    #         if isinstance(child, nn.Dropout):
    #             print(f"{prefix}{name}: p={child.p}, training={child.training}")
    #         print_dropout_status(child, prefix + name + ".")
    
    # print_dropout_status(model)

    # for layer_path in target_layers:
    #     module = model.get_submodule(layer_path)
    #     hook = module.register_forward_hook(
    #         lambda m, i, o, ln=layer_path: record_layer_io(m, i, o, ln)
    #     )
    #     hooks.append(hook)

    
    # if is_profiler:
    #     profiler = profile(
    #         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #         schedule=torch.profiler.schedule(
    #         ),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
    #         record_shapes=False,
    #         with_stack=False
    #     )

    #     # profiler_args = {
    #     #     "schedule": torch.profiler.schedule(
    #     #     )
    #     # }

    #     trainer = KTrainer(
    #         model=model,
    #         train_dataset=train_dataset,
    #         data_collator=DataCollatorForSeq2Seq(
    #             tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    #         ),
    #         callbacks=[ProfilerCallback(profiler)]
    #     )

    #     with profiler:
    #         trainer.train()

    #     print("Training finished. Exporting profiler data...")
    #     with open("profiler_output.txt", "w") as f:
    #         f.write(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    #   profiler.export_chrome_trace("trace.json")
    
    

    # verify_lora_layers(model)

    # model.save_pretrained(save_adapter_path)

    '''
    ----------------------- START: Lora Test -----------------------
    

    # for name, module in model.named_modules():
    #     if "q_proj" in name or "kv_a_proj" in name or "o_proj" in name:
    #         print(name)

    # print_model_params(model)

    # model = KTransformersLinearLora()

    # inspect_device(model, '/home/yj/ktransformers/device1.txt')
    # with open('/home/yj/ktransformers/device1.txt', 'a') as file:
    #     file.write(f"Base model device: {model.base_model.device}\n")
        # file.write(f"LoRA adapter device: {model.lora_config['target_modules'].device}\n")
    # print(f"Base model device: {model.base_model.device}") 
    # print(f"LoRA adapter device: {model.lora_config['target_modules'].device}") 


    # model = model.to('cuda')

    # for name, module in model.named_modules():
    #     module.register_forward_hook(forward_hook)

    # for name, parms in model.named_parameters():	
    #     # parms.requires_grad = True
    #     print('-->name:', name)
    #     print('-->para:', parms)
    #     print('-->grad_requirs:',parms.requires_grad)
    #     print('-->grad_fn:',parms.grad_fn)
    #     print('-->grad_value:',parms.grad)
    #     print("===")

    # output = model(input_ids=torch.tensor([[1,2,3]], dtype=torch.int32, device="cuda:0"))
    # loss = output.logits.mean()

    # dot = make_dot(loss, params=dict(model.named_parameters()))
    # dot.render("KT_compute_graph", format="svg")

    # inspect_device(model, '/home/yj/ktransformers/device2.txt')
    # with open('/home/yj/ktransformers/device2.txt', 'a') as file:
    #     file.write(f"Base model device: {model.base_model.device}\n")
        # file.write(f"LoRA adapter device: {model.lora_config['target_modules'].device}\n")
    # print(f"Base model device: {model.base_model.device}") 
    # print(f"LoRA adapter device: {model.lora_config['target_modules'].device}") 

    # print_lora_params(model)

    # trainer = KTrainer(
    #     model=model,
    #     train_dataset=train_dataset,
    #     args=transformers.TrainingArguments(
    #         output_dir=save_adapter_path,
    #         per_device_train_batch_size=1,
    #         gradient_accumulation_steps=16,
    #         num_train_epochs=10,
    #         learning_rate=3e-4,
    #         fp16=False,
    #         logging_steps=10,
    #         save_steps=200,
    #         dataloader_drop_last=True,
    #         ddp_find_unused_parameters=False 
    #     ),
    #     data_collator=DataCollatorForSeq2Seq(
    #         tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    #     ),
    # )

    # model(input_ids=torch.tensor([[1,2,3]], dtype=torch.int32, device="cuda:0"))

    # trainer.train()

    # print_lora_params(model)

    # model = model.merge_and_unload()
    ----------------------- END: Lora Test -----------------------

    '''