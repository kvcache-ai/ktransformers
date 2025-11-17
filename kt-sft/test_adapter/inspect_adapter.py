# -*- coding: utf-8 -*-
"""
inspect_adapter.py  ‒  查看 LoRA / Adapter checkpoint 信息
------------------------------------------------------------
示例：
  python inspect_adapter.py ./checkpoint
  python inspect_adapter.py ./checkpoint --show-params            # 打印全部权重行
  python inspect_adapter.py ./checkpoint --param lora_A.weight    # 只看某个权重
  python inspect_adapter.py ./checkpoint --dump-all               # 导出所有张量
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file as safe_load
from tabulate import tabulate


def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def human_readable(num: int) -> str:
    for unit in ["", "K", "M", "B"]:
        if abs(num) < 1000:
            return f"{num:,.0f}{unit}"
        num /= 1000
    return f"{num:.1f}T"


def inspect_adapter_weights(weight_path: Path):
    """
    读取 adapter_model.safetensors / .bin / .pt
    返回 (rows, total_params, state) 三元组
    """
    if weight_path.suffix == ".safetensors":
        state = safe_load(str(weight_path))
    else:
        state = torch.load(str(weight_path), map_location="cpu")

    rows, total = [], 0
    for name, tensor in state.items():
        n = tensor.numel()
        total += n
        rows.append([
            name,
            list(tensor.shape),
            str(tensor.dtype).replace("torch.", ""),
            human_readable(n)
        ])
    rows.sort(key=lambda x: x[0])
    return rows, total, state


def maybe_print_optimizer(optimizer_pt: Path, max_keys: int = 20):
    try:
        opt_state = torch.load(str(optimizer_pt), map_location="cpu")
    except Exception as e:
        print(f"[optimizer.pt] 读取失败：{e}")
        return
    print("\n====== optimizer.pt 结构 (部分) ======")
    if isinstance(opt_state, dict):
        for i, k in enumerate(opt_state.keys()):
            if i >= max_keys:
                print("... (省略)")
                break
            print(f"{k}: type={type(opt_state[k])}")
    else:
        print(f"type={type(opt_state)} 非典型，请自行查看。")


def maybe_print_scheduler(scheduler_pt: Path, max_keys: int = 20):
    try:
        sch_state = torch.load(str(scheduler_pt), map_location="cpu")
    except Exception as e:
        print(f"[scheduler.pt] 读取失败：{e}")
        return
    print("\n====== scheduler.pt 结构 (部分) ======")
    if isinstance(sch_state, dict):
        for i, (k, v) in enumerate(sch_state.items()):
            if i >= max_keys:
                print("... (省略)")
                break
            print(f"{k}: type={type(v)}")
    else:
        print(f"type={type(sch_state)} 非典型，请自行查看。")


def maybe_print_rng(rng_pth: Path):
    try:
        rng = torch.load(str(rng_pth), map_location="cpu")
    except Exception as e:
        print(f"[rng_state.pth] 读取失败：{e}")
        return
    print("\n====== rng_state.pth 键列表 ======")
    if isinstance(rng, dict):
        for k in rng.keys():
            print(f"- {k}")
    else:
        print(f"type={type(rng)} 非典型，请自行查看。")


def dump_tensors(state: dict, out_dir="tensor_dump"):
    """
    将 state 的每个张量写入 txt（repr）并可选保存二进制 .pt
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    torch.set_printoptions(sci_mode=False, linewidth=180)

    for name, tensor in state.items():
        safe_name = name.replace("/", "_")
        txt_path = out_dir / f"{safe_name}.txt"
        with open(txt_path, "w") as f:
            f.write(repr(tensor))

        # 若需要二进制，取消下一行注释
        # torch.save(tensor, out_dir / f"{safe_name}.pt")

    print(f"[done] 已把 {len(state)} 个张量写入 {out_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="检查 LoRA / Adapter checkpoint 内容")
    parser.add_argument("ckpt_dir", type=str,
                        help="包含 adapter_config.json / adapter_model.safetensors 的目录")
    parser.add_argument("--show-params", action="store_true",
                        help="打印所有权重摘要（默认只显示前 30 行）")
    parser.add_argument("--param", type=str,
                        help="仅打印指定参数的完整张量")
    parser.add_argument("--dump-all", action="store_true",
                        help="把所有张量完整写入文件夹")
    args = parser.parse_args()

    d = Path(args.ckpt_dir).expanduser()
    if not d.exists():
        raise FileNotFoundError(d)

    # ========== adapter_config.json ==========
    cfg_path = d / "adapter_config.json"
    if cfg_path.exists():
        print("====== adapter_config.json ======")
        print(json.dumps(load_json(cfg_path), indent=2, ensure_ascii=False))
    else:
        print("未找到 adapter_config.json")

    # ========== trainer_state.json ==========
    ts_path = d / "trainer_state.json"
    if ts_path.exists():
        ts = load_json(ts_path)
        print("\n====== trainer_state.json (节选) ======")
        sel = {k: ts.get(k, None) for k in
               ["global_step", "best_metric", "best_model_checkpoint", "log_history"]}
        if isinstance(sel.get("log_history"), list) and len(sel["log_history"]) > 3:
            sel["log_history"] = sel["log_history"][-3:]
        print(json.dumps(sel, indent=2, ensure_ascii=False))
    else:
        print("\n未找到 trainer_state.json")

    # ========== adapter_model.* ==========
    st_path = next((d / n for n in
                   ["adapter_model.safetensors", "adapter_model.bin", "adapter_model.pt"]
                   if (d / n).exists()), None)

    if st_path is None:
        print("\n未找到 adapter_model.* (safetensors/bin/pt)")
        state = {}
    else:
        rows, total, state = inspect_adapter_weights(st_path)

        # 若用户指定 --param，仅打印该张量
        if args.param is not None:
            if args.param not in state:
                raise KeyError(f"参数 {args.param!r} 不存在！")
            torch.set_printoptions(sci_mode=False, linewidth=180, profile="full")
            print(f"\n====== {args.param} 的完整张量 ======")
            print(state[args.param])
            return  # 提前结束

        print(f"\n====== {st_path.name} 中的可训练参数（共 {human_readable(total)} 个元素）======")
        if args.show_params:
            print(tabulate(rows, headers=["参数名", "形状", "dtype", "元素数"], tablefmt="github"))
        else:
            head = rows[:30]
            print(tabulate(head, headers=["参数名", "形状", "dtype", "元素数"], tablefmt="github"))
            if len(rows) > 30:
                print(f"... 还有 {len(rows) - 30} 个参数未展示，使用 --show-params 查看全部。")

        # --dump-all 时将所有张量写文件
        if args.dump_all:
            dump_tensors(state, out_dir=f"{st_path.stem}_dump")

    # ========== 其它 state_dict ==========
    if (d / "optimizer.pt").exists():
        maybe_print_optimizer(d / "optimizer.pt")
    if (d / "scheduler.pt").exists():
        maybe_print_scheduler(d / "scheduler.pt")
    if (d / "rng_state.pth").exists():
        maybe_print_rng(d / "rng_state.pth")

    print("\nDone.")


if __name__ == "__main__":
    main()
