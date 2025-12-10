import os
import ast
import argparse
from coverage import Coverage


def main():
    parser = argparse.ArgumentParser(
        description="统计某个类在 .coverage 数据中的行覆盖率"
    )
    parser.add_argument(
        "--data-file",
        default=".coverage",
        help="coverage 数据文件路径（默认 ./.coverage）",
    )
    parser.add_argument(
        "--file",
        dest="file_pattern",
        default="ktransformers/operators/ascend/ascend_attention.py",
        help=(
            "要统计的源码文件路径（可用结尾匹配，默认 "
            "ktransformers/operators/ascend/ascend_attention.py）"
        ),
    )
    parser.add_argument(
        "--class",
        dest="class_name",
        default="KDeepseekV2AttentionW8A8A2Serve",
        help="要统计的类名（默认 KDeepseekV2AttentionW8A8A2Serve）",
    )

    args = parser.parse_args()

    if not os.path.exists(args.data_file):
        print(f"找不到 coverage 数据文件: {args.data_file}")
        raise SystemExit(1)

    cov = Coverage(data_file=args.data_file)
    cov.load()
    data = cov.get_data()

    file_pattern_norm = os.path.normpath(args.file_pattern)

    target_file = None
    for f in data.measured_files():
        f_norm = os.path.normpath(f)
        if f_norm.endswith(file_pattern_norm) or file_pattern_norm in f_norm:
            target_file = f
            break

    if not target_file:
        print(
            f"没有在 coverage 数据里找到匹配文件: {args.file_pattern}\n"
            f"实际记录的文件有:"
        )
        for f in data.measured_files():
            print("  ", f)
        raise SystemExit(1)

    print("使用的源码文件:", target_file)
    executed_lines = set(data.lines(target_file) or [])
    try:
        with open(target_file, "r", encoding="utf-8") as f:
            source_text = f.read()
    except OSError as e:
        print(f"无法打开源码文件 {target_file}: {e}")
        raise SystemExit(1)

    source_lines = source_text.splitlines()
    tree = ast.parse(source_text)

    class_start = None
    class_end = None

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == args.class_name:
            class_start = node.lineno
            max_lineno = node.lineno
            for sub in ast.walk(node):
                ln = getattr(sub, "end_lineno", getattr(sub, "lineno", None))
                if ln is not None and ln > max_lineno:
                    max_lineno = ln
            class_end = max_lineno
            break

    if class_start is None:
        print(f"在源码 {target_file} 中没有找到类 {args.class_name}")
        raise SystemExit(1)

    print(
        f"类 {args.class_name} 行范围: {class_start} ~ {class_end}"
    )

    total = 0
    covered = 0
    missed_lines = []

    for lineno in range(class_start, class_end + 1):
        line = source_lines[lineno - 1].strip()
        # 跳过空行和纯注释
        if not line or line.startswith("#"):
            continue

        total += 1
        if lineno in executed_lines:
            covered += 1
        else:
            missed_lines.append(lineno)

    percent = (covered / total * 100) if total > 0 else 0.0

    print(
        f"类 {args.class_name} 覆盖: {covered}/{total} 行, 覆盖率 = {percent:.1f}%"
    )
    if missed_lines:
        print("未覆盖行号:", missed_lines)
    else:
        print("该类所有有效代码行均被覆盖")


if __name__ == "__main__":
    main()

