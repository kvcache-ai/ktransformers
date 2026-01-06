#!/usr/bin/env python3
"""
Test global URL search in README.md
"""

from pathlib import Path
import sys

sys.path.insert(0, "python")

from kt_kernel.cli.utils.repo_detector import extract_repo_from_global_search

# Test case 1: README with multiple URLs, no frontmatter
test_readme_1 = """
# Model Card

This model is based on [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B).

## Usage

Download from https://huggingface.co/deepseek-ai/DeepSeek-V3

## License

See https://huggingface.co/meta-llama/Llama-3.1-70B for details.
"""

# Test case 2: README with ModelScope URLs
test_readme_2 = """
# 模型介绍

访问 https://modelscope.cn/models/Qwen/Qwen2-7B 查看更多。

## 下载

从 https://modelscope.cn/models/deepseek/deepseek-v3 下载模型。
"""

# Test case 3: Mixed URLs
test_readme_3 = """
# Model

Based on https://huggingface.co/Qwen/Qwen3-30B-A3B

Also available at https://modelscope.cn/models/Qwen/Qwen3-30B-A3B

Final version: https://huggingface.co/final/model-v2
"""


def test_global_search():
    """Test global URL search"""

    # Create temp directory
    temp_dir = Path("/tmp/test_repo_detect")
    temp_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("Testing Global URL Search")
    print("=" * 80)

    # Test 1
    readme1 = temp_dir / "README1.md"
    readme1.write_text(test_readme_1)

    print("\n[Test 1] Multiple HuggingFace URLs:")
    print("-" * 80)
    print(test_readme_1)
    print("-" * 80)
    result = extract_repo_from_global_search(readme1)
    print(f"Result: {result}")
    print(f"Expected: Last URL (meta-llama/Llama-3.1-70B, huggingface)")

    # Test 2
    readme2 = temp_dir / "README2.md"
    readme2.write_text(test_readme_2)

    print("\n[Test 2] Multiple ModelScope URLs:")
    print("-" * 80)
    print(test_readme_2)
    print("-" * 80)
    result = extract_repo_from_global_search(readme2)
    print(f"Result: {result}")
    print(f"Expected: Last URL (deepseek/deepseek-v3, modelscope)")

    # Test 3
    readme3 = temp_dir / "README3.md"
    readme3.write_text(test_readme_3)

    print("\n[Test 3] Mixed URLs:")
    print("-" * 80)
    print(test_readme_3)
    print("-" * 80)
    result = extract_repo_from_global_search(readme3)
    print(f"Result: {result}")
    print(f"Expected: Last URL (final/model-v2, huggingface)")

    print("\n" + "=" * 80)
    print("Tests completed!")
    print("=" * 80)

    # Cleanup
    readme1.unlink()
    readme2.unlink()
    readme3.unlink()
    temp_dir.rmdir()


if __name__ == "__main__":
    test_global_search()
