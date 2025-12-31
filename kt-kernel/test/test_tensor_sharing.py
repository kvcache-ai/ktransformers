"""
快速测试工具：检查 PyTorch tensor 操作是否共享内存
"""

import torch


class TensorSharingChecker:
    """检查 tensor 操作是否共享内存的工具类"""

    @staticmethod
    def check(tensor1, tensor2, name1="tensor1", name2="tensor2", verbose=True):
        """
        检查两个 tensor 是否共享内存

        参数:
            tensor1: 第一个 tensor
            tensor2: 第二个 tensor
            name1: 第一个 tensor 的名称
            name2: 第二个 tensor 的名称
            verbose: 是否打印详细信息

        返回:
            dict: 包含检查结果的字典
        """
        result = {
            'same_object': tensor1 is tensor2,
            'same_data_ptr': tensor1.data_ptr() == tensor2.data_ptr(),
            'same_storage': tensor1.storage().data_ptr() == tensor2.storage().data_ptr(),
            'shares_memory': False
        }

        # 判断是否共享内存
        result['shares_memory'] = (
            result['same_object'] or
            result['same_data_ptr'] or
            result['same_storage']
        )

        if verbose:
            print(f"\n{'='*60}")
            print(f"检查 {name1} 和 {name2} 的内存关系")
            print(f"{'='*60}")
            print(f"是否为同一对象 ({name1} is {name2}): {result['same_object']}")
            print(f"数据指针相同: {result['same_data_ptr']}")
            print(f"  - {name1}.data_ptr(): {tensor1.data_ptr()}")
            print(f"  - {name2}.data_ptr(): {tensor2.data_ptr()}")
            print(f"存储对象相同: {result['same_storage']}")
            print(f"  - {name1}.storage().data_ptr(): {tensor1.storage().data_ptr()}")
            print(f"  - {name2}.storage().data_ptr(): {tensor2.storage().data_ptr()}")
            print(f"\n结论: {'共享内存 ✓' if result['shares_memory'] else '不共享内存 ✗'}")
            print(f"{'='*60}\n")

        return result

    @staticmethod
    def test_modification(tensor1, tensor2, name1="tensor1", name2="tensor2"):
        """
        通过修改测试是否共享内存

        参数:
            tensor1: 第一个 tensor
            tensor2: 第二个 tensor
            name1: 第一个 tensor 的名称
            name2: 第二个 tensor 的名称

        返回:
            bool: 是否共享内存
        """
        print(f"\n通过修改测试 {name1} 和 {name2} 是否共享内存:")
        print(f"原始 {name1}: {tensor1}")
        print(f"原始 {name2}: {tensor2}")

        # 保存原始值
        original_value = tensor1.flatten()[0].item()

        # 修改 tensor1
        test_value = 99999.0
        tensor1.flatten()[0] = test_value

        print(f"\n修改 {name1}[0] = {test_value}")
        print(f"修改后 {name1}: {tensor1}")
        print(f"修改后 {name2}: {tensor2}")

        # 检查 tensor2 是否也被修改
        shares_memory = tensor2.flatten()[0].item() == test_value

        # 恢复原始值
        tensor1.flatten()[0] = original_value

        print(f"\n结论: {'共享内存（tensor2 也被修改）✓' if shares_memory else '不共享内存（tensor2 未被修改）✗'}\n")

        return shares_memory


def demo_usage():
    """演示如何使用 TensorSharingChecker"""
    checker = TensorSharingChecker()

    print("=" * 80)
    print("TensorSharingChecker 使用示例")
    print("=" * 80)

    # 示例 1: a = b
    print("\n示例 1: a = b")
    b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    a = b
    checker.check(a, b, "a", "b")
    checker.test_modification(a, b, "a", "b")

    # 示例 2: a = b.clone()
    print("\n示例 2: a = b.clone()")
    b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    a = b.clone()
    checker.check(a, b, "a", "b")
    checker.test_modification(a, b, "a", "b")

    # 示例 3: a = b.detach()
    print("\n示例 3: a = b.detach()")
    b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, requires_grad=True)
    a = b.detach()
    checker.check(a, b, "a", "b")
    checker.test_modification(a, b, "a", "b")

    # 示例 4: a = b.view(...)
    print("\n示例 4: a = b.view(-1)")
    b = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    a = b.view(-1)
    checker.check(a, b, "a", "b")
    checker.test_modification(a, b, "a", "b")

    # 示例 5: a = b[:]
    print("\n示例 5: a = b[:]")
    b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    a = b[:]
    checker.check(a, b, "a", "b")
    checker.test_modification(a, b, "a", "b")

    # 示例 6: a[:] = b（先创建 a，再拷贝）
    print("\n示例 6: a[:] = b")
    a = torch.zeros(5, dtype=torch.float32)
    b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    a[:] = b
    checker.check(a, b, "a", "b")
    checker.test_modification(a, b, "a", "b")


def interactive_test():
    """交互式测试"""
    print("\n" + "=" * 80)
    print("交互式内存共享测试")
    print("=" * 80)
    print("\n你可以手动测试任意 tensor 操作是否共享内存")
    print("\n示例代码模板:")
    print("""
import torch
from test_tensor_sharing import TensorSharingChecker

checker = TensorSharingChecker()

# 创建测试 tensor
b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)

# 执行你想测试的操作
a = b.view(-1)  # 替换为你想测试的操作

# 检查是否共享内存
checker.check(a, b, "a", "b")
checker.test_modification(a, b, "a", "b")
""")


def batch_test_all_operations():
    """批量测试所有常见操作"""
    print("\n" + "=" * 80)
    print("批量测试所有常见操作")
    print("=" * 80)

    checker = TensorSharingChecker()
    results = {}

    operations = [
        ("a = b", lambda b: b),
        ("a = b.clone()", lambda b: b.clone()),
        ("a = b.detach()", lambda b: b.detach()),
        ("a = b.view(-1)", lambda b: b.view(-1)),
        ("a = b[:]", lambda b: b[:]),
        ("a = b.reshape(-1)", lambda b: b.reshape(-1)),
    ]

    print("\n{:<30} | {:<15} | {:<15} | {:<15}".format(
        "操作", "同一对象", "共享数据指针", "共享存储"
    ))
    print("-" * 80)

    for op_name, op_func in operations:
        b = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        if "detach" in op_name:
            b.requires_grad = True

        try:
            a = op_func(b)
            result = checker.check(a, b, "a", "b", verbose=False)
            results[op_name] = result

            print("{:<30} | {:<15} | {:<15} | {:<15}".format(
                op_name,
                "✓" if result['same_object'] else "✗",
                "✓" if result['same_data_ptr'] else "✗",
                "✓" if result['same_storage'] else "✗"
            ))
        except Exception as e:
            print(f"{op_name:<30} | Error: {e}")

    # 测试就地操作
    print("\n\n就地操作测试:")
    print("{:<30} | {:<15}".format("操作", "共享内存（修改测试）"))
    print("-" * 50)

    # a[:] = b
    a = torch.zeros((2, 3), dtype=torch.float32)
    b = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    a_copy = a.clone()
    a[:] = b
    shares = checker.test_modification(a, b, "a", "b")
    print("{:<30} | {:<15}".format("a[:] = b", "✓" if shares else "✗"))

    # a.copy_(b)
    a = torch.zeros((2, 3), dtype=torch.float32)
    b = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    a.copy_(b)
    shares = checker.test_modification(a, b, "a", "b")
    print("{:<30} | {:<15}".format("a.copy_(b)", "✓" if shares else "✗"))

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    # 运行演示
    demo_usage()

    # 批量测试
    batch_test_all_operations()

    # 显示交互式测试说明
    interactive_test()
