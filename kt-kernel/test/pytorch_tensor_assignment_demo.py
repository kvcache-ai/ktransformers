"""
PyTorch Tensor 赋值行为详解
演示不同赋值方式下的内存共享和数据拷贝行为
"""

import torch
import numpy as np


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def check_memory_sharing(tensor1, tensor2, name1="a", name2="b"):
    """检查两个 tensor 是否共享内存"""
    # 使用 data_ptr() 检查底层数据指针
    ptr1 = tensor1.data_ptr()
    ptr2 = tensor2.data_ptr()
    shares_memory = ptr1 == ptr2

    print(f"{name1}.data_ptr(): {ptr1}")
    print(f"{name2}.data_ptr(): {ptr2}")
    print(f"共享内存: {'是' if shares_memory else '否'}")

    # 也可以使用 storage().data_ptr() 检查存储对象
    storage_ptr1 = tensor1.storage().data_ptr()
    storage_ptr2 = tensor2.storage().data_ptr()
    shares_storage = storage_ptr1 == storage_ptr2
    print(f"共享存储对象: {'是' if shares_storage else '否'}")

    return shares_memory


def demo_case_1():
    """案例 1: a = b (引用赋值)"""
    print_section("案例 1: a = b (引用赋值)")

    b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    print(f"原始 b: {b}")

    a = b  # 这是 Python 的引用赋值
    print(f"执行 a = b 后:")
    print(f"a: {a}")
    print(f"a is b: {a is b}")  # True，a 和 b 是同一个对象

    check_memory_sharing(a, b)

    # 修改 a，b 也会改变
    print("\n修改 a[0] = 999:")
    a[0] = 999
    print(f"a: {a}")
    print(f"b: {b}")  # b 也被修改了

    print("\n结论: a = b 是 Python 层面的引用赋值，a 和 b 指向同一个对象")


def demo_case_2():
    """案例 2: a = b.clone() (深拷贝)"""
    print_section("案例 2: a = b.clone() (深拷贝)")

    b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    print(f"原始 b: {b}")

    a = b.clone()  # 创建数据的深拷贝
    print(f"执行 a = b.clone() 后:")
    print(f"a: {a}")
    print(f"a is b: {a is b}")  # False

    check_memory_sharing(a, b)

    # 修改 a，b 不会改变
    print("\n修改 a[0] = 999:")
    a[0] = 999
    print(f"a: {a}")
    print(f"b: {b}")  # b 不变

    print("\n结论: clone() 创建完全独立的数据副本，修改互不影响")


def demo_case_3():
    """案例 3: a = b.detach() (分离计算图，但共享数据)"""
    print_section("案例 3: a = b.detach() (分离计算图，但共享数据)")

    b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, requires_grad=True)
    print(f"原始 b: {b}")
    print(f"b.requires_grad: {b.requires_grad}")

    a = b.detach()  # 分离计算图，但共享数据
    print(f"\n执行 a = b.detach() 后:")
    print(f"a: {a}")
    print(f"a.requires_grad: {a.requires_grad}")  # False
    print(f"a is b: {a is b}")  # False，不是同一个对象

    check_memory_sharing(a, b)

    # 修改 a，b 也会改变（共享数据）
    print("\n修改 a[0] = 999:")
    a[0] = 999
    print(f"a: {a}")
    print(f"b: {b}")  # b 也被修改了

    print("\n结论: detach() 创建新的 tensor 对象但共享底层数据，用于从计算图分离")


def demo_case_4():
    """案例 4: a[:] = b (就地赋值/拷贝)"""
    print_section("案例 4: a[:] = b (就地赋值/拷贝)")

    a = torch.tensor([10, 20, 30, 40, 50], dtype=torch.float32)
    b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    print(f"原始 a: {a}")
    print(f"原始 b: {b}")

    # 记录原始的 data_ptr
    original_a_ptr = a.data_ptr()

    a[:] = b  # 将 b 的数据拷贝到 a 的内存中
    print(f"\n执行 a[:] = b 后:")
    print(f"a: {a}")
    print(f"b: {b}")
    print(f"a 的内存地址是否改变: {original_a_ptr != a.data_ptr()}")  # False，未改变

    check_memory_sharing(a, b)

    # 修改 b，a 不会改变
    print("\n修改 b[0] = 999:")
    b[0] = 999
    print(f"a: {a}")  # a 不变
    print(f"b: {b}")

    print("\n结论: a[:] = b 将 b 的数据拷贝到 a 的现有内存，不共享数据")


def demo_case_5():
    """案例 5: a[0] = b (2D tensor 的行赋值)"""
    print_section("案例 5: a[0] = b (2D tensor 的行赋值)")

    a = torch.zeros((3, 5), dtype=torch.float32)
    b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    print(f"原始 a:\n{a}")
    print(f"原始 b: {b}")

    a[0] = b  # 将 b 的数据拷贝到 a 的第一行
    print(f"\n执行 a[0] = b 后:")
    print(f"a:\n{a}")
    print(f"b: {b}")

    # 检查 a[0] 和 b 是否共享内存
    print("\n检查 a[0] 和 b 的内存关系:")
    check_memory_sharing(a[0], b, "a[0]", "b")

    # 修改 b，a[0] 不会改变
    print("\n修改 b[0] = 999:")
    b[0] = 999
    print(f"a:\n{a}")  # a 不变
    print(f"b: {b}")

    print("\n结论: a[0] = b 将 b 的数据拷贝到 a 的第一行，不共享数据")


def demo_case_6():
    """案例 6: a.copy_(b) (就地拷贝方法)"""
    print_section("案例 6: a.copy_(b) (就地拷贝方法)")

    a = torch.tensor([10, 20, 30, 40, 50], dtype=torch.float32)
    b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    print(f"原始 a: {a}")
    print(f"原始 b: {b}")

    # 记录原始的 data_ptr
    original_a_ptr = a.data_ptr()
    original_a_id = id(a)

    result = a.copy_(b)  # 将 b 的数据拷贝到 a 的内存中
    print(f"\n执行 a.copy_(b) 后:")
    print(f"a: {a}")
    print(f"b: {b}")
    print(f"返回值 is a: {result is a}")  # True，返回 a 自身
    print(f"a 对象 id 是否改变: {id(a) != original_a_id}")  # False
    print(f"a 的内存地址是否改变: {a.data_ptr() != original_a_ptr}")  # False

    check_memory_sharing(a, b)

    # 修改 b，a 不会改变
    print("\n修改 b[0] = 999:")
    b[0] = 999
    print(f"a: {a}")  # a 不变
    print(f"b: {b}")

    print("\n结论: copy_(b) 是就地操作，将 b 的数据拷贝到 a，不共享数据")


def demo_case_7():
    """案例 7: a = b.view(...) (视图操作，共享数据)"""
    print_section("案例 7: a = b.view(...) (视图操作，共享数据)")

    b = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    print(f"原始 b (shape {b.shape}):\n{b}")

    a = b.view(6)  # 改变形状，但共享数据
    print(f"\n执行 a = b.view(6) 后:")
    print(f"a (shape {a.shape}): {a}")
    print(f"a is b: {a is b}")  # False

    check_memory_sharing(a, b)

    # 修改 a，b 也会改变
    print("\n修改 a[0] = 999:")
    a[0] = 999
    print(f"a: {a}")
    print(f"b:\n{b}")  # b 也被修改了

    # view 的限制：只能在内存连续的 tensor 上使用
    print("\n注意: view() 要求 tensor 在内存中是连续的")
    c = b.t()  # 转置后不连续
    print(f"c.is_contiguous(): {c.is_contiguous()}")
    try:
        d = c.view(6)
    except RuntimeError as e:
        print(f"c.view(6) 抛出异常: {e}")

    # 使用 reshape 可以处理不连续的 tensor
    d = c.reshape(6)
    print(f"\nc.reshape(6) 成功: {d}")
    print(f"reshape 是否共享内存 (取决于是否连续):")
    check_memory_sharing(c, d, "c", "d")

    print("\n结论: view() 创建共享数据的新视图，要求内存连续；reshape() 更灵活")


def demo_case_8():
    """案例 8: a = b[:] (切片操作，共享数据)"""
    print_section("案例 8: a = b[:] (切片操作，共享数据)")

    b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    print(f"原始 b: {b}")

    a = b[:]  # 切片操作，共享数据
    print(f"\n执行 a = b[:] 后:")
    print(f"a: {a}")
    print(f"a is b: {a is b}")  # False

    check_memory_sharing(a, b)

    # 修改 a，b 也会改变
    print("\n修改 a[0] = 999:")
    a[0] = 999
    print(f"a: {a}")
    print(f"b: {b}")  # b 也被修改了

    # 对比部分切片
    print("\n\n部分切片 c = b[1:3]:")
    c = b[1:3]
    print(f"c: {c}")
    print("检查 c 和 b 的内存关系:")
    # 虽然 data_ptr 不同，但它们共享同一个存储对象
    print(f"c.data_ptr(): {c.data_ptr()}")
    print(f"b.data_ptr(): {b.data_ptr()}")
    print(f"c.storage().data_ptr(): {c.storage().data_ptr()}")
    print(f"b.storage().data_ptr(): {b.storage().data_ptr()}")
    print(f"共享存储对象: {c.storage().data_ptr() == b.storage().data_ptr()}")

    print("\n修改 c[0] = 777:")
    c[0] = 777
    print(f"c: {c}")
    print(f"b: {b}")  # b[1] 也被修改了

    print("\n结论: 切片操作创建共享底层存储的视图，修改会互相影响")


def demo_additional_cases():
    """补充案例: 其他重要的赋值场景"""
    print_section("补充案例: 其他重要的赋值场景")

    # 1. tensor.data
    print("1. 使用 .data 属性:")
    b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, requires_grad=True)
    a = b.data  # 返回 tensor，不追踪梯度，但共享数据
    print(f"a = b.data")
    print(f"a.requires_grad: {a.requires_grad}")
    check_memory_sharing(a, b)
    a[0] = 999
    print(f"修改 a[0] 后，b: {b}\n")

    # 2. torch.as_tensor
    print("\n2. torch.as_tensor (尽可能避免拷贝):")
    np_array = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    a = torch.as_tensor(np_array)
    print(f"从 NumPy 数组创建: torch.as_tensor(np_array)")
    print(f"a: {a}")
    a[0] = 999
    print(f"修改 a[0] 后，np_array: {np_array}")  # NumPy 数组也被修改
    print("注意: as_tensor 会与 NumPy 数组共享内存\n")

    # 3. torch.tensor (总是拷贝)
    print("\n3. torch.tensor (总是拷贝数据):")
    np_array2 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    b = torch.tensor(np_array2)
    print(f"从 NumPy 数组创建: torch.tensor(np_array2)")
    print(f"b: {b}")
    b[0] = 999
    print(f"修改 b[0] 后，np_array2: {np_array2}")  # NumPy 数组不变
    print("注意: torch.tensor 总是创建数据副本\n")

    # 4. tensor.new_tensor
    print("\n4. tensor.new_tensor (拷贝数据):")
    original = torch.tensor([1, 2, 3])
    new = original.new_tensor([4, 5, 6])
    print(f"original: {original}")
    print(f"new: {new}")
    check_memory_sharing(original, new)

    # 5. tensor.reshape vs view
    print("\n\n5. reshape vs view:")
    b = torch.arange(12).reshape(3, 4)
    print(f"b:\n{b}")
    print(f"b.is_contiguous(): {b.is_contiguous()}")

    # transpose 会使 tensor 不连续
    c = b.t()
    print(f"\nc = b.t():\n{c}")
    print(f"c.is_contiguous(): {c.is_contiguous()}")

    # view 要求连续
    try:
        d = c.view(-1)
        print("c.view(-1) 成功")
    except RuntimeError as e:
        print(f"c.view(-1) 失败: {type(e).__name__}")

    # reshape 可以处理不连续的情况（可能会拷贝）
    e = c.reshape(-1)
    print(f"c.reshape(-1) 成功: {e}")
    shares = c.storage().data_ptr() == e.storage().data_ptr()
    print(f"c 和 e 共享存储: {shares}")


def demo_summary():
    """总结表格"""
    print_section("总结对比表")

    print("""
操作                    | 共享数据 | 创建新对象 | 用途/说明
-----------------------|---------|-----------|----------------------------------
a = b                  | 是      | 否        | Python 引用赋值，a 和 b 是同一对象
a = b.clone()          | 否      | 是        | 深拷贝，创建独立副本
a = b.detach()         | 是      | 是        | 从计算图分离，共享数据
a[:] = b               | 否      | 否        | 就地拷贝数据到 a 的内存
a[0] = b               | 否      | 否        | 拷贝 b 到 a 的某一行/元素
a.copy_(b)             | 否      | 否        | 就地拷贝方法，返回 a 自身
a = b.view(...)        | 是      | 是        | 改变形状的视图，要求内存连续
a = b[:]               | 是      | 是        | 切片视图，共享底层存储
a = b.reshape(...)     | 视情况   | 是        | 灵活的形状变换，可能拷贝
a = b.data             | 是      | 是*       | 访问底层数据，不追踪梯度
torch.tensor(data)     | 否      | 是        | 总是创建新的数据副本
torch.as_tensor(data)  | 视情况   | 是        | 尽可能避免拷贝（如 NumPy 数组）

* 对于 b.data，返回的仍是 tensor 类型，只是不追踪梯度

关键判断标准:
1. 检查是否共享内存: 使用 data_ptr() 或 storage().data_ptr()
2. 是否是同一对象: 使用 is 运算符
3. 就地操作的特征: 方法名以 _ 结尾（如 copy_、add_）
4. 修改测试: 修改一个 tensor，观察另一个是否变化

常见误区:
- detach() 会共享数据，不是深拷贝！如需独立副本用 clone()
- view() 和切片都共享数据，修改会互相影响
- a[:] = b 是拷贝，但 a = b[:] 是共享
- reshape() 在内存连续时共享数据，不连续时会拷贝
""")


def main():
    """主函数"""
    print("PyTorch Tensor 赋值行为完整演示")
    print("=" * 80)

    demo_case_1()
    demo_case_2()
    demo_case_3()
    demo_case_4()
    demo_case_5()
    demo_case_6()
    demo_case_7()
    demo_case_8()
    demo_additional_cases()
    demo_summary()

    print("\n" + "=" * 80)
    print("演示完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
