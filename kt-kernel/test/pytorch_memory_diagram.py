"""
创建 PyTorch tensor 赋值行为的可视化图表
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def create_memory_sharing_comparison():
    """创建内存共享对比图"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('PyTorch Tensor 赋值操作：内存共享对比', fontsize=16, fontweight='bold')

    operations = [
        ('a = b', True, 'Python引用\n同一对象'),
        ('a = b.clone()', False, '深拷贝\n独立数据'),
        ('a = b.detach()', True, '分离计算图\n共享数据'),
        ('a = b.view(...)', True, '视图操作\n共享数据'),
        ('a = b[:]', True, '切片\n共享数据'),
        ('a[:] = b', False, '就地拷贝\n独立数据'),
        ('a.copy_(b)', False, '就地拷贝\n独立数据'),
        ('a = b.reshape(...)', None, '视情况\n可能共享')
    ]

    for idx, (operation, shares_memory, description) in enumerate(operations):
        ax = axes[idx // 4, idx % 4]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # 绘制操作标题
        ax.text(5, 9.5, operation, ha='center', va='top',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='black'))

        # 绘制描述
        ax.text(5, 8.5, description, ha='center', va='top',
                fontsize=9, style='italic')

        if shares_memory is True:
            # 共享内存的情况
            # 绘制共享的内存块
            memory_box = FancyBboxPatch((3, 3), 4, 2,
                                       boxstyle="round,pad=0.1",
                                       facecolor='yellow',
                                       edgecolor='red',
                                       linewidth=2)
            ax.add_patch(memory_box)
            ax.text(5, 4, 'Memory\n[1,2,3,4,5]', ha='center', va='center',
                   fontsize=9, fontweight='bold')

            # 绘制 a 和 b 指向同一内存
            ax.text(2, 6.5, 'a', ha='center', va='center',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='circle', facecolor='lightgreen'))
            ax.text(8, 6.5, 'b', ha='center', va='center',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='circle', facecolor='lightgreen'))

            # 箭头指向共享内存
            arrow1 = FancyArrowPatch((2, 6), (4, 5),
                                    arrowstyle='->', mutation_scale=20,
                                    color='red', linewidth=2)
            arrow2 = FancyArrowPatch((8, 6), (6, 5),
                                    arrowstyle='->', mutation_scale=20,
                                    color='red', linewidth=2)
            ax.add_patch(arrow1)
            ax.add_patch(arrow2)

            # 标注
            ax.text(5, 1.5, '共享内存', ha='center', va='center',
                   fontsize=10, color='red', fontweight='bold')

        elif shares_memory is False:
            # 不共享内存的情况
            # 绘制 a 的内存
            memory_box_a = FancyBboxPatch((0.5, 3), 3.5, 2,
                                         boxstyle="round,pad=0.1",
                                         facecolor='lightgreen',
                                         edgecolor='green',
                                         linewidth=2)
            ax.add_patch(memory_box_a)
            ax.text(2.25, 4, 'Memory A\n[1,2,3,4,5]', ha='center', va='center',
                   fontsize=9, fontweight='bold')

            # 绘制 b 的内存
            memory_box_b = FancyBboxPatch((6, 3), 3.5, 2,
                                         boxstyle="round,pad=0.1",
                                         facecolor='lightblue',
                                         edgecolor='blue',
                                         linewidth=2)
            ax.add_patch(memory_box_b)
            ax.text(7.75, 4, 'Memory B\n[1,2,3,4,5]', ha='center', va='center',
                   fontsize=9, fontweight='bold')

            # a 指向其内存
            ax.text(2.25, 6.5, 'a', ha='center', va='center',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='circle', facecolor='lightgreen'))
            arrow_a = FancyArrowPatch((2.25, 6), (2.25, 5),
                                     arrowstyle='->', mutation_scale=20,
                                     color='green', linewidth=2)
            ax.add_patch(arrow_a)

            # b 指向其内存
            ax.text(7.75, 6.5, 'b', ha='center', va='center',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='circle', facecolor='lightblue'))
            arrow_b = FancyArrowPatch((7.75, 6), (7.75, 5),
                                     arrowstyle='->', mutation_scale=20,
                                     color='blue', linewidth=2)
            ax.add_patch(arrow_b)

            # 标注
            ax.text(5, 1.5, '独立内存', ha='center', va='center',
                   fontsize=10, color='green', fontweight='bold')

        else:  # None - 视情况而定
            # 绘制两种可能性
            # 左侧：连续时共享
            ax.text(2.5, 7, '连续时:', ha='center', fontsize=9, fontweight='bold')
            memory_shared = FancyBboxPatch((1, 4.5), 3, 1.5,
                                          boxstyle="round,pad=0.1",
                                          facecolor='yellow',
                                          edgecolor='red',
                                          linewidth=1.5)
            ax.add_patch(memory_shared)
            ax.text(2.5, 5.25, 'Shared', ha='center', va='center', fontsize=8)

            # 右侧：不连续时拷贝
            ax.text(7.5, 7, '不连续时:', ha='center', fontsize=9, fontweight='bold')
            memory_a_sep = FancyBboxPatch((6, 4.5), 1.3, 1.5,
                                         boxstyle="round,pad=0.05",
                                         facecolor='lightgreen',
                                         edgecolor='green',
                                         linewidth=1.5)
            memory_b_sep = FancyBboxPatch((7.7, 4.5), 1.3, 1.5,
                                         boxstyle="round,pad=0.05",
                                         facecolor='lightblue',
                                         edgecolor='blue',
                                         linewidth=1.5)
            ax.add_patch(memory_a_sep)
            ax.add_patch(memory_b_sep)
            ax.text(6.65, 5.25, 'A', ha='center', va='center', fontsize=8)
            ax.text(8.35, 5.25, 'B', ha='center', va='center', fontsize=8)

            ax.text(5, 1.5, '取决于内存布局', ha='center', va='center',
                   fontsize=10, color='orange', fontweight='bold')

    plt.tight_layout()
    plt.savefig('/mnt/data/djw/experts_sched/ktransformers/kt-kernel/test/tensor_memory_sharing.png',
                dpi=300, bbox_inches='tight')
    print("图表已保存到: tensor_memory_sharing.png")
    plt.close()


def create_decision_flowchart():
    """创建决策流程图"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    fig.suptitle('PyTorch Tensor 赋值：如何选择正确的操作', fontsize=14, fontweight='bold')

    # 起点
    start_box = FancyBboxPatch((3.5, 10.5), 3, 0.8,
                              boxstyle="round,pad=0.1",
                              facecolor='lightblue',
                              edgecolor='black',
                              linewidth=2)
    ax.add_patch(start_box)
    ax.text(5, 10.9, '需要赋值操作', ha='center', va='center',
           fontsize=11, fontweight='bold')

    # 决策1：是否需要独立数据
    decision1 = FancyBboxPatch((3, 9), 4, 0.8,
                              boxstyle="round,pad=0.1",
                              facecolor='lightyellow',
                              edgecolor='black',
                              linewidth=2)
    ax.add_patch(decision1)
    ax.text(5, 9.4, '需要独立的数据副本？', ha='center', va='center',
           fontsize=10, fontweight='bold')

    arrow = FancyArrowPatch((5, 10.5), (5, 9.8),
                           arrowstyle='->', mutation_scale=20,
                           color='black', linewidth=1.5)
    ax.add_patch(arrow)

    # 左分支：需要独立副本
    ax.text(2, 8.5, '是', ha='center', fontsize=9, color='green', fontweight='bold')
    arrow_yes = FancyArrowPatch((3.5, 9), (2, 8.5),
                               arrowstyle='->', mutation_scale=20,
                               color='green', linewidth=1.5)
    ax.add_patch(arrow_yes)

    # 子决策：是否需要梯度
    decision2 = FancyBboxPatch((0.5, 7), 3, 0.8,
                              boxstyle="round,pad=0.1",
                              facecolor='lightgreen',
                              edgecolor='green',
                              linewidth=2)
    ax.add_patch(decision2)
    ax.text(2, 7.4, '需要保留梯度？', ha='center', va='center',
           fontsize=9, fontweight='bold')
    arrow_to_grad = FancyArrowPatch((2, 8.2), (2, 7.8),
                                   arrowstyle='->', mutation_scale=15,
                                   color='green', linewidth=1.5)
    ax.add_patch(arrow_to_grad)

    # 梯度相关选择
    result_box1 = FancyBboxPatch((0.2, 5.8), 1.3, 0.8,
                                boxstyle="round,pad=0.05",
                                facecolor='#90EE90',
                                edgecolor='green',
                                linewidth=1.5)
    ax.add_patch(result_box1)
    ax.text(0.85, 6.2, 'clone()', ha='center', va='center',
           fontsize=9, fontweight='bold')

    result_box2 = FancyBboxPatch((1.7, 5.8), 2, 0.8,
                                boxstyle="round,pad=0.05",
                                facecolor='#90EE90',
                                edgecolor='green',
                                linewidth=1.5)
    ax.add_patch(result_box2)
    ax.text(2.7, 6.2, 'detach().clone()', ha='center', va='center',
           fontsize=9, fontweight='bold')

    ax.text(0.85, 5.3, '保留梯度', ha='center', fontsize=8)
    ax.text(2.7, 5.3, '不需梯度', ha='center', fontsize=8)

    # 右分支：可以共享数据
    ax.text(6.5, 8.5, '否', ha='center', fontsize=9, color='red', fontweight='bold')
    arrow_no = FancyArrowPatch((6.5, 9), (6.5, 8.5),
                              arrowstyle='->', mutation_scale=20,
                              color='red', linewidth=1.5)
    ax.add_patch(arrow_no)

    # 子决策：是否需要改变形状
    decision3 = FancyBboxPatch((5, 7), 3, 0.8,
                              boxstyle="round,pad=0.1",
                              facecolor='#FFE4E1',
                              edgecolor='red',
                              linewidth=2)
    ax.add_patch(decision3)
    ax.text(6.5, 7.4, '需要改变形状？', ha='center', va='center',
           fontsize=9, fontweight='bold')
    arrow_to_shape = FancyArrowPatch((6.5, 8.2), (6.5, 7.8),
                                    arrowstyle='->', mutation_scale=15,
                                    color='red', linewidth=1.5)
    ax.add_patch(arrow_to_shape)

    # 形状相关选择
    result_box3 = FancyBboxPatch((4.5, 5.8), 1.5, 0.8,
                                boxstyle="round,pad=0.05",
                                facecolor='#FFB6C1',
                                edgecolor='red',
                                linewidth=1.5)
    ax.add_patch(result_box3)
    ax.text(5.25, 6.2, 'view()', ha='center', va='center',
           fontsize=9, fontweight='bold')

    result_box4 = FancyBboxPatch((6.2, 5.8), 2, 0.8,
                                boxstyle="round,pad=0.05",
                                facecolor='#FFB6C1',
                                edgecolor='red',
                                linewidth=1.5)
    ax.add_patch(result_box4)
    ax.text(7.2, 6.2, 'reshape()', ha='center', va='center',
           fontsize=9, fontweight='bold')

    ax.text(5.25, 5.3, '确保连续', ha='center', fontsize=8)
    ax.text(7.2, 5.3, '更灵活', ha='center', fontsize=8)

    # 其他常见场景
    other_box = FancyBboxPatch((1, 3.5), 8, 0.6,
                              boxstyle="round,pad=0.1",
                              facecolor='lightgray',
                              edgecolor='black',
                              linewidth=1.5)
    ax.add_patch(other_box)
    ax.text(5, 3.8, '其他常见场景', ha='center', va='center',
           fontsize=10, fontweight='bold')

    # 场景列表
    scenarios = [
        ('就地更新现有tensor', 'a[:] = b 或 a.copy_(b)', 2.5),
        ('只需要引用', 'a = b', 1.7),
        ('从计算图分离', 'a = b.detach()', 0.9),
        ('切片/索引', 'a = b[start:end]', 0.1),
    ]

    for text, operation, y_pos in scenarios:
        ax.text(1.5, y_pos, f'• {text}:', ha='left', fontsize=9, fontweight='bold')
        ax.text(5.5, y_pos, operation, ha='left', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))

    plt.tight_layout()
    plt.savefig('/mnt/data/djw/experts_sched/ktransformers/kt-kernel/test/tensor_decision_flowchart.png',
                dpi=300, bbox_inches='tight')
    print("流程图已保存到: tensor_decision_flowchart.png")
    plt.close()


def create_summary_table():
    """创建总结表格"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')

    # 表格数据
    operations = [
        ['a = b', '✓', '✓', '✓', 'Python引用赋值'],
        ['a = b.clone()', '✗', '✗', '✗', '深拷贝，创建独立副本'],
        ['a = b.detach()', '✗', '✓', '✓', '分离计算图，共享数据'],
        ['a[:] = b', '✗', '✗', '✗', '就地拷贝数据到a'],
        ['a[i] = b', '✗', '✗', '✗', '索引赋值（拷贝）'],
        ['a.copy_(b)', '✗', '✗', '✗', '就地拷贝方法'],
        ['a = b.view(...)', '✗', '✓', '✓', '视图（要求连续）'],
        ['a = b[:]', '✗', '✓', '✓', '切片（共享存储）'],
        ['a = b.reshape(...)', '✗', '视情况', '视情况', '灵活形状变换'],
        ['a = b.data', '✗', '✓', '✓', '访问数据（不推荐）'],
    ]

    headers = ['操作', '同一对象', '共享数据指针', '共享存储', '说明']

    # 创建表格
    table = ax.table(cellText=operations,
                    colLabels=headers,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])

    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # 设置表头样式
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')

    # 设置行样式
    for i in range(1, len(operations) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
            else:
                cell.set_facecolor('#F2F2F2')

            # 高亮共享内存的情况
            if j in [1, 2, 3]:
                text = cell.get_text().get_text()
                if text == '✓':
                    cell.set_facecolor('#FFE4E1')
                    cell.set_text_props(color='red', weight='bold')
                elif text == '✗':
                    cell.set_facecolor('#E0FFE0')
                    cell.set_text_props(color='green', weight='bold')

    plt.title('PyTorch Tensor 赋值操作完整对比表',
             fontsize=14, fontweight='bold', pad=20)

    plt.savefig('/mnt/data/djw/experts_sched/ktransformers/kt-kernel/test/tensor_summary_table.png',
                dpi=300, bbox_inches='tight')
    print("对比表已保存到: tensor_summary_table.png")
    plt.close()


if __name__ == "__main__":
    print("正在生成 PyTorch tensor 赋值行为可视化图表...")

    create_memory_sharing_comparison()
    create_decision_flowchart()
    create_summary_table()

    print("\n所有图表生成完成！")
    print("\n生成的文件:")
    print("1. tensor_memory_sharing.png - 内存共享对比图")
    print("2. tensor_decision_flowchart.png - 操作选择决策流程图")
    print("3. tensor_summary_table.png - 完整对比表")
