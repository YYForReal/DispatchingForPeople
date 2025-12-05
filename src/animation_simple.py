import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import copy

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import main as main
from material import MaterialPackage
import file_utils

def show_static_map():
    """显示静态地图，避免动画相关的问题"""

    # 重新加载数据
    file_utils.read_data()
    map_nodes = copy.deepcopy(main.map_nodes_backup)

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))

    # 绘制受灾点
    x1 = [node.x for node in map_nodes[:main.AFFECTED_NUMBER]]
    y1 = [node.y for node in map_nodes[:main.AFFECTED_NUMBER]]
    affected_scatter = ax.scatter(x1, y1, c='red', marker="s", s=100, label='受灾点', alpha=0.7)

    # 绘制补给点
    x2 = [node.x for node in map_nodes[main.AFFECTED_NUMBER:]]
    y2 = [node.y for node in map_nodes[main.AFFECTED_NUMBER:]]
    supple_scatter = ax.scatter(x2, y2, s=100, c='green', marker='^', label='补给点', alpha=0.7)

    # 绘制石头障碍物
    for stone in main.stone_list:
        circle = plt.Circle((stone.x, stone.y), stone.radius,
                          color='gray', alpha=0.5, label='障碍物')
        ax.add_artist(circle)

    # 绘制连接线
    draw_lines(map_nodes, ax)

    # 设置参数
    ax.set_title("应急救援物资调度地图", fontsize=16, fontweight='bold')
    ax.set_xlabel("x轴（单位：百米）", fontsize=12)
    ax.set_ylabel("y轴（单位：百米）", fontsize=12)
    ax.set_xlim(-10, 170)
    ax.set_ylim(-10, 170)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # 字体设置已在文件开头配置

    # 标注节点信息
    for i, node in enumerate(map_nodes):
        if i < main.AFFECTED_NUMBER:
            label = f"受灾点 {node.name}\n需求: {node.need.A_material.quantity:.1f}"
        else:
            label = f"补给点 {node.name}"

        ax.annotate(label, (node.x, node.y),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # 保存图像
    plt.tight_layout()
    plt.savefig('../data/disaster_map_static.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("静态地图已保存到 ../data/disaster_map_static.png")

    # 也可以保存为PDF格式
    plt.savefig('../data/disaster_map_static.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("静态地图已保存到 ../data/disaster_map_static.pdf")

    plt.close()

def draw_lines(nodes, ax):
    """绘制节点之间的连接线"""
    for i in range(len(nodes) - 1):
        for j in range(i + 1, len(nodes)):
            if main.distance_matrix_copy[i][j] != main.MAX_INF:
                # 只绘制直接相连的节点
                x = [nodes[i].x, nodes[j].x]
                y = [nodes[i].y, nodes[j].y]
                ax.plot(x, y, 'b--', alpha=0.2, linewidth=0.5)

def print_statistics():
    """打印统计信息"""
    print("\n=== 地图统计信息 ===")
    print(f"受灾点数量: {main.AFFECTED_NUMBER}")
    print(f"补给点数量: {main.SUPPLE_NUMBER}")
    print(f"障碍物数量: {len(main.stone_list)}")

    total_emergency = sum(node.need.A_material.quantity for node in main.map_nodes_backup[:main.AFFECTED_NUMBER])
    total_regular = sum(node.need.B_material.quantity for node in main.map_nodes_backup[:main.AFFECTED_NUMBER])
    total_equipment = sum(node.need.C_material.quantity for node in main.map_nodes_backup[:main.AFFECTED_NUMBER])

    print(f"\n总需求量:")
    print(f"  应急物资: {total_emergency:.2f}")
    print(f"  常规物资: {total_regular:.2f}")
    print(f"  设备物资: {total_equipment:.2f}")

if __name__ == "__main__":
    print("生成静态救援地图...")
    print_statistics()
    show_static_map()
    print("\n地图生成完成！")