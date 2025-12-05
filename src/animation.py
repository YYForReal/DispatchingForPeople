import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端避免 Tkinter 问题
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import copy

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# import main
import main as main

import os
from material import MaterialPackage
import json
import itertools
import file_utils
stone_list = main.stone_list
distance_matrix = copy.deepcopy(main.distance_matrix_copy)

# 是否需要看上次的动画
repeat = True



# 读取配置文件
f = open('config.json', encoding="utf-8")
config_dic = json.load(f)

# 地图上的全部节点数量（受灾点 + 补给点）
ROBOT_A_CAPACITY = config_dic["ROBOT_A_CAPACITY"]
ROBOT_B_CAPACITY = config_dic["ROBOT_B_CAPACITY"]
ROBOT_C_CAPACITY = config_dic["ROBOT_C_CAPACITY"]

total_time = 0
end_flag = False


def update(frame, nodes, robots, affected_scatter ,robot_scatter, ax):
    global end_flag
    if end_flag == True:
        return False

    global total_time
    total_time += 1
    if total_time >= main.MAX_TIME and end_flag == False:
        print("结束")
        end_flag = True
        return False

    # 更新线段
    # draw_lines(nodes,ax)

    for robot in robots:
        robot.move(stone_list)

    x_nodes = [node.x for node in nodes]
    y_nodes = [node.y for node in nodes]

    x_robots = [robot.x for robot in robots]
    y_robots = [robot.y for robot in robots]

    new_positions = np.c_[x_robots, y_robots]
    robot_scatter.set_offsets(new_positions)



    # scatter.set_offsets([x_nodes + x_robots, y_nodes + y_robots])

    # 设置颜色
    color_list = []
    for node in main.map_nodes[:main.AFFECTED_NUMBER]:
        # 如果仍有需求
        if node.need.cal_anxiety_rate() > 0:
            color_list.append("red")
        else:
            color_list.append("yellow")

    affected_scatter.set_color(color_list)

    # marker_list = [[] + ['s' for node in range(main.AFFECTED_NUMBER)] +
    #                ['^' for node in range(main.SUPPLE_NUMBER)] +
    #                ['o' for robot in robots]]

    # marker_list = ['s'] * num + ['^'] * (len(nodes) - num)

    # scatter.set_marker(marker_list)

    # 设置形状

    # # scatter._markers = marker_list
    # for x in marker_list:
    #     scatter.set_marker(x)
    # scatter.set_color(['red' for node in range(main.AFFECTED_NUMBER)] +
    #                   ['green' for node in range(main.SUPPLE_NUMBER)] +
    #                   ['blue' for robot in robots])

    # 创图形
    # fig, ax = plt.subplots(figsize=(600, 560), dpi=100)
    # fig.canvas.manager.window.move(50, 50)
    #

def show(nodes, robots):
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))


    # 绘制受灾点
    x1 = [node.x for node in nodes[:main.AFFECTED_NUMBER]]
    y1 = [node.y for node in nodes[:main.AFFECTED_NUMBER]]
    affected_scatter = ax.scatter(x1, y1, c='red',marker="s")

    # 绘制补给点
    x2 = [node.x for node in nodes[main.AFFECTED_NUMBER:]]
    y2 = [node.y for node in nodes[main.AFFECTED_NUMBER:]]
    supple_scatter = ax.scatter(x2, y2, s=30, c='green',marker='^')

    # 绘制机器人
    x3 = [robot.x for robot in robots]
    y3 = [robot.y for robot in robots]
    robot_scatter = ax.scatter(x3, y3, c='blue',marker="o")

    # 绘制节点之间的线段,
    draw_lines(nodes, ax)

    # 绘制石头（圆形）
    x = [stone.x for stone in stone_list]
    y = [stone.y for stone in stone_list]
    radii = [stone.radius for stone in stone_list]
    for i, (xi, yi, radius) in enumerate(zip(x, y, radii)):
        circle = plt.Circle((xi, yi), radius, alpha=0.5)
        ax.add_artist(circle)

    # 设置参数
    ax.set_title("突发性灾难地图")
    ax.set_xlabel("x轴（单位：百米）")
    ax.set_ylabel("y轴（单位：百米）")
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 160)
    # 字体设置已在文件开头配置

    # 标注节点的name
    for node in nodes:
        ax.text(node.x + 2, node.y + 2, node.name, ha='center', va='center', fontsize=10)

    # 保存静态图像
    plt.savefig('../data/disaster_map.png', dpi=300, bbox_inches='tight')
    print("地图已保存到 ../data/disaster_map.png")

    # 注释掉动画部分，避免显示问题
    # input("输入回车：开始绘制动画")
    # 创建动画
    # ani = animation.FuncAnimation(fig, update, frames=100, repeat=True, fargs=(nodes, robots, affected_scatter,robot_scatter, ax), interval=20)
    # 显示动画
    # plt.show()


def draw_lines(nodes, ax):
    x = [node.x for node in nodes]
    y = [node.y for node in nodes]
    # 绘制节点之间的线段,
    for i in range(len(nodes) - 1):
        for j in range(i + 1, len(nodes)):
            if distance_matrix[i][j] != main.MAX_INF:
                # print("非无穷远")
                ax.plot([x[i], x[j]], [y[i], y[j]], linestyle='--', linewidth=0.5)
            # else:
            # print(f"节点{i}与节点{j}距离为无穷远")
    # 绘制节点之间的连线
    # for node1, node2 in itertools.combinations(nodes, 2):
    #     ax.plot([node1.x, node2.x], [node1.y, node2.y],linestyle='--', linewidth=0.5)

if __name__ == '__main__':
    chromosome = None
    # 复用结果
    if repeat:
        main.init_all()
        path = file_utils.read_data()["best_chromosome"]
        print(path)
        chromosome = main.Chromosome(path)
    else:
        chromosome = main.main()

    main.IS_ANIMATION = True
    print(str(chromosome))

    map_nodes = main.map_nodes
    main.reset_all_need()
    robots = []
    start = None
    destination = None
    tasks = list()
    # 构造机器人及其任务
    for i in range(len(chromosome.path)):
        if np.issubdtype(type(chromosome.path[i]), np.integer) == False:
            # 如果是字母的话就是机器人
            robot = main.Robot(tasks, speed=1,
                               max_carry=MaterialPackage(ROBOT_A_CAPACITY, ROBOT_B_CAPACITY, ROBOT_C_CAPACITY),
                               name=chromosome.path[i])
            robots.append(robot)
            tasks = list()
        else:
            # 如果是数字就是一个地点
            tasks.append(chromosome.path[i])

    for node in main.map_nodes_backup[:main.AFFECTED_NUMBER]:
        print(f"{node.name} need: {str(node.need)}")

    print("robots tasks:")
    for robot in robots:
        print("robot:", end=" ")
        for index in robot.tasks:
            print(map_nodes[index].name, end=" ")
        # print(str( ) ,end=" ")
        print("")


    show(map_nodes, robots)
