import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import copy

# import main
import main as main

import os
from material import MaterialPackage
import json
import itertools

stone_list = main.stone_list
distance_matrix = copy.deepcopy(main.distance_matrix_copy)

# 读取配置文件
f = open('config.json', encoding="utf-8")
config_dic = json.load(f)

# 地图上的全部节点数量（受灾点 + 补给点）
ROBOT_A_CAPACITY = config_dic["ROBOT_A_CAPACITY"]
ROBOT_B_CAPACITY = config_dic["ROBOT_B_CAPACITY"]
ROBOT_C_CAPACITY = config_dic["ROBOT_C_CAPACITY"]

total_time = 0
end_flag = False


def update(frame, nodes, robots, scatter, ax):
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

    scatter.set_offsets(np.c_[x_nodes + x_robots, y_nodes + y_robots])

    color_list = []
    for node in main.map_nodes[:main.AFFECTED_NUMBER]:
        # 如果仍有需求
        if node.need.cal_anxiety_rate() > 0:
            color_list.append("red")
        else:
            color_list.append("yellow")

    scatter.set_color(color_list +
                      ['green' for node in range(main.SUPPLE_NUMBER)] +
                      ['blue' for robot in robots])

    # scatter.set_color(['red' for node in range(main.AFFECTED_NUMBER)] +
    #                   ['green' for node in range(main.SUPPLE_NUMBER)] +
    #                   ['blue' for robot in robots])


def show(nodes, robots):
    # 创建节点数据
    x = [node.x for node in nodes]
    y = [node.y for node in nodes]

    # 创建图形
    fig, ax = plt.subplots()
    # fig, ax = plt.subplots(figsize=(600, 560), dpi=100)
    # fig.canvas.manager.window.move(50, 50)

    scatter = ax.scatter(x, y, c='red')

    # 绘制节点之间的线段,
    draw_lines(nodes, ax)
    # for i in range(len(nodes) - 1):
    #     for j in range(i + 1, len(nodes)):
    #         if distance_matrix[i][j] != main.MAX_INF:
    #             # print("非无穷远")
    #             ax.plot([x[i], x[j]], [y[i], y[j]], linestyle='--', linewidth=0.5)
    #         # else:
    #             print(f"节点{i}与节点{j}距离为无穷远")

    # 绘制石头
    # 创建节点数据
    x = [stone.x for stone in stone_list]
    y = [stone.y for stone in stone_list]
    radii = [stone.radius for stone in stone_list]

    # 绘制石头圆圈
    for i, (xi, yi, radius) in enumerate(zip(x, y, radii)):
        circle = plt.Circle((xi, yi), radius, alpha=0.5)
        ax.add_artist(circle)

    # 绘制节点之间的连线
    # for node1, node2 in itertools.combinations(nodes, 2):
    #     ax.plot([node1.x, node2.x], [node1.y, node2.y],linestyle='--', linewidth=0.5)

    # 设置参数
    ax.set_title("地图节点")
    ax.set_xlabel("x轴")
    ax.set_ylabel("y轴")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 180)
    # 设置绘制的参数
    plt.rcParams["font.sans-serif"] = "SimHei"
    plt.rcParams["axes.unicode_minus"] = False
    # plt.title("地图节点")
    # plt.xlabel("纬度")
    # plt.ylabel("经度")

    # 标注节点的name
    for node in nodes:
        ax.text(node.x + 2, node.y + 2, node.name, ha='center', va='center', fontsize=10)

    input("输入回车：开始绘制动画")

    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=100, repeat=True, fargs=(nodes, robots, scatter, ax), interval=20)

    # 显示动画
    plt.show()


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


if __name__ == '__main__':
    # 引入数据
    chromosome = main.main()
    main.IS_ANIMATION = True
    print(str(chromosome))
    # print("Look")
    # print(chromosome.path)
    # print("node")
    # main2.map_nodes = copy.deepcopy(main2.map_nodes_backup)
    # map_nodes = copy.deepcopy(main.map_nodes_backup)

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
