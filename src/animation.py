import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import copy
import main

total_time = 0
def update(frame,nodes,robots,scatter):
    for robot in robots:
        robot.move()

    x_nodes = [node.x for node in nodes]
    y_nodes = [node.y for node in nodes]

    x_robots = [robot.x for robot in robots]
    y_robots = [robot.y for robot in robots]

    scatter.set_offsets(np.c_[x_nodes + x_robots, y_nodes + y_robots])
    scatter.set_color(['red' for node in range(main.AFFECTED_NUMBER)] +
                      ['green' for node in range(main.SUPPLE_NUMBER)] +
                      ['blue' for robot in robots])

def show(nodes,robots):

    # 创建节点数据
    x = [node.x for node in nodes]
    y = [node.y for node in nodes]

    # 创建图形
    fig, ax = plt.subplots()
    # fig, ax = plt.subplots(figsize=(600, 560), dpi=100)
    # fig.canvas.manager.window.move(50, 50)

    scatter = ax.scatter(x, y, c='red')
    # 设置参数
    ax.set_title("地图节点")
    ax.set_xlabel("x轴")
    ax.set_ylabel("y轴")
    ax.set_xlim(0,180)
    ax.set_ylim(0,180)
    # 设置绘制的参数
    plt.rcParams["font.sans-serif"] = "SimHei"
    plt.rcParams["axes.unicode_minus"] = False
    # plt.title("地图节点")
    # plt.xlabel("纬度")
    # plt.ylabel("经度")

    # 标注节点的name
    for node in nodes:
        ax.text(node.x + 2, node.y + 2, node.name, ha='center', va='center', fontsize=10)

    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=100, repeat=True,fargs=(nodes,robots,scatter),interval=20)

    # 显示动画
    plt.show()

if __name__ == '__main__':
    # 引入数据
    chromosome = main.main()
    # print("Look")
    # print(chromosome.path)
    # print("node")
    # main2.map_nodes = copy.deepcopy(main2.map_nodes_backup)
    map_nodes = copy.deepcopy(main.map_nodes_backup)
    map_nodes = main.map_nodes
    main.resetAllNeed()
    robots = []
    start = None
    destination = None
    tasks = list()
    # 构造机器人及其任务
    for i in range(len(chromosome.path)):
        if type(chromosome.path[i]) != int:
            # 如果是字母的话就是机器人
            robot = main.Robot(tasks, speed=1, max_carry=120)
            robots.append(robot)
            tasks = list()
        else:
            # 如果是数字就是一个地点
            tasks.append(chromosome.path[i])

    for node in main.map_nodes_backup[:main.AFFECTED_NUMBER]:
        print("need:" + str(node.need))

    print("robots tasks:")
    for robot in robots:
        print(str(robot.tasks) ,end=" ")
        print(robot.carry)

    show(map_nodes,robots)