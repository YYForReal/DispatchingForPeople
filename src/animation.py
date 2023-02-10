import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import copy
import main2

total_time = 0
def update(frame,nodes,robots,scatter):
    for robot in robots:
        robot.move()

    x_nodes = [node.x for node in nodes]
    y_nodes = [node.y for node in nodes]

    x_robots = [robot.x for robot in robots]
    y_robots = [robot.y for robot in robots]

    scatter.set_offsets(np.c_[x_nodes + x_robots, y_nodes + y_robots])
    scatter.set_color(['red' for node in range(main2.affected_number)]  +
                      ['green' for node in range(main2.supple_number)]  +
                      ['blue' for robot in robots])

def show(nodes,robots):

    # 创建节点数据
    x = [node.x for node in nodes]
    y = [node.y for node in nodes]

    # 创建图形
    fig, ax = plt.subplots()
    scatter = ax.scatter(x, y, c='red')

    # 设置参数
    ax.set_title("地图节点")
    ax.set_xlabel("纬度")
    ax.set_ylabel("经度")
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
    ani = animation.FuncAnimation(fig, update, frames=100, repeat=True,fargs=(nodes,robots,scatter))

    # 显示动画
    plt.show()

if __name__ == '__main__':
    # 引入数据
    chromosome = main2.main()
    print("Look")
    print(chromosome.path)
    print("node")
    # main2.map_nodes = copy.deepcopy(main2.map_nodes_backup)
    map_nodes = copy.deepcopy(main2.map_nodes_backup)
    map_nodes = main2.map_nodes
    main2.resetAllNeed()
    robots = []
    start = None
    destination = None
    tasks = list()
    # 构造机器人及其任务
    for i in range(len(chromosome.path)):
        if type(chromosome.path[i]) != int:
            # 如果是字母的话就是机器人
            robot = main2.Robot(tasks, speed=1, max_carry=150)
            robots.append(robot)
            tasks = list()
        else:
            # 如果是数字就是一个地点
            tasks.append(chromosome.path[i])
    print("robots")
    print(robots)

    show(map_nodes,robots)