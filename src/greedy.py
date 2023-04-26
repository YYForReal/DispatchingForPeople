import random
import numpy as np
import copy

from map import Node, map_nodes, AFFECTED_NUMBER, SUPPLE_NUMBER, showMap, stone_list
from selection import selection, selection_roulette, selection_championships
import json
from material import MaterialPackage
import heapq
import os
import time

# 文件操作类
import file_utils

NUM_NODES = AFFECTED_NUMBER + SUPPLE_NUMBER
IS_ANIMATION = False

"""
总：我们定义Chromosome类来表示染色体,并定义了函数
    calculate_fitness：计算适应度
    mutate：表示变异
    crossover：表示交叉
    selection 表示选择新一代个体 ==> 当成模块
待优化点：
    0. 地图节点的数据 随机
    1. 完成多次调度的编码、解码
    2. 完善适应度函数，更好地计算个体的适应度    
    3. 添加遗传算法算子：不同的交叉策略、变异、置换...
    4. 新的停止条件：收敛条件、运行时间限制
    5. 优化参数：交叉概率、变异概率、种群大小
    6. 算法结合：模拟退火算法...    

    7. 当前的物资度量是统一的，以后可以改成列表形式的数据  衣物、被褥、水、食物、帐篷...  
    8. 可视化
    9. 染色体的基因需要拓展成一个新的对象
    10. 计算适应度时的终止时间


整体设想：
    已知：初始的机器人的数量及出发点（必定是补给点）
    染色体编码：仅使用受灾点进行编码
    当抵达目的受灾点时，选择最适合的补给点进行补给后再继续前往受灾点。
    计算染色体适应度即模拟如上。

应急救援物资 主要三类
生活资料类：衣被，毯子、方便食品、救灾帐篷、饮水器械
救生器械类：救生舟、救生船、探生工具，顶升设备，小型起重机器
救治物品类：救治药品和医疗器械。

大多数情况下少数几类物资占据供应量的75%~95%
如01年印度地震救援活动，外界供应救援物资按照重量划分90%是基本生活用品
供应量：水、药品、加氯消毒药片、帐篷、食物

"""

# 读取配置文件
config_dic = file_utils.read_config()

RANDOM_MODE = config_dic["RANDOM_MODE"]  # 复用模式，若复用=2，则存取数据。可以基于上次的最优染色体做进一步优化。
USE_OLD_RESULT = config_dic["USE_OLD_RESULT"]  # 是否使用上一次的结果
ROBOT_NUMBER = config_dic["ROBOT_NUMBER"]  # 机器人数量
CHROMOSOME_NUMBER = config_dic["CHROMOSOME_NUMBER"]  # 染色体（方案）数量
CROSSOVER_RATE = config_dic["CROSSOVER_RATE"]  # 交叉概率
MUTATION_RATE = config_dic["MUTATION_RATE"]  # 变异概率
MAX_TIME = config_dic["MAX_TIME"]  # 设置每一个调度的最大时间限制
MAX_GENERATION = config_dic["MAX_GENERATION"]  # 设置最大迭代次数(如果方案数量筛选到只剩一个就提前结束了)
# ANXIETY_RATE = config_dic["ANXIETY_RATE"]  # 焦虑幂指速率

ROBOT_A_CAPACITY = config_dic["ROBOT_A_CAPACITY"]
ROBOT_B_CAPACITY = config_dic["ROBOT_B_CAPACITY"]
ROBOT_C_CAPACITY = config_dic["ROBOT_C_CAPACITY"]

now_time = 0  # 当前时间
anxiety_arr = [0] * len(map_nodes)  # 各地点的人群累计焦虑值
map_nodes_backup = copy.deepcopy(map_nodes)  # 作一个深拷贝,用于后续恢复

# 图节点
n = len(map_nodes)
distance_matrix = np.zeros((n, n))  # 图的邻接矩阵（各节点的路径代价）
distance_matrix_copy = np.zeros((n, n))  # 图的初始邻接矩阵
distance_matrix_initial = np.zeros((n, n))  # 初始可视化图上的距离

# 佛洛依德思想 --> i->j节点的路径
path = [[[j] if distance_matrix[i][j] != float('inf') else [] for j in range(NUM_NODES)] for i in range(NUM_NODES)]

# least_distance_supple_node_index = [0] * AFFECTED_NUMBER  # 距离受灾点最近的补给点映射 ：补给点下标
priority_supple_node_index = [[0] * SUPPLE_NUMBER for _ in range(AFFECTED_NUMBER)]  # 距离受灾点最近的补给点优先级 ：补给点下标
least_node_index = [[0] * NUM_NODES for _ in range(NUM_NODES)]  # 距离每个节点最近的节点列表

MAX_INF = 999999999

# 当前仍有需求的节点下标列表
now_need_node_index_list = list(range(AFFECTED_NUMBER))

# 模拟调度时的状态
state = {
    # 所有补给点是否物资不足
    "isAllSuppleEnd": False
}

# 贪心的救援顺序
def get_sorted_need_arr():
    # 对map_nodes数组按需求从大到小排序
    sorted_nodes = sorted(range(AFFECTED_NUMBER), key=lambda i: map_nodes[i].need, reverse=True)
    return sorted_nodes


def reset_dispactching_data():
    global now_time, anxiety_arr
    now_time = 0
    anxiety_arr = [0] * len(map_nodes)

# 重置所有受灾点需求
def reset_all_need():
    for i in range(AFFECTED_NUMBER):
        map_nodes[i].reset_need()
        # print(map_nodes[i].need)



def evaluate_fitness(path):
    robots = []
    running_robots = []
    start = None
    destination = None
    tasks = list()

    # 构造机器人及其任务
    for i in range(len(path)):
        if np.issubdtype(type(path[i]), np.integer) == False:
            # 如果是字母的话就是机器人
            # distance = distances[start][destination]
            robot = Robot(tasks, speed=1,
                          max_carry=MaterialPackage(ROBOT_A_CAPACITY, ROBOT_B_CAPACITY, ROBOT_C_CAPACITY),
                          start=AFFECTED_NUMBER, name=path[i])
            # robot = Robot(start, destination, distance, speed)
            robots.append(robot)
            running_robots.append(robot)
            tasks = list()
        else:
            # 如果是数字就是一个地点
            tasks.append(path[i])

    # 每次调度前重置调度数据
    global now_time
    now_time = 0
    global anxiety_arr
    for i in range(len(anxiety_arr)):
        anxiety_arr[i] = 0
    global now_need_node_index_list
    now_need_node_index_list = list(range(AFFECTED_NUMBER))
    global distance_matrix
    distance_matrix = copy.deepcopy(distance_matrix_copy)

    # 当前假设是机器人跑完所有的路程即结束，限定时间段
    # 每轮机器人抵达目标时均会记录各个顶点的焦虑度
    while now_time < MAX_TIME:
        if not running_robots:
            break
        for robot in running_robots:
            canContinue = robot.move(stone_list)
            if robot.task_index >= len(robot.tasks):
                running_robots.remove(robot)
        now_time += 1

    for i in range(AFFECTED_NUMBER):
        anxiety_arr[i] += map_nodes[i].cal_people_anxiety(now_time)

    return [now_time,sum(anxiety_arr)]


# 定义运输机器
class Robot(Node):
    def __init__(self, tasks, speed=1, max_carry=100, start=AFFECTED_NUMBER, name=""):
        """
        :param tasks: 目的地任务序列
        :param speed: 速度: 约 1.2 m/s
        :param max_carry: 最大携带物资量: 约 150kg
        :param start: 当前的机器人的出发点（也是初始的位置）
        """
        Node.__init__(self, map_nodes[start].x, map_nodes[start].y, name)
        self.name = name
        self.tasks = copy.copy(tasks)  # 总共的任务队列
        self.max_carry = max_carry  # 最大携带容量
        self.speed = speed  # 运输机器的速度

        self.start = start  # 当前位处的节点下标（初始位置）

        if len(tasks) > 0:
            self.destination = tasks[0]  # 标记目的地
        else:
            self.destination = self.start
        self.distance = 0  # 这段路行驶路程
        self.elapsed_distance = 0  # 已走总路程

        self.carry = copy.deepcopy(max_carry)  # 当前携带容量,由于初始位于补给点，所以默认是满的
        self.task_index = 0  # 对应染色体的任务下标
        self.x = map_nodes[start].x
        self.y = map_nodes[start].y

        # 机器人当前的状态
        self.state = {
            "is_backing": False,  # 是否处于撤回状态(遇到石头了)
            "need_supply": False  # 是否需要补给
        }

    # 移动
    def move(self, stone_list):
        """
        模拟机器人移动
        :return:
        """
        # 如果处于回撤状态
        if self.state["is_backing"]:
            self.back()
            return True

        # 移动时如果出发点和目的地一样，就表示结束？
        # if self.start == self.destination:
        if self.task_index >= len(self.tasks):
            # print("Finish:",end=" ")
            # 本身任务完成，且地图上所有补给点均无物资，结束。
            if state["isAllSuppleEnd"] == True:
                print(f"robot {self.name} is finish")
                return False

            # 检测是否存在其他未完成的节点，前往该节点进行补给。同时反馈给染色体，动态增加实际染色体的路径
            # 遍历当前节点的优先节点列表
            for index in least_node_index[self.start]:
                # 如果出现了还有其他节点有需求，就去帮忙，同时选择的帮忙节点是优先级最高的（最近）
                if index in now_need_node_index_list:
                    self.tasks.append(index)
                    self.destination = self.tasks[self.task_index]
                    # print(f"准备前往{index} == {self.destination}")
                    # print(self.tasks)
            return True

        self.distance += self.speed
        self.elapsed_distance += self.speed

        # print(f"robot move from {self.start} to {self.destination}")
        if self.distance >= distance_matrix[self.start][self.destination]:
            self.arrive()
        else:
            percentage = self.distance / distance_matrix[self.start][self.destination]
            target_x = map_nodes[self.destination].x
            target_y = map_nodes[self.destination].y
            self.x = map_nodes[self.start].x + percentage * (target_x - map_nodes[self.start].x)
            self.y = map_nodes[self.start].y + percentage * (target_y - map_nodes[self.start].y)

            # 是否在石头内
            for stone in stone_list:
                if stone.calculate_distance(self) <= stone.radius:
                    distance_matrix[self.start][self.destination] = MAX_INF
                    # print(f"{self.start} 和 {self.destination}堵塞。。。")
                    # 修改目标，返回
                    self.state["is_backing"] = True
                    # input("发现石头，开始回撤")
        return True

    # 机器人撤回时的行动
    def back(self):
        # 路程减少
        self.distance -= self.speed
        self.elapsed_distance -= self.speed

        # 计算坐标
        percentage = self.distance / distance_matrix_copy[self.start][self.destination]
        target_x = map_nodes[self.destination].x
        target_y = map_nodes[self.destination].y
        self.x = map_nodes[self.start].x + percentage * (target_x - map_nodes[self.start].x)
        self.y = map_nodes[self.start].y + percentage * (target_y - map_nodes[self.start].y)

        # 如果回撤到起点
        if self.distance <= 0:
            self.state["is_backing"] = False
            # TODO：检测是否可以绕路抵达（默认连通可达，若不连通，则可视为两个区域，需要分别应用该算法）
            # print(str(self.start) + " --》 " + str(self.destination))
            # print("当前任务不可达:" + str(self.task_index))
            # print(self.tasks)
            # input("回撤到起点")
            # now_start =
            # 优先路径依旧是直达，且本身不可达，即不连通
            try:
                if len(path[self.start][self.destination]) == 1:
                    # print("del before  tasks: " + str(self.tasks) + " -- idx: " + str(self.task_index))
                    # print("不连通")
                    # 删除对应任务
                    del self.tasks[self.task_index]
                    # print("del after  tasks: " + str(self.tasks) + " -- idx: " + str(self.task_index))
                    # if self.task_index >= len(self.tasks):
                    #     input()
                    # self.tasks.remove(self.task_index)
                    self.destination = self.start
                else:
                    # 绕路抵达
                    # print("连通，非直达")
                    self.tasks[self.task_index:self.task_index] = path[self.task_index][self.destination][:-1]
                    # print(self.tasks)
                    # print(self.task_index)
                    self.destination = self.tasks[self.task_index]
            except Exception as e:
                print(path)
                print(f"len: {len(path)}")
                print(e)
                input("报错啦")

            # new_task_path = path[self.task_index][:-1]
            # print(new_task_path)
            # self.tasks.insert(self.task_index,new_task_path)
            # print(self.tasks)
            # input()

    # 抵达目的地
    def arrive(self):
        global now_need_node_index_list

        """
        模拟机器人抵达目的地：重置行驶的这段路程，清算物资，判定是否需要行驶到补给点
        :return:
        """
        # 机器人抵达目的地后的重新调度 Show
        # print(f"Robot arrive from {map_nodes[self.start].name} to {map_nodes[self.destination].name}")
        self.distance = 0  # 重置 下一段路已经行驶的路程

        old_start = self.start
        self.start = self.destination  # 更新 下一段路的起始点

        self.x = map_nodes[self.start].x
        self.y = map_nodes[self.start].y
        index = self.destination  #
        arrive_node = map_nodes[index]
        if arrive_node.is_supple:
            # 如果到达的是补给点
            # TODO： 比较补给量，这里先默认补给充裕,
            arrive_node.supple(self)
            # 当前的目标节点
            des = self.tasks[self.task_index]

            # 检测下一步移动的需求
            # 如果下一步的是补给点，则跳过
            # 如果下一步是受灾点，肯定返回前往

            # if IS_ANIMATION:
            #     print("是否为了补给：" + str(self.state["need_supply"]))
            #     print(f"是否没有需求了：{map_nodes[des].is_supple == False and map_nodes[des].need == 0 and self.task_index < len(self.tasks) - 1}")
            #     print(f"不是补给，是顺路吗？）-- {map_nodes[des].is_supple and self.task_index < len(self.tasks) - 1}")
            #     input(self.name + "抵达目的地 -- ")

            # 如果原先是为了补给才抵达的补给点
            if self.state["need_supply"]:
                self.state["need_supply"] = False
                # 已经没有需求了，就不需要再返回去了
                if map_nodes[des].is_supple == False and map_nodes[des].need == 0 and self.task_index < len(
                        self.tasks) - 1:
                    self.task_index += 1
            # 如果不是为了补给（那就是为了中转）
            elif map_nodes[des].is_supple and self.task_index < len(self.tasks) - 1:
                self.task_index += 1

            # self.carry = copy.deepcopy(self.max_carry)
            self.destination = self.tasks[self.task_index]  # 设置前往受灾点，继续进行物资补给

            pass
        else:
            # print(f"carry:{self.carry} , need: {map_nodes[self.destination].need}")
            # 如果到达的是受灾点：

            global anxiety_arr
            anxiety_arr[index] += arrive_node.cal_people_anxiety(now_time)
            arrive_node.last_time_visit = now_time

            is_enough = self.carry - arrive_node.need

            if is_enough:
                now_need_node_index_list = [x for x in now_need_node_index_list if
                                            x != self.destination]  # 清除数值为当前目的地的元素
                # print("剩余的需求点：" + str(now_need_node_index_list))
                self.task_index += 1
                # 已完成本机器人的任务，需要增加一个
                if self.task_index >= len(self.tasks):
                    # 需要增加一个新的需求点
                    if len(now_need_node_index_list) > 0:
                        randomIndex = random.randint(0, len(now_need_node_index_list) - 1)
                        append_node = now_need_node_index_list[randomIndex]
                        if append_node not in self.tasks:
                            self.tasks.append(append_node)
                            # print("push a node : " + str(now_need_node_index_list[randomIndex]))
                    # print(self.tasks)
                    # print(self.task_index)
                    pass
                else:
                    self.destination = self.tasks[self.task_index]
            else:
                # TODO： 是否可以按百分比决策
                # TODO： （暂时）先返回补给点，清空受灾点所有需求后再进行下一个受灾点的补给
                # TODO： 返回哪一个补给点也是问题：暂定是路径优先
                global state
                # 此处按照优先级进行寻找前往的补给点（前提：存在物资）
                if state["isAllSuppleEnd"]:
                    print("所有补给点 无物资")
                    # self.task_index += 1
                    return

                for i in range(len(priority_supple_node_index[self.start])):
                    ready_supple_node_index = priority_supple_node_index[self.start][i]
                    ready_supple_node = map_nodes[ready_supple_node_index]
                    if ready_supple_node.hasMaterial():
                        # TODO: 使用迪杰斯特拉 | 弗洛伊德 || 规划一下路径 --------------------------------------------------------- ！！！！！！！！！！！！

                        # print("原任务")
                        # print(self.tasks)
                        # print(self.task_index)
                        # print(f"原本--> {ready_supple_node_index}")
                        # print(f"从节点{self.start}到{ready_supple_node_index}:")
                        # print(path[self.start][ready_supple_node_index])

                        # task_index = 1
                        # A = [4, 5, 6]
                        # 直达
                        if len(path[self.start][ready_supple_node_index]) == 1:
                            self.destination = ready_supple_node_index
                        else:
                            self.tasks[self.task_index:self.task_index] = path[self.start][ready_supple_node_index]
                            self.destination = self.tasks[self.task_index]

                        # print("新任务")
                        # print(self.tasks)
                        # print(self.task_index)
                        # print(self.destination)
                        # print(str(ready_supple_node) + " 物资充足")

                        # 设置机器人当前的状态为返程补给状态（如果没有设置，则表示直接经过该部分）
                        self.state["need_supply"] = True
                        break
                    else:
                        # print(str(ready_supple_node) + " 物资不足，切换到下一个优先补给点")
                        if i == len(priority_supple_node_index[self.start]) - 1:
                            state["isAllSuppleEnd"] = True
                            print("所有物资已消耗完毕")
                pass

    def __str__(self):
        return self.name + '->'.join(self.tasks) + "    now : " + str(self.task_index)


# 初始化 地图
def init_map():
    global distance_matrix_copy
    global distance_matrix_initial
    global distance_matrix

    # 读取原本的json文件内容
    original_data = file_utils.read_data()
    # 读取带有路径的邻接矩阵
    read_distance_matrix = original_data.get("distance_matrix")
    read_distance_matrix_initial = original_data.get("distance_matrix_initial")

    # 如果需要复用数据 且 具有复用的数据
    if RANDOM_MODE == 2 and read_distance_matrix and read_distance_matrix_initial:
        distance_matrix = np.array(read_distance_matrix)
        distance_matrix_initial = np.array(read_distance_matrix_initial)
        distance_matrix_copy = copy.deepcopy(distance_matrix)
    else:
        # 初始化地图的邻接矩阵
        for i in range(len(map_nodes) - 1):
            for j in range(i + 1, len(map_nodes)):
                a = map_nodes[i]
                b = map_nodes[j]

                dis = a.calculate_distance(b)
                distance_matrix_initial[i][j] = dis
                distance_matrix_initial[j][i] = dis

                # 50%的概率随机堵塞两节点之间的路径
                if random.random() < 0.5:
                    dis = MAX_INF
                distance_matrix[i][j] = dis
                distance_matrix[j][i] = dis

    print("初始化地图节点数据:邻接矩阵 完成")
    distance_matrix_copy = copy.deepcopy(distance_matrix)
    file_utils.set_property("distance_matrix", distance_matrix.tolist())
    file_utils.set_property("distance_matrix_initial", distance_matrix_initial.tolist())

    # 初始化每个节点到另外一个节点的最近路径
    for k in range(NUM_NODES):
        for i in range(NUM_NODES):
            for j in range(NUM_NODES):
                # 如果中间节点路径可达
                if distance_matrix[i][k] != MAX_INF and distance_matrix[k][j] != MAX_INF:
                    # 且距离更近
                    if distance_matrix[i][j] > distance_matrix[i][k] + distance_matrix[k][j]:
                        distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j]
                        path[i][j] = path[i][k] + path[k][j]

    print("各节点之间的路径: " + str(path))

    global priority_supple_node_index
    global least_node_index
    for i in range(AFFECTED_NUMBER):
        heap = []
        for j in range(AFFECTED_NUMBER, len(map_nodes)):
            if j not in priority_supple_node_index[i]:
                heapq.heappush(heap, (distance_matrix[i][j], j))
            if len(heap) > SUPPLE_NUMBER:
                heapq.heappop(heap)
        for j in range(SUPPLE_NUMBER):
            if heap:
                _, min_node_index = heapq.heappop(heap)
                priority_supple_node_index[i][j] = min_node_index
            else:
                priority_supple_node_index[i][j] = -1

    for i in range(NUM_NODES):
        heap = []
        for j in range(NUM_NODES):
            if j != i:
                heapq.heappush(heap, (distance_matrix[i][j], j))
            if len(heap) > NUM_NODES - 1:
                heapq.heappop(heap)
        for j in range(NUM_NODES - 1):
            _, least_node_index[i][j] = heapq.heappop(heap)

    print("各节点优先寻找补给点列表: " + str(priority_supple_node_index))
    print("各节点优先寻找节点列表: " + str(least_node_index))



def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]


# 定义主函数
def main():
    # 初始化地图数据，邻接矩阵 distance_matrix
    init_map()
    start_time = time.time()  # 记录程序开始时间
    need_arr = get_sorted_need_arr()
    print(need_arr)

    path = need_arr + ['a'] + need_arr  + ['b'] + need_arr + ['c']
    print(path)
    now_time, fitness =  evaluate_fitness(path)
    end_time = time.time()  # 记录程序开始时间
    run_time = end_time - start_time

    # file_utils.append_greedy_result(run_time, str(''.join(path_arr)), best_chromosome.fitness, str(best_chromosome))
    print("程序运行时间为：", run_time, "秒")
    print("程序调度时间：",str(now_time),"秒")
    print("焦虑度：", str(fitness))
    input("回车确认")

    return path


if __name__ == '__main__':
    main()

    # 设置最佳染色体
    # best_path = []
    # for x in best_chromosome.path:
    #     if np.issubdtype(type(x), np.integer) == False:
    #         best_path.append(x)
    #     else:
    #         best_path.append(int(x))
    # file_utils.set_property("best_chromosome", best_path)
    # file_utils.set_property("best_chromosome_fitness", best_chromosome.fitness)

