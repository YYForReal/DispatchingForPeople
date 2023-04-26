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
ANXIETY_RATE = config_dic["ANXIETY_RATE"]  # 焦虑幂指速率

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


def reset_dispactching_data():
    global now_time, anxiety_arr
    now_time = 0
    anxiety_arr = [0] * len(map_nodes)


# 返回中转的节点？
# def go_to_node(i,j):
#     if distance_matrix[i][j] == MAX_INF:
#         path = []
#         for k in range(len(map_nodes)):
#             if (distance_matrix[i][k] + distance_matrix[k][j])


# 定义染色体结构
class Chromosome:
    def __init__(self, path):
        self.path = path  # 路径： abc1 ef2 ght3
        self.fitness = None
        # print(f"Chromosome: {self.path}")

    # 版本1：计算适应度
    def calculate_fitness(self):
        distance = 0
        for i in range(len(self.path) - 1):
            # print(f"{str(self.path[i])}  -- {str(self.path[i+1])}")
            distance += distance_matrix[int(self.path[i])][int(self.path[i + 1])]
        self.fitness = distance

    # 版本2：计算焦虑度 TODO
    def calculate_anxiety(self):
        anxiety = 0
        cost_time = 0
        for i in range(len(self.path) - 1):
            # print(f"{str(self.path[i])}  -- {str(self.path[i+1])}")
            anxiety += distance_matrix[int(self.path[i])][int(self.path[i + 1])]
        self.fitness = anxiety

    # 版本3：计算适应度
    def evaluate_fitness(self):
        robots = []
        running_robots = []
        start = None
        destination = None
        tasks = list()

        # 构造机器人及其任务
        for i in range(len(self.path)):
            # if type(self.path[i]) != int: # 踩坑点：不能判断出numpy.int32类型 ,isinstance(self.path[i],int)  这个也不可以
            if np.issubdtype(type(self.path[i]), np.integer) == False:
                # 如果是字母的话就是机器人
                # distance = distances[start][destination]
                robot = Robot(tasks, speed=1,
                              max_carry=MaterialPackage(ROBOT_A_CAPACITY, ROBOT_B_CAPACITY, ROBOT_C_CAPACITY),
                              start=AFFECTED_NUMBER, name=self.path[i])
                # robot = Robot(start, destination, distance, speed)
                robots.append(robot)
                running_robots.append(robot)
                tasks = list()
            else:
                # 如果是数字就是一个地点
                tasks.append(self.path[i])

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
        # print("before path" + str(self.path))

        # 当前假设是机器人跑完所有的路程即结束，限定时间段
        # 每轮机器人抵达目标时均会记录各个顶点的焦虑度
        while now_time < MAX_TIME:
            if not running_robots:
                break
            for robot in running_robots:
                canContinue = robot.move(stone_list)
                if robot.task_index >= len(robot.tasks):
                    # if isTest:
                    #     print(robot)
                    #     input("移除robot")
                    running_robots.remove(robot)
                # if canContinue:
                #     running_robots.remove(robot)
            now_time += 1

        for i in range(AFFECTED_NUMBER):
            anxiety_arr[i] += map_nodes[i].cal_people_anxiety(now_time)

        # if now_time == MAX_TIME:
        # print("到达时间上限");

        # TODO: 最后计算一次各个节点的焦虑度


        self.path = []
        for i in range(len(robots)):
            robot = robots[i]
            self.path.extend(robot.tasks)
            self.path.extend(robot.name)

        # 去重
        for i in range(len(self.path) - 1, 0, -1):
            if self.path[i] == self.path[i - 1]:
                self.path.pop(i)

        # print("after path" + str(self.path))

        # print(str(self) + " 最终焦虑度：" + str(anxiety_arr))
        # total_time = sum(robot.elapsed_time for robot in robots)
        # self.fitness = total_time
        self.fitness = sum(anxiety_arr)
        return now_time

    def __str__(self):
        path = self.path

        temp_list = []
        result = []

        for item in path:
            if isinstance(item, str):
                result.append(item + ':' + '->'.join(map(str, temp_list)))
                temp_list = []
            else:
                temp_list.append(item)
        return "[chromosome] " + '\n'.join(result) + '\n'
        # return "->".join([str(i) for i in self.path])


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
                if map_nodes[des].is_supple == False and map_nodes[des].need == 0 and self.task_index < len(self.tasks) - 1:
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

                        # 设置机器人当前的状态为返程补给状态
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
    if USE_OLD_RESULT == True and RANDOM_MODE == 2 and read_distance_matrix and read_distance_matrix_initial:
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
    file_utils.set_property("distance_matrix",distance_matrix.tolist())
    file_utils.set_property("distance_matrix_initial",distance_matrix_initial.tolist())



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

    # 初始化各个节点的最近补给点
    # for i in range(AFFECTED_NUMBER):
    #     min_node_index = None
    #     min_distance = 2 ** 30
    #     for j in range(AFFECTED_NUMBER, len(map_nodes)):
    #         if distance_matrix[i][j] < min_distance:
    #             min_node_index = j
    #             min_distance = distance_matrix[i][j]
    #     least_distance_supple_node_index[i] = min_node_index
    # print("初始化各节点最近补给点: " + str(least_distance_supple_node_index))


# 定义 计算适应度函数
def calculate_fitness(chromosomes):
    # print("计算适应度")
    for chromosome in chromosomes:
        # 每次计算适应度前先恢复一下地图数据
        global map_nodes
        map_nodes = copy.deepcopy(map_nodes_backup)
        # chromosome.calculate_fitness()
        chromosome.evaluate_fitness()
        # print(f"{str(chromosome)} fitness: {str(chromosome.fitness)}")


# 定义变异函数
def mutate(chromosome):
    mutate_position(chromosome)
    mutate_kind(chromosome)


# 第一种变异：交换本体的位置
def mutate_position(chromosome):
    # 不交换机器人的位置,确保交换的是地点,但是若长时间没选择到，则变异失败
    loop = 0
    while loop < 100:
        # 随机选择两个位置并交换其中的值
        i, j = random.sample(range(len(chromosome.path)), 2)
        if type(chromosome.path[i]) == int and type(chromosome.path[j]) == int:
            chromosome.path[i], chromosome.path[j] = chromosome.path[j], chromosome.path[i]
            break
        loop += 1


# 第二种变异，随机选择节点位置
def mutate_kind(chromosome, range_number=AFFECTED_NUMBER):
    # 若长时间没选择到受灾节点，则变异失败
    loop = 0
    while loop < 100:
        # 随机选择两个位置并交换其中的值
        i = random.sample(range(len(chromosome.path)), 1)[0]
        if type(chromosome.path[i]) == int:
            chromosome.path[i] = random.randint(0, range_number - 1)
            break
        loop += 1


# 将染色体的路径换成路径的列表。（多机器人的额外转化）
def get_list_from_path(path):
    lists = []
    temp_list = []
    for i in range(len(path)):
        if isinstance(path[i], str) and path[i].isalpha():
            temp_list.append(path[i])
            lists.append(temp_list)
            temp_list = []
        else:
            temp_list.append(path[i])
    return lists


def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]


# 定义交叉函数
def crossover(chromosome1, chromosome2):
    path1 = chromosome1.path
    path2 = chromosome2.path
    # 获取两端输入为list的列表 [[1,2,3,'a'],[4,'b'],[5,'6']]
    path1 = get_list_from_path(path1)
    path2 = get_list_from_path(path2)

    # 随机选择交叉点.交换两个方案之间的机器人调度地点，同时保持机器人不变，防止重复调度
    # cross_point = random.randint(1, len(chromosome1.path) - 1)
    index1 = random.randint(0, len(path1) - 1)
    index2 = random.randint(0, len(path2) - 1)
    path1[index1][-1], path2[index2][-1] = path2[index2][-1], path1[index1][-1]
    path1[index1], path2[index2] = path2[index2], path1[index1]
    # chromosome1.path = flatten_list(path1)
    # chromosome2.path = flatten_list(path2)

    # 交叉两个染色体
    # new_chromosome1 = Chromosome(chromosome1.path[:cross_point] + chromosome2.path[cross_point:])
    # new_chromosome2 = Chromosome(chromosome2.path[:cross_point] + chromosome1.path[cross_point:])
    new_chromosome1 = Chromosome(flatten_list(path1))
    new_chromosome2 = Chromosome(flatten_list(path2))

    return new_chromosome1, new_chromosome2


# TODO: 生成初始的路径（轮盘赌思想选同一个区域块的）
def generate_path(nodes, index):
    # 定义受灾节点列表
    nodes = copy.copy([{node} for node in nodes[:AFFECTED_NUMBER]])
    # 假设已经存在邻接矩阵 distance_matrix
    path = [0]  # 起点为0
    copied_matrix = distance_matrix[:AFFECTED_NUMBER, :AFFECTED_NUMBER]
    copied_matrix = np.where(copied_matrix == 0, 1e-10, copied_matrix)  # 将距离矩阵中为0的数值置为一个极小的数字 1e-10，后续要取倒数

    while len(path) < AFFECTED_NUMBER:
        # 计算当前节点到其他节点的距离，并计算每个节点的权重
        last_node = path[-1]
        distances = copied_matrix[last_node]
        weights = 1 / distances

        # weights[last_node] = 0
        # 已经遍历过的节点权重置为0
        for visited in path:
            weights[visited] = 0

        # 使用加权随机算法选择下一个要遍历的节点
        total_weight = np.sum(weights)
        probs = weights / total_weight
        next_node = np.random.choice(np.arange(AFFECTED_NUMBER), p=probs)

        # 将选择的节点加入到 path 列表中
        path.append(next_node)

    return path


# 初始化种群-染色体
def init_population(n_chromosomes, n_cities):
    """
    种群初始化 (数字是地点、字母是机器人)
    :param n_chromosomes: 染色体的个数，即给出的初始随机方案数
    :param n_cities: 受灾点的个数，用于路径的选择，
    :return: 返回一个染色体的列表
    """
    population = []
    # 确定初始条件下每个机器人的负责区域，先分类，然后插进入
    letters = [chr(97 + i) for i in range(ROBOT_NUMBER)]  # 生成字母列表：代表机器人的名字
    insert_positions = sorted(random.sample(range(2, n_cities - 1), ROBOT_NUMBER - 1))

    for i in range(n_chromosomes):
        # TODO: 调优方面可以在初始化上逼近均匀分配
        # path = list(range(n_cities))
        # random.shuffle(path)

        path = generate_path(map_nodes, i)

        # print("受灾点路径生成：" + str(path))

        # 多机器人要用多个字母截断
        path.extend(letters[ROBOT_NUMBER - 1])  # 先插入最后一个，确保机器人能够调度完所有的节点
        # 逆序 插入
        for j in range(ROBOT_NUMBER - 2, -1, -1):
            path.insert(insert_positions[j], letters[j])
        population.append(Chromosome(path))

        # print("全部路径生成：" + str(path))

    return population


# 交叉和变异
def crossover_and_mutate(population):
    new_population = []
    for i in range(0, len(population), 2):
        # 如果只剩最后一个，就不交叉变异了
        if i + 1 >= len(population):
            new_population.append(population[i])
            break
        # 随机决定是否进行交叉
        if random.random() < CROSSOVER_RATE:
            # 进行交叉
            offspring1, offspring2 = crossover(population[i], population[i + 1])
        else:
            offspring1, offspring2 = population[i], population[i + 1]
        # 随机决定是否进行变异
        if random.random() < MUTATION_RATE:
            mutate(offspring1)
        if random.random() < MUTATION_RATE:
            mutate(offspring2)
        new_population.append(offspring1)

        new_population.append(offspring2)
    return new_population


# 展示当前的种群染色体（方案）: 路径 + 适应度
def show_population(population):
    for i in range(len(population)):
        chromosome = population[i]
        print(f"{str(chromosome)}适应度：{str(chromosome.fitness)}\n")


# 重置所有受灾点需求
def reset_all_need():
    for i in range(AFFECTED_NUMBER):
        map_nodes[i].reset_need()
        # print(map_nodes[i].need)


# 定义主函数
def main():

    # 初始化地图数据，邻接矩阵 distance_matrix
    init_map()

    # 初始化种群
    population = init_population(CHROMOSOME_NUMBER, AFFECTED_NUMBER)

    # 如果有复用数据
    if USE_OLD_RESULT == True and RANDOM_MODE == 2:
        # 读取原本的json文件内容
        original_data = file_utils.read_data()
        # 混入上次的最佳染色体
        best_chromosome = original_data.get("best_chromosome")
        if best_chromosome is not None:
            print("use old result:" + str(best_chromosome))
            population.pop()
            population.append(Chromosome(best_chromosome))



    # 计算初始种群的适应度值
    calculate_fitness(population)
    # 留下一个适应度最好的
    best_chromosome = population[0]
    # 进行遗传算法迭代
    print("开始遗传算法迭代")
    start_time = time.time()  # 记录程序开始时间

    for i in range(MAX_GENERATION):

        # 选择新一代个体
        new_population = selection_championships(population)
        if len(new_population) == 0:  # 如果筛到最后没了，提前终止
            break
        # print("----------------------------------------")
        # print("选择新一代个体")
        # show_population(new_population)

        # 进行交叉和变异
        new_population = crossover_and_mutate(new_population)

        # 重新计算 更新后种群的适应度
        calculate_fitness(new_population)

        # print("进行交叉变异后")
        # show_population(new_population)

        # 更新种群
        population = new_population

        # 输出最优解
        best_chromosome = min(min(population, key=lambda x: x.fitness), best_chromosome, key=lambda x: x.fitness)
        if best_chromosome not in population:
            # 增加当前最优解到种群里面，同时移除最差的染色体
            population.append(best_chromosome)
            bad_chromosome = max(population, key=lambda x: x.fitness)
            population.remove(bad_chromosome)

        if i % (MAX_GENERATION / 10) == 0:
            print(f"第{i}次迭代")
            print(f'best path: {best_chromosome}')
            print(f'best fitness: {best_chromosome.fitness}')


    end_time = time.time()  # 记录程序开始时间
    run_time = end_time - start_time
    path_arr = [str(x) for x in best_chromosome.path]  # 将数字转换为字符串

    file_utils.append_result(run_time,str(''.join(path_arr)),best_chromosome.fitness,str(best_chromosome))
    print("程序运行时间为：", run_time, "秒")
    print(f'最佳染色体: {best_chromosome}')
    print(f'最佳适应度: {best_chromosome.fitness}')
    input("回车确认")

    return best_chromosome


if __name__ == '__main__':
    best_chromosome = main()

    # 设置最佳染色体
    best_path = []
    for x in best_chromosome.path:
        if np.issubdtype(type(x), np.integer) == False:
            best_path.append(x)
        else:
            best_path.append(int(x))
    file_utils.set_property("best_chromosome",best_path)
    file_utils.set_property("best_chromosome_fitness", best_chromosome.fitness)

