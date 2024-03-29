import random
import numpy as np
import copy

from map import map_nodes, random_node_list, random_node_supple_list, Node, AffectedNode, SuppleNode
from selection import selection, selection_roulette, selection_championships

# 使用随机生成的数据进行模拟
map_nodes = copy.deepcopy(random_node_list)
map_nodes.extend(random_node_supple_list)
affected_number = len(random_node_list)
supple_number = len(random_node_supple_list)
print(f"受灾点个数：{affected_number}")
for i in range(len(map_nodes)):
    node = map_nodes[i]
    if node["is_supple"] == True:
        map_nodes[i] = SuppleNode(node["x"], node["y"], node["name"])
    else:
        map_nodes[i] = AffectedNode(node["x"], node["y"], node["name"], node["population"], node["magnitude"])

map_nodes_backup = copy.deepcopy(map_nodes)  # 作一个深拷贝，后续恢复

print(map_nodes_backup[0].need)
print(map_nodes[0].need)

map_nodes[0].need = 0

print(map_nodes_backup[0].need)
print(map_nodes[0].need)

map_nodes = copy.deepcopy(map_nodes_backup)

print(map_nodes_backup[0].need)
print(map_nodes[0].need)

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
    
    7. 当前的物资度量是统一的，以后可以改成列表形式的数据    
    8. 可视化
    
    
整体设想：
    已知：初始的机器人的数量及出发点（必定是补给点）
    染色体编码：仅使用受灾点进行编码
    当抵达目的受灾点时，选择最适合的补给点进行补给后再继续前往受灾点。
    计算染色体适应度即模拟如上。
"""

CHROMOSOME_NUMBER = 8  # 染色体（方案）数量
CROSSOVER_RATE = 0.8  # 交叉概率
MUTATION_RATE = 0.3  # 变异概率
MAX_TIME = 1000  # 设置最大迭代次数(如果方案数量筛选到只剩一个就提前结束了)
ANXIETY_RATE = 1.1  # 焦虑幂指速率
now_time = 0  # 当前时间

distance_matrix = [0] * len(map_nodes)  # 图的邻接矩阵（各节点的路径代价）
# TODO: 设置一个距离受灾点最近的补给点映射
least_distance_supple_node_index = [0] * affected_number  # 存储 受灾点 距离最近的补给点下标
for i in range(len(distance_matrix)):
    distance_matrix[i] = [0] * len(map_nodes)

anxiety_arr = [0] * len(map_nodes)  # 各地点的人群累计焦虑值


# 定义染色体结构
class Chromosome:
    def __init__(self, path):
        self.path = path  # 路径： abc1 ef2 ght3
        self.fitness = None
        print(f"Chromosome: {self.path}")

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
        start = None
        destination = None
        tasks = list()
        # 构造机器人及其任务
        for i in range(len(self.path)):
            # if self.path[i].isalpha():
            if type(self.path[i]) != int:
                # 如果是字母的话就是机器人
                # distance = distances[start][destination]
                robot = Robot(tasks, speed=1, max_carry=150)
                # robot = Robot(start, destination, distance, speed)
                robots.append(robot)
                tasks = list()
            else:
                # 如果是数字就是一个地点
                tasks.append(self.path[i])

        total_time = 0
        while True:
            if not robots:
                break
            for robot in robots:
                robot.move()
                if robot.task_index >= len(robot.tasks):
                    robots.remove(robot)
            total_time += 1

        # total_time = sum(robot.elapsed_time for robot in robots)
        self.fitness = total_time
        return total_time

    def __str__(self):
        return "->".join([str(i) for i in self.path])


# 定义运输机器
class Robot:
    def __init__(self, tasks, speed=1, max_carry=100, start=affected_number):
        """
        :param tasks: 目的地任务序列
        :param speed: 速度: 约 1.2 m/s
        :param max_carry: 最大携带物资量: 约 150kg
        :param start: 当前的机器人的出发点（也是初始的位置）
        """
        self.tasks = tasks  # 总共的任务队列
        self.max_carry = max_carry  # 最大携带容量
        self.speed = speed  # 运输机器的速度

        self.start = start  # 当前位处的节点下标（初始位置）
        self.destination = tasks[0]  # 标记目的地
        self.distance = 0  # 这段路行驶路程
        self.elapsed_distance = 0  # 已走总路程
        self.carry = max_carry  # 当前携带容量,由于初始位于补给点，所以默认是满的
        self.task_index = 0  # 对应染色体的任务下标
        self.x = map_nodes[start].x
        self.y = map_nodes[start].y

    def move(self,stone_list):
        """
        模拟机器人移动
        :return:
        """
        # 移动时如果出发点和目的地一样，就表示结束？
        if self.start == self.destination:
            pass

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

    def arrive(self):
        """
        模拟机器人抵达目的地：重置行驶的这段路程，清算物资，判定是否需要行驶到补给点
        :return:
        """
        # 机器人抵达目的地后的重新调度
        """
            补给点：  
        """
        print(f"Robot arrive from {map_nodes[self.start].name} to {map_nodes[self.destination].name}")
        self.distance = 0  # 重置 下一段路已经行驶的路程
        old_start = self.start
        self.start = self.destination  # 更新 下一段路的起始点
        self.x = map_nodes[self.start].x
        self.y = map_nodes[self.start].y
        index = self.destination  #
        arrive_node = map_nodes[index]
        if arrive_node.is_supple:
            # 如果到达的是补给点
            # TODO： 比较补给量，这里先默认补给充裕,直接拉满
            self.carry = self.max_carry
            self.destination = self.tasks[self.task_index]  # 设置前往受灾点，继续进行物资补给
            pass
        else:
            print(f"carry:{self.carry} , need: {map_nodes[self.destination].need}")
            # 如果到达的是受灾点：
            if arrive_node.need > self.carry:
                print("物资不足")
                # 如果 受灾点的需求 比 运输物资 大
                # TODO： 是否可以按百分比决策
                arrive_node.need -= self.carry
                self.carry = 0
                # TODO： （暂时）先返回补给点，清空受灾点所有需求后再进行下一个受灾点的补给
                # TODO： 返回哪一个补给点也是问题：暂定是路径优先
                self.destination = least_distance_supple_node_index[self.destination]
                pass
            else:
                # 物资充足
                self.carry -= arrive_node.need
                print(f"物资充足:剩余：{self.carry}")

                arrive_node.need = 0
                self.task_index += 1
                if self.task_index >= len(self.tasks):
                    print("finish its tasks")
                else:
                    self.destination = self.tasks[self.task_index]

        print(f"Robot go to  {map_nodes[self.destination].name} \n")


# 初始化 地图
def init_map():
    # 计算距离每个节点最近的其他节点下标（除了自己之外）
    # least_distance_supple_node = [0] * affected_number
    # 计算邻接矩阵
    for i in range(len(map_nodes) - 1):
        for j in range(i + 1, len(map_nodes)):
            a = map_nodes[i]
            # a = Node(a['x'], a['y'])
            b = map_nodes[j]
            # b = Node(b['x'], b['y'])

            dis = a.calculate_distance(b)
            distance_matrix[i][j] = dis
            distance_matrix[j][i] = dis
            # i_distance.append(dis)
            # print(f"{str(a)} -- {str(b)}  :  {str(dis)}")
        # distance_matrix.append(i_distance)

    print("初始化地图节点数据:邻接矩阵 完成")
    # print(distance_matrix)

    for i in range(affected_number):
        min_node_index = None
        min_distance = 2 ** 30
        for j in range(affected_number, len(map_nodes)):
            if distance_matrix[i][j] < min_distance:
                min_node_index = j
                min_distance = distance_matrix[i][j]
        least_distance_supple_node_index[i] = min_node_index

    print("初始化地图节点数据:最近补给点 完成")
    print(least_distance_supple_node_index)


# 定义 计算适应度函数
def calculate_fitness(chromosomes):
    print("计算适应度")
    for chromosome in chromosomes:
        # 每次计算适应度前先恢复一下地图数据
        global map_nodes
        map_nodes = copy.deepcopy(map_nodes_backup)
        print(map_nodes_backup[0].need)
        print(map_nodes[0].need)
        # chromosome.calculate_fitness()
        chromosome.evaluate_fitness()
        print(f"{str(chromosome)} fitness: {str(chromosome.fitness)}")


# 定义变异函数
def mutate(chromosome):
    # 随机选择两个位置并交换其中的值
    i, j = random.sample(range(len(chromosome.path)), 2)
    # 不交换机器人的位置,确保交换的是地点
    if type(chromosome.path[i]) == int and type(chromosome.path[j]) == int:
        chromosome.path[i], chromosome.path[j] = chromosome.path[j], chromosome.path[i]


# 定义交叉函数
def crossover(chromosome1, chromosome2):
    # 随机选择交叉点
    cross_point = random.randint(1, len(chromosome1.path) - 1)
    # 交叉两个染色体
    new_chromosome1 = Chromosome(chromosome1.path[:cross_point] + chromosome2.path[cross_point:])
    new_chromosome2 = Chromosome(chromosome2.path[:cross_point] + chromosome1.path[cross_point:])
    return new_chromosome1, new_chromosome2


# 初始化种群-染色体
def init_population(n_chromosomes, n_cities):
    """
    种群初始化 (数字是地点、字母是机器人)
    :param n_chromosomes: 染色体的个数，即给出的初始随机方案数
    :param n_cities: 受灾点的个数，用于路径的选择，
    :return: 返回一个染色体的列表
    """
    population = []
    for i in range(n_chromosomes):
        path = list(range(n_cities))
        random.shuffle(path)
        print("路径生成：" + str(path))

        # TODO: 目前是单机器人的表示，多机器人要用多个字母截断
        # TODO: 调优方面可以在初始化上逼近均匀分配
        path.extend("a")
        population.append(Chromosome(path))
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
        print(f"{str(chromosome)} : {str(chromosome.fitness)}")


# 重置所有受灾点需求
def resetAllNeed():
    for i in range(len(random_node_list)):
        map_nodes[i].reset_need()


# 定义主函数
def main():
    # 初始化地图数据，邻接矩阵 distance_matrix
    init_map()
    # 初始化种群
    population = init_population(CHROMOSOME_NUMBER, affected_number)
    # 计算初始种群的适应度值
    calculate_fitness(population)
    # 留下一个适应度最好的
    best_chromosome = population[0]
    # 进行遗传算法迭代
    print("开始遗传算法迭代")
    for i in range(MAX_TIME):

        # 选择新一代个体
        new_population = selection(population)
        if len(new_population) == 0:  # 如果筛到最后没了，提前终止
            break;
        best_chromosome = population[0]
        print("----------------------------------------")
        print("选择新一代个体")
        show_population(new_population)

        # 进行交叉和变异
        new_population = crossover_and_mutate(new_population)

        # 重新计算 更新后种群的适应度
        calculate_fitness(new_population)
        print("进行交叉变异后")
        show_population(new_population)

        # 更新种群
        population = new_population

        # 输出最优解
        # TODO: 注意这里每轮的最优解可能被去除（交叉变异没保存） 需要采取精英保留策略
        best = min(population, key=lambda x: x.fitness)
        print(f'best path: {best}')
        print(f'best fitness: {best.fitness}')

    return best_chromosome


if __name__ == '__main__':
    main()
