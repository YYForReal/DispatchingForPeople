import random
import numpy as np
from map import map_nodes,random_node_list,random_node_supple_list,Node,AffectedNode,SuppleNode
from selection import selection,selection_roulette,selection_championships

# 使用随机生成的数据进行模拟
map_nodes = random_node_list
map_nodes.extend(random_node_supple_list)

for i in range(len(map_nodes)):
    node = map_nodes[i]
    if node["is_supple"] == True:
        map_nodes[i] = SuppleNode(node["x"],node["y"],node["name"])
    else:
        map_nodes[i] = AffectedNode(node["x"],node["y"],node["name"],node["population"],node["magnitude"])

# print(map_nodes)
"""
总：我们定义Chromosome类来表示染色体,并定义了函数
    calculate_fitness：计算适应度
    mutate：表示变异
    crossover：表示交叉
    selection 表示选择新一代个体 ==> 当成模块
待优化点：
    0. 地图节点的数据
    1. 完成多次调度的编码、解码
    2. 完善适应度函数，更好地计算个体的适应度    
    3. 添加遗传算法算子：不同的交叉策略、变异、置换...
    4. 新的停止条件：收敛条件、运行时间限制
    5. 优化参数：交叉概率、变异概率、种群大小
    6. 算法结合：模拟退火算法...
"""

CHROMOSOME_NUMBER = 10  # 染色体（方案）数量
CROSSOVER_RATE = 0.8  # 交叉概率
MUTATION_RATE = 0.3  # 变异概率
MAX_TIME = 1000  # 设置最大迭代次数(如果方案数量筛选到只剩一个就提前结束了)
ANXIETY_RATE = 1.1  # 焦虑幂指速率
now_time = 0 # 当前时间


distance_matrix = [0] * len(map_nodes) # 图的邻接矩阵（各节点的路径代价）
for i in range(len(distance_matrix)):
    distance_matrix[i] = [0] * len(map_nodes)

anxiety_arr = [0] * len(map_nodes) # 各地点的人群累计焦虑值



# 定义染色体结构
class Chromosome:
    def __init__(self, path):
        self.path = path
        self.fitness = None

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
        start = self.path[0]
        destination = self.path[1]

        for i in range(1, len(self.path)):
            distance = distances[start][destination]
            robot = Robot(start, destination, distance, speed)
            robots.append(robot)

        while True:
            for robot in robots:
                robot.move()
                if robot.destination == "D":
                    robots.remove(robot)
            if not robots:
                break
        total_time = sum(robot.elapsed_time for robot in robots)
        return total_time

    def __str__(self):
        return "->".join([str(i) for i in self.path])

# 定义运输机器
class Robot:
    def __init__(self, start, destination, distance, speed=1,carry=0):
        """
        :param start: 起始点
        :param destination: 目的地
        :param distance: 当前路程
        :param speed: 速度: 约 1.2 m/s
        :param carry: 携带物资量: 约 150kg
        """
        self.start = start
        self.destination = destination
        self.distance = distance
        self.speed = speed
        self.elapsed_time = 0
        self.carry = carry
        self.task = None # 对应染色体的路径下标

    def move(self):
        """
        模拟机器人移动
        :return:
        """
        self.elapsed_time += self.speed
        if self.elapsed_time >= self.distance:
            self.arrive()

    def arrive(self):
        """
        模拟机器人抵达目的地
        :return:
        """
        # 机器人抵达目的地后的重新调度
        index = self.destination
        node = map_nodes[index]
        if node.is_supple == True:
            # 如果到达的是补给点：TODO
            pass
        else:
            # 如果到达的是受灾点：TODO
            pass




# 初始化 地图 邻接矩阵
def init_map():
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
    print("初始化地图节点数据(邻接矩阵)")
    # print(distance_matrix)


# 定义 计算适应度函数
def calculate_fitness(chromosomes):
    print("计算适应度")
    for chromosome in chromosomes:
        chromosome.calculate_fitness()
        print(f"{str(chromosome)} fitness: {str(chromosome.fitness)}")


# 定义变异函数
def mutate(chromosome):
    # 随机选择两个位置并交换其中的值
    i, j = random.sample(range(len(chromosome.path)), 2)
    chromosome.path[i], chromosome.path[j] = chromosome.path[j], chromosome.path[i]


# 定义交叉函数
def crossover(chromosome1, chromosome2):
    # 随机选择交叉点
    cross_point = random.randint(1, len(chromosome1.path) - 1)
    # 交叉两个染色体
    new_chromosome1 = Chromosome(chromosome1.path[:cross_point] + chromosome2.path[cross_point:])
    new_chromosome2 = Chromosome(chromosome2.path[:cross_point] + chromosome1.path[cross_point:])
    return new_chromosome1, new_chromosome2

# 种群初始化
def init_population(n_chromosomes, n_cities):
    population = []
    for i in range(n_chromosomes):
        path = list(range(n_cities))
        random.shuffle(path)
        print("路径生成："+str(path))
        population.append(Chromosome(path))
    return population


# 交叉和变异
def crossover_and_mutate(population):
    new_population = []
    for i in range(0, len(population), 2):
        # 如果只剩最后一个，就不交叉变异了
        if i+1 >= len(population):
            new_population.append(population[i])
            break

        # 随机决定是否进行交叉
        if random.random() < CROSSOVER_RATE:
            # 进行交叉
            offspring1, offspring2 = crossover(population[i], population[i+1])
        else:
            offspring1, offspring2 = population[i], population[i+1]
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


# 定义主函数
def main():
    # 初始化地图数据，邻接矩阵 distance_matrix
    init_map()
    # 初始化种群
    population = init_population(CHROMOSOME_NUMBER, len(map_nodes))
    # 计算初始种群的适应度值
    calculate_fitness(population)
    # 进行遗传算法迭代
    print("开始遗传算法迭代")
    for i in range(MAX_TIME):

        # 选择新一代个体
        new_population = selection(population)
        if len(new_population) == 0:  # 如果筛到最后没了，提前终止
            break;
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
        best = min(population, key=lambda x: x.fitness)
        print(f'best path: {best}')
        print(f'best fitness: {best.fitness}')

if __name__ == '__main__':
    main()

