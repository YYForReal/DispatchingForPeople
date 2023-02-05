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
    
    7. 当前的物资度量是统一的，以后可以改成列表形式的数据    
    
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

# TODO: 设置一个距离受灾点最近的补给点映射

anxiety_arr = [0] * len(map_nodes) # 各地点的人群累计焦虑值



# 定义染色体结构
class Chromosome:
    def __init__(self, path):
        self.path = path # 路径： abc1 ef2 ght3
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
                robot = Robot(tasks,speed=1,max_carry=150)
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
                print("robot move")
                if robot.task_index >= len(robot.tasks):
                    robots.remove(robot)
            total_time+=1

        # total_time = sum(robot.elapsed_time for robot in robots)
        self.fitness = total_time
        return total_time

    def __str__(self):
        return "->".join([str(i) for i in self.path])

# 定义运输机器
class Robot:
    def __init__(self, tasks, speed=1,max_carry=100,position = 0):
        """
        :param tasks: 目的地任务序列
        :param speed: 速度
        :param max_carry: 携带物资量
        :param position: 当前的机器人所在位置
        """
        self.tasks = tasks # 总共的任务队列
        self.max_carry = max_carry  # 最大携带容量
        self.speed = speed # 运输机器的速度
        self.position = position # 当前位处的节点下标（初始位置）

        # self.start = tasks[0]
        self.destination = tasks[0]
        self.distance = 0 # 这段路行驶路程
        self.elapsed_distance = 0 # 已走总路程
        self.carry = 0 # 当前携带容量
        self.task_index = 1 # 对应染色体的任务下标

    def move(self):
        """
        模拟机器人移动
        :return:
        """
        self.distance += self.speed
        self.elapsed_distance += self.speed
        if self.distance >= distance_matrix[self.position][self.destination]:
            self.arrive()

    def arrive(self):
        """
        模拟机器人抵达目的地：重置行驶的这段路程，清算物资，判定是否需要行驶到补给点
        :return:
        """
        # 机器人抵达目的地后的重新调度
        self.distance = 0
        index = self.destination
        arrive_node = map_nodes[index]
        if arrive_node.is_supple:
            # 如果到达的是补给点
            # TODO： 补给

            pass
        else:
            # 如果到达的是受灾点：
            if arrive_node.need > self.carry:
                # 如果 受灾点的需求 比 运输物资 大
                # TODO： 是否可以按百分比决策
                arrive_node.need -= self.carry
                self.carry = 0
                # TODO： （暂时）先返回补给点，清空受灾点所有需求后再进行下一个受灾点的补给

                pass
            else:
                # 物资充足
                self.carry -= arrive_node.need
                arrive_node.need = 0
                task_index +=1


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
        # chromosome.calculate_fitness()
        chromosome.evaluate_fitness()
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

# 种群初始化 (数字是地点、字母是机器人)
def init_population(n_chromosomes, n_cities):
    population = []
    for i in range(n_chromosomes):
        path = list(range(n_cities))
        random.shuffle(path)
        print("路径生成："+str(path))
        path.extend("a")
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

