import random
import numpy as np
import math
from config import map_nodes
# print(dir(config))

"""
总：我们定义Chromosome类来表示染色体,并定义了calculate_fitness.mutate.crossover和selection函数来分别计算适应度，变异
优化点：
    0. 地图节点的数据
    1. 完成多次调度的编码、解码
    2. 完善适应度函数，更好地计算个体的适应度    
    3. 添加遗传算法算子：不同的交叉策略、变异、置换...
    4. 新的停止条件：收敛条件、运行时间限制
    5. 优化参数：交叉概率、变异概率、种群大小
    6. 算法结合：模拟退火算法...
"""
chromosome_number = 10 # 染色体（方案）数量
crossover_rate = 0.8 # 交叉概率
mutation_rate = 0.3 # 变异概率
n_iterations = 1000 # 设置最大迭代次数

distance_matrix = [0] * len(map_nodes) # 图的邻接矩阵（各节点的路径代价）
for i in range(len(distance_matrix)):
    distance_matrix[i] = [0] * len(map_nodes)





class Node:
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def calculate_distance(self,other_node):
        return math.sqrt(pow(self.x - other_node.x,2) + pow(self.y - other_node.y,2))

    def __str__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

# 定义染色体结构
class Chromosome:
    def __init__(self, path):
        self.path = path
        self.fitness = None

    # 计算适应度
    def calculate_fitness(self):
        distance = 0
        for i in range(len(self.path) - 1):
            # print(f"{str(self.path[i])}  -- {str(self.path[i+1])}")
            distance += distance_matrix[int(self.path[i])][int(self.path[i + 1])]
        self.fitness = distance

    def __str__(self):
        return "->".join([str(i) for i in self.path])

# 初始化 地图 邻接矩阵
def init_map():
    for i in range(len(map_nodes)):
        for j in range(i + 1, len(map_nodes)):
            a = map_nodes[i]
            a = Node(a['x'], a['y'])
            b = map_nodes[j]
            b = Node(b['x'], b['y'])
            dis = a.calculate_distance(b)
            distance_matrix[i][j] = dis
            distance_matrix[j][i] = dis
            # i_distance.append(dis)
            print(f"{str(a)} -- {str(b)}  :  {str(dis)}")
        # distance_matrix.append(i_distance)
    print("初始化地图节点数据:")
    print(distance_matrix)


# 定义 计算适应度函数
def calculate_fitness(population):
    print("计算适应度")
    for chromosome in population:
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


# 选择 - 定义选择新一代个体的函数
def selection(population):
    # 按照适应度值从小到大排序
    population.sort(key=lambda x: x.fitness)
    # 选择适应度值最优的前50%个体作为新一代种群
    return population[:int(len(population) /2)]


# 选择 - 轮盘赌算法
def selection_roulette(population):
    """
    这个函数首先计算种群中每个染色体的适应度总和，然后计算每个染色体的适应度概率。接着计算累积概率。
    对于新一代种群中的每个个体，生成一个随机数并在累积概率表中找到第一个大于该随机数的染色体并将其加入新一代种群中。
    """
    # 计算适应度总和
    fitness_sum = sum(chromosome.fitness for chromosome in population)
    # 计算概率
    probabilities = [chromosome.fitness / fitness_sum for chromosome in population]
    # 计算累积概率
    cumulative_probabilities = [probabilities[0]]
    for i in range(1, len(probabilities)):
        cumulative_probabilities.append(cumulative_probabilities[i-1] + probabilities[i])
    # 生成新一代种群
    new_population = []
    for i in range(len(population)):
        r = random.random()
        for j in range(len(cumulative_probabilities)):
            if r < cumulative_probabilities[j]:
                new_population.append(population[j])
                break
    return new_population

# 选择 - 锦标赛算法
def selection_championships(population, tournament_size):
    new_population = []
    for i in range(len(population)):
        # 随机选择参赛个体
        competitors = random.sample(population, tournament_size)
        # 选择适应度最高的个体
        competitors.sort(key=lambda x: x.fitness, reverse=True)
        new_population.append(competitors[0])
    return new_population


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
        if random.random() < crossover_rate:
            # 进行交叉
            offspring1, offspring2 = crossover(population[i], population[i+1])
        else:
            offspring1, offspring2 = population[i], population[i+1]
        # 随机决定是否进行变异
        if random.random() < mutation_rate:
            mutate(offspring1)
        if random.random() < mutation_rate:
            mutate(offspring2)
        new_population.append(offspring1)
        new_population.append(offspring2)
    return new_population


# 展示当前的染色体（方案）: 路径 + 适应度
def show_population(population):
    for i in range(len(population)):
        chromosome = population[i]
        print(f"{str(chromosome)} : {str(chromosome.fitness)}")


# 定义主函数
def main():
    # 初始化地图数据，邻接矩阵 distance_matrix
    init_map()
    # 初始化种群
    population = init_population(chromosome_number,len(map_nodes))
    # 计算初始种群的适应度值
    calculate_fitness(population)
    # 进行遗传算法迭代
    print("开始遗传算法迭代")
    for i in range(n_iterations):
        # 选择新一代个体
        print("选择新一代个体")
        new_population = selection(population)
        if len(new_population) == 0: # 如果筛到最后没了，提前终止
            break;
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

