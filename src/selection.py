import random
"""
    这里存放一些遗传算法中染色体选择的不同方法
"""
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
def selection_championships(chromosomes, tournament_size = 3):
    new_chromosomes = []
    for i in range(len(chromosomes)):
        competitors = random.sample(chromosomes, tournament_size)
        # 选择适应度最高（焦虑度最低）的个体
        competitors.sort(key=lambda x: x.fitness, reverse=False)
        new_chromosomes.append(competitors[0])
    return new_chromosomes
