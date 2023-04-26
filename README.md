# DispatchingForPeople

## 目录结构

- data：存放csv地图节点的数据、实验记录数据、生成结果的数据
- src：存放源代码
  - animation.py：使用matplotlib的可视化程序，呈现遗传算法的可行解。可接收来自main.py计算的数据或者复用之前的结果。
  - animation_greedy.py：使用matplotlib的可视化程序，呈现贪心策略的可行解。
  - config.json：配置程序及遗传算法的相关参数。
  - file_utils.py：辅助操作文件的工具类。
  - greedy.py：对比算法，使用贪心策略求解的程序。
  - main.py：程序整体框架，用于接收来自map生成的地图节点数据，并计算最佳调度策略。
  - map.py：生成地图节点。
  - material.py：三种物资类及物资包裹集合类的声明。
  - selection.py：遗传算法中选择部分的方法实现。

- mind：存放程序模型的思维导图。
- version：存储过往的版本

### 版本

- version2.10：实现遗传的框架。

- version3.13：实现机器人的拓展和可视化表示。
- version4.1：优化染色体的动态编码，更改染色体的初始化及表示。
- version4.8：完善物资包裹、石头障碍。增加了物资的表示方式，同时以package为单位进行传输。
- version4.16：重构代码提取文件工具类，实验数据的存储和恢复。
- src：将完全补给改为单次补给，增加对比的贪心策略。优化动画展示。



## 总说明

 我们定义Chromosome类来表示染色体,并定义了calculate_fitness.mutate.crossover和selection函数来分别计算适应度，变异

### 优化方向

0. 地图节点的数据
1. 完成多次调度的编码、解码
2. 完善适应度函数，更好地计算个体的适应度    
3. 添加遗传算法算子：不同的交叉策略、变异、置换...
4. 新的停止条件：收敛条件、运行时间限制
5. 优化参数：交叉概率、变异概率、种群大小
6. 算法结合：模拟退火算法...

### 机器人 设计思路

存储起始点，目的点，任务目的序列，当前行驶的任务目的下标，当前任务序列序号

当机器人运输到目的地进行补给后，
- 若有剩余的物资，优先运输到下一个受灾点；
    - 或顺路前往最近最适合的物资补给点 
    - 继续前往下一个受灾点
- 若不足，则返回补给点后，将这个受灾目的点重新进入任务序列。


### 染色体编码原理


因为补给点的移动是可以根据受灾点的行驶任务进行动态决定的

所以可以排除补给点的编码，仅仅使用受灾点进行染色体编码。



### 项目配置文件说明

/src/config.json：

- `RANDOM_MODE`:随机模式。
  - 1 代表完全随机（仅限制地图节点个数），数据导出
  - 2 读取数据运行
  - 3 代表限制补给点和受灾点个数（TODO）
  - 4 代表限制机器人个数（TODO）
  
~~~json
{
  "RANDOM_MODE": 1, // 生成数据的方式
  "NODE_NUMBER": 10, // 节点数量
  "AFFECTED_NUMBER": 7, // 受灾点数量
  "SUPPLE_NUMBER": 3, // 补给点数量
  "ROBOT_NUMBER": 3, // 机器人数量
  "CHROMOSOME_NUMBER": 20, // 染色体方案数
  "CROSSOVER_RATE": 0.8, // 交叉概率
  "MUTATION_RATE": 0.3, // 变异概率
  "MAX_TIME": 10000, // 最长时间 
  "MAX_GENERATION":100, // 最大迭代次数
  "ANXIETY_RATE": 1.1 // 焦虑幂指速率
}
~~~


### 问题

#### 物资细化

1. 每个机器人的运输能力一样吗?
2. 每个机器人是专门运输一种还是多种？
  - 如果是一种 ==> 问题 化归
  - 如果是多种。

      - 每次运输量怎么确定？ 

        - 按固定比例装载

        - 还是每次装载一种

        - 还是根据目的地的需求进行装载

        - 还是根据当前补给点进行装载 
4. 每种物资都有一定的单位重量，如何确保尽可能地满载。

为了降低焦虑度，理论上需要优先装载焦虑系数高的，
但是考虑现实，肯定需要同时装载衣物，那么通讯设备是否需要？


#### 石头

程序虽然有计算，但是难表现，原本的遗传算法本身就是在全局范围下跑的，第一次染色体的调度如果因为到了石头处折返，则人群焦虑度势必升高，
最终选出来的是很大概率上是”碰巧“没有走到石头的调度。

#### 调度判定

（同质化问题）

节点存在需求数量quantity。当其他机器人调度完自己的任务后，下一步就到有需求的节点处。假设节点1存在需求，机器人A，B，C...相继执行完自己的任务，都会前往去补给节点1。假设每个机器人都携带不定量随机的物资，如何设计并判定是否其他机器人在A未抵达时也需要前往节点1？

### 优化之处

#### 提前计算优先补给点

1. 在计算每个节点的最近补给点时，我们可以按照优先级依次检查节点需要寻找的补给点，直到找到一个物资充足的补给点为止。
   为了实现这一过程，我们可以将上述代码中的内层循环改为一个 while 循环，并在循环中检查当前的补给点是否在该节点的优先寻找补给点列表中。 
   具体而言，我们可以先将 least_distance_supple_node_index 初始化为全零二维数组，然后对于每个节点 i，按照优先级依次遍历该节点需要寻找的补给点下标，直到找到一个物资充足的补给点为止。
   - 如果找到了充足的补给点，就将其下标存储到 `least_distance_supple_node_index[i][j]` 中，并终止循环； 
   - 否则，将下一个补给点的下标作为循环的继续条件，继续遍历。 
   
   这种方法可以让我们避免重复计算每个节点的最近补给点，同时也可以方便地实现多个优先级寻找补给点的功能。如果我们需要更新补给点的物资信息，只需要在更新补给点列表的同时更新 supple_node_index 数组即可。


#### 染色体动态增长的自适应编码

因为初始不知道每一个节点是否可达，是否需要多次调度，为了便于初始化，可以多次调度，在调度过程中若机器人走完了全程，仍有需求，则增加任务，前往对应的需求点

#### 预估需求量

预估需求量：当某个机器人预估抵达某一个节点的时候，就提前计算出对应的剩余需求量，如果没有剩余，则其他机器人就无需前往看上去还有需求的地点。
（实际上已经有人承包了。）


1. 邻接矩阵，表示不可达的时候是设置成无穷大好还是-1好？
2. 我需要表示一个机器人行驶途中发现两节点堵塞，报告路径不可达，此时机器人需要返回到原出发节点，如果我设置成了路径无穷大，那么机器人会在节点中间无法返回。如何解决这个问题？
3. 节点存在需求数量quantity。当其他机器人调度完自己的任务后，下一步就到有需求的节点处。假设节点1存在需求，机器人A，B，C...相继执行完自己的任务，都会前往去补给节点1。假设每个机器人都携带不定量随机的物资，如何设计并判定是否其他机器人在A未抵达时也需要前往节点1？
