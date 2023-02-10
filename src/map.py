import random
import matplotlib.pyplot as plt

import math

# 定义地图节点
class Node:
    def __init__(self,x,y,name=None):
        self.x = x # 纬度
        self.y = y # 经度
        self.name = name # 地名
        self.is_supple = False # 是否是补给点

    def calculate_distance(self,other_node):
        """
        计算与其他节点的距离
        :param other_node:  其他节点
        :return: 返回与其他节点的距离
        """
        return math.sqrt(pow(self.x - other_node.x,2) + pow(self.y - other_node.y,2))

    def __str__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

# 定义受灾点
class AffectedNode(Node):
    def __init__(self,x, y, name=None,population = 0,magnitude = 0):
        Node.__init__(self,x,y,name)
        self.population = population # 人口
        self.magnitude = magnitude # 震级
        self.anxiety = 0 # 总焦虑度
        self.last_time_visit = 0 # 距离上次访问的时间
        # 推测 物资需求
        self.reset_need()

    # 定义该地人群的焦虑函数
    def cal_people_anxiety():
        """
        :param magnitude: 震级
        :param population: 人口
        :param last_time: 距离上次得到救援的时间
        :return: 返回时刻的人群焦虑度 : TODO 暂时不使用 math.exp()
        """
        return self.magnitude * self.population * self.last_time_visit

    def __str__(self):
        return "受灾点" + Node.__str__(self)

    def reset_need(self):
        self.need = self.population * 1

# 定义补给点
class SuppleNode(Node):
    def __init__(self,x, y, name=None,material = 150):
        Node.__init__(self,x,y,name)
        self.is_supple = True
        self.material = material

    def __str__(self):
        return "补给点" + Node.__str__(self)


# 地图节点的数据
"""
     地名     纬度         经度           震级      半径
    水磨镇：30.972311，103.395026    ---- 8.0 级    89
    绵虒镇：31.351174，103.528645    ---- 6.3 级    71
    清平镇：31.563442，103.986821    ---- 4.4 级    49
    响岩镇：32.081993，104.708695    ---- 6.8 级    76
    马角镇：32.081993，105.008102    ---- 6.6 级    74
"""
map_nodes = [
    # 汶川县
    {
        "name": '水磨镇',
        "x": 103.395026,
        "y": 30.972311,
        "magnitude": 8.0,
        "population": 11935
    },
    {
        "name": '绵虒镇',
        "x": 103.528645,
        "y": 31.351174,
        "magnitude": 6.3,
        "population": 8606
    },
    # 绵竹市
    {
        "name": '清平镇',
        "x": 103.986821,
        "y": 31.563442,
        "magnitude": 4.4,
        "population": 20488
    },
    {
        "name": '响岩镇',
        "x": 104.708695,
        "y": 32.081993,
        "magnitude": 6.8,
        "population": 10589  # 11年末
    },
    {
        "name": '马角镇',
        "x": 105.008102,
        "y": 32.081993,
        "magnitude": 6.6,
        "population": 21217  # 11年末
    },

]

# 补给站
supply_node = [
    {
        "name": '水磨镇',
        "x": 103.395026,
        "y": 30.972311,
        "robot_num": 1,
        "material": {
            "food": 11935
        }
    }
]

random_node_list = []
random_node_supple_list = []
any_supple = False
# 随机生成的节点数据 26个 TODO: 如何随机 符合现实依据
for i in range(ord('a'), ord('z') + 1):
    name = chr(i)
    x = round(random.uniform(0, 90),3)
    y = round(random.uniform(0, 180),3)
    magnitude = round(random.uniform(1, 8),2)
    population = round(random.uniform(1, 100),2)
    is_supple = True if random.random() < 0.2 else False

    # 如果到了最后一个还没有补给点，我们确保必定有一个补给点
    if name == "z" and any_supple == False:
        is_supple = True

    new_node = {
        "name": name,
        "x": x,
        "y": y,
        "magnitude": magnitude,
        "population": population,
        "is_supple": is_supple,
    }
    if (is_supple):
        any_supple = True
        random_node_supple_list.append(new_node)
    else:
        random_node_list.append(new_node)

if __name__ == '__main__':
    # 设置字体的属性
    plt.rcParams["font.sans-serif"] = "SimHei"
    plt.rcParams["axes.unicode_minus"] = False
    plt.title("地图节点")
    plt.xlabel("纬度")
    plt.ylabel("经度")

    # 设置绘制的列表1
    show_nodes = random_node_list
    # 绘制 受灾点的散点图 数据 s 大小 c 颜色 marker 样式
    x = [d["x"] for d in show_nodes]
    y = [d["y"] for d in show_nodes]
    plt.scatter(x, y, s=16, c="m", marker="s")
    # 绘制 地名 标注
    for i, d in enumerate(show_nodes):
        plt.annotate(d["name"], (x[i] + 0.5, y[i] + 2))

    # 设置绘制的列表2
    show_nodes = random_node_supple_list
    # 绘制 补给点的散点图 数据 s 大小 c 颜色 marker 样式
    x = [d["x"] for d in show_nodes]
    y = [d["y"] for d in show_nodes]
    plt.scatter(x, y, s=28, c="g", marker="p")
    # 绘制 地名 标注
    for i, d in enumerate(show_nodes):
        plt.annotate(d["name"], (x[i] + 0.5, y[i] + 2))

    plt.show()
