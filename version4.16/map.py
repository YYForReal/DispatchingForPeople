import random
import matplotlib.pyplot as plt
import math
import json
import copy
import os
import numpy as np
from material import Material, AMaterial, BMaterial, CMaterial, MaterialPackage

# 文件操作类
import  file_utils


# import builtins
#
#
# def do_nothing(*args, **kwargs):
#     pass
# # 将print函数重新绑定为一个空函数
# builtins.print = do_nothing
# # 以下所有的print语句都不会有任何输出
# print("This is a test")
# print("Another test")

data_output_filename = "../data/data.json"

# 读取配置文件
config_dic = file_utils.read_config()

# 地图上的全部节点数量（受灾点 + 补给点）
NODE_NUMBER = config_dic["NODE_NUMBER"]
RANDOM_MODE = config_dic["RANDOM_MODE"]
AFFECTED_NUMBER = None
SUPPLE_NUMBER = None


# class Material:
#     def __init__(self, name, category, weight_per_unit , quantity = 0 ,anxiety_factor=0):
#         self.name = name
#         self.category = category
#         self.weight_per_unit = weight_per_unit
#         self.anxiety_factor = anxiety_factor
#         self.quantity = quantity # 补给点为数量、受灾点为需求
#
#     def cal_anxiety_rate(self):
#         return self.anxiety_factor * self.quantity


# 定义地图节点
class Node:

    def __init__(self, x, y, name=None):
        self.x = x  # 纬度
        self.y = y  # 经度
        self.name = name  # 地名
        self.is_supple = False  # 是否是补给点

    def calculate_distance(self, other_node):
        """
        计算与其他节点的距离
        :param other_node:  其他节点
        :return: 返回与其他节点的距离
        """
        return math.sqrt(pow(self.x - other_node.x, 2) + pow(self.y - other_node.y, 2))

    def __str__(self):
        return str(self.name) + "：(" + str(self.x) + "," + str(self.y) + ")"


# 定义受灾点
class AffectedNode(Node):

    def __init__(self, x, y, name=None, population=0, damage=0):
        Node.__init__(self, x, y, name)
        self.population = population  # 人口
        self.damage = damage  # 震级
        self.anxiety = 0  # 总焦虑度
        # self.last_time_visit = 0  # 距离上次访问的时间
        # self.need = self.population * self.damage * 10
        self.reset_need()

    # 定义该地人群的焦虑函数
    def cal_people_anxiety(self, nowtime):
        """
        通过人口、需求和距离上次得到救援的时间来 计算这段期间累增的焦虑度
        :param nowtime: 当前时间
        :return: 返回时刻的人群焦虑度 : TODO 暂时不使用 math.exp()
        """
        # anxiety = self.need * self.population * (nowtime - self.last_time_visit)
        # anxiety_material_rate = sum(material.cal_anxiety_rate() for material in self.need)
        # 最后计算为平方
        anxiety = 0.01 * (self.need.cal_anxiety_rate() + 1)  *  pow((nowtime - self.last_time_visit),2)


        # anxiety = (self.need + 1) * 0.01 * (nowtime - self.last_time_visit) * (nowtime - self.last_time_visit)

        # print(f"计算 {self.name} 焦虑度 -- 当前时间：{nowtime} -- 上次访问时间 {self.last_time_visit} -- 本次焦虑 {anxiety}")

        return anxiety

    def __str__(self):
        return "受灾点" + Node.__str__(self)

    def reset_need(self):
        # TODO: 物资需求的推测 物资系数 因素

        self.last_time_visit = 0
        factor = 5 * self.population * (self.damage * 0.1 + 1)
        self.need = MaterialPackage(factor * 8, factor * 1.5, factor * 0.5)

        # self.need = [
        #     Material("紧急物资","A",1,   , 1),
        #     Material("生活物资","B",3,  self.population * self.damage * 2 , 0.5),
        #     Material("通讯设备","C",10, self.population * self.damage * 1 , 0.1)
        # ]


# 定义补给点
class SuppleNode(Node):
    def __init__(self, x, y, name=None, material_package=MaterialPackage(4000, 1000, 500)):
        Node.__init__(self, x, y, name)
        self.is_supple = True
        self.material_package = material_package

    def supple(self, robot):
        # print("before supple:" + str(self) + str(self.material_package))
        # print("before supple:" + "robot" + str(robot.carry))
        minA = min(self.material_package.A_material.quantity,
                   robot.max_carry.A_material.quantity - robot.carry.A_material.quantity)
        minB = min(self.material_package.B_material.quantity,
                   robot.max_carry.B_material.quantity - robot.carry.B_material.quantity)
        minC = min(self.material_package.C_material.quantity,
                   robot.max_carry.C_material.quantity - robot.carry.C_material.quantity)
        transfer_package = MaterialPackage(minA, minB, minC)
        robot.carry = robot.carry + transfer_package
        try:
            assert self.material_package - transfer_package == True
            # print("after supple:" + str(self) + str(self.material_package))
            # print("after supple:" + "robot:" + str(robot.carry))
        except Exception as e:
            print("补给点无法供给物资（却依旧补给）")

    def hasMaterial(self):

        return self.material_package.A_material.quantity > 0 or self.material_package.B_material.quantity > 0 or self.material_package.C_material.quantity > 0

    def __str__(self):
        return "补给点" + Node.__str__(self)


class Stone(Node):
    def __init__(self, x=0, y=0, name=None, radius=1, data=None):
        Node.__init__(self, x, y, name)
        self.radius = radius
        if data is not None:
            self.set_json(data)

    def get_json(self):
        return {
            "x": self.x,
            "y": self.y,
            "name": self.name,
            "radius": self.radius
        }

    def set_json(self, data):
        self.x = data["x"]
        self.y = data["y"]
        self.radius = data["radius"]


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
    {
        "name": '水磨镇',
        "x": 103.395026,
        "y": 30.972311,
        "damage": 8.0,
        "population": 11935
    },
    {
        "name": '绵虒镇',
        "x": 103.528645,
        "y": 31.351174,
        "damage": 6.3,
        "population": 8606
    },
    {
        "name": '清平镇',
        "x": 103.986821,
        "y": 31.563442,
        "damage": 4.4,
        "population": 20488
    },
    {
        "name": '响岩镇',
        "x": 104.708695,
        "y": 32.081993,
        "damage": 6.8,
        "population": 10589
    },
    {
        "name": '马角镇',
        "x": 105.008102,
        "y": 32.081993,
        "damage": 6.6,
        "population": 21217
    },
]

stone_list = []
for i in range(5):
    x = random.uniform(0, 180)
    y = random.uniform(0, 180)
    radius = random.randint(1, 10)
    stone = Stone(x, y, f"stone_{i}", radius)
    stone_list.append(stone)

stone_json_list = [stone.get_json() for stone in stone_list]
print(stone_json_list)
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

# 判断数据初始化的方式
if RANDOM_MODE == 1:
    # 不限制受灾点和补给点个数
    random_node_list = []
    random_node_supple_list = []
    any_supple = False

    # 随机生成的节点数据 NODE_NUMBER 个 TODO: 如何随机 符合现实依据
    # for i in range(ord('A'), ord('A') + NODE_NUMBER):
    for i in range(0, NODE_NUMBER):
        x = round(np.random.uniform(0, 180), 3)
        y = round(np.random.uniform(0, 180), 3)
        damage = round(np.random.uniform(1, 7), 2)  # 原本模拟的震级，这里需要再度抽象为损毁程度 ==> 已抽象为 损毁程度
        population = round(np.random.uniform(0.1, 5), 2)  # 假设以千人为单位，100 - 5000
        is_supple = True if random.random() < 0.2 else False

        # 如果到了最后一个还没有补给点，我们确保必定有一个补给点
        # if i == ord('A') + NODE_NUMBER - 1 and any_supple == False:
        if NODE_NUMBER >= 2 and i >= NODE_NUMBER - 2 and any_supple == False:
            is_supple = True
        new_node = {
            "name": '',
            "x": x,
            "y": y,
            "damage": damage,
            "population": population,
            "is_supple": is_supple,
        }
        if is_supple:
            any_supple = True
            random_node_supple_list.append(new_node)
        else:
            random_node_list.append(new_node)

    # 修改导出的地图节点数据
    map_nodes = copy.deepcopy(random_node_list)
    map_nodes.extend(random_node_supple_list)
    for i in range(len(map_nodes)):
        map_nodes[i]['name'] = i

    AFFECTED_NUMBER = len(random_node_list)
    SUPPLE_NUMBER = len(random_node_supple_list)
elif RANDOM_MODE == 2:
    # 读取JSON文件数据
    data = file_utils.read_data()
    map_nodes = data["map_nodes"]
    AFFECTED_NUMBER = data["AFFECTED_NUMBER"]
    SUPPLE_NUMBER = data["SUPPLE_NUMBER"]
    stone_json_list = data["stone_list"]
    stone_list = []
    for stone_json in stone_json_list:
        new_stone = Stone()
        new_stone.set_json(stone_json)
        stone_list.append(new_stone)

# 写入这次的数据
print("写入数据...")
file_utils.set_property("map_nodes",map_nodes)
file_utils.set_property("AFFECTED_NUMBER",AFFECTED_NUMBER)
file_utils.set_property("SUPPLE_NUMBER",SUPPLE_NUMBER)
file_utils.set_property("stone_list",stone_json_list)


print(f"初始化受灾点个数：{AFFECTED_NUMBER}")
print(f"初始化补给点个数：{SUPPLE_NUMBER}")

# 初始化结束后的对象构造
for i in range(len(map_nodes)):
    node = map_nodes[i]
    if node["is_supple"] == True:
        map_nodes[i] = SuppleNode(node["x"], node["y"], node["name"])
        print("node1: " + str(map_nodes[i].material_package))
    else:
        map_nodes[i] = AffectedNode(node["x"], node["y"], node["name"], node["population"], node["damage"])
        print("node2: " + str(map_nodes[i].need))


def showMap():
    # 设置字体的属性
    plt.rcParams["font.sans-serif"] = "SimHei"
    plt.rcParams["axes.unicode_minus"] = False
    plt.title("地图节点")
    plt.xlabel("x轴")
    plt.ylabel("y轴")

    # 设置绘制的列表1
    show_nodes = map_nodes[:AFFECTED_NUMBER]
    # 绘制 受灾点的散点图 数据 s 大小 c 颜色 marker 样式
    x = [d.x for d in show_nodes]
    y = [d.y for d in show_nodes]
    plt.scatter(x, y, s=16, c="m", marker="s")
    # 绘制 地名 标注
    for i, d in enumerate(show_nodes):
        plt.annotate(d.name, (x[i] + 0.5, y[i] + 2))

    # 设置绘制的列表2
    show_nodes = map_nodes[AFFECTED_NUMBER:]
    # 绘制 补给点的散点图 数据 s 大小 c 颜色 marker 样式
    x = [d.x for d in show_nodes]
    y = [d.y for d in show_nodes]
    plt.scatter(x, y, s=28, c="g", marker="p")
    # 绘制 地名 标注
    for i, d in enumerate(show_nodes):
        plt.annotate(d.name, (x[i] + 0.5, y[i] + 2))

    plt.show()


if __name__ == '__main__':
    showMap()
