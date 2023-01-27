import random
import matplotlib.pyplot as plt
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
       "population":8606
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
        "magnitude": 6.6    ,
        "population": 21217 # 11年末
    },

]

# 补给站
supply_node = [
    {

    }
]

random_node_list = []

# 随机生成的节点数据
for i in range(ord('a'),ord('z')+1):
    name = chr(i)
    x = random.uniform(0,90)
    y = random.uniform(0,180)
    magnitude = random.uniform(1,8)
    population = random.randint(1,100)
    random_node_list.append({
        "name":name,
        "x":x,
        "y":y,
        "magnitude":magnitude,
        "population":population
    })


if __name__ == '__main__':
    # 设置字体的属性
    plt.rcParams["font.sans-serif"] = "SimHei"
    plt.rcParams["axes.unicode_minus"] = False
    plt.title("地图节点")
    plt.xlabel("纬度")
    plt.ylabel("经度")

    # 设置绘制的列表
    show_nodes = random_node_list
    # 绘制 受灾点的散点图
    x = [d["x"] for d in show_nodes]
    y = [d["y"] for d in show_nodes]

    plt.scatter(x, y)

    for i, d in enumerate(show_nodes):
        plt.annotate(d["name"], (x[i], y[i]))

    plt.show()
