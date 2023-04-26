import errno
import os
import json
import csv


data_file_name = '../data/data.json'
config_file_name = "./config.json"
result_file_name = "../data/result.csv"

csv_header = ['程序执行时间', '染色体路径', '适应度', '染色体任务说明']


# 默认在data.json中设置属性
def set_property(key, value):
    # 检查目录是否存在，如果不存在则创建它
    if not os.path.exists(os.path.dirname(data_file_name)):
        try:
            os.makedirs(os.path.dirname(data_file_name))
        except OSError as exc:  # 防止多个线程同时创建目录
            if exc.errno != errno.EEXIST:
                raise

    # 读取原本的json文件内容
    original_data = {}
    with open(data_file_name, 'r') as f:
        original_data = json.load(f)

    original_data[key] = value

    # print(original_data)
    with open(data_file_name, "w") as f:
        json.dump(original_data, f, indent=4)


def read_config():
    f = open(config_file_name, encoding="utf-8")
    config_dic = json.load(f)
    return config_dic

def read_data():
    data = {}
    with open(data_file_name, 'r') as f:
        data = json.load(f)
    return data

# 记录结果
def append_result(run_time,path,fitness,desc):
    # 如果文件不存在，则创建文件并写入表头
    if not os.path.exists(result_file_name):
        with open(result_file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)

    # 将记录写入csv文件
    with open(result_file_name, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([run_time, path, fitness,desc])

# 记录路径
# def append_matrix_json(name, distance_matrix):
#     # 读取JSON文件中的原始数据，并将其转换为Python对象
#     with open(file_path, 'r') as f:
#         data = json.load(f)
#
#     # 将新的distance_matrix添加到Python对象中
#     data['distance_matrix'] = distance_matrix
#
#     # 将Python对象转换回JSON格式，并将其写回到原始的JSON文件中
#     with open(file_path, 'w') as f:
#         json.dump(data, f, indent=4)