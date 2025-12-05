#!/usr/bin/env python3
"""
生成应急救援物资调度的 GIF 动画（简化版，仅使用 matplotlib）
"""

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import copy
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import main as main_module
from material import MaterialPackage
import file_utils

class SimpleTransportAnimation:
    """简化的物资运输动画类"""

    def __init__(self):
        # 重新加载数据
        file_utils.read_data()
        self.map_nodes = copy.deepcopy(main_module.map_nodes_backup)
        self.stone_list = main_module.stone_list
        self.robots = []

        # 动画参数
        self.fig_size = (14, 10)
        self.dpi = 100

        # 颜色配置
        self.colors = {
            'affected': '#FF6B6B',      # 受灾点 - 红色
            'supply': '#51CF66',         # 补给点 - 绿色
            'stone': '#868E96',          # 障碍物 - 灰色
            'robot': '#339AF0',          # 机器人 - 蓝色
            'path': '#FFD43B',           # 路径 - 黄色
            'background': '#F8F9FA'      # 背景 - 浅灰
        }

        print("简化动画系统初始化完成")

    def create_robots(self):
        """创建机器人和任务分配"""
        # 尝试从文件读取最优染色体
        try:
            data = file_utils.read_data()
            saved_chromosome = data.get("best_chromosome")
            if saved_chromosome:
                print(f"使用保存的最优染色体: {saved_chromosome}")
                path = saved_chromosome
            else:
                raise ValueError("未找到保存的染色体")
        except:
            print("未找到保存的染色体，使用测试染色体")
            # 创建一个测试染色体：3个机器人，分配不同任务
            path = [0, 1, 'a', 2, 3, 'b', 4, 5, 'c', 6, 7]

        # 解析染色体，创建机器人
        tasks = []
        robot_count = 0
        robot_colors = ['#FF6B9D', '#C44569', '#F8961E', '#F9844A', '#43AA8B']

        for item in path:
            if isinstance(item, str):
                # 机器人标记
                if tasks:
                    start_node = main_module.AFFECTED_NUMBER + (robot_count % main_module.SUPPLE_NUMBER)
                    start_pos = [self.map_nodes[start_node].x, self.map_nodes[start_node].y]

                    robot = {
                        'name': f"Robot_{robot_count}",
                        'position': list(start_pos),
                        'start_position': list(start_pos),
                        'tasks': tasks.copy(),
                        'current_task': 0,
                        'color': robot_colors[robot_count % len(robot_colors)],
                        'progress': 0.0
                    }
                    self.robots.append(robot)

                    robot_count += 1
                    tasks = []
            else:
                # 任务节点
                tasks.append(item)

        # 处理最后一个机器人的任务
        if tasks:
            start_node = main_module.AFFECTED_NUMBER + (robot_count % main_module.SUPPLE_NUMBER)
            start_pos = [self.map_nodes[start_node].x, self.map_nodes[start_node].y]

            robot = {
                'name': f"Robot_{robot_count}",
                'position': list(start_pos),
                'start_position': list(start_pos),
                'tasks': tasks.copy(),
                'current_task': 0,
                'color': robot_colors[robot_count % len(robot_colors)],
                'progress': 0.0
            }
            self.robots.append(robot)

        print(f"创建了 {len(self.robots)} 个机器人")
        for robot in self.robots:
            print(f"  {robot['name']}: 任务 {robot['tasks']}")

    def update_frame(self, frame_num):
        """更新动画帧"""
        # 更新机器人位置
        for robot in self.robots:
            if robot['current_task'] < len(robot['tasks']):
                # 计算到目标节点的位置
                target_node = robot['tasks'][robot['current_task']]
                target_pos = [self.map_nodes[target_node].x, self.map_nodes[target_node].y]

                # 插值计算当前位置
                t = robot['progress']
                robot['position'] = [
                    robot['start_position'][0] + (target_pos[0] - robot['start_position'][0]) * t,
                    robot['start_position'][1] + (target_pos[1] - robot['start_position'][1]) * t
                ]

                # 更新进度
                robot['progress'] += 0.05  # 每帧移动5%
                if robot['progress'] >= 1.0:
                    robot['progress'] = 0.0
                    robot['current_task'] += 1
                    robot['start_position'] = list(robot['position'])

        return self.draw_frame(frame_num)

    def draw_frame(self, frame_num):
        """绘制当前帧"""
        # 清除之前的绘制
        self.ax.clear()

        # 绘制受灾点
        x1 = [node.x for node in self.map_nodes[:main_module.AFFECTED_NUMBER]]
        y1 = [node.y for node in self.map_nodes[:main_module.AFFECTED_NUMBER]]
        self.ax.scatter(x1, y1, s=300, c=self.colors['affected'], marker='s',
                       label='受灾点', alpha=0.8, edgecolors='white', linewidth=2, zorder=3)

        # 绘制补给点
        x2 = [node.x for node in self.map_nodes[main_module.AFFECTED_NUMBER:]]
        y2 = [node.y for node in self.map_nodes[main_module.AFFECTED_NUMBER:]]
        self.ax.scatter(x2, y2, s=300, c=self.colors['supply'], marker='^',
                       label='补给点', alpha=0.8, edgecolors='white', linewidth=2, zorder=3)

        # 绘制障碍物
        for stone in self.stone_list:
            circle = plt.Circle((stone.x, stone.y), stone.radius,
                              color=self.colors['stone'], alpha=0.6, zorder=1)
            self.ax.add_patch(circle)

        # 绘制连接线
        for i in range(len(self.map_nodes) - 1):
            for j in range(i + 1, len(self.map_nodes)):
                if main_module.distance_matrix_copy[i][j] != main_module.MAX_INF:
                    x = [self.map_nodes[i].x, self.map_nodes[j].x]
                    y = [self.map_nodes[i].y, self.map_nodes[j].y]
                    self.ax.plot(x, y, 'gray', alpha=0.2, linewidth=0.5, zorder=0)

        # 绘制机器人
        for robot in self.robots:
            # 绘制机器人
            self.ax.scatter(robot['position'][0], robot['position'][1],
                          s=200, c=robot['color'], marker='o',
                          alpha=0.9, edgecolors='white', linewidth=2, zorder=5)

            # 绘制机器人名称
            self.ax.annotate(robot['name'], (robot['position'][0], robot['position'][1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, fontweight='bold', color='black', zorder=6)

            # 绘制路径轨迹
            if robot['current_task'] > 0:
                path_x = [robot['tasks'][0]]
                path_y = [robot['tasks'][0]]
                for i in range(min(robot['current_task'], len(robot['tasks']))):
                    node = self.map_nodes[robot['tasks'][i]]
                    path_x.append(node.x)
                    path_y.append(node.y)

                if len(path_x) > 1:
                    self.ax.plot(path_x, path_y, color=robot['color'], alpha=0.5,
                               linewidth=2, linestyle='--', zorder=2)

        # 设置图形属性
        self.ax.set_title(f'应急救援物资调度 - 时间步: {frame_num}', fontsize=16, fontweight='bold')
        self.ax.set_xlabel('x轴（单位：百米）', fontsize=12)
        self.ax.set_ylabel('y轴（单位：百米）', fontsize=12)
        self.ax.set_xlim(-10, 170)
        self.ax.set_ylim(-10, 170)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right', fontsize=10)

        # 添加节点标签
        for i, node in enumerate(self.map_nodes):
            if i < main_module.AFFECTED_NUMBER:
                label = f"受灾点{i}\n需求:{node.need.A_material.quantity:.0f}"
            else:
                label = f"补给点{i-main_module.AFFECTED_NUMBER}"

            self.ax.annotate(label, (node.x, node.y),
                           xytext=(0, -25), textcoords='offset points',
                           fontsize=8, ha='center',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                           zorder=4)

        # 添加统计信息
        info_text = f"机器人数量: {len(self.robots)}\n"
        info_text += f"受灾点: {main_module.AFFECTED_NUMBER}\n"
        info_text += f"补给点: {main_module.SUPPLE_NUMBER}"

        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def create_animation_gif(self, frames=60, output_path='../data/transport_animation.gif'):
        """创建 GIF 动画"""
        print(f"开始创建 {frames} 帧动画，输出到: {output_path}")

        # 创建图形
        self.fig, self.ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        self.fig.patch.set_facecolor(self.colors['background'])

        # 创建动画
        anim = animation.FuncAnimation(
            self.fig, self.update_frame, frames=frames,
            interval=100,  # 每帧100毫秒
            repeat=True,
            blit=False
        )

        # 保存为 GIF
        try:
            # 使用 pillow writer
            writer = animation.PillowWriter(fps=10)
            anim.save(output_path, writer=writer)
            print(f"GIF 动画已保存到: {output_path}")

            # 获取文件大小
            file_size = os.path.getsize(output_path) / (1024*1024)  # MB
            print(f"文件大小: {file_size:.2f} MB")
        except Exception as e:
            print(f"保存 GIF 时出错: {e}")
            # 尝试保存为 MP4
            mp4_path = output_path.replace('.gif', '.mp4')
            try:
                anim.save(mp4_path, writer='ffmpeg', fps=10)
                print(f"已保存为 MP4 格式: {mp4_path}")
            except:
                print("无法保存动画，尝试保存单帧图像")
                for i in range(0, frames, 10):  # 每10帧保存一张
                    self.update_frame(i)
                    frame_path = f'../data/animation_frame_{i:03d}.png'
                    plt.savefig(frame_path, dpi=self.dpi, bbox_inches='tight',
                               facecolor='white', edgecolor='none')
                    print(f"已保存帧: {frame_path}")

        plt.close()

def main():
    """主函数"""
    print("=== 应急救援物资调度 GIF 动画生成器（简化版） ===")

    # 创建动画对象
    anim = SimpleTransportAnimation()

    try:
        # 创建机器人和任务
        anim.create_robots()

        # 创建 GIF 动画
        anim.create_animation_gif(frames=80, output_path='../data/transport_animation.gif')

        # 创建较短的快速版本
        print("\n生成快速版本...")
        anim.create_animation_gif(frames=40, output_path='../data/transport_animation_fast.gif')

    except Exception as e:
        print(f"生成动画时出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n动画生成完成！")
    print("生成的文件:")
    print("- ../data/transport_animation.gif (标准版本)")
    print("- ../data/transport_animation_fast.gif (快速版本)")

if __name__ == "__main__":
    main()