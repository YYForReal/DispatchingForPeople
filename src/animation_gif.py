#!/usr/bin/env python3
"""
生成应急救援物资调度的 GIF 动画
"""

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import copy
import imageio
import os
from PIL import Image, ImageDraw, ImageFont
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import main as main
from material import MaterialPackage
import file_utils

class TransportAnimation:
    """物资运输动画类"""

    def __init__(self):
        # 重新加载数据
        file_utils.read_data()
        self.map_nodes = copy.deepcopy(main.map_nodes_backup)
        self.stone_list = main.stone_list
        self.robots = []

        # 动画参数
        self.fig_size = (14, 10)
        self.dpi = 100
        self.fps = 2  # 每秒帧数
        self.frames = []

        # 颜色配置
        self.colors = {
            'affected': '#FF6B6B',      # 受灾点 - 红色
            'supply': '#51CF66',         # 补给点 - 绿色
            'stone': '#868E96',          # 障碍物 - 灰色
            'robot': '#339AF0',          # 机器人 - 蓝色
            'path': '#FFD43B',           # 路径 - 黄色
            'background': '#F8F9FA'      # 背景 - 浅灰
        }

        # 创建输出目录
        self.temp_dir = '../data/temp_frames'
        os.makedirs(self.temp_dir, exist_ok=True)

        print("动画系统初始化完成")

    def create_robot(self, start_pos, tasks, name="Robot"):
        """创建机器人对象"""
        return {
            'name': name,
            'position': list(start_pos),
            'start_position': list(start_pos),
            'tasks': tasks.copy(),
            'current_task': 0,
            'carrying': MaterialPackage(0, 0, 0),
            'path': [],
            'color': np.random.choice(['#FF6B9D', '#C44569', '#F8961E', '#F9844A', '#43AA8B'])
        }

    def calculate_path(self, from_node, to_node):
        """计算两个节点之间的路径"""
        if from_node == to_node:
            return [from_node]

        # 使用已有的路径信息
        try:
            path_nodes = main.paths[from_node][to_node]
            if len(path_nodes) > 1:
                return path_nodes
            else:
                return [from_node, to_node]
        except:
            return [from_node, to_node]

    def simulate_transport(self):
        """模拟物资运输过程"""
        print("开始模拟物资运输...")

        # 尝试从文件读取最优染色体
        try:
            data = file_utils.read_data()
            saved_chromosome = data.get("best_chromosome")
            if saved_chromosome:
                print(f"使用保存的最优染色体: {saved_chromosome}")
                chromosome = type('Chromosome', (), {'path': saved_chromosome})()
            else:
                raise ValueError("未找到保存的染色体")
        except:
            print("未找到保存的染色体，使用测试染色体")
            # 创建一个测试染色体：3个机器人，分配不同任务
            chromosome = type('Chromosome', (), {
                'path': [0, 1, 'a', 2, 3, 'b', 4, 5, 'c', 6, 7]
            })()

        # 解析染色体，创建机器人
        tasks = []
        robot_count = 0

        for item in chromosome.path:
            if isinstance(item, str):
                # 机器人标记
                if tasks:
                    robot_count += 1
                    start_node = main.AFFECTED_NUMBER + (robot_count % main.SUPPLE_NUMBER)
                    self.robots.append(
                        self.create_robot(
                            self.map_nodes[start_node].get_pos(),
                            tasks,
                            f"Robot_{robot_count}"
                        )
                    )
                    tasks = []
            else:
                # 任务节点
                tasks.append(item)

        # 处理最后一个机器人的任务
        if tasks:
            robot_count += 1
            start_node = main.AFFECTED_NUMBER + (robot_count % main.SUPPLE_NUMBER)
            self.robots.append(
                self.create_robot(
                    self.map_nodes[start_node].get_pos(),
                    tasks,
                    f"Robot_{robot_count}"
                )
            )

        print(f"创建了 {len(self.robots)} 个机器人")
        for robot in self.robots:
            print(f"  {robot['name']}: 任务 {robot['tasks']}")

    def create_frame(self, frame_num, time_step):
        """创建单帧动画"""
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        fig.patch.set_facecolor(self.colors['background'])

        # 绘制受灾点
        x1 = [node.x for node in self.map_nodes[:main.AFFECTED_NUMBER]]
        y1 = [node.y for node in self.map_nodes[:main.AFFECTED_NUMBER]]
        ax.scatter(x1, y1, s=200, c=self.colors['affected'], marker='s',
                  label='受灾点', alpha=0.8, edgecolors='white', linewidth=2)

        # 绘制补给点
        x2 = [node.x for node in self.map_nodes[main.AFFECTED_NUMBER:]]
        y2 = [node.y for node in self.map_nodes[main.AFFECTED_NUMBER:]]
        ax.scatter(x2, y2, s=200, c=self.colors['supply'], marker='^',
                  label='补给点', alpha=0.8, edgecolors='white', linewidth=2)

        # 绘制障碍物
        for stone in self.stone_list:
            circle = plt.Circle((stone.x, stone.y), stone.radius,
                              color=self.colors['stone'], alpha=0.6, label='障碍物' if stone == self.stone_list[0] else "")
            ax.add_patch(circle)

        # 绘制连接线
        for i in range(len(self.map_nodes) - 1):
            for j in range(i + 1, len(self.map_nodes)):
                if main.distance_matrix_copy[i][j] != main.MAX_INF:
                    x = [self.map_nodes[i].x, self.map_nodes[j].x]
                    y = [self.map_nodes[i].y, self.map_nodes[j].y]
                    ax.plot(x, y, 'gray', alpha=0.2, linewidth=0.5)

        # 更新机器人位置
        for robot in self.robots:
            # 简单的移动模拟
            if robot['current_task'] < len(robot['tasks']):
                target_node = robot['tasks'][robot['current_task']]
                target_pos = self.map_nodes[target_node].get_pos()

                # 简单插值移动
                t = min(time_step / 20.0, 1.0)  # 20步到达目标
                current_pos = [
                    robot['start_position'][0] + (target_pos[0] - robot['start_position'][0]) * t,
                    robot['start_position'][1] + (target_pos[1] - robot['start_position'][1]) * t
                ]
                robot['position'] = current_pos

                if t >= 1.0:
                    robot['current_task'] += 1
                    if robot['current_task'] < len(robot['tasks']):
                        robot['start_position'] = list(current_pos)
            else:
                current_pos = robot['position']

            # 绘制机器人
            ax.scatter(current_pos[0], current_pos[1], s=150, c=robot['color'],
                      marker='o', alpha=0.9, edgecolors='white', linewidth=2, zorder=5)

            # 绘制机器人名称
            ax.annotate(robot['name'], (current_pos[0], current_pos[1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, fontweight='bold', color='white')

            # 绘制路径轨迹
            if robot['current_task'] > 0:
                path_x = [robot['tasks'][0]]
                path_y = [robot['tasks'][0]]
                for i in range(min(robot['current_task'], len(robot['tasks']))):
                    node = self.map_nodes[robot['tasks'][i]]
                    path_x.append(node.x)
                    path_y.append(node.y)

                if len(path_x) > 1:
                    ax.plot(path_x, path_y, color=robot['color'], alpha=0.5,
                           linewidth=2, linestyle='--')

        # 设置图形属性
        ax.set_title(f'应急救援物资调度 - 时间步: {time_step}', fontsize=16, fontweight='bold')
        ax.set_xlabel('x轴（单位：百米）', fontsize=12)
        ax.set_ylabel('y轴（单位：百米）', fontsize=12)
        ax.set_xlim(-10, 170)
        ax.set_ylim(-10, 170)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)

        # 添加节点标签
        for i, node in enumerate(self.map_nodes):
            if i < main.AFFECTED_NUMBER:
                label = f"受灾点{i}\n需求:{node.need.A_material.quantity:.0f}"
            else:
                label = f"补给点{i-main.AFFECTED_NUMBER}"

            ax.annotate(label, (node.x, node.y),
                       xytext=(0, -25), textcoords='offset points',
                       fontsize=8, ha='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        # 添加统计信息
        info_text = f"机器人数量: {len(self.robots)}\n"
        info_text += f"受灾点: {main.AFFECTED_NUMBER}\n"
        info_text += f"补给点: {main.SUPPLE_NUMBER}"

        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        # 保存帧
        frame_path = os.path.join(self.temp_dir, f'frame_{frame_num:03d}.png')
        plt.savefig(frame_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        return frame_path

    def generate_frames(self, max_frames=50):
        """生成所有动画帧"""
        print(f"开始生成 {max_frames} 帧动画...")

        frames = []
        for i in range(max_frames):
            frame_path = self.create_frame(i, i)
            frames.append(frame_path)

            if i % 10 == 0:
                print(f"已生成 {i+1}/{max_frames} 帧")

        print("所有帧生成完成")
        return frames

    def create_gif(self, frames, output_path='../data/transport_animation.gif'):
        """创建 GIF 动画"""
        print(f"正在创建 GIF 动画: {output_path}")

        images = []
        for frame_path in frames:
            try:
                img = Image.open(frame_path)
                images.append(img)
            except Exception as e:
                print(f"无法加载帧 {frame_path}: {e}")

        if images:
            # 保存为 GIF
            images[0].save(
                output_path,
                save_all=True,
                append_images=images[1:],
                duration=1000//self.fps,  # 每帧持续时间（毫秒）
                loop=0,  # 无限循环
                optimize=True
            )
            print(f"GIF 动画已保存到: {output_path}")

            # 获取文件大小
            file_size = os.path.getsize(output_path) / (1024*1024)  # MB
            print(f"文件大小: {file_size:.2f} MB")
        else:
            print("没有有效的帧可以创建 GIF")

    def cleanup(self):
        """清理临时文件"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print("临时文件已清理")

def main():
    """主函数"""
    print("=== 应急救援物资调度 GIF 动画生成器 ===")

    # 创建动画对象
    anim = TransportAnimation()

    try:
        # 模拟运输过程
        anim.simulate_transport()

        # 生成动画帧
        frames = anim.generate_frames(max_frames=40)

        # 创建 GIF
        anim.create_gif(frames)

        # 可选：创建更高帧率的版本
        print("\n生成高帧率版本...")
        anim.create_gif(frames[:20], '../data/transport_animation_fast.gif')

    except Exception as e:
        print(f"生成动画时出错: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理临时文件
        anim.cleanup()

    print("\n动画生成完成！")
    print("生成的文件:")
    print("- ../data/transport_animation.gif (标准速度)")
    print("- ../data/transport_animation_fast.gif (快速版本)")

if __name__ == "__main__":
    main()