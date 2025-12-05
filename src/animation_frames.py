#!/usr/bin/env python3
"""
生成应急救援物资调度的动画帧（PNG序列）
"""

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import copy
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import main as main_module
from material import MaterialPackage
import file_utils

class FrameGenerator:
    """动画帧生成器"""

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

        # 创建输出目录
        self.output_dir = '../data/animation_frames'
        os.makedirs(self.output_dir, exist_ok=True)

        print("动画帧生成器初始化完成")

    def create_robots(self):
        """创建机器人和任务分配"""
        # 使用测试染色体：3个机器人
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
                        'progress': 0.0,
                        'completed_tasks': []
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
                'progress': 0.0,
                'completed_tasks': []
            }
            self.robots.append(robot)

        print(f"创建了 {len(self.robots)} 个机器人")
        for robot in self.robots:
            print(f"  {robot['name']}: 任务 {robot['tasks']}")

    def update_robots(self, time_step):
        """更新机器人位置"""
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
                robot['progress'] += 0.08  # 每帧移动8%
                if robot['progress'] >= 1.0:
                    robot['progress'] = 0.0
                    robot['completed_tasks'].append(robot['tasks'][robot['current_task']])
                    robot['current_task'] += 1
                    robot['start_position'] = list(robot['position'])

    def draw_frame(self, frame_num):
        """绘制单帧"""
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        fig.patch.set_facecolor(self.colors['background'])

        # 绘制受灾点
        x1 = [node.x for node in self.map_nodes[:main_module.AFFECTED_NUMBER]]
        y1 = [node.y for node in self.map_nodes[:main_module.AFFECTED_NUMBER]]
        ax.scatter(x1, y1, s=300, c=self.colors['affected'], marker='s',
                  label='受灾点', alpha=0.8, edgecolors='white', linewidth=2, zorder=3)

        # 绘制补给点
        x2 = [node.x for node in self.map_nodes[main_module.AFFECTED_NUMBER:]]
        y2 = [node.y for node in self.map_nodes[main_module.AFFECTED_NUMBER:]]
        ax.scatter(x2, y2, s=300, c=self.colors['supply'], marker='^',
                  label='补给点', alpha=0.8, edgecolors='white', linewidth=2, zorder=3)

        # 绘制障碍物
        for stone in self.stone_list:
            circle = plt.Circle((stone.x, stone.y), stone.radius,
                              color=self.colors['stone'], alpha=0.6, zorder=1)
            ax.add_patch(circle)

        # 绘制连接线
        for i in range(len(self.map_nodes) - 1):
            for j in range(i + 1, len(self.map_nodes)):
                if main_module.distance_matrix_copy[i][j] != main_module.MAX_INF:
                    x = [self.map_nodes[i].x, self.map_nodes[j].x]
                    y = [self.map_nodes[i].y, self.map_nodes[j].y]
                    ax.plot(x, y, 'gray', alpha=0.2, linewidth=0.5, zorder=0)

        # 绘制机器人
        for robot in self.robots:
            # 绘制机器人
            ax.scatter(robot['position'][0], robot['position'][1],
                      s=200, c=robot['color'], marker='o',
                      alpha=0.9, edgecolors='white', linewidth=2, zorder=5)

            # 绘制机器人名称
            ax.annotate(robot['name'], (robot['position'][0], robot['position'][1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, fontweight='bold', color='black', zorder=6)

            # 绘制路径轨迹
            if robot['completed_tasks']:
                path_x = [robot['tasks'][0]]
                path_y = [robot['tasks'][0]]
                for i in robot['completed_tasks']:
                    node = self.map_nodes[i]
                    path_x.append(node.x)
                    path_y.append(node.y)

                if len(path_x) > 1:
                    ax.plot(path_x, path_y, color=robot['color'], alpha=0.5,
                           linewidth=2, linestyle='--', zorder=2)

            # 绘制当前位置到目标的路径
            if robot['current_task'] < len(robot['tasks']):
                target_node = robot['tasks'][robot['current_task']]
                target_pos = [self.map_nodes[target_node].x, self.map_nodes[target_node].y]
                ax.plot([robot['position'][0], target_pos[0]],
                       [robot['position'][1], target_pos[1]],
                       color=robot['color'], alpha=0.7, linewidth=2, linestyle=':', zorder=2)

        # 设置图形属性
        ax.set_title(f'应急救援物资调度 - 时间步: {frame_num}', fontsize=16, fontweight='bold')
        ax.set_xlabel('x轴（单位：百米）', fontsize=12)
        ax.set_ylabel('y轴（单位：百米）', fontsize=12)
        ax.set_xlim(-10, 170)
        ax.set_ylim(-10, 170)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)

        # 添加节点标签
        for i, node in enumerate(self.map_nodes):
            if i < main_module.AFFECTED_NUMBER:
                label = f"受灾点{i}\n需求:{node.need.A_material.quantity:.0f}"
            else:
                label = f"补给点{i-main_module.AFFECTED_NUMBER}"

            ax.annotate(label, (node.x, node.y),
                       xytext=(0, -25), textcoords='offset points',
                       fontsize=8, ha='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                       zorder=4)

        # 添加统计信息
        info_text = f"机器人数量: {len(self.robots)}\n"
        info_text += f"受灾点: {main_module.AFFECTED_NUMBER}\n"
        info_text += f"补给点: {main_module.SUPPLE_NUMBER}\n"

        completed_count = sum(len(robot['completed_tasks']) for robot in self.robots)
        total_tasks = sum(len(robot['tasks']) for robot in self.robots)
        info_text += f"任务进度: {completed_count}/{total_tasks}"

        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        return fig, ax

    def generate_frames(self, total_frames=50):
        """生成所有动画帧"""
        print(f"开始生成 {total_frames} 帧动画...")

        for frame in range(total_frames):
            # 更新机器人位置
            self.update_robots(frame)

            # 绘制帧
            fig, ax = self.draw_frame(frame)

            # 保存帧
            frame_path = os.path.join(self.output_dir, f'frame_{frame:04d}.png')
            plt.savefig(frame_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()

            if frame % 10 == 0:
                print(f"已生成 {frame+1}/{total_frames} 帧: {frame_path}")

        print(f"所有动画帧生成完成！保存在: {self.output_dir}")

        # 生成一个简单的动画说明文件
        self.create_animation_info(total_frames)

    def create_animation_info(self, total_frames):
        """创建动画说明文件"""
        info_path = os.path.join(self.output_dir, 'animation_info.txt')
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write("应急救援物资调度动画\n")
            f.write("=" * 30 + "\n")
            f.write(f"总帧数: {total_frames}\n")
            f.write(f"机器人数量: {len(self.robots)}\n")
            f.write(f"受灾点数量: {main_module.AFFECTED_NUMBER}\n")
            f.write(f"补给点数量: {main_module.SUPPLE_NUMBER}\n")
            f.write(f"障碍物数量: {len(self.stone_list)}\n\n")

            f.write("机器人任务分配:\n")
            for i, robot in enumerate(self.robots):
                f.write(f"  {robot['name']}: {robot['tasks']}\n")

            f.write("\n使用方法:\n")
            f.write("1. 使用图像查看器按顺序查看帧文件\n")
            f.write("2. 或使用工具将帧序列转换为视频/GIF\n")
            f.write("3. 例如使用 ffmpeg: ffmpeg -r 10 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p animation.mp4\n")

        print(f"动画说明文件已保存: {info_path}")

def main():
    """主函数"""
    print("=== 应急救援物资调度动画帧生成器 ===")

    # 创建帧生成器
    generator = FrameGenerator()

    try:
        # 创建机器人和任务
        generator.create_robots()

        # 生成动画帧
        generator.generate_frames(total_frames=60)

    except Exception as e:
        print(f"生成动画帧时出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n动画帧生成完成！")
    print("输出目录: ../data/animation_frames/")
    print("包含文件:")
    print("- frame_0000.png 到 frame_0059.png (60帧)")
    print("- animation_info.txt (动画说明)")

if __name__ == "__main__":
    main()