#!/usr/bin/env python3
"""
完整的应急救援物资调度动画生成器
展示真实的物资运输过程：补给 -> 装载 -> 运输 -> 卸载 -> 返回
"""

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import math

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import main as main_module
from material import MaterialPackage, AMaterial, BMaterial, CMaterial
import file_utils

class CompleteDispatchAnimation:
    """完整物资调度动画生成器"""

    def __init__(self):
        # 重新加载数据
        file_utils.read_data()
        self.map_nodes = copy.deepcopy(main_module.map_nodes_backup)
        self.stone_list = main_module.stone_list
        self.robots = []
        self.time_step = 0
        self.completed_deliveries = []

        # 动画参数
        self.fig_size = (16, 12)
        self.dpi = 120
        self.total_frames = 300

        # 颜色配置
        self.colors = {
            'affected': '#FF6B6B',      # 受灾点 - 红色
            'supply': '#51CF66',         # 补给点 - 绿色
            'stone': '#868E96',          # 障碍物 - 灰色
            'robot_empty': '#339AF0',    # 空载机器人 - 蓝色
            'robot_loaded': '#FF8C00',   # 满载机器人 - 橙色
            'path_empty': '#A5D8FF',     # 空载路径 - 浅蓝
            'path_loaded': '#FFA94D',    # 满载路径 - 浅橙
            'background': '#F8F9FA'      # 背景 - 浅灰
        }

        # 创建输出目录
        self.output_dir = '../data/complete_animation'
        os.makedirs(self.output_dir, exist_ok=True)

        print("完整物资调度动画生成器初始化完成")

    def create_realistic_robots(self):
        """创建真实场景的机器人和调度计划"""

        # 定义三种类型的机器人
        robot_types = [
            {'name': 'A型运输车', 'capacity': 80, 'color': '#FF6B9D', 'speed': 1.2},
            {'name': 'B型运输车', 'capacity': 15, 'color': '#C44569', 'speed': 1.5},
            {'name': 'C型运输车', 'capacity': 5, 'color': '#F8961E', 'speed': 2.0}
        ]

        # 分析所有受灾点的需求
        affected_nodes = self.map_nodes[:main_module.AFFECTED_NUMBER]
        total_needs = []
        for i, node in enumerate(affected_nodes):
            need = {
                'node': i,
                'emergency': node.need.A_material.quantity,
                'regular': node.need.B_material.quantity,
                'equipment': node.need.C_material.quantity
            }
            total_needs.append(need)
            print(f"受灾点 {i}: 应急={need['emergency']:.1f}, 常规={need['regular']:.1f}, 设备={need['equipment']:.1f}")

        # 创建运输任务序列
        transport_tasks = []
        for i, need in enumerate(total_needs):
            # 每个受灾点可能需要多次运输
            total_load = need['emergency'] + need['regular'] + need['equipment']

            # 根据总需求计算需要的运输次数
            trips_needed = max(1, int(math.ceil(total_load / 50)))  # 假设平均载重50

            for trip in range(trips_needed):
                task = {
                    'target': i,
                    'emergency_load': min(need['emergency'] / trips_needed, 20),
                    'regular_load': min(need['regular'] / trips_needed, 10),
                    'equipment_load': min(need['equipment'] / trips_needed, 5)
                }
                transport_tasks.append(task)

        print(f"生成了 {len(transport_tasks)} 个运输任务")

        # 分配任务给机器人
        robot_count = 4
        for i in range(robot_count):
            robot_type = robot_types[i % len(robot_types)]
            start_supply = main_module.AFFECTED_NUMBER + (i % main_module.SUPPLE_NUMBER)

            # 为每个机器人分配任务
            robot_tasks = []
            for j in range(i, len(transport_tasks), robot_count):
                robot_tasks.append(transport_tasks[j])

            robot = {
                'id': i,
                'name': f'{robot_type["name"]}_{i}',
                'type': robot_type['name'],
                'capacity': robot_type['capacity'],
                'speed': robot_type['speed'],
                'color': robot_type['color'],
                'start_supply': start_supply,
                'position': [self.map_nodes[start_supply].x, self.map_nodes[start_supply].y],
                'target': start_supply,
                'state': 'idle',  # idle, loading, transporting_to_affected, unloading, returning, loading_at_supply
                'tasks': robot_tasks,
                'current_task': 0,
                'load': MaterialPackage(0, 0, 0),
                'delivery_history': [],
                'progress': 0.0
            }
            self.robots.append(robot)

        print(f"创建了 {len(self.robots)} 个机器人:")
        for robot in self.robots:
            print(f"  {robot['name']}: 载重{robot['capacity']}, 任务数{len(robot['tasks'])}")

    def calculate_path_distance(self, from_node, to_node):
        """计算两个节点之间的距离"""
        if from_node == to_node:
            return 0
        try:
            return main_module.distance_matrix_copy[from_node][to_node]
        except:
            dx = self.map_nodes[to_node].x - self.map_nodes[from_node].x
            dy = self.map_nodes[to_node].y - self.map_nodes[from_node].y
            return math.sqrt(dx*dx + dy*dy)

    def update_robot_state(self, robot, time_step):
        """更新单个机器人的状态和位置"""

        # 状态机逻辑
        if robot['state'] == 'idle':
            # 空闲状态，选择下一个任务
            if robot['current_task'] < len(robot['tasks']):
                task = robot['tasks'][robot['current_task']]

                # 计算最佳补给点（优先选择距离目标受灾点最近的补给点）
                best_supply = None
                min_distance = float('inf')

                for supply_idx in range(main_module.AFFECTED_NUMBER, len(self.map_nodes)):
                    distance = self.calculate_path_distance(supply_idx, task['target'])
                    if distance < min_distance:
                        min_distance = distance
                        best_supply = supply_idx

                robot['target_supply'] = best_supply
                robot['target'] = best_supply
                robot['state'] = 'moving_to_supply'
                robot['start_position'] = list(robot['position'])
                robot['progress'] = 0.0

                print(f"{robot['name']} 开始前往补给点 {best_supply-main_module.AFFECTED_NUMBER} 为受灾点 {task['target']} 准备物资")

        elif robot['state'] == 'moving_to_supply':
            # 前往补给点
            if robot['target'] is not None:
                target_pos = [self.map_nodes[robot['target']].x, self.map_nodes[robot['target']].y]
                robot['progress'] += 0.05 * robot['speed']

                if robot['progress'] >= 1.0:
                    robot['position'] = target_pos
                    robot['state'] = 'loading'
                    robot['loading_progress'] = 0.0
                    print(f"{robot['name']} 到达补给点，开始装载物资")
                else:
                    # 插值移动
                    robot['position'][0] = robot['start_position'][0] + (target_pos[0] - robot['start_position'][0]) * robot['progress']
                    robot['position'][1] = robot['start_position'][1] + (target_pos[1] - robot['start_position'][1]) * robot['progress']

        elif robot['state'] == 'loading':
            # 装载物资
            task = robot['tasks'][robot['current_task']]
            robot['loading_progress'] += 0.1

            if robot['loading_progress'] >= 1.0:
                # 完成装载
                robot['load'] = MaterialPackage(
                    task['emergency_load'],
                    task['regular_load'],
                    task['equipment_load']
                )
                robot['state'] = 'transporting_to_affected'
                robot['target'] = task['target']
                robot['start_position'] = list(robot['position'])
                robot['progress'] = 0.0
                print(f"{robot['name']} 装载完成，前往受灾点 {task['target']}")

        elif robot['state'] == 'transporting_to_affected':
            # 运输到受灾点
            if robot['target'] is not None:
                task = robot['tasks'][robot['current_task']]
                target_pos = [self.map_nodes[robot['target']].x, self.map_nodes[robot['target']].y]
                robot['progress'] += 0.05 * robot['speed']

                if robot['progress'] >= 1.0:
                    robot['position'] = target_pos
                    robot['state'] = 'unloading'
                    robot['unloading_progress'] = 0.0
                    print(f"{robot['name']} 到达受灾点 {task['target']}, 开始卸载")
                else:
                    # 插值移动
                    robot['position'][0] = robot['start_position'][0] + (target_pos[0] - robot['start_position'][0]) * robot['progress']
                    robot['position'][1] = robot['start_position'][1] + (target_pos[1] - robot['start_position'][1]) * robot['progress']

        elif robot['state'] == 'unloading':
            # 卸载物资
            robot['unloading_progress'] += 0.1

            if robot['unloading_progress'] >= 1.0:
                # 完成卸载
                task = robot['tasks'][robot['current_task']]
                delivery = {
                    'time': time_step,
                    'from': robot['target_supply'],
                    'to': task['target'],
                    'load': robot['load']
                }
                robot['delivery_history'].append(delivery)
                self.completed_deliveries.append(delivery)

                # 更新受灾点需求
                affected_node = self.map_nodes[task['target']]
                affected_node.need.A_material.quantity -= task['emergency_load']
                affected_node.need.B_material.quantity -= task['regular_load']
                affected_node.need.C_material.quantity -= task['equipment_load']

                robot['load'] = MaterialPackage(0, 0, 0)
                robot['current_task'] += 1
                robot['state'] = 'idle'
                robot['progress'] = 0.0
                print(f"{robot['name']} 卸载完成，任务进度: {robot['current_task']}/{len(robot['tasks'])}")

    def draw_frame(self, frame_num):
        """绘制单帧动画"""
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        fig.patch.set_facecolor(self.colors['background'])

        # 绘制受灾点（显示剩余需求）
        x1 = [node.x for node in self.map_nodes[:main_module.AFFECTED_NUMBER]]
        y1 = [node.y for node in self.map_nodes[:main_module.AFFECTED_NUMBER]]

        # 根据剩余需求调整颜色深度
        remaining_needs = [node.need.A_material.quantity + node.need.B_material.quantity + node.need.C_material.quantity
                          for node in self.map_nodes[:main_module.AFFECTED_NUMBER]]
        max_need = max(remaining_needs) if remaining_needs else 1

        for i, (x, y, need) in enumerate(zip(x1, y1, remaining_needs)):
            alpha = 0.3 + 0.7 * (need / max_need)  # 需求越多颜色越深
            ax.scatter(x, y, s=400, c=self.colors['affected'], marker='s',
                      alpha=alpha, edgecolors='white', linewidth=2, zorder=3)

        # 绘制补给点
        x2 = [node.x for node in self.map_nodes[main_module.AFFECTED_NUMBER:]]
        y2 = [node.y for node in self.map_nodes[main_module.AFFECTED_NUMBER:]]
        ax.scatter(x2, y2, s=400, c=self.colors['supply'], marker='^',
                  alpha=0.8, edgecolors='white', linewidth=2, zorder=3)

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
            # 根据载重状态选择颜色
            color = self.colors['robot_loaded'] if robot['load'].A_material.quantity > 0 else self.colors['robot_empty']

            # 绘制机器人
            ax.scatter(robot['position'][0], robot['position'][1],
                      s=250, c=color, marker='o',
                      alpha=0.9, edgecolors='white', linewidth=2, zorder=5)

            # 绘制机器人标签
            label = f"{robot['name']}\n{robot['state']}"
            ax.annotate(label, (robot['position'][0], robot['position'][1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, fontweight='bold', color='black', zorder=6)

            # 绘制运载信息
            if robot['load'].A_material.quantity > 0:
                load_text = f"载重: {robot['load'].A_material.quantity:.1f}"
                ax.annotate(load_text, (robot['position'][0], robot['position'][1]),
                           xytext=(5, -20), textcoords='offset points',
                           fontsize=7, color='red', zorder=6)

            # 绘制路径轨迹
            if robot['delivery_history']:
                path_x = []
                path_y = []
                for delivery in robot['delivery_history'][-3:]:  # 只显示最近3次
                    from_pos = [self.map_nodes[delivery['from']].x, self.map_nodes[delivery['from']].y]
                    to_pos = [self.map_nodes[delivery['to']].x, self.map_nodes[delivery['to']].y]
                    path_x.extend([from_pos[0], to_pos[0]])
                    path_y.extend([from_pos[1], to_pos[1]])

                if len(path_x) > 1:
                    ax.plot(path_x, path_y, color=robot['color'], alpha=0.5,
                           linewidth=2, linestyle='--', zorder=2)

        # 设置图形属性
        ax.set_title(f'应急救援物资调度完整过程 - 时间步: {frame_num}', fontsize=18, fontweight='bold')
        ax.set_xlabel('x轴（单位：百米）', fontsize=14)
        ax.set_ylabel('y轴（单位：百米）', fontsize=14)
        ax.set_xlim(-10, 170)
        ax.set_ylim(-10, 170)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=12)

        # 添加节点标签（显示剩余需求）
        for i, node in enumerate(self.map_nodes):
            if i < main_module.AFFECTED_NUMBER:
                total_remaining = node.need.A_material.quantity + node.need.B_material.quantity + node.need.C_material.quantity
                label = f"受灾点{i}\n剩余需求: {total_remaining:.1f}"
            else:
                label = f"补给点{i-main_module.AFFECTED_NUMBER}"

            ax.annotate(label, (node.x, node.y),
                       xytext=(0, -30), textcoords='offset points',
                       fontsize=9, ha='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                       zorder=4)

        # 添加统计信息面板
        info_text = f"机器人数量: {len(self.robots)}\n"
        info_text += f"受灾点: {main_module.AFFECTED_NUMBER}\n"
        info_text += f"补给点: {main_module.SUPPLE_NUMBER}\n"
        info_text += f"已完成运输: {len(self.completed_deliveries)}\n\n"

        info_text += "各机器人状态:\n"
        for robot in self.robots:
            info_text += f"• {robot['name']}: {robot['state']}\n"

        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        return fig, ax

    def generate_complete_animation(self):
        """生成完整的物资调度动画"""
        print(f"开始生成 {self.total_frames} 帧完整物资调度动画...")

        # 创建机器人和任务
        self.create_realistic_robots()

        for frame in range(self.total_frames):
            # 更新所有机器人状态
            for robot in self.robots:
                self.update_robot_state(robot, frame)

            # 绘制帧
            fig, ax = self.draw_frame(frame)

            # 保存帧
            frame_path = os.path.join(self.output_dir, f'complete_frame_{frame:04d}.png')
            plt.savefig(frame_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()

            if frame % 20 == 0:
                completed_tasks = sum(robot['current_task'] for robot in self.robots)
                total_tasks = sum(len(robot['tasks']) for robot in self.robots)
                progress = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
                print(f"已生成 {frame+1}/{self.total_frames} 帧, 任务进度: {progress:.1f}%")

        print(f"完整物资调度动画生成完成！保存在: {self.output_dir}")

        # 生成动画说明文件
        self.create_animation_report()

    def create_animation_report(self):
        """生成动画报告"""
        report_path = os.path.join(self.output_dir, 'complete_animation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("应急救援物资调度完整过程报告\n")
            f.write("=" * 40 + "\n\n")

            f.write(f"动画规格:\n")
            f.write(f"- 总帧数: {self.total_frames}\n")
            f.write(f"- 机器人数量: {len(self.robots)}\n")
            f.write(f"- 受灾点数量: {main_module.AFFECTED_NUMBER}\n")
            f.write(f"- 补给点数量: {main_module.SUPPLE_NUMBER}\n")
            f.write(f"- 完成运输次数: {len(self.completed_deliveries)}\n\n")

            f.write("机器人详情:\n")
            for robot in self.robots:
                f.write(f"\n{robot['name']}:\n")
                f.write(f"  类型: {robot['type']}\n")
                f.write(f"  载重: {robot['capacity']}\n")
                f.write(f"  总任务: {len(robot['tasks'])}\n")
                f.write(f"  完成任务: {robot['current_task']}\n")
                f.write(f"  运输历史: {len(robot['delivery_history'])} 次\n")

            f.write(f"\n运输记录:\n")
            for i, delivery in enumerate(self.completed_deliveries):
                f.write(f"  {i+1}. 从补给点{delivery['from']-main_module.AFFECTED_NUMBER} 到受灾点{delivery['to']}")
                f.write(f" (应急:{delivery['load'].A_material.quantity:.1f}, ")
                f.write(f"常规:{delivery['load'].B_material.quantity:.1f}, ")
                f.write(f"设备:{delivery['load'].C_material.quantity:.1f})\n")

            f.write(f"\n视频生成命令:\n")
            f.write(f"cd {self.output_dir}\n")
            f.write(f"ffmpeg -r 10 -i complete_frame_%04d.png \\\n")
            f.write(f'       -vf "scale=iw-1:ih,scale=trunc(iw/2)*2:trunc(ih/2)*2" \\\n')
            f.write(f'       -c:v libx264 -pix_fmt yuv420p complete_dispatch.mp4\n')

        print(f"动画报告已保存: {report_path}")

def main():
    """主函数"""
    print("=== 完整应急救援物资调度动画生成器 ===")

    # 创建动画生成器
    animator = CompleteDispatchAnimation()

    try:
        # 生成完整动画
        animator.generate_complete_animation()

    except Exception as e:
        print(f"生成动画时出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n完整物资调度动画生成完成！")
    print(f"输出目录: ../data/complete_animation/")
    print("包含文件:")
    print("- complete_frame_0000.png 到 complete_frame_0119.png (120帧)")
    print("- complete_animation_report.txt (详细报告)")

if __name__ == "__main__":
    main()