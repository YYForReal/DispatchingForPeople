#!/usr/bin/env python3
"""
基于遗传算法最优解的应急救援物资调度动画
使用 main.py 找到的最优调度策略生成动画
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

class GAOptimizedAnimation:
    """基于遗传算法最优解的物资调度动画"""

    def __init__(self):
        # 重新加载数据
        file_utils.read_data()
        self.map_nodes = copy.deepcopy(main_module.map_nodes_backup)
        self.stone_list = main_module.stone_list
        self.robots = []
        self.time_step = 0
        self.completed_deliveries = []

        # 动画参数
        self.fig_size = (18, 14)
        self.dpi = 120
        self.total_frames = 500

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
        self.output_dir = '../data/ga_optimized_animation'
        os.makedirs(self.output_dir, exist_ok=True)

        # 读取遗传算法最优解
        self.ga_data = file_utils.read_data()
        self.best_chromosome = self.ga_data.get("best_chromosome", [])
        self.best_fitness = self.ga_data.get("best_chromosome_fitness", 0)

        print("基于遗传算法最优解的动画生成器初始化完成")
        print(f"最优染色体: {self.best_chromosome}")
        print(f"最优适应度（总焦虑度）: {self.best_fitness:.2f}")

    def decode_chromosome_to_robots(self):
        """解码遗传算法染色体为机器人和任务分配"""
        if not self.best_chromosome:
            print("未找到遗传算法最优解，使用默认调度策略")
            return self.create_default_robots()

        print("解码遗传算法最优解...")

        # 定义机器人类型（与main.py保持一致）
        ROBOT_A_CAPACITY = main_module.ROBOT_A_CAPACITY  # 80
        ROBOT_B_CAPACITY = main_module.ROBOT_B_CAPACITY  # 15
        ROBOT_C_CAPACITY = main_module.ROBOT_C_CAPACITY  # 5

        robot_types = [
            {'name': 'A型机器人', 'capacity': ROBOT_A_CAPACITY, 'color': '#FF6B9D', 'speed': 1.0},
            {'name': 'B型机器人', 'capacity': ROBOT_B_CAPACITY, 'color': '#C44569', 'speed': 1.3},
            {'name': 'C型机器人', 'capacity': ROBOT_C_CAPACITY, 'color': '#F8961E', 'speed': 1.6}
        ]

        # 解码染色体并过滤无效节点
        valid_tasks = []
        for gene in self.best_chromosome:
            if isinstance(gene, int) and gene < main_module.AFFECTED_NUMBER:
                valid_tasks.append(gene)

        print(f"有效任务节点: {valid_tasks}")

        # 分配任务给机器人
        robot_assignments = []
        robot_count = min(4, len(valid_tasks))  # 最多4个机器人

        for i in range(robot_count):
            robot_type = robot_types[i % len(robot_types)]
            start_supply = main_module.AFFECTED_NUMBER + (i % main_module.SUPPLE_NUMBER)

            # 为每个机器人分配任务
            robot_tasks = []
            for j in range(i, len(valid_tasks), robot_count):
                robot_tasks.append(valid_tasks[j])

            robot = {
                'id': i,
                'name': f"{robot_type['name']}_{i}",
                'type': robot_type['name'],
                'capacity': robot_type['capacity'],
                'speed': robot_type['speed'],
                'color': robot_type['color'],
                'start_supply': start_supply,
                'position': [self.map_nodes[start_supply].x, self.map_nodes[start_supply].y],
                'target': start_supply,
                'state': 'idle',
                'tasks': robot_tasks.copy(),
                'current_task': 0,
                'load': MaterialPackage(0, 0, 0),
                'delivery_history': [],
                'progress': 0.0
            }
            robot_assignments.append(robot)

            print(f"机器人 {robot['name']}: 载重{robot['capacity']}, 任务 {robot_tasks}")

        return robot_assignments

    def create_default_robots(self):
        """创建默认机器人（当没有遗传算法结果时使用）"""
        robot_types = [
            {'name': 'A型机器人', 'capacity': 80, 'color': '#FF6B9D', 'speed': 1.0},
            {'name': 'B型机器人', 'capacity': 15, 'color': '#C44569', 'speed': 1.3},
            {'name': 'C型机器人', 'capacity': 5, 'color': '#F8961E', 'speed': 1.6}
        ]

        robots = []
        for i in range(3):
            robot_type = robot_types[i % len(robot_types)]
            start_supply = main_module.AFFECTED_NUMBER + (i % main_module.SUPPLE_NUMBER)

            robot = {
                'id': i,
                'name': f"{robot_type['name']}_{i}",
                'type': robot_type['name'],
                'capacity': robot_type['capacity'],
                'speed': robot_type['speed'],
                'color': robot_type['color'],
                'start_supply': start_supply,
                'position': [self.map_nodes[start_supply].x, self.map_nodes[start_supply].y],
                'target': start_supply,
                'state': 'idle',
                'tasks': list(range(i, min(i+3, main_module.AFFECTED_NUMBER))),
                'current_task': 0,
                'load': MaterialPackage(0, 0, 0),
                'delivery_history': [],
                'progress': 0.0
            }
            robots.append(robot)

        return robots

    def calculate_optimal_load(self, target_node, robot_capacity):
        """计算针对目标节点的最优装载量"""
        # 确保目标节点是受灾点
        if target_node >= main_module.AFFECTED_NUMBER:
            return MaterialPackage(0, 0, 0)

        affected_node = self.map_nodes[target_node]

        # 获取受灾点的需求
        emergency_need = affected_node.need.A_material.quantity
        regular_need = affected_node.need.B_material.quantity
        equipment_need = affected_node.need.C_material.quantity

        # 按优先级比例分配物资
        total_need = emergency_need + regular_need + equipment_need

        if total_need <= 0:
            return MaterialPackage(0, 0, 0)

        # 计算各类物资的装载比例
        emergency_ratio = emergency_need / total_need
        regular_ratio = regular_need / total_need
        equipment_ratio = equipment_need / total_need

        # 根据机器人载重分配
        emergency_load = min(emergency_need, robot_capacity * emergency_ratio)
        regular_load = min(regular_need, robot_capacity * regular_ratio)
        equipment_load = min(equipment_need, robot_capacity * equipment_ratio)

        # 确保不超过载重
        total_load = emergency_load + regular_load + equipment_load
        if total_load > robot_capacity:
            scale = robot_capacity / total_load
            emergency_load *= scale
            regular_load *= scale
            equipment_load *= scale

        return MaterialPackage(emergency_load, regular_load, equipment_load)

    def find_best_supply_point(self, target_node):
        """为给定的目标节点找到最佳补给点"""
        min_distance = float('inf')
        best_supply = None

        for supply_idx in range(main_module.AFFECTED_NUMBER, len(self.map_nodes)):
            distance = self.calculate_distance(supply_idx, target_node)
            if distance < min_distance:
                min_distance = distance
                best_supply = supply_idx

        return best_supply

    def calculate_distance(self, from_node, to_node):
        """计算两个节点之间的距离"""
        if from_node == to_node:
            return 0
        try:
            dist = main_module.distance_matrix_copy[from_node][to_node]
            return dist if dist != main_module.MAX_INF else float('inf')
        except:
            dx = self.map_nodes[to_node].x - self.map_nodes[from_node].x
            dy = self.map_nodes[to_node].y - self.map_nodes[from_node].y
            return math.sqrt(dx*dx + dy*dy)

    def update_robot_state(self, robot, time_step):
        """更新单个机器人的状态（完整调度状态机）"""

        if robot['state'] == 'idle':
            # 空闲状态，选择下一个任务
            if robot['current_task'] < len(robot['tasks']):
                target_node = robot['tasks'][robot['current_task']]

                # 找到最佳补给点
                best_supply = self.find_best_supply_point(target_node)

                robot['target_supply'] = best_supply
                robot['target'] = best_supply
                robot['state'] = 'moving_to_supply'
                robot['start_position'] = list(robot['position'])
                robot['progress'] = 0.0

                print(f"{robot['name']} 前往补给点 {best_supply-main_module.AFFECTED_NUMBER} 为受灾点 {target_node} 准备物资")

        elif robot['state'] == 'moving_to_supply':
            # 前往补给点
            if robot['target'] is not None:
                target_pos = [self.map_nodes[robot['target']].x, self.map_nodes[robot['target']].y]
                robot['progress'] += 0.03 * robot['speed']  # 调整速度

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
            robot['loading_progress'] += 0.08

            if robot['loading_progress'] >= 1.0:
                # 计算最优装载量
                target_node = robot['tasks'][robot['current_task']]
                robot['load'] = self.calculate_optimal_load(target_node, robot['capacity'])

                robot['state'] = 'transporting_to_affected'
                robot['target'] = target_node
                robot['start_position'] = list(robot['position'])
                robot['progress'] = 0.0

                total_load = robot['load'].A_material.quantity + robot['load'].B_material.quantity + robot['load'].C_material.quantity
                print(f"{robot['name']} 装载完成({total_load:.1f}单位)，前往受灾点 {target_node}")

        elif robot['state'] == 'transporting_to_affected':
            # 运输到受灾点
            if robot['target'] is not None:
                target_pos = [self.map_nodes[robot['target']].x, self.map_nodes[robot['target']].y]
                robot['progress'] += 0.03 * robot['speed']

                if robot['progress'] >= 1.0:
                    robot['position'] = target_pos
                    robot['state'] = 'unloading'
                    robot['unloading_progress'] = 0.0
                    print(f"{robot['name']} 到达受灾点 {robot['target']}, 开始卸载")
                else:
                    # 插值移动
                    robot['position'][0] = robot['start_position'][0] + (target_pos[0] - robot['start_position'][0]) * robot['progress']
                    robot['position'][1] = robot['start_position'][1] + (target_pos[1] - robot['start_position'][1]) * robot['progress']

        elif robot['state'] == 'unloading':
            # 卸载物资
            robot['unloading_progress'] += 0.08

            if robot['unloading_progress'] >= 1.0:
                # 完成卸载
                target_node = robot['tasks'][robot['current_task']]
                delivery = {
                    'time': time_step,
                    'from': robot['target_supply'],
                    'to': target_node,
                    'load': robot['load']
                }
                robot['delivery_history'].append(delivery)
                self.completed_deliveries.append(delivery)

                # 更新受灾点需求
                affected_node = self.map_nodes[target_node]
                affected_node.need.A_material.quantity -= robot['load'].A_material.quantity
                affected_node.need.B_material.quantity -= robot['load'].B_material.quantity
                affected_node.need.C_material.quantity -= robot['load'].C_material.quantity

                robot['load'] = MaterialPackage(0, 0, 0)
                robot['current_task'] += 1
                robot['state'] = 'idle'
                robot['progress'] = 0.0

                total_delivered = robot['load'].A_material.quantity + robot['load'].B_material.quantity + robot['load'].C_material.quantity
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
            ax.scatter(x, y, s=500, c=self.colors['affected'], marker='s',
                      alpha=alpha, edgecolors='white', linewidth=2, zorder=3)

        # 绘制补给点
        x2 = [node.x for node in self.map_nodes[main_module.AFFECTED_NUMBER:]]
        y2 = [node.y for node in self.map_nodes[main_module.AFFECTED_NUMBER:]]
        ax.scatter(x2, y2, s=500, c=self.colors['supply'], marker='^',
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
                      s=300, c=color, marker='o',
                      alpha=0.9, edgecolors='white', linewidth=2, zorder=5)

            # 绘制机器人标签
            label = f"{robot['name']}\n{robot['state']}"
            ax.annotate(label, (robot['position'][0], robot['position'][1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold', color='black', zorder=6)

            # 绘制运载信息
            if robot['load'].A_material.quantity > 0:
                load_text = f"载重: {robot['load'].A_material.quantity + robot['load'].B_material.quantity + robot['load'].C_material.quantity:.1f}"
                ax.annotate(load_text, (robot['position'][0], robot['position'][1]),
                           xytext=(5, -25), textcoords='offset points',
                           fontsize=7, color='red', zorder=6)

            # 绘制路径轨迹
            if robot['delivery_history']:
                path_x = []
                path_y = []
                for delivery in robot['delivery_history']:
                    from_pos = [self.map_nodes[delivery['from']].x, self.map_nodes[delivery['from']].y]
                    to_pos = [self.map_nodes[delivery['to']].x, self.map_nodes[delivery['to']].y]
                    path_x.extend([from_pos[0], to_pos[0]])
                    path_y.extend([from_pos[1], to_pos[1]])

                if len(path_x) > 1:
                    ax.plot(path_x, path_y, color=robot['color'], alpha=0.5,
                           linewidth=2, linestyle='--', zorder=2)

        # 设置图形属性
        ax.set_title(f'遗传算法优化调度 - 时间步: {frame_num} | 最优适应度: {self.best_fitness:.0f}',
                   fontsize=20, fontweight='bold')
        ax.set_xlabel('x轴（单位：百米）', fontsize=16)
        ax.set_ylabel('y轴（单位：百米）', fontsize=16)
        ax.set_xlim(-10, 170)
        ax.set_ylim(-10, 170)
        ax.grid(True, alpha=0.3)

        # 添加节点标签（显示剩余需求）
        for i, node in enumerate(self.map_nodes):
            if i < main_module.AFFECTED_NUMBER:
                total_remaining = node.need.A_material.quantity + node.need.B_material.quantity + node.need.C_material.quantity
                label = f"受灾点{i}\n剩余需求: {total_remaining:.1f}"
            else:
                label = f"补给点{i-main_module.AFFECTED_NUMBER}"

            ax.annotate(label, (node.x, node.y),
                       xytext=(0, -35), textcoords='offset points',
                       fontsize=10, ha='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                       zorder=4)

        # 添加统计信息面板
        info_text = f"基于遗传算法最优解调度\n"
        info_text += f"{'='*25}\n"
        info_text += f"机器人数量: {len(self.robots)}\n"
        info_text += f"受灾点: {main_module.AFFECTED_NUMBER}\n"
        info_text += f"补给点: {main_module.SUPPLE_NUMBER}\n"
        info_text += f"已完成运输: {len(self.completed_deliveries)}\n\n"

        info_text += "各机器人状态:\n"
        for robot in self.robots:
            progress = (robot['current_task'] / len(robot['tasks']) * 100) if len(robot['tasks']) > 0 else 0
            info_text += f"• {robot['name']}: {robot['state']} ({progress:.0f}%)\n"

        # 显示染色体信息（简化显示）
        chromosome_preview = f"长度:{len(self.best_chromosome)}, 前10个:{self.best_chromosome[:10]}..."
        info_text += f"\n最优染色体: {chromosome_preview}"

        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        return fig, ax

    def generate_ga_optimized_animation(self):
        """生成基于遗传算法最优解的动画"""
        print(f"开始生成 {self.total_frames} 帧遗传算法优化动画...")

        # 创建基于遗传算法结果的机器人
        self.robots = self.decode_chromosome_to_robots()

        for frame in range(self.total_frames):
            # 更新所有机器人状态
            for robot in self.robots:
                self.update_robot_state(robot, frame)

            # 绘制帧
            fig, ax = self.draw_frame(frame)

            # 保存帧
            frame_path = os.path.join(self.output_dir, f'ga_frame_{frame:04d}.png')
            plt.savefig(frame_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()

            if frame % 30 == 0:
                completed_tasks = sum(robot['current_task'] for robot in self.robots)
                total_tasks = sum(len(robot['tasks']) for robot in self.robots)
                progress = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
                print(f"已生成 {frame+1}/{self.total_frames} 帧, 任务进度: {progress:.1f}%")

        print(f"遗传算法优化动画生成完成！保存在: {self.output_dir}")

        # 生成动画报告
        self.create_ga_animation_report()

    def format_chromosome_display(self, chromosome, max_items_per_line=15):
        """格式化染色体数组显示，每行最多显示max_items_per_line个元素"""
        if len(chromosome) <= max_items_per_line:
            return str(chromosome)

        lines = []
        for i in range(0, len(chromosome), max_items_per_line):
            chunk = chromosome[i:i + max_items_per_line]
            # 添加位置索引
            start_pos = i
            end_pos = min(i + max_items_per_line, len(chromosome))
            lines.append(f"  位置{start_pos:2d}-{end_pos-1:2d}: {chunk}")

        return "[\n" + ",\n".join(lines) + "\n]"

    def create_ga_animation_report(self):
        """生成遗传算法动画报告"""
        report_path = os.path.join(self.output_dir, 'ga_animation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("遗传算法优化调度动画报告\n")
            f.write("=" * 40 + "\n\n")

            f.write(f"遗传算法优化结果:\n")
            f.write(f"- 最优适应度（总焦虑度）: {self.best_fitness:.2f}\n")
            f.write(f"- 染色体长度: {len(self.best_chromosome)}\n")
            f.write(f"- 染色体编码（每行15个元素，带位置索引）:\n")
            f.write(f"{self.format_chromosome_display(self.best_chromosome)}\n")
            f.write(f"- 动画帧数: {self.total_frames}\n")
            f.write(f"- 机器人数量: {len(self.robots)}\n")
            f.write(f"- 受灾点数量: {main_module.AFFECTED_NUMBER}\n")
            f.write(f"- 补给点数量: {main_module.SUPPLE_NUMBER}\n")
            f.write(f"- 完成运输次数: {len(self.completed_deliveries)}\n\n")

            f.write("机器人详情:\n")
            for robot in self.robots:
                f.write(f"\n{robot['name']}:\n")
                f.write(f"  类型: {robot['type']}\n")
                f.write(f"  载重: {robot['capacity']}\n")
                f.write(f"  速度: {robot['speed']}\n")
                f.write(f"  分配任务: {robot['tasks']}\n")
                f.write(f"  完成任务: {robot['current_task']}/{len(robot['tasks'])}\n")
                f.write(f"  运输历史: {len(robot['delivery_history'])} 次\n")

            f.write(f"\n运输记录:\n")
            for i, delivery in enumerate(self.completed_deliveries):
                f.write(f"  {i+1}. 机器人从补给点{delivery['from']-main_module.AFFECTED_NUMBER} 到受灾点{delivery['to']}")
                f.write(f" (应急:{delivery['load'].A_material.quantity:.1f}, ")
                f.write(f"常规:{delivery['load'].B_material.quantity:.1f}, ")
                f.write(f"设备:{delivery['load'].C_material.quantity:.1f})\n")

            f.write(f"\n视频生成命令:\n")
            f.write(f"cd {self.output_dir}\n")
            f.write(f"ffmpeg -r 10 -i ga_frame_%04d.png \\\n")
            f.write(f'       -vf "scale=iw-1:ih,scale=trunc(iw/2)*2:trunc(ih/2)*2" \\\n')
            f.write(f'       -c:v libx264 -pix_fmt yuv420p ga_optimized_dispatch.mp4\n')

        print(f"遗传算法动画报告已保存: {report_path}")

def main():
    """主函数"""
    print("=== 基于遗传算法最优解的物资调度动画生成器 ===")

    # 创建动画生成器
    animator = GAOptimizedAnimation()

    try:
        # 生成遗传算法优化动画
        animator.generate_ga_optimized_animation()

    except Exception as e:
        print(f"生成动画时出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n遗传算法优化动画生成完成！")
    print(f"输出目录: ../data/ga_optimized_animation/")
    print("包含文件:")
    print("- ga_frame_0000.png 到 ga_frame_0299.png (300帧)")
    print("- ga_animation_report.txt (遗传算法优化报告)")

if __name__ == "__main__":
    main()