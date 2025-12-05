# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

DispatchingForPeople 是一个使用遗传算法优化应急救援物资调度的系统。系统通过模拟机器人从补给点向受灾点运输三类物资（生活资料类、救生器械类、救治物品类），寻找最优调度策略以最小化人群焦虑度。

## 运行命令

### 基本运行
```bash
# 运行主程序（遗传算法求解）
cd src
python3 main.py

# 运行贪心策略对比算法
python3 greedy.py

# 查看遗传算法可视化
python3 animation.py

# 查看贪心策略可视化
python3 animation_greedy.py
```

### 配置说明
程序运行模式通过 `src/config.json` 配置：
- `RANDOM_MODE: 1`: 随机生成地图数据并运行
- `RANDOM_MODE: 2`: 复用已有数据运行（data/data.json）
- `USE_OLD_RESULT`: 是否使用上次的最优染色体结果

## 核心架构

### 主要模块结构
- **map.py**: 地图节点生成，包含受灾点和补给点的坐标、距离计算、障碍物（石头）处理
- **main.py**: 遗传算法主框架，染色体编码、适应度计算、遗传算子实现
- **material.py**: 三类物资定义（A/B/C类）及物资包裹管理，支持需求量和焦虑度计算
- **selection.py**: 遗传算法选择策略（轮盘赌、锦标赛选择）
- **animation.py**: 使用matplotlib可视化遗传算法求解过程和结果
- **greedy.py**: 贪心策略对比算法
- **file_utils.py**: 文件I/O工具类，支持配置读取、数据保存和恢复

### 遗传算法设计
- **染色体编码**: 仅使用受灾点序列进行编码，补给点通过动态决策确定
- **适应度函数**: 基于总焦虑度，考虑时间衰减（焦虑幂指速率1.1）
- **遗传算子**: 交叉、变异、多种选择策略
- **约束处理**: 机器人载重限制、路径可达性、物资需求动态更新

### 物资调度逻辑
1. **三种机器人类型**: A类(80单位)、B类(15单位)、C类(5单位)载重能力
2. **动态补给策略**: 机器人根据受灾点需求动态选择最适合的补给点
3. **需求预估**: 支持预估抵达时的剩余需求量，避免重复调度
4. **多物资协调**: 支持同时装载多种物资，按优先级分配

### 数据文件
- `data/data.json`: 地图节点数据（坐标、类型、物资需求/供给）
- `data/result.csv`: 实验记录和结果数据
- `src/config.json`: 算法和程序参数配置

### 版本管理
项目采用版本目录管理：
- `src`: 当前稳定版本
- `versionX.X`: 历史版本，包含重要功能迭代记录

## 开发注意事项

### 算法参数调优
关键参数在 `config.json` 中：
- `CHROMOSOME_NUMBER`: 种群大小（建议50）
- `CROSSOVER_RATE`: 交叉概率（建议0.8-0.9）
- `MUTATION_RATE`: 变异概率（建议0.3-0.5）
- `MAX_GENERATION`: 最大迭代次数（建议200-300）

### 性能优化点
1. 邻接矩阵处理：不可达路径用-1表示，避免无穷大导致机器人无法返回
2. 预估需求量：计算机器人抵达时的实际需求，减少无效调度
3. 染色体动态编码：根据实际需求增加任务序列长度

### 问题调试
- 检查 `src/map.py:stone_list` 处理路径阻塞逻辑
- 验证物资需求更新机制（`material.py:next_quantity`）
- 确认机器人载重约束和补给策略合理性

#### Matplotlib 中文字体设置
项目中已配置中文字体支持，避免中文显示为方框（□）：
```python
# 字体设置优先级
plt.rcParams['font.sans-serif'] = [
    'WenQuanYi Micro Hei',  # 优先使用
    'Noto Sans CJK JP',     # 备选
    'DejaVu Sans'           # 最后备选
]
plt.rcParams['axes.unicode_minus'] = False
```

如果遇到中文显示问题，可运行 `test_chinese_font.py` 测试字体可用性。

### 动画生成功能

项目支持生成应急救援物资调度的动画效果：

#### 动画帧生成器（推荐）
```bash
# 生成动画帧序列
python animation_frames.py
```

**功能特点**：
- 生成 60 帧高质量 PNG 图像序列
- 显示机器人从补给点到受灾点的移动过程
- 包含路径轨迹、任务进度、物资需求信息
- 支持中文标签和图例
- 自动生成动画说明文件

**输出文件**：
- `../data/animation_frames/frame_0000.png` 到 `frame_0059.png` - 动画帧序列
- `../data/animation_frames/animation_info.txt` - 动画说明和使用方法

#### GIF 动画生成器（实验性）
```bash
# 生成 GIF 动画（需要额外依赖）
python animation_gif_simple.py
```

**注意**：GIF 生成可能需要安装额外的依赖包（如 Pillow），建议使用帧序列生成器。

#### 完整物资调度动画（新增）
```bash
# 生成完整的物资调度过程动画
python animation_complete.py
```

**完整调度过程特点**：
- 真实的物资调度模拟：补给点装载 → 运输 → 受灾点卸载 → 返回补给
- 三种机器人类型：A型(80单位)、B型(15单位)、C型(5单位)
- 智能路径规划：自动选择最近补给点
- 动态需求管理：实时更新受灾点剩余需求
- 完整状态机：idle → loading → transporting → unloading → idle
- 实时统计信息：运输进度、任务完成情况

**输出文件**：
- `../data/complete_animation/complete_frame_*.png` - 120帧高清动画序列
- `../data/complete_animation/complete_animation_report.txt` - 详细调度报告
- `../data/complete_dispatch.mp4` - 完整调度过程视频（12秒，高清）