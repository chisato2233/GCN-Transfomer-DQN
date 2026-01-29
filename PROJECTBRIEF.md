# SAGIN 智能路由项目 - GCN-Transformer-DQN

## 项目概述

本项目实现了基于 **GCN-Transformer-DQN** 的空天地一体化网络 (SAGIN, Space-Air-Ground Integrated Network) 智能路由系统。通过结合图卷积网络 (GCN)、Transformer 架构和深度 Q 网络 (DQN)，在异构网络中实现自适应、高效的数据包路由决策。

### 核心创新

- **V3 Per-Neighbor Q-Value 架构**：每个动作的 Q 值从对应邻居的特征直接计算，避免 GCN 聚合导致的邻居特征丢失
- **Transformer 时序编码**：利用路由历史的时序上下文增强 Q 值计算
- **三层联合优化**：同时考虑能量、拥塞和跨层切换代价

---

## 网络架构

### 三层网络模型

```
空间层 (Space):   LEO 卫星 (高度: 550 km)
  ├── 覆盖半径: 500 km
  ├── 带宽: 100 MHz (星间), 50 MHz (星-无人机)
  └── 能量: 稳定供电

空中层 (Air):     无人机 UAV (高度: 100-500 m)
  ├── 数量: 6 节点 (可配置)
  ├── 覆盖半径: 5 km
  ├── 带宽: 10-20 MHz
  ├── 能量: 0.5-1.0 (关键约束)
  └── 移动性: 随机 2D 运动 (0-10 m/s)

地面层 (Ground):  地面基站 (高度: 0 m)
  ├── 数量: 10 节点 (可配置)
  ├── 覆盖半径: 50 km
  └── 能量: 稳定供电
```

### 连接规则

| 连接类型 | 条件 |
|---------|------|
| 卫星-卫星 | 始终连接 (星间链路) |
| 卫星-无人机 | 500 km 覆盖范围内 |
| 无人机-无人机 | 10 km 内 (2x 覆盖范围) |
| 无人机-地面 | 无人机覆盖范围内 (5 km) |
| 地面-地面 | 2000 m 内 (本地网状网络) |
| 卫星-地面 | **不直连** (必须通过空中层中继) |

---

## 算法设计

### V3 Per-Neighbor Q-Value Agent

```
PerNeighborQNetwork
├── TemporalContextEncoder (Transformer)
│   ├── 输入投影: 6-dim -> 32-dim
│   ├── 位置编码: 可学习
│   ├── TransformerEncoderLayer: 2 heads, 1 layer
│   └── 输出: 全局时序上下文 [batch, 32]
│
└── Q-Value Head
    ├── Per-Neighbor 特征编码器: 14-dim -> 64-dim
    ├── 时序上下文扩展到所有邻居
    ├── Q-Head: [64+32] -> 1 Q-value per neighbor
    └── Dueling 架构: V + (A - mean(A))
```

### 特征维度

**邻居拓扑特征 (14-dim)**:
- 路由特征 (8-dim): 到目标距离、改进量、度、两跳前瞻、是否目标、时延、带宽、是否已访问
- 三层特征 (6-dim): 节点类型独热 (3)、能量、拥塞、跨层切换

**简化历史 (10 步, 6-dim/步)**:
- 到目标距离、改进量、是否环路、是否卫星、是否无人机、是否地面站

### 奖励函数

```
总奖励 = Σ:
  1. 进度奖励:  (距离改进 / 初始距离) × 5.0
  2. 接近奖励:  (1 - 当前距离/初始距离) × 0.3
  3. 目标接近:  +1.5 (距离 < 15%)
  4. 时延惩罚:  -α × 0.1 × delay/100
  5. 环路惩罚:  -2.0 (重访节点)
  6. 跳数惩罚:  -0.1 × (hops - 3)
  7. 远离惩罚:  -0.5 (距离增加时)
  8. UAV 能量:  -1.0 (能量 < 30%), -0.3 (能量 < 50%)
  9. 拥塞惩罚:  -0.5 × (queue_length/100)
  10. 跨层代价: -0.1 (跨层), +0.3 (卫星有利)
  11. 带宽奖励: +0.2 (>50Mbps), +0.1 (>20Mbps)

成功到达: +30.0
超时惩罚: -5.0
```

### DQN 超参数

| 参数 | 值 |
|-----|-----|
| 学习率 | 5e-5 |
| 软更新 tau | 0.005 |
| 折扣因子 gamma | 0.99 |
| Epsilon 起始/终止 | 1.0 / 0.05 |
| Epsilon 衰减 | 0.995/episode |
| Replay Buffer | 100K |
| Batch Size | 128 |
| 梯度裁剪 | max_grad_norm=1.0 |
| 训练轮次 | 3000 episodes |
| 评估频率 | 每 50 episodes |
| 早停耐心 | 800 episodes (热身 400) |

---

## 项目结构

```
GCN-Transfomer-DQN/
├── configs/                              # 配置文件
│   ├── routing_config.yaml              # 主配置 (3卫星+6无人机+10地面)
│   ├── routing_config_starlink.yaml     # Starlink 星座配置
│   ├── routing_config_complex.yaml      # 复杂拓扑配置
│   └── routing_config_realworld.yaml    # 真实数据配置
├── data/                                # 真实卫星与地面数据
│   ├── starlink_tle.txt                 # Starlink TLE 轨道根数
│   ├── iridium_tle.txt                  # Iridium 星座 TLE
│   ├── oneweb_tle.txt                   # OneWeb 星座 TLE
│   ├── ground_stations.json             # 20 个全球地面站
│   └── uav_trajectories.json            # 无人机巡逻轨迹
├── src/
│   ├── agents/
│   │   └── dqn_gcn_transformer_v3.py    # V3 核心智能体 (644 行)
│   ├── env/
│   │   ├── sagin_env.py                 # Gym 仿真环境 (841 行)
│   │   └── network_topology.py          # SAGIN 拓扑模型 (514 行)
│   ├── data/
│   │   ├── tle_parser.py               # TLE 轨道传播 (259 行)
│   │   └── topology_builder.py         # 真实拓扑构建 (485 行)
│   ├── baselines/
│   │   ├── traditional_routing.py      # OSPF, AODV, ECMP 基线
│   │   ├── gnn_baselines.py            # GAT-DRL, GraphSAGE-DQN 基线
│   │   └── q_routing.py                # Q-Routing, PQ-Routing 基线
│   └── experiments/
│       ├── train_v3.py                 # V3 训练脚本
│       ├── evaluate_v3.py              # 对比评估
│       ├── train_ablation.py           # 消融实验
│       └── evaluate_comprehensive.py   # 综合评估
├── docker-compose.yml                  # Docker 服务 (GPU 训练/Jupyter/TensorBoard)
├── Dockerfile                          # 多阶段 Docker 构建
└── requirements.txt                    # 依赖 (PyTorch 2.0+, PyG 2.4+, Gymnasium 等)
```

---

## 实验结果

### 基线对比 (来自 evaluation_comprehensive.yaml)

| 方法 | 平均跳数 | 成功率 | 平均奖励 | 说明 |
|------|---------|--------|---------|------|
| **V3 Per-Neighbor (ours)** | **3.68** | **95.5%** | **45.39** | DRL 方法 |
| ECMP | 4.13 | 98.5% | 54.08 | 等价多路径负载均衡 |
| AODV | 3.625 | 98.0% | 52.26 | 按需路由 |
| OSPF | 5.075 | 95.0% | 42.33 | 最短路径 (全局信息) |
| Greedy | 5.085 | 92.5% | 35.06 | 贪心最近邻 |
| Q-Routing | 21.595 | 52.5% | -122.72 | 经典 Q-Routing |
| PQ-Routing | 34.085 | 33.5% | -215.66 | 预测 Q-Routing |
| Random | 24.555 | 64.0% | -91.71 | 随机选择 |

### 关键发现

- V3 实现了**接近最优的跳数** (3.68 vs OSPF 5.075)，同时仅需**局部信息**
- 成功率 95.5% 与依赖全局拓扑信息的 OSPF (95.0%) 持平
- 显著优于经典 Q-Routing 和 Random 基线
- ECMP 和 AODV 在当前场景下表现较好，但它们依赖全局/泛洪机制

---

## 真实数据支持

### 卫星数据
- **Starlink**: 8000+ 颗 (倾角 53.1, 高度 550km)
- **Iridium**: 66 颗 (倾角 86.4, 高度 780km)
- **OneWeb**: 600+ 颗 (倾角 87.9, 高度 1200km)
- 使用 SGP4 简化轨道力学计算卫星位置

### 地面站
20 个全球地面站: 北京、上海、深圳、成都、西安、武汉、广州、杭州、南京、天津、重庆、哈尔滨、乌鲁木齐、拉萨、昆明、纽约、洛杉矶、伦敦、东京、悉尼

### 无人机轨迹
4 种类型: 巡逻 (城市监控)、中继 (农村覆盖)、应急 (灾害响应)、编队配置

---

## 技术栈

| 类别 | 依赖 | 版本 |
|-----|------|------|
| 深度学习 | PyTorch | 2.0.0+ |
| 图神经网络 | PyTorch Geometric | 2.4.0+ |
| 强化学习环境 | Gymnasium | 0.29.0+ |
| 图算法 | NetworkX | 3.0+ |
| 配置管理 | PyYAML | 6.0+ |
| 数值计算 | NumPy, SciPy | 1.24.0+, 1.10.0+ |
| 可视化 (可选) | TensorBoard, Seaborn | 2.13.0+, 0.12.0+ |

---

## 使用方法

### 训练
```bash
python src/experiments/train_v3.py --config configs/routing_config.yaml
```

### 评估
```bash
python src/experiments/evaluate_v3.py --checkpoint logs/v3_xxx/checkpoints/best_model.pt
```

### Docker 部署
```bash
docker-compose up sagin                                    # GPU 训练
docker-compose --profile jupyter up jupyter                 # 交互开发
docker-compose --profile monitoring up tensorboard          # 可视化
```

---

## 当前状态

- V3 Per-Neighbor Q-Value Agent 已实现并训练
- 基线对比实验已完成
- 真实卫星数据集成已完成 (Starlink/Iridium/OneWeb TLE)
- 消融实验框架已搭建
- 综合评估系统已实现

## 待改进方向

- V3 的成功率 (95.5%) 和平均奖励 (45.39) 低于 ECMP (98.5%, 54.08) 和 AODV (98.0%, 52.26)
- 可考虑引入 GCN 全局拓扑感知与 Per-Neighbor 的混合架构
- 进一步优化奖励函数设计
- 在更大规模网络和真实卫星轨道数据上验证泛化性能
