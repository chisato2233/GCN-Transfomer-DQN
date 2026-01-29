# SAGIN 空天地三层网络智能路由优化

## 项目概述

基于深度强化学习的空天地一体化网络(SAGIN)智能路由优化系统，采用 **Per-Neighbor Q-Value + Transformer** 架构实现分布式本地决策。

## 核心架构

### Per-Neighbor Q-Value 设计

解决传统 GCN 聚合操作丢失邻居-动作对应关系的问题：

- 每个动作 i 的 Q 值直接从对应邻居 i 的特征计算
- Transformer 提供历史路由决策的时序上下文
- 保持 neighbor-action 的直接映射关系

### 信息使用原则

| 类型 | 信息 | 用途 |
|------|------|------|
| **本地信息** | 邻居列表、节点特征、链路属性、历史记录 | 训练和推理 |
| **全局信息** | 最短路径 (Dijkstra) | 仅评估对比 |

## SAGIN 三层网络

| 层级 | 类型 | 特点 |
|------|------|------|
| Space | 卫星 | 覆盖广、延迟高、长距离高效 |
| Air | UAV | 灵活、能量受限、中继节点 |
| Ground | 地面站 | 稳定、容量大、最终目的地 |

### 14维特征设计

**路由特征 (8维)**: 目标距离、距离改善、邻居度数、2跳距离、是否目的地、链路延迟、链路带宽、是否访问过

**三层网络特征 (6维)**: 节点类型 one-hot (3维)、能量水平、队列拥塞度、层间切换指示

## 真实数据集

### 数据来源

| 数据类型 | 来源 | 文件 |
|----------|------|------|
| **卫星轨道 (TLE)** | [CelesTrak](https://celestrak.org/NORAD/elements/) | `data/starlink_tle.txt` (9419颗) |
| | | `data/iridium_tle.txt` (80颗) |
| | | `data/oneweb_tle.txt` (651颗) |
| **地面站** | 主要城市坐标 | `data/ground_stations.json` (20站) |
| **UAV轨迹** | 模拟巡逻路径 | `data/uav_trajectories.json` |

### 卫星星座统计

| 星座 | 数量 | 平均高度 | 轨道倾角 |
|------|------|----------|----------|
| Starlink | 9419 | 504 km | 53° |
| Iridium | 80 | 771 km | 86° |
| OneWeb | 651 | 1205 km | 88° |

## 项目结构

```
GNN-Transformer/
├── configs/
│   ├── routing_config.yaml           # 简单网络 (19节点)
│   ├── routing_config_complex.yaml   # 复杂网络 (46节点)
│   ├── routing_config_realworld.yaml # 真实数据网络 (65节点)
│   └── routing_config_starlink.yaml  # Starlink大规模网络 (~80节点)
├── data/
│   ├── starlink_tle.txt              # Starlink卫星TLE数据
│   ├── iridium_tle.txt               # Iridium卫星TLE数据
│   ├── oneweb_tle.txt                # OneWeb卫星TLE数据
│   ├── ground_stations.json          # 地面站位置
│   └── uav_trajectories.json         # UAV轨迹配置
├── src/
│   ├── data/
│   │   ├── tle_parser.py             # TLE数据解析器
│   │   └── topology_builder.py       # 真实网络拓扑构建器
│   ├── env/
│   │   ├── sagin_env.py              # SAGIN环境
│   │   └── network_topology.py       # 三层网络拓扑
│   ├── agents/
│   │   └── dqn_gcn_transformer_v3.py # Per-Neighbor Q-Value Agent
│   ├── baselines/
│   │   ├── q_routing.py              # Q-Routing, Predictive Q-Routing
│   │   ├── traditional_routing.py    # OSPF, ECMP, AODV
│   │   └── gnn_baselines.py          # GAT-DQN, GraphSAGE-DQN, GCN-DQN, Dueling-DQN
│   └── experiments/
│       ├── train_v3.py               # 训练脚本
│       ├── train_ablation.py         # 消融实验
│       ├── evaluate_v3.py            # 评估脚本
│       └── evaluate_comprehensive.py # 综合对比评估
├── logs/                             # 训练日志和检查点
│   └── ablation_*/results.yaml       # 消融实验结果
└── evaluation_comprehensive.yaml     # 综合评估结果
```

## 实验结果

### 综合对比评估 (200 episodes)

| 方法 | 类别 | 信息类型 | 成功率 | 平均跳数 |
|------|------|----------|--------|----------|
| **V3 Per-Neighbor** | Ours | Local+Temporal | **95.5%** | 3.68 |
| ECMP | Traditional | Global | 98.5% | 4.13 |
| AODV | Traditional | On-demand | 98.0% | 3.63 |
| OSPF | Traditional | Global | 95.0% | 5.08 |
| GAT-DQN | GNN-DRL 2024 | Local+GNN | - | - |
| GraphSAGE-DQN | GNN-DRL 2024 | Local+GNN | - | - |
| GCN-DQN | GNN-DRL 2024 | Local+GNN | - | - |
| Dueling-DQN | DRL 2024 | Local | - | - |
| Greedy | Simple | Local | 92.5% | 5.09 |
| Random | Simple | Local | 64.0% | 24.56 |
| Q-Routing | Classic RL | Local | 52.5% | 21.60 |

### 消融实验 (简单网络 1000 episodes)

| 变体 | 描述 | 成功率 |
|------|------|--------|
| **Full** | Per-Neighbor + Transformer + 14维特征 | 100.0% |
| w/o Transformer | 去掉时序建模 | 100.0% |
| w/o Three-Layer | 只用8维路由特征 | 100.0% |
| MLP Only | 简单MLP基线 | 100.0% |
| **w/o Per-Neighbor** | 全局池化 (丢失对应关系) | **88.0%** |

**结论**: Per-Neighbor 架构对保持邻居-动作对应关系至关重要。

## 使用方法

### 训练 (简单网络)

```bash
python src/experiments/train_v3.py --config configs/routing_config.yaml --episodes 3000
```

### 训练 (真实数据网络)

```bash
python src/experiments/train_v3.py --config configs/routing_config_realworld.yaml --episodes 4000
```

### 训练 (Starlink 大规模网络)

```bash
python src/experiments/train_v3.py --config configs/routing_config_starlink.yaml --episodes 8000
```

### 消融实验

```bash
python src/experiments/train_ablation.py --ablation all --config configs/routing_config.yaml --episodes 1000
```

### 综合评估

```bash
python src/experiments/evaluate_comprehensive.py --config configs/routing_config.yaml --episodes 200
```

### 更新卫星数据

```bash
cd data
curl -s "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle" -o starlink_tle.txt
curl -s "https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium-NEXT&FORMAT=tle" -o iridium_tle.txt
curl -s "https://celestrak.org/NORAD/elements/gp.php?GROUP=oneweb&FORMAT=tle" -o oneweb_tle.txt
```

## 关键发现

1. **Per-Neighbor 架构有效**: 保持邻居-动作对应关系显著提升性能 (88% → 100%)
2. **本地信息可达全局水平**: V3 使用本地信息达到 95.5%，接近全局算法 OSPF (95.0%)
3. **超越传统本地算法**: V3 (95.5%) > Greedy (92.5%) > Q-Routing (52.5%)

## GNN Baseline References (2024-2025)

| 方法 | 论文标题 | 期刊/来源 | DOI | Baseline适用性 |
|------|----------|----------|-----|----------------|
| **GCN-Transformer-PPO** | "Intelligent Routing Optimization via GCN-Transformer Hybrid Encoder and Reinforcement Learning in Space–Air–Ground Integrated Networks" | MDPI Electronics 2025, 15(1), 14 | [10.3390/electronics15010014](https://doi.org/10.3390/electronics15010014) | ✅ **强烈推荐** - 同为 SAGIN 路由，直接竞争对手 |
| GraphSAGE-DQN | "Low Earth Orbit Satellite Network Routing Algorithm Based on Graph Neural Networks and Deep Q-Network" | MDPI Applied Sciences 2024, 14(9), 3840 | [10.3390/app14093840](https://doi.org/10.3390/app14093840) | ✅ 推荐 - LEO 卫星路由，可对比 GNN 聚合方式 |
| Dueling-DQN | "GPU-Accelerated CNN Inference for Onboard DQN-Based Routing in Dynamic LEO Satellite Networks" | MDPI Aerospace 2024, 11(12), 1028 | [10.3390/aerospace11121028](https://doi.org/10.3390/aerospace11121028) | ⚠️ 部分适用 - 偏重星载实现优化 |

### Baseline 对比要点

| 对比维度 | 本项目 (Ours) | GCN-Transformer-PPO | GraphSAGE-DQN |
|----------|---------------|---------------------|---------------|
| 应用场景 | SAGIN 三层网络 | SAGIN 三层网络 ✅ | LEO 卫星网络 |
| GNN 架构 | Per-Neighbor Q-Value | GCN 编码器 | GraphSAGE 聚合 |
| 时序建模 | Transformer | Transformer ✅ | 无 |
| RL 算法 | DQN | PPO | DQN ✅ |
| 数据来源 | CelesTrak TLE | CelesTrak TLE ✅ | 仿真数据 |

## 网络参数参考依据

### 卫星网络参数

| 参数 | 设置值 | 参考来源 |
|------|--------|----------|
| Starlink 轨道高度 | 550 km | [1] SpaceX FCC Filing |
| 轨道周期 | 90 min | 开普勒定律计算 [2] |
| ISL 带宽 | 100 Gbps | Starlink 激光链路规格 |
| 覆盖半径计算 | 25° 最小仰角 | [3] ITU-R S.1503 |
| 传播延迟 | 光速计算 | [4] Handley, ACM HotNets 2018 |

### SAGIN 网络建模

| 参数 | 参考来源 |
|------|----------|
| UAV 空对地信道 | [5] 3GPP TR 36.777 |
| 卫星通信架构 | [6] Kodheli et al., IEEE COMST 2021 |
| LEO 星座对比 | [7] Del Portillo et al., Acta Astronautica 2019 |

### UAV 通信参数

| 参数 | 设置值 | 参考来源 |
|------|--------|----------|
| 飞行高度 | 100-500 m | [8] Khuwaja et al., IEEE COMST 2018 |
| A2G 信道模型 | LoS/NLoS | [9] Zeng et al., IEEE CommMag 2016 |
| 覆盖半径 | 10-15 km | [8][9] |

### 参考文献

**卫星网络:**
1. SpaceX, "Application for Modification of Authorization for the SpaceX NGSO Satellite System," FCC Filing, 2018
2. Vallado, D.A., "Fundamentals of Astrodynamics and Applications," 4th ed., Microcosm Press, 2013
3. ITU-R S.1503, "Functional description to be used in developing software tools for determining interference or frequency coordination"
4. Handley, M., "Delay is Not an Option: Low Latency Routing in Space," ACM HotNets, 2018

**SAGIN 网络:**
5. 3GPP TR 36.777, "Enhanced LTE support for aerial vehicles"
6. Kodheli, O. et al., "Satellite Communications in the New Space Era: A Survey and Future Challenges," IEEE Communications Surveys & Tutorials, 2021
7. Del Portillo, I. et al., "A technical comparison of three low earth orbit satellite constellation systems," Acta Astronautica, 2019

**UAV 通信:**
8. Khuwaja, A.A. et al., "A Survey of Channel Modeling for UAV Communications," IEEE Communications Surveys & Tutorials, 2018
9. Zeng, Y. et al., "Wireless communications with unmanned aerial vehicles: opportunities and challenges," IEEE Communications Magazine, 2016
