# 需求规格说明书 (Requirements Specification)

## GCN-Transformer-DRL SAGIN 智能路由算法

---

## 1. 项目概述

| 项目 | 内容 |
|------|------|
| **项目名称** | 基于GCN+Transformer+深度强化学习的空天地一体化网络智能路由算法 |
| **目标期刊** | IEEE级别期刊 |
| **交付时间** | 2026年10月前见刊 |
| **硬件配置** | RTX 4080 Super (16GB VRAM) |
| **交付范围** | 技术部分全包（实验、算法、代码、建模、结果） |

---

## 2. 问题定义

### 2.1 核心问题

**在空天地一体化网络(SAGIN)中，如何在缺乏全局网络状态信息的条件下，利用深度强化学习方法实现高效、稳定的智能路由决策，从而提升网络整体通信性能。**

### 2.2 研究重点

1. 如何对空天地一体化通信网络进行统一建模
2. 如何将路由决策问题抽象为强化学习问题
3. 如何设计合理的状态空间、动作空间与奖励函数
4. 如何构建并训练适用于动态拓扑环境的深度强化学习路由模型

### 2.3 问题特点

- **动作空间**：离散（从邻居节点集合中选择下一跳）
- **决策方式**：分布式（每个节点基于局部信息决策）
- **网络特性**：拓扑动态变化、节点异构、链路状态时变

---

## 3. 网络模型

### 3.1 三层网络架构

| 层级 | 节点类型 | 特点 | 功能 |
|------|----------|------|------|
| 空间层 (Space) | 低轨道(LEO)卫星 | 覆盖范围广、移动速度快 | 骨干网络、广域覆盖 |
| 空中层 (Air) | 无人机(UAV)/高空平台 | 灵活部署、动态调整 | 中继转发、局部覆盖增强 |
| 地面层 (Ground) | 地面基站、用户节点 | 固定/低速移动 | 数据接入与汇聚 |

### 3.2 动态图网络模型

网络表示为随时间变化的动态图：

```
G(t) = (V, E(t))
```

其中：
- `V`: 网络节点集合（卫星、无人机、地面基站、用户）
- `E(t)`: 时刻 t 存在的通信链路集合

### 3.3 链路属性

每条链路 e ∈ E(t) 具有以下动态属性：

| 属性 | 符号 | 说明 |
|------|------|------|
| 带宽 | B_e(t) | 可用传输带宽 |
| 时延 | D_e(t) | 传输时延 |
| 丢包率 | L_e(t) | 数据包丢失概率 |

### 3.4 网络参数（默认配置）

| 参数 | 值 | 说明 |
|------|-----|------|
| 卫星数量 | 3 | LEO卫星 |
| UAV数量 | 6 | 中继无人机 |
| 地面节点数量 | 10 | 基站+用户 |
| 总节点数 | 19 | 可配置 |
| 卫星高度 | 550-1200 km | LEO轨道 |
| UAV高度 | 100-500 m | 低空飞行 |
| 区域大小 | 10km × 10km | 地面区域 |
| 时隙长度 | 1 秒 | 决策周期 |

---

## 4. MDP建模（核心）

### 4.1 状态空间 (State)

每个节点观测的**局部网络状态**：

```python
s_t = {
    # 当前节点状态
    queue_length,         # 当前节点队列长度
    node_energy,          # 节点剩余能量（可选）

    # 邻居链路状态
    neighbor_bandwidth[], # 到各邻居节点的链路带宽
    neighbor_delay[],     # 到各邻居节点的链路时延
    neighbor_loss_rate[], # 到各邻居节点的丢包率

    # 邻居节点状态
    neighbor_queue[],     # 邻居节点的队列长度
    neighbor_congestion[],# 邻居节点的拥塞程度

    # 目标信息
    destination_id,       # 目的节点标识
    packet_priority,      # 数据包优先级（可选）
}
```

**状态向量维度**：取决于最大邻居数 K_max

```python
state_dim = 2 + 5 * K_max + 2  # 基础状态 + 邻居信息 + 目标信息
```

### 4.2 动作空间 (Action) - 离散

**在每一个决策时刻，智能体从当前节点的可达邻居节点集合中选择一个作为下一跳节点。**

```python
A = {0, 1, 2, ..., K-1}  # K为当前节点的邻居数量
```

- **动作类型**：离散（Discrete）
- **动作含义**：选择第 i 个邻居作为下一跳
- **动态动作空间**：不同节点可能有不同数量的邻居

**动作掩码 (Action Masking)**：
- 对于无效邻居（不可达/已断开），使用掩码将其Q值设为负无穷
- 保证只选择有效的下一跳节点

### 4.3 奖励函数 (Reward)

综合考虑多种通信性能指标的奖励函数：

```python
R = -α * D - β * L - γ * P
```

其中：
| 符号 | 含义 | 计算方式 |
|------|------|----------|
| D | 端到端时延 | 从源到目的的累计时延 |
| L | 丢包惩罚 | 丢包时给予负奖励 |
| P | 路径拥塞惩罚 | 选择拥塞路径的惩罚 |
| α, β, γ | 权重系数 | 可根据业务需求调整 |

**详细奖励设计**：

```python
def calculate_reward(state, action, next_state, done):
    reward = 0

    # 1. 时延惩罚 (越小越好)
    hop_delay = get_link_delay(current_node, next_hop)
    reward -= alpha * hop_delay

    # 2. 丢包惩罚
    if packet_dropped:
        reward -= beta * PACKET_LOSS_PENALTY

    # 3. 拥塞惩罚 (选择拥塞节点的惩罚)
    congestion_level = get_congestion(next_hop)
    reward -= gamma * congestion_level

    # 4. 成功到达奖励
    if reached_destination:
        reward += SUCCESS_BONUS

    # 5. 环路惩罚 (避免数据包在网络中循环)
    if loop_detected:
        reward -= LOOP_PENALTY

    return reward
```

**默认权重参数**：
| 参数 | 值 | 说明 |
|------|-----|------|
| α | 1.0 | 时延权重 |
| β | 10.0 | 丢包权重 |
| γ | 0.5 | 拥塞权重 |
| SUCCESS_BONUS | 10.0 | 成功到达奖励 |
| LOOP_PENALTY | 5.0 | 环路惩罚 |

### 4.4 状态转移

```python
# 状态转移过程
1. 智能体观测当前节点的局部状态 s_t
2. 选择下一跳动作 a_t (邻居索引)
3. 数据包转发到下一跳节点
4. 网络状态更新（队列、链路状态变化）
5. 观测新状态 s_{t+1}
6. 计算奖励 r_t
```

### 4.5 Episode定义

- **开始**：数据包在源节点生成
- **结束**：数据包到达目的节点 或 超过最大跳数 或 数据包丢失
- **最大跳数**：max_hops = 15（防止无限循环）

---

## 5. 神经网络架构

### 5.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        输入层                                │
│  节点特征矩阵 X ∈ R^(N×F) + 邻接矩阵 A + 历史状态序列          │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    GCN编码器 (2-3层)                          │
│  - 输入: 节点特征 + 邻接关系                                  │
│  - 输出: 节点嵌入 H ∈ R^(N×d_gcn)                            │
│  - 作用: 捕获网络拓扑结构和节点间关系                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                 Transformer编码器                            │
│  - 输入: 历史GCN嵌入序列 [H_{t-T}, ..., H_t]                 │
│  - 输出: 时序感知特征 Z ∈ R^d_transformer                    │
│  - 作用: 捕获时序依赖和网络动态变化趋势                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    特征融合层                                 │
│  - 融合当前节点状态 + GCN-Transformer特征                     │
│  - 输出: 融合特征 F ∈ R^d_fusion                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Q网络 (DQN)                               │
│  - 输入: 融合特征 F                                          │
│  - 输出: Q(s,a) ∈ R^K (每个邻居的Q值)                        │
│  - 动作选择: argmax_a Q(s,a) with action mask               │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 GCN编码器配置

```yaml
gcn:
  num_layers: 2
  hidden_dim: 64
  output_dim: 64
  activation: relu
  dropout: 0.1
  use_batch_norm: true
  aggregation: mean  # mean/sum/max
```

### 5.3 Transformer配置

```yaml
transformer:
  num_layers: 2
  d_model: 64
  num_heads: 4
  d_ff: 256
  dropout: 0.1
  max_seq_length: 10  # 历史时隙数
  positional_encoding: sinusoidal
```

### 5.4 Q网络配置

```yaml
q_network:
  hidden_layers: [256, 128, 64]
  activation: relu
  output_dim: K_max  # 最大邻居数
  dueling: true      # 使用Dueling架构
```

---

## 6. 算法选择

### 6.1 主算法：DQN及其变体

**选择理由**：
- 路由决策动作空间具有**离散特性**（选择下一跳节点）
- 基于价值函数的方法适合离散动作空间
- DQN及其变体成熟稳定，易于调试

**算法变体**：

| 算法 | 特点 | 适用场景 |
|------|------|----------|
| DQN | 基础版本 | 基线对比 |
| Double DQN | 减少Q值过估计 | 推荐使用 |
| Dueling DQN | 分离状态价值和优势函数 | 推荐使用 |
| Prioritized Experience Replay | 重要样本优先学习 | 加速收敛 |

### 6.2 推荐配置：Dueling Double DQN + PER

```python
算法特性：
1. Double DQN: 使用两个网络分离动作选择和Q值评估
2. Dueling架构: V(s) + A(s,a) - mean(A)
3. PER: 基于TD误差的优先级采样
```

### 6.3 算法流程

```python
for episode in range(num_episodes):
    # 1. 初始化：生成数据包，设置源节点和目的节点
    state = env.reset()

    for step in range(max_steps):
        # 2. 获取当前节点的邻居集合
        neighbors = env.get_neighbors(current_node)
        action_mask = env.get_action_mask()

        # 3. Q网络计算各邻居的Q值
        q_values = q_network(state)
        q_values[~action_mask] = -inf  # 掩码无效动作

        # 4. ε-greedy选择动作
        if random() < epsilon:
            action = random_choice(valid_actions)
        else:
            action = argmax(q_values)

        # 5. 执行动作，转发数据包
        next_state, reward, done = env.step(action)

        # 6. 存储经验
        replay_buffer.add(state, action, reward, next_state, done)

        # 7. 从经验池采样训练
        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            train_step(batch)

        state = next_state
        if done:
            break
```

---

## 7. 训练参数

### 7.1 DQN参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 学习率 | 1e-4 | Adam优化器 |
| 折扣因子 γ | 0.99 | 重视长期回报 |
| 批量大小 | 64 | 经验回放采样大小 |
| 经验池容量 | 100,000 | 存储历史经验 |
| 目标网络更新频率 | 100 steps | 硬更新 |
| 训练轮数 | 5000 | 总episode数 |
| 每轮最大步数 | 50 | 最大跳数 |

### 7.2 探索策略

```yaml
exploration:
  type: epsilon_greedy
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 0.995  # 每episode衰减
```

### 7.3 优先经验回放参数

```yaml
per:
  alpha: 0.6      # 优先级指数
  beta_start: 0.4 # 重要性采样初始值
  beta_end: 1.0   # 重要性采样最终值
  beta_frames: 100000  # beta退火步数
```

---

## 8. 评估指标

### 8.1 主要性能指标

| 指标 | 定义 | 目标 |
|------|------|------|
| **平均端到端时延** | Σ D_i / N_success | 最小化 |
| **数据包投递率** | N_success / N_total | 最大化 |
| **网络吞吐量** | 单位时间成功传输的数据量 | 最大化 |
| **算法收敛速度** | 达到稳定性能所需episode数 | 最小化 |

### 8.2 辅助指标

| 指标 | 定义 |
|------|------|
| 平均跳数 | 成功路由的平均跳数 |
| 路径最优性 | 与最短路径的跳数比 |
| 负载均衡度 | 各节点流量的均衡程度 |
| 适应性 | 拓扑变化后的性能恢复速度 |

---

## 9. 基线算法对比

| 算法 | 描述 | 类型 |
|------|------|------|
| **Dijkstra** | 全局最短路径算法 | 传统算法 |
| **OSPF** | 链路状态路由协议 | 传统算法 |
| **AODV** | 按需距离矢量路由 | 自组织网络 |
| **Q-Routing** | 经典Q学习路由 | RL路由 |
| **DQN (w/o GCN)** | 不使用GCN的DQN | 消融对比 |
| **DQN (w/o Transformer)** | 不使用Transformer的DQN | 消融对比 |
| **Random** | 随机选择下一跳 | 基线 |

---

## 10. 实验设计

### 10.1 实验场景

| 场景 | 节点数 | 特点 | 目的 |
|------|--------|------|------|
| 小规模静态 | 10 | 拓扑固定 | 验证算法基本功能 |
| 中规模动态 | 20 | 拓扑周期变化 | 默认测试场景 |
| 大规模动态 | 50 | 高动态拓扑 | 测试可扩展性 |
| 高移动性 | 30 | 快速拓扑变化 | 测试适应能力 |

### 10.2 消融实验

1. **GCN影响**：对比有/无GCN
2. **Transformer影响**：对比有/无Transformer
3. **历史长度影响**：T = {5, 10, 15, 20}
4. **DQN变体对比**：DQN vs Double DQN vs Dueling DQN
5. **奖励权重敏感性**：不同α, β, γ组合

### 10.3 动态场景测试

1. **链路失效**：随机断开链路，测试路由适应性
2. **节点失效**：随机移除节点，测试路由恢复
3. **流量突发**：突然增加流量，测试拥塞处理
4. **拓扑变化**：卫星移动导致的拓扑变化

### 10.4 可视化输出

1. **训练曲线**：Episode Reward vs Episode
2. **投递率曲线**：Packet Delivery Rate vs Episode
3. **时延分布**：CDF曲线
4. **路由路径可视化**：网络拓扑 + 路由路径
5. **对比图**：不同算法性能柱状图/折线图

---

## 11. 代码实现清单

### 11.1 核心文件

| 文件 | 内容 |
|------|------|
| `src/env/sagin_env.py` | 路由环境实现 |
| `src/env/network_topology.py` | 动态网络拓扑 |
| `src/agents/dqn.py` | DQN/Double DQN/Dueling DQN |
| `src/agents/networks.py` | 神经网络模块 |
| `src/models/gcn.py` | GCN编码器 |
| `src/models/transformer.py` | Transformer编码器 |
| `src/utils/replay_buffer.py` | 经验回放（含PER） |
| `configs/routing_config.yaml` | 路由参数配置 |

### 11.2 环境接口

```python
class SAGINRoutingEnv(gym.Env):
    """SAGIN智能路由环境"""

    def __init__(self, config):
        self.observation_space = spaces.Box(...)  # 局部状态空间
        self.action_space = spaces.Discrete(K_max)  # 离散动作空间

    def reset(self):
        """重置环境，生成新的数据包路由任务"""
        return initial_state

    def step(self, action):
        """执行路由决策，转发数据包到下一跳"""
        return next_state, reward, done, info

    def get_neighbors(self, node_id):
        """获取节点的邻居列表"""
        return neighbor_list

    def get_action_mask(self):
        """获取有效动作掩码"""
        return action_mask
```

---

## 12. 创新点总结

1. **GCN捕获空间拓扑**：利用图卷积网络学习节点间的结构关系
2. **Transformer捕获时序动态**：利用自注意力机制学习网络状态的时序变化
3. **分布式路由决策**：每个节点基于局部信息独立决策
4. **多QoS奖励函数**：融合时延、丢包率、拥塞的综合奖励
5. **动作掩码机制**：处理动态变化的邻居集合

---

## 13. 参考文献

1. **客户需求文档**: 基于深度强化学习的空天地一体化通信网络智能路由研究方案

2. **DQN原始论文**:
   - Mnih et al., "Human-level control through deep reinforcement learning," Nature, 2015

3. **Double DQN**:
   - Van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning," AAAI, 2016

4. **Dueling DQN**:
   - Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning," ICML, 2016

5. **GCN**:
   - Kipf and Welling, "Semi-Supervised Classification with Graph Convolutional Networks," ICLR, 2017

---

**文档版本**: v2.0
**创建日期**: 2026-01-09
**最后更新**: 2026-01-09
**重大变更**: 从任务卸载模型改为智能路由模型
