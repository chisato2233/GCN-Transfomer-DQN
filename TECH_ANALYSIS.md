# 两篇参考论文技术深度解读

## 概览对比

| 维度 | Paper 1 (Jia - IEEE TVT) | Paper 2 (Huynh - IEEE IoT J) |
|------|--------------------------|------------------------------|
| **问题** | SFC服务功能链调度 | 任务卸载资源分配 |
| **算法** | DDQN (离散动作) | DDPG/CMADDPG (连续动作) |
| **创新点** | RTEG时间扩展图 | 多智能体CTDE框架 |
| **网络规模** | 30 UAV + 2卫星 | 3 UAV + 1卫星 |

---

# Paper 2 详细解读: Multi-Agent RL for SAGIN

## 1. 系统模型

### 1.1 网络架构
```
                    ┌─────────────┐
                    │   卫星      │  ← 边缘服务器
                    │ (处理任务)   │
                    └──────┬──────┘
                           │ Air-to-Space
              ┌────────────┼────────────┐
              │            │            │
         ┌────┴────┐  ┌────┴────┐  ┌────┴────┐
         │  UAV 1  │  │  UAV 2  │  │  UAV M  │  ← 中继
         └────┬────┘  └────┬────┘  └────┬────┘
              │            │            │ Ground-to-Air
    ┌─────────┼─────────┐  │  ┌─────────┼─────────┐
    │    │    │    │    │  │  │    │    │    │    │
   GU1  GU2  ...  GUn   │ ... │   GU1  GU2  ...  GUn
                        │     │
                   地面用户群组
```

### 1.2 任务模型
每个地面用户 i 的任务表示为三元组：
```
(Si, Ci, Di)
 │   │   └─ 延迟容忍度 (秒)
 │   └───── CPU周期需求 (cycles)
 └───────── 任务数据量 (bits)
```

### 1.3 信道模型

**Ground-to-UAV (G2U):**
```python
# 信道增益
G_gu = G0 / d_gu^2  # G0为参考距离1m时的增益

# 信噪比
SNR_gu = P_tr * G_gu / sigma_0^2

# 传输速率 (Shannon公式)
r_gu = B * log2(1 + SNR_gu)
```

**UAV-to-Satellite (U2S) - Shadowed-Rician衰落:**
```python
# 信道增益包含阴影Rician衰落
h_j = sqrt((d0/dj)^α) * SR(ω, δ, ε)
# ω: 直射分量平均功率
# δ: 散射分量半平均功率
# ε: Nakagami参数
```

---

## 2. 优化问题

### 2.1 目标函数
最大化**可靠任务卸载比例**：

```
max  (1/N) * Σ φi(xi,j, bi,j, bj) * xi,j
```

其中：
- `φi = 1` 如果总延迟 Li ≤ 延迟容忍 Di，否则为0
- `xi,j` 是GU i卸载到UAV j的任务比例

### 2.2 约束条件

| 约束 | 公式 | 含义 |
|------|------|------|
| 能量约束 | Ei ≤ Ei_max | GU能量预算 |
| 带宽约束1 | Σbi,j ≤ Bj_max | UAV带宽限制 |
| 带宽约束2 | Σbj ≤ B_SAT | 卫星带宽限制 |
| QoS约束1 | ri,j ≥ ri,j_min | G2U最小速率 |
| QoS约束2 | rj ≥ rj_min | U2S最小速率 |
| 计算约束 | Σxi,j*fi ≤ F_max | 卫星计算能力 |

### 2.3 延迟模型

```python
总延迟 = 本地处理延迟 + G2U传输延迟 + U2S传输延迟 + 边缘处理延迟

L_total = L_GU + L_G2U + L_U2S + L_ES

其中:
L_GU = (1-xi,j) * Ci / f_GU      # 本地处理
L_G2U = xi,j * Si / r_gu          # 上行传输到UAV
L_U2S = xi,j * Si / r_us          # UAV到卫星
L_ES = xi,j * Ci / f_ES           # 卫星边缘计算
```

---

## 3. DDPG算法详解

### 3.1 MDP建模

#### 状态空间 (State Space)
```python
state = {
    R_off(t),    # 任务完成率
    V_con(t),    # 约束违反数
    U_util(t)    # 卫星资源利用率
}

# 资源利用率计算
U_proc(t) = Σ(xi,j * f_ES) / F_max
```

#### 动作空间 (Action Space)
```python
action = {
    xi,j(t),     # 卸载比例 ∈ [0, 1]
    bi,j(t),     # G2U带宽分配 ∈ [0, Bj_max]
    bj(t)        # U2S带宽分配 ∈ [0, B_SAT]
}
```

#### 奖励函数 (Reward Function)
```python
reward = Σ Σ (φi * xi,j - λE*ΨE - λB*ΨB - λr*Ψr - λF*ΨF)

# 各惩罚项:
# ΨE: 能量约束违反程度
# ΨB: 带宽约束违反程度
# Ψr: 速率约束违反程度
# ΨF: 计算约束违反程度
```

### 3.2 DDPG网络架构

```
┌─────────────────────────────────────────────────────────┐
│                      DDPG 框架                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ┌─────────────┐         ┌─────────────┐              │
│   │   Actor     │         │   Critic    │              │
│   │   网络      │         │   网络      │              │
│   │ μ(s|θ)     │         │ Q(s,a|φ)   │              │
│   └──────┬──────┘         └──────┬──────┘              │
│          │                       │                      │
│          │ 输出动作              │ 评估Q值              │
│          ▼                       ▼                      │
│   ┌──────────────────────────────────────┐             │
│   │           经验回放缓冲区              │             │
│   │        (st, at, rt, st+1)           │             │
│   └──────────────────────────────────────┘             │
│                                                         │
│   ┌─────────────┐         ┌─────────────┐              │
│   │ Target Actor│         │Target Critic│              │
│   │   μ'(θ')    │         │   Q'(φ')   │              │
│   └─────────────┘         └─────────────┘              │
│         ↑ 软更新                 ↑ 软更新               │
│         θ' ← τθ + (1-τ)θ'       φ' ← τφ + (1-τ)φ'     │
└─────────────────────────────────────────────────────────┘
```

### 3.3 核心公式

**Q值更新:**
```python
y_t = r_t + γ * Q'(s_{t+1}, μ'(s_{t+1}|θ')|φ')

# Critic损失函数
L(φ) = E[(y_t - Q(s_t, a_t|φ))^2]
```

**Actor更新:**
```python
∇θJ ≈ (1/N) * Σ ∇a Q(s,a|φ)|_{a=μ(s)} * ∇θ μ(s|θ)
```

### 3.4 超参数设置 (论文中)

| 参数 | 值 |
|------|-----|
| Actor学习率 | 10^-5 |
| Critic学习率 | 10^-4 |
| 折扣因子 γ | 0.99 |
| 软更新系数 τ | 0.05 |
| Batch size | 256 |
| Replay buffer | 10^6 |

---

## 4. 多智能体CMADDPG

### 4.1 CTDE机制
**Centralized Training, Decentralized Execution**

```
训练阶段 (集中式):
┌────────────────────────────────────────┐
│         中央Critic (全局信息)          │
│  Q(s1,s2,...,sM, a1,a2,...,aM)        │
└────────────────────────────────────────┘
           ↑ 评估
┌──────┐ ┌──────┐     ┌──────┐
│Actor1│ │Actor2│ ... │ActorM│  ← 各UAV独立
└──────┘ └──────┘     └──────┘

执行阶段 (分布式):
┌──────┐ ┌──────┐     ┌──────┐
│Actor1│ │Actor2│ ... │ActorM│
│ (s1) │ │ (s2) │     │ (sM) │  ← 仅用本地观测
└──────┘ └──────┘     └──────┘
```

### 4.2 多智能体更新公式

**各Agent的Target Q值:**
```python
y_m = r_m + γ * Q'_φm(s', a')

# 其中 a' = {μ'_θ1(s'1), ..., μ'_θM(s'M)}
# 包含所有Agent的目标动作
```

**Critic损失:**
```python
L(φm) = E[(Q_φm(s, a) - y_m)^2]
```

**Actor梯度:**
```python
∇θm J ≈ (1/|R|) * Σ ∇θm μθm(sm) * ∇am Q_φm(s,a)|_{am=μθm(sm)}
```

---

# Paper 1 详细解读: SFC Dynamic Scheduling

## 1. RTEG (可重构时间扩展图)

### 1.1 核心思想
将动态网络按时隙划分，每个时隙内网络拓扑准静态：

```
时隙 t=1        时隙 t=2        时隙 t=3
┌─────┐        ┌─────┐        ┌─────┐
│ n1  │        │ n1  │        │ n1  │
└──┬──┘        └──┬──┘        └──┬──┘
   │               │               │
   ▼               ▼               ▼
┌─────┐        ┌─────┐        ┌─────┐
│ n2  │   ──▶  │ n2  │   ──▶  │ n2  │
└─────┘   存储  └─────┘   存储  └─────┘
                链路Lt          链路Lt
```

### 1.2 链路类型
- `L_gu`: Ground-to-UAV
- `L_ug`: UAV-to-Ground
- `L_uu`: UAV-to-UAV
- `L_us`: UAV-to-Satellite
- `L_ss`: Satellite-to-Satellite
- `L_sg`: Satellite-to-Ground
- `L_t`: 存储链路 (同节点跨时隙)

## 2. SFC调度模型

### 2.1 VNF序列约束
```
SFC: f1 → f2 → f3 → ... → fl

# VNF必须按顺序执行
t_{f_{m+1}} - t_{f_m} ≥ 处理时间
```

### 2.2 资源约束
```python
# 计算资源约束
Σ Σ x_{ni,fm} * σ_{fm} ≤ C_ni

# 存储约束
Σ ρ_{k,(ni,ni+1)} * δk ≤ U_ni

# 信道容量约束
Σ z_{k,(ni,nj)} * δk ≤ r_{(ni,nj)} * τ
```

## 3. DDQN算法

### 3.1 与DDPG的区别

| 特性 | DDQN (Paper 1) | DDPG (Paper 2) |
|------|----------------|----------------|
| 动作空间 | 离散 (选择节点) | 连续 (比例/带宽) |
| 输出 | 各动作Q值 | 确定性动作 |
| 网络 | 单Q网络 | Actor + Critic |

### 3.2 状态设计
```python
state = {
    F_k: (k, v_k, C_k),  # SFC信息
    N: (η1, η2, ..., ηI)  # 节点资源占用
}

# v_k: VNF状态
#   0 = 传输中
#   1 = 处理中
#   2 = 存储等待中
```

### 3.3 奖励设计
```python
R_k = c0 - c1 * t_c(t) - c2 * t_w(t)

# t_c: 传输时间消耗
# t_w: 等待时间消耗
# c0, c1, c2: 权重系数
```

---

# 你的项目如何融合GCN + Transformer

## 建议的架构

```
┌─────────────────────────────────────────────────────────────┐
│                   GCN-Transformer-RL 架构                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────────────────────────────────────────────────┐    │
│  │              输入层                                 │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │    │
│  │  │网络拓扑图│  │节点特征  │  │历史状态  │         │    │
│  │  │邻接矩阵A │  │X (资源等)│  │序列      │         │    │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘         │    │
│  └───────┼─────────────┼─────────────┼───────────────┘    │
│          │             │             │                     │
│          ▼             ▼             │                     │
│  ┌────────────────────────┐         │                     │
│  │         GCN           │         │                     │
│  │  图卷积特征提取        │         │                     │
│  │  h = GCN(X, A)        │         │                     │
│  └───────────┬───────────┘         │                     │
│              │                      │                     │
│              ▼                      ▼                     │
│  ┌────────────────────────────────────────┐              │
│  │            Transformer                  │              │
│  │    时序特征建模 + 注意力机制             │              │
│  │    z = Transformer([h; history])       │              │
│  └───────────────────┬────────────────────┘              │
│                      │                                    │
│                      ▼                                    │
│  ┌────────────────────────────────────────┐              │
│  │              RL决策层                   │              │
│  │  ┌─────────┐       ┌─────────┐        │              │
│  │  │  Actor  │       │  Critic │        │              │
│  │  │ π(z)    │       │ Q(z,a)  │        │              │
│  │  └────┬────┘       └────┬────┘        │              │
│  └───────┼─────────────────┼─────────────┘              │
│          │                 │                             │
│          ▼                 ▼                             │
│      路由决策           价值评估                          │
└─────────────────────────────────────────────────────────────┘
```

## 各模块的具体作用

### GCN模块
```python
# 输入:
#   - 网络拓扑邻接矩阵 A [N×N]
#   - 节点特征 X [N×F] (带宽、时延、能量等)

# 输出:
#   - 节点嵌入 H [N×D]

# 作用:
#   - 捕获网络结构信息
#   - 聚合邻居节点特征
#   - 替代人工设计的状态特征
```

### Transformer模块
```python
# 输入:
#   - 历史GCN嵌入序列 [T×N×D]
#   - 或历史状态序列 [T×S]

# 输出:
#   - 时序感知特征 [D']

# 作用:
#   - 建模网络动态变化规律
#   - 自注意力捕获长期依赖
#   - 预测未来网络状态趋势
```

### RL模块
```python
# 沿用Paper2的DDPG/CMADDPG框架
# 但输入改为GCN+Transformer的融合特征
```

---

## 实现优先级建议

1. **Week 1**: 先实现基础DDPG (不含GCN/Transformer)
2. **Week 2**: 加入GCN，验证图特征的效果
3. **Week 3**: 加入Transformer，完成融合
4. **Week 4**: 多智能体扩展 + 实验

---

## 关键代码片段参考

### GCN层实现
```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCNEncoder(nn.Module):
    def __init__(self, in_features, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
```

### Transformer编码器
```python
class TemporalEncoder(nn.Module):
    def __init__(self, d_model, nhead=4, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        # x: [seq_len, batch, d_model]
        return self.transformer(x)
```

### 融合模型
```python
class GCNTransformerRL(nn.Module):
    def __init__(self, node_features, gcn_hidden, gcn_out,
                 seq_len, transformer_dim, action_dim):
        super().__init__()
        self.gcn = GCNEncoder(node_features, gcn_hidden, gcn_out)
        self.temporal = TemporalEncoder(gcn_out)
        self.actor = nn.Sequential(
            nn.Linear(gcn_out, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(gcn_out + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
```
