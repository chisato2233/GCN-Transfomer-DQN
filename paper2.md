This article has been accepted for publication in IEEE Internet of Things Journal. This is the author's version which has not been fully edited and

content may change prior to final publication. Citation information: DOI 10.1109/JIOT.2025.3598597

# Multi-Agent Reinforcement Learning for Optimal Resource Allocation in Space-Air-Ground Integrated Networks



1



Dang Van Huynh, _Member, IEEE,_ Saeed R. Khosravirad, _Senior Member, IEEE,_ Simon L. Cotton, _Fellow, IEEE,_
Hyundong Shin, _Fellow, IEEE,_ and Trung Q. Duong, _Fellow, IEEE_



_**Abstract**_ **—This paper addresses the problem of reliable task**
**offloading in space-air-ground integrated network (SAGIN)-**
**assisted edge computing systems, with the goal of maximising**
**the ratio of tasks successfully offloaded and executed within**
**quality-of-service (QoS) constraints. In the considered system,**
**ground users offload computation tasks to a satellite-mounted**
**edge server via unmanned aerial vehicles (UAVs) acting as**
**relays. The formulated optimisation problem jointly consid-**
**ers task offloading portions and bandwidth allocations across**
**ground-to-air and air-to-space links, subject to constraints on**
**transmission rates, total bandwidth, energy budgets, and the**
**satellite’s computational capacity. The resulting problem is non-**
**linear, non-convex, and mixed-integer, making it challenging to**
**solve with traditional optimisation techniques. To this end, we**
**propose a deep reinforcement learning (DRL)-based solution**
**to learn optimal offloading and resource allocation policies in**
**dynamic environments. Furthermore, to enhance scalability and**
**decentralised coordination, we develop a multi-agent DRL frame-**
**work that enables cooperative decision-making across UAVs.**
**Simulation results demonstrate that both the single-agent and**
**multi-agent approaches achieve stable training performance, and**
**the proposed method improves the reliable task offloading ratio**
**by up to two times compared to benchmark schemes, while also**
**achieving more efficient resource utilisation in complex SAGIN**
**scenarios.**


_**Index Terms**_ **—6G, multi-agent reinforcement learning, non-**
**terrestrial networks, space-air-ground integrated networks**


D. V. Huynh is with the Faculty of Engineering and Applied Science, Memorial University, St. John’s, NL A1B 3X5, Canada (e-mail: vdhuynh@mun.ca).
S. R. Khosravirad is with Nokia Bell Labs, Murray Hill, NJ 07964 USA
(e-mail: saeed.khosravirad@nokia-bell-labs.com).
S. L. Cotton is with the Centre for Wireless Innovation (CWI), School of
Electronics, Electrical Engineering and Computer Science, Queen’s University
Belfast, BT3 9DT, Belfast, U.K (email: simon.cotton@qub.ac.uk).
H. Shin is with Department of Electronics and Information Convergence
Engineering, Kyung Hee University, 1732 Deogyeong-daero, Giheung-gu,
Yongin-si, Gyeonggi-do 17104, Korea (e-mail: hshin@khu.ac.kr).
T. Q. Duong is with the Faculty of Engineering and Applied Science,
Memorial University, St. John’s, NL A1C 5S7, Canada, and with the School of
Electronics, Electrical Engineering and Computer Science, Queen’s University
Belfast, Belfast, U.K, and also with the Department of Electronic Engineering,
Kyung Hee University, Yongin-si, Gyeonggi-do 17104, South Korea (e-mail:
tduong@mun.ca).
This paper has been accepted in part for presentation at the IEEE International Conference on Communications 8–12 June 2025, Montreal, Canada.
The work of T. Q. Duong was supported in part by the Canada Excellence
Research Chair (CERC) Program CERC-2022-00109 and in part by the
Natural Sciences and Engineering Research Council of Canada (NSERC)
Discovery Grant Program RGPIN-2025-04941. The work of S. L. Cotton was
supported by the U.K. Engineering and Physical Sciences Research Council
(EPSRC) through the EPSRC Hub on All Spectrum Connectivity under Grant
EP/X040569/1 and Grant EP/Y037197/1. The work of H. Shin was supported
in part by National Research Foundation of Korea (NRF) grant funded by the
Korean government (MSIT) (RS-2025 00556064).
Corresponding authors are Trung Q. Duong and Hyundong Shin.



I. INTRODUCTION


Space-air-ground integrated networks (SAGIN) are emerging as a key technology for achieving ubiquitous connectivity
in 6G networks [1], [2]. By integrating space, aerial, and
terrestrial components, SAGIN can provide seamless wireless
coverage across vast geographical areas, including hard-toreach locations, making it essential for critical services such
as remote surveillance, environmental monitoring, and disaster
management [2]. However, SAGIN presents several challenges, which have attracted considerable attention from the
research community [3]. One such challenge lies in optimising
resource management across both communication resources
(e.g., bandwidth allocation, transmission power) and computing resources (e.g., processing capacity, storage, energy) [4]–

[6]. The high attenuation and long-distance nature of satellite
communications, combined with the resource limitations of
ground devices used for remote operation, make the design
of efficient solutions for SAGIN-assisted systems particularly
complex [2]. Addressing these challenges will be crucial for
fully realising the potential of SAGIN in real-world applications [7], [8].
Recently, the integration of SAGIN-based communication
with edge and cloud computing, driven by the need to support
emerging services that demand low latency and high computational power, has gained significant attention [9]–[18].
The convergence of these key technologies will unlock the
full potential of next-generation wireless networks, providing
the ability to not only deliver global coverage but at the
same time enhance computational capacity to meet complex
service demands. Satellites equipped with edge servers to
process computational tasks offloaded from ground users are
a key element of this integration. These satellites, with their
powerful computing resources, are able to process complex
tasks and provide timely responses to users on the ground,
greatly reducing latency compared to routing from remote
locations via ground based telecommunications infrastructure,
which in some cases may not be available.
The benefits of using SAGIN assisted edge computing are
not open-ended. A major challenge here is long-distance transmissions between satellites and ground users which presents
considerable challenges in maintaining efficient communication, making joint optimisation of communication and computing resources a difficult and multi-faceted problem. The
dynamic nature of user demands, network conditions, and
resource availability requires sophisticated strategies for balancing these resources. These challenges are drawing significant interest in advanced resource optimisation for dynamic



Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY MADRAS. Downloaded on September 04,2025 at 03:05:46 UTC from IEEE Xplore. Restrictions apply.

© 2025 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,

but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.


This article has been accepted for publication in IEEE Internet of Things Journal. This is the author's version which has not been fully edited and

content may change prior to final publication. Citation information: DOI 10.1109/JIOT.2025.3598597



SAGIN-enabled edge computing.


_A. Related Works_

Recent research efforts have focused on the joint optimisation of communication and computation resources in SAGINassisted edge and cloud computing systems, addressing the
growing need for efficient and adaptive service provisioning
in highly dynamic and heterogeneous environments [10], [12]–

[14], [17]–[19]. The fundamental research problems typically revolve around optimising task offloading decisions,
bandwidth allocation, service placement, and computational
resource management to improve overall system performance
under strict constraints on latency, energy consumption, and
reliability. To address these challenges, a range of optimisation and learning-based approaches have been proposed.
In particular, deep reinforcement learning (DRL) has been
widely explored as a promising solution for dynamic traffic
offloading and resource allocation across the satellite, aerial,
and terrestrial segments of SAGINs [10], [12]. DRL-based
methods are well-suited to capture the stochasticity of user
demands and channel conditions, often leveraging hybrid
discrete-continuous action space frameworks to enable flexible
and adaptive decision-making in real time.
In parallel, efforts have been made to optimise the placement
of services and selection of servers in satellite-terrestrial
edge environments, where the goal is to minimise end-to-end
latency and operational costs while satisfying task requirements and resource constraints [14]. Related contributions
have addressed the co-optimisation of transmission power and
computational resources to reduce energy usage while ensuring
timely task execution in satellite and high-altitude platformassisted systems [17]. Further developments in the field have
focused on mobile edge computing within satellite-based IoT
networks, where the joint management of multi-task offloading
and resource allocation is shown to significantly reduce system
latency [18]. Additionally, satellite edge computing has been
investigated for real-time, high-resolution Earth observation
tasks, enabling onboard processing to alleviate downlink bandwidth demands and improve resource utilisation efficiency

[13].
A range of optimisation methods has been proposed to
address these challenges, spanning from traditional mathematical formulations to more adaptive learning-based strategies.
Classical approaches often rely on deterministic models to
formulate and solve resource scheduling and allocation problems under known system constraints. For instance, resource
scheduling across edge–cloud infrastructures in SAGINs has
been addressed through graph-based modelling and centralised
optimisation, where the objective is to allocate bandwidth and
computation resources efficiently for vehicular services [11].
Similarly, service placement and server selection in integrated
satellite–terrestrial systems have been jointly optimised to
minimise latency and resource usage through mixed-integer
nonlinear programming techniques [14]. Moreover, satellite
edge computing frameworks have been proposed for latencycritical applications such as Earth observation, where mathematical scheduling is used to minimise transmission delays
while maintaining high-resolution image fidelity [13].



2


In contrast, modern solutions are increasingly shifting towards data-driven methods that offer better adaptability in
highly dynamic environments. AI-driven frameworks have
been proposed for end-to-end network management in SAGINs, enabling real-time decision-making through predictive
analytics and context-aware control [20], [21]. Among these,
deep reinforcement learning (DRL) has emerged as a leading
technique due to its ability to learn optimal policies through
interaction with the environment. Various DRL algorithms
have been adopted to solve these challenging problems such
as proximal policy optimisation [18], Q-learning [10], soft
actor-critic (SAC) algorithm [12]. For instance, a PPO-based
approach has been applied to for joint task offloading and
resource allocation in satellite-based IoT systems in [18],
where agents learn to balance communication load and computational demand dynamically. These methods demonstrate
strong potential in capturing the temporal variability and
stochastic nature of SAGIN environments, offering a more
scalable and autonomous approach to network optimisation
compared to traditional rule-based systems.
To address the increasing scale and complexity of decisionmaking in SAGIN-enabled networks, recent studies have
explored multi-agent reinforcement learning (MARL) as a
promising paradigm for decentralised yet coordinated control

[22]–[24]. By enabling multiple collaborative agents such as
UAVs, satellites, and ground nodes, MARL offers a scalable framework for solving distributed optimisation problems
across heterogeneous network segments. In the context of
6G subnetwork management, MARL has been applied for
dynamic resource allocation, where agents learn joint policies to maximise network utility under varying traffic and
channel conditions [23]. Cooperative MARL techniques have
also been used to optimise ITS data offloading and computation in satellite-aided architectures by leveraging attention
mechanisms and proximal policy optimisation, significantly
enhancing system responsiveness and task distribution across
space and ground segments [24].
The potential of MARL extends to aerial edge networks as
well [25]–[27]. In particular, shared critic-based actor-critic
architectures have been proposed, enabling joint UAV trajectory planning while minimising training complexity [25]. In
line with this, cluster-based MARL algorithms have emerged
for cooperative task scheduling in large-scale SAGINs, where
UAVs are grouped to collectively learn efficient service policies within and across clusters, thereby improving resource
utilisation and system throughput [26]. In addition, coordinated task offloading strategies have demonstrated substantial
energy efficiency gains in multi-UAV scenarios for aerial edge
networks [27]. These advancements underline the ability of
MARL to handle the distributed, dynamic, and multi-objective
nature of SAGIN resource management, paving the way for
more autonomous and intelligent 6G infrastructures.
In summary, while significant progress has been made in
optimising resource allocation, task offloading, and service
deployment in SAGIN-assisted systems, ensuring the reliability of task execution under dynamic and resource-constrained
conditions remains a critical and insufficiently addressed issue.
Motivated by these gaps, this work focuses on developing



Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY MADRAS. Downloaded on September 04,2025 at 03:05:46 UTC from IEEE Xplore. Restrictions apply.

© 2025 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,

but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.


This article has been accepted for publication in IEEE Internet of Things Journal. This is the author's version which has not been fully edited and

content may change prior to final publication. Citation information: DOI 10.1109/JIOT.2025.3598597



scalable and intelligent solutions that explicitly maximise reliable task offloading while considering the complex interactions
between communication and computation resources across the
space-air-ground infrastructure.


_B. Motivation and Contributions_


While significant progress has been made in optimising
resource allocation, task offloading, and service deployment
in SAGIN-based systems [10], [12], [14], [18], a critical
challenge that remains insufficiently addressed is the reliable
offloading and execution of tasks, particularly under uncertain
and resource-constrained conditions. This issue becomes especially important in remote and mission-critical scenarios, such
as emergency response and disaster recovery, where timely and
dependable task completion is essential. Existing solutions,
including those leveraging deep reinforcement learning [12],

[18] and multi-agent collaboration [24], [27], primarily focus
on improving general system performance metrics like latency
or energy efficiency. However, they often overlook the importance of guaranteeing task reliability, especially when network
dynamics, user heterogeneity, and limited edge resources interact unpredictably. Moreover, while recent MARL approaches
enable scalable coordination across UAVs and edge nodes [25],

[26], they typically lack explicit mechanisms to ensure that
offloaded tasks are completed successfully within their QoS
constraints. Motivated by these limitations, this work addresses
the problem of reliable task offloading in SAGIN-assisted edge
computing. Our objective is to maximise the ratio of tasks
that are not only offloaded but also completed within their
latency and resource bounds, thereby ensuring QoS guarantees
in complex and dynamic network environments.
The main contributions of this work are summarised as
follows:


_•_ We formulate a reliability-driven task offloading problem
in SAGIN-assisted edge computing, aiming to maximise
the ratio of tasks that are successfully offloaded and
completed within their delay and resource constraints.

_•_ We propose a DRL-based solution that jointly optimises
bandwidth allocation and task offloading portions, while
satisfying energy consumption budgets and QoS requirements.

_•_ To enhance scalability and coordination across network
layers, we develop a multi-agent DRL solution that enables distributed decision-making for edge nodes, leveraging cooperation to improve system-wide reliability.

_•_ We conduct extensive simulations to evaluate the performance of the proposed methods. Numerical results
confirm the effectiveness of our proposed solutions in
terms of training convergence, maximising reliable task
offloading, and efficient resource utilisation under dynamic network conditions.


_C. Paper Structure and Notations_


The remainder of this paper is structured as follows. Section II presents the system model and problem formulation,
including detailed descriptions of the wireless transmission
model, latency model, energy consumption model, and the



3


reliability-driven task offloading model. The formal optimisation problem is then expressed based on these models.
Section III introduces the transformation of the original problem into a reinforcement learning framework, enabling the
development of a DRL-based solution. A classical singleagent DRL approach is proposed in Section III. Section IV
extends the solution to a multi-agent framework, leveraging
the centralised training and distributed execution (CTDE)
mechanism to enhance scalability and coordination. Section V
provides simulation results and discussions, demonstrating
the effectiveness and advantages of the proposed solutions.
Finally, Section VI concludes the paper by summarising the
main findings and highlighting promising directions for future
research.
Throughout this paper, scalar values are denoted by regular
lowercase letters, while vectors are represented by bold lowercase letters. Parameters or variables associated with ground
user (GU) _i_ and UAV _j_ are denoted by _xi,j_, where _x_ indicates
a specific parameter or variable under consideration. The key
notations used in this paper are summarised in Table I.


TABLE I
KEY NOTATIONS.

|Notation|Definition|
|---|---|
|_N_<br>_M_<br>_Ci_<br>_Di_<br>_xi,j_<br>_bi,j_<br>_bj_<br>_Pi_,_ Pj_<br>_hGA_<br>_i,j_<br>_hAS_<br>_j_<br>_N_0<br>_rGA_<br>_i,j_<br>_rAS_<br>_j_<br>_Ei_<br>_Li_|Number of ground users (GUs)<br>Number of UAVs<br>Required CPU cycles for task_ i_<br>Delay tolerance for task_ i_<br>Offoading portion of GU_ i_ to UAV_ j_<br>Bandwidth allocated between GU_ i_ and UAV_ j_<br>Bandwidth allocated between UAV_ j_ and the satellite<br>Transmit power of GU_ i_ and UAV_ j_, respectively<br>Channel gain between GU_ i_ and UAV_ j_<br>Channel gain between UAV_ j_ and the satellite<br>Noise power spectral density<br>Transmission rate from GU_ i_ to UAV_ j_<br>Transmission rate from UAV_ j_ to the satellite<br>Energy budget of GU_ i_<br>Latency for task processed by UAV_ j_ or the satellite|



II. SYSTEM MODEL AND PROBLEM FORMULATION


In this paper, we consider a SAGIN model which consists
of _N_ ground users (GUs), _M_ relay UAVs, and one satellite
associated with an edge server to process offloaded tasks from
the GUs. Fig. 1 provides an illustration of the considered
system model. We assume that the formation of GU-UAV
networks is conducted in advanced, where the _j_ -th UAV only
serves a finite number of GUs in its coverage. A computational
task generated from the _i_ -th GU is characterised by a tuple
of three parameters ( _Si, Ci, Di_ ), denoting the task size (bits),
required CPU cycles (cycles), and delay tolerance (seconds),
respectively. Due to limitation in the available computing
capacity, as well as energy budget, the GUs have to offload
the task to the satellite’s edge server to process.


_A. Wireless Transmission Model_


In this paper, all devices considered in the system model
are single-antenna devices. Frequency division multiple access



Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY MADRAS. Downloaded on September 04,2025 at 03:05:46 UTC from IEEE Xplore. Restrictions apply.

© 2025 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,

but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.


This article has been accepted for publication in IEEE Internet of Things Journal. This is the author's version which has not been fully edited and

content may change prior to final publication. Citation information: DOI 10.1109/JIOT.2025.3598597


4


UAV to the satellite. Here, _bi,j_ is bandwidth allocated for the
link from the _i_ -th GU to the _j_ -th UAV, and _bj_ is bandwidth
allocated for the link from the _j_ -th UAV to the satellite. _Pi_
and _Pj_ are the transmission power of the _i_ -th GU and the _j_ -th
UAV, respectively. _hi,j_ is channel gain between the _i_ -th GU
and the _j_ -th UAV, and _hj_ is the channel gain between the UAV
and the satellite. _N_ 0 is the noise spectral density.


_B. Latency Model_


As illustrated in Fig. 1, the _i_ -th GU offloads a portion of _xi,j_
of the computational task to the _j_ -th UAV, then the UAV forwards it to the satellite’s edge server for processing. Therefore,
the latency of the _i_ -th task consists of four components: local
processing latency                        - _L_ [GU] _i_ �, GU-to-UAV transmission latency
( _Li,j_ ), UAV-to-satellite transmission latency ( _Li,j_ ), and edge
processing latency                        - _L_ [ES] _i,j_ �, which calculated as follows:



Fig. 1. An illustration of SAGIN-enabled edge computing systems.



(FDMA) method is utilised for wireless transmissions in the
system. We aim to develop an optimal design for bandwidth
allocations, which guarantees the transmission QoS, meets the
desired latency constraints, and improves the reliability of task
offloading.
_1) Channel model for space-to-air transmissions:_ We adopt
the shadowed-Rician fading (SRF) model to describe the channel of space-to-air transmissions [28]. The channel gain _hj_ is
expressed as _hj_ = ~~�~~ ( _d_ 0 _/dj_ ) ~~_[α]_~~ SR( _ω, δ, ϵ_ ), where SR ( _·, ·, ·_ )



_L_ [GU] _i_ [(] _[x][i,j]_ [) =] [(1] _[ −]_ _[x][i,j]_ [)] _[C][i]_ _,_

_fi_ [GU]

_xi,jSi_
_Li,j_ ( _xi,j, bi,j_ ) =
_ri,j_ ( _bi,j_ ) _[,]_

_Li,j_ ( _xi,j, bj_ ) = _[x][i,j][S][i]_

_rj_ ( _bj_ ) _[,]_

_L_ [ES] _i,j_ [(] _[x][i,j]_ [) =] _[x][i,j][C][i]_ _._

_fi_ [ES]



(3)


(4)


(5)


(6)



expressed as _hj_ = ~~�~~ ( _d_ 0 _/dj_ ) ~~_[α]_~~ SR( _ω, δ, ϵ_ ), where SR ( _·, ·, ·_ )

represent the SRF parameters. Here _dj_ is the distance between
the _j_ -th UAV and the satellite; _d_ 0 = 1 is a reference distance
and _α_ is the path loss exponent for the space-to-air link; _ω_
is the average power of the direct LoS component; _δ_ is the
half average power of the scatter portion; _ϵ_ is the Nakagami
_m_ -parameter for the scattered NLoS components.
_2) Channel model for air-to-ground transmissions:_ In this
work, we consider light-of-sight (LOS) links between the _i_ -th
GU and _j_ -th UAVs so we can model the channel gain _hi,j_ as
_hi,j_ = - _βi,j_ ( _di,j_ ) _gi,j_ [12]. Here, _βi,j_ ( _di,j_ ) = - _d_ 0 _/di,j_ - _α_

represents the large-scale fading, including distance-based path
loss and shadowing, where _α_ and _d_ 0 = 1 m are the path
loss exponent for the air-to-ground links and the reference
distance, respectively. _gi,j ∼_ Rician( _K_ ) is the small-scale
fading component, where _K_ is the Rician factor defining the
ratio of power of the direct LoS path to the power contributed
by the scattered paths.
_3) Transmission schemes:_ The transmission rate of the _i_ -th
GU to the _j_ -th UAV is given by



_βi,j_ ( _di,j_ ) _gi,j_ [12]. Here, _βi,j_ ( _di,j_ ) = - _d_ 0 _/di,j_ - _α_




_._ (1)



As a result, the total latency for a task completely offloaded
and processed is given by


_Li_ = _L_ [GU] _i_ + _Li,j_ + _Li,j_ + _L_ [ES] _i,j_ _[.]_ (7)


_C. Energy Consumption Model_


To handle the limitation on the energy budget of the GUs,
we model the energy consumption of the _i_ -th GU as follows

_Ei_ ( _xi,j, bi,j_ ) = _θ_ (1 _−_ _xi,j_ ) _Ci_  - _fi_ [GU] �2 + _xi,jri,jSiPi_ _,_ (8)


which includes two components: energy consumption for local
processing and energy consumption for the transmission. Since
the task is partially offloaded a portion of _xi,j_ to the UAVs, the
_i_ -th GU only processes the remaining portion of (1 _−_ _xi,j_ ).
Here, _θ_ is the parameter used to calculate the computation
energy of GUs, which varies according to the CPU used [29].


_D. Reliable Task Offloading Definition_


In this paper, we propose a reliability-driven approach
for optimal design of joint task offloading and bandwidth
allocations in SAGIN-assisted edge computing. The reliable
metric is developed based on a binary indicator _ϕi_ = _{_ 0 _,_ 1 _}_ .
Specifically, a task is considered reliably offloaded if the total
latency _Li_ is less than or equal to its delay tolerance _Di_,
mathematically expressed as



_ri,j_ ( _bi,j_ ) = _bi,j_ log2




1 + _[P][i][h][i,j]_

_N_ 0 _bi,j_



Similarly, the transmission rate of the _j_ -th UAV to the
satellite is expressed as




_._ (2)



_rj_ ( _bj_ ) = _bj_ log2




1 + _[P][j][h][j]_

_N_ 0 _bj_



where _ri,j_ denotes the transmission rate of the _i_ -th GU to
the _j_ -th UAV, while _rj_ is the transmission rate from the _j_ -th



_ϕi_ ( _xi,j, bi,j, bj_ ) =




1 _,_ if _Li ≤_ _Di,_
(9)
0 _,_ otherwise _._



Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY MADRAS. Downloaded on September 04,2025 at 03:05:46 UTC from IEEE Xplore. Restrictions apply.

© 2025 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,

but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.


This article has been accepted for publication in IEEE Internet of Things Journal. This is the author's version which has not been fully edited and

content may change prior to final publication. Citation information: DOI 10.1109/JIOT.2025.3598597



_E. Problem Formulation_


Based on the above representation of the system model, the
optimisation problem formulated in this paper is given by (10).
Here, the objective of the problem is to maximise the average
reliable task offloading ratio, ensuring the tasks are completely
offloaded and processed within their delay tolerances by optimising the variables of offloading portions, i.e., **x** ≜ _{xi,j}∀i,j_
and bandwidth allocations, i.e., **b** ≜ _{bi,j, bj}∀i,j_ .



(10a)


(10b)


(10c)


(10d)


(10e)


(10f)


(10g)



1
**P1:** maximise
**x** _,_ **b** _N_



_N_

- _ϕi_ ( _xi,j, bi,j, bj_ ) _xi,j,_


_i_ =1



5


action space ( _A_ ), and the reward function ( _R_ ). We first start
with the design of the state space.
_1) State space:_ The state space _S_ is composed of necessary
information of the system at the time step _t_, observed by the
agent to select next action, including the following system
parameters:


_•_ Task size _Si_ ( _t_ ): The size of the computational task
generated by the _i_ -th GU, measured in bits.

_•_ Required CPU cycles _Ci_ ( _t_ ): The number of CPU cycles
required to process the task generated by the _i_ -th GU,
measured in cycles/second.

_•_ Delay tolerance _Di_ ( _t_ ): The maximum allowable latency
for the task generated by the _i_ -th GU, measured in
seconds.

_•_ Bandwidth allocations _bi,j_ ( _t_ ) and _bj_ ( _t_ ): The bandwidth
allocated to the _i_ -th GU for communication with the _j_ -th
UAV and for the communication of the _j_ -th UAV with
the satellite, respectively.

_•_ Channel conditions _hi,j_ ( _t_ ) and _hj_ ( _t_ ) :The wireless channel gains from the _i_ -th GU to the _j_ -th UAV, and from the
_j_ -th UAV and the satellite, respectively.

_•_ Energy consumption _Ei_ ( _t_ ): The current energy consumption of the _i_ -th GU, measured in joules.


It is important to note that the DDPG agent can learn
more effectively with a concentrated state space, instead of
discrete information. The concentrated state space can incorporate constraint violations as part of the state, making it
easier for the agent to learn feasible solutions, focusing on
optimising the key metrics that matter. Therefore, we propose
the concentrated expression of the state space _S_ as follows


**st** = _{R_ off( _t_ ) _, V_ con( _t_ ) _, U_ util( _t_ ) _},_ (11)


where:


_• R_ off( _t_ ) is the task completion ratio at time _t_, indicating
the percentage of tasks completed within the allowed
delay;

_• V_ con( _t_ ) represents the number of system constraint violations up to time _t_, with penalties applied according to _**λ**_
for each violation;

_• U_ proc( _t_ ) is the utilisation of the computing capacity at the
satellite’s edge server, calculated as:


      - _N_
_i_ =1 _[x][i,j][f]_ _i_ [ ES]
_U_ proc( _t_ ) = _F_ [max] _,_ (12)

where _xi,j_ is the offloading portion, _fi_ [ES] is the allocated
processing rate for task offloaded from the _i_ -th GU,
and _F_ [max] is the maximum computing capacity of the
satellite’s edge server.


_2) Action space:_ The action space _A_ consists of the following decisions made by the agent


_•_ Offloading portion _xi,j_ ( _t_ ): The portion of the task offloaded from the _i_ -th GU to the _j_ -th UAV;

_•_ Bandwidth allocation _bi,j_ ( _t_ ): The bandwidth allocated to
the _i_ -th GU for communication with the _j_ -th UAV;

_•_ Satellite bandwidth allocation _bj_ ( _t_ ): The bandwidth allocated to the _j_ -th UAV for air-to-space communications.



s.t. _Ei_ ( _xi,j, bi,j_ ) _≤_ _Ei_ [max] _, ∀i,_


_N_

 - _bi,j ≤_ _Bj_ [max] _, ∀j,_


_i_ =1


_M_

 - _bj ≤_ _B_ [SAT] _,_


_j_ =1

_ri,j ≥_ _ri,j_ [min] _[,][ ∀][i, j,]_

_rj ≥_ _rj_ [min] _, ∀j,_



_N_



_i_ =1



_M_

- _xi,jfi_ [ES] _≤_ _F_ [max] _._


_j_ =1



In (10), (10b) represents the constraint of the energy budget
of the GUs. Constraints (10c) and (10d) are constraints for
the bandwidth allocations of GU-to-UAV links and UAV-tosatellite links, respectively. Constraints (10e) and (10f) are the
QoS requirements for the transmission rates. Lastly, constraint
(10g) guarantees the computing capacity of the satellite’s edge
server against exceeding maximum setting.


III. PROPOSED DDPG-BASED SOLUTION


It is obvious that the problem given in (10) comprises of
non-linearities, non-convexity, coupled constraints, and binary
indicators in the objective function, which make the problem
challenging for classical optimisation methods to find optimal solutions effectively. In contrast, DRL offers a flexible,
scaleable, and adaptive approach to learning optimal policies
in dynamic environments, making it an attractive solution for
solving the presented problem. By leveraging exploration and
function approximation, DRL can find near-optimal solutions
that meet the problem’s constraints while maximising the
task offloading ratio. Therefore, we propose a DRL-based
optimisation solution to tackle the formulated problem. More
specifically, the optimisation variables of the problem include
the task offloading portions and the bandwidth allocations,
which are all continuous variables. Consequently, the deep
deterministic policy gradient (DDPG) algorithm is selected to
develop the solution for this paper.


_A. Reinforcement Learning Representation_


We are in the position of transforming the original problem (10) into a problem that can be solved by DRL-based
algorithms. To solve a problem with DRL algorithms, the
optimisation problem needs to be reformulated as a Markov
decision process (MDP) formulation, including state space ( _S_ ),



Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY MADRAS. Downloaded on September 04,2025 at 03:05:46 UTC from IEEE Xplore. Restrictions apply.

© 2025 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,

but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.


This article has been accepted for publication in IEEE Internet of Things Journal. This is the author's version which has not been fully edited and

content may change prior to final publication. Citation information: DOI 10.1109/JIOT.2025.3598597



Thus, the action space _A_ is represented as


_A_ = _{xi,j_ ( _t_ ) _, bi,j_ ( _t_ ) _, bj_ ( _t_ ) _}∀i,j._ (13)


where _xi,j_ ( _t_ ) _∈_ [0 _,_ 1] represents the offloading portion, and
_bi,j_ ( _t_ ) _∈_ [0 _, Bj_ [max] ], _bj_ ( _t_ ) _∈_ [0 _, B_ [SAT] ] represents the
bandwidth allocations at the time step _t_ .
_3) Reward function:_ The reward _rt_ at time step _t_ is designed to encourage efficient task offloading, minimise latency,
and reduce energy consumption. The reward is calculated as



_M_

- ( _ϕixi,j −_ _λ_ EΨE _−_ _λ_ BΨB _−_ _λ_ rΨr _−_ _λ_ FΨF) _,_ (14)


_j_ =1



6


predicted _Q_ -value _Q_ ( _st, at|φ_ ) and the target _Q_ -value _yt_, given
by

           _L_ ( _φ_ ) = E ( _yt −_ _Q_ ( _st, at|φ_ )) [2][�] _._ (17)


It is important to note that, in the implementation of DDPG
algorithm, the replay buffer is a crucial component. The replay
buffer works as a memory buffer that stores the agent’s experiences from interacting with the environment. Each experience
is stored in the relay buffer as a tuple ( _st, at, rt, st_ +1). By
sampling random mini-batches from the buffer, DDPG trains
more effectively, reusing valuable experiences from previous
interactions. In summary, the proposed DDPG-based algorithm
for solving the problem formulated in (10) is provided in
Algorithm 1.


**Algorithm 1** : Proposed DDPG-based Algorithm for Solving
**P1** (10).

1: Initialise actor and critic networks i.e., _µ_ ( _s|θ_ ) and
_Q_ ( _s, a|φ_ ), with random trainable parameters _θ_ and _φ_,
respectively.
2: Initialise target networks _µ_ _[′]_ and _Q_ _[′]_ with trainable param_′_ _′_
eters _θ_ _←_ _θ_, _φ_ _←_ _φ_, respectively;
3: Initialise a replay buffer _R_ ;
4: Initialise Ornstein-Uhlenbeck noise _O_ for exploration;
5: **for** each episode **do**
6: Initialise a random process _O_ for action exploration;
7: Receive initial state _s_ 1;
8: **for** each step _t_ **do**
9: Select action _at_ = _µ_ ( _st|θ_ ) + _Ot_ (with noise for
exploration);
10: Execute _at_ and calculate _rt_ and _st_ +1;
11: Store ( _st, at, rt, st_ +1) in _R_ ;
12: Sample _Nb_ transitions ( _sℓ, aℓ, rℓ, s′ℓ_ +1) _′_ from _R_ ;
13: Set _yℓ_ = _rℓ_ + _γQ_ _[′]_ ( _sℓ_ +1 _, µ_ _[′]_ ( _sℓ_ +1 _|θ_ ) _|φ_ );
14: Update critic by minimising:



_rt_ =



_N_



_i_ =1



where:

_• ϕi_ is an indicator function that equals 1 if the total task
latency _Li ≤_ _Di_, and 0 otherwise, defined in (9);

_• xi,j_ is the offloaded portion generated by the _i_ -th GU;

_• λ_ E _, λ_ B _, λ_ r and _λ_ F are introduced weighting factors for
penalising energy consumption, bandwidth budget, minimum transmission rates, and the satellite’s computing
budget, respectively;

_•_ ΨE, ΨB, Ψr, ΨE present how much the constraints in (10)
are violated.

By designing the reward function in this way, the agent
is encouraged to maximise the reliable task offloading portion while penalising high energy consumption, task latency,
exceeding bandwidth budget and computing capacity, thereby
efficiently finding the optimal solution for the original problem
(10).


_B. Implementation of the Proposed DDPG-based Solution_


The proposed solution is constructed from the DDPG
algorithm, implemented with the framework of actor-critic
networks. The actor network is responsible for mapping the
current state of the environment to a continuous action, which
is fully connected and consists of three layers: the input layer,
hidden layer, and output layer. The hidden layer in this network
works as an approximator for the policy function. On the other
hand, the critic network evaluates how good a particular action
is for a given state by estimating the _Q_ -value (i.e., the expected
cumulative reward), expressed as (15) [30]. It takes both the
current state and the action as input, and outputs a scalar value.



_L_ = [1]

_Nb_


15: Update the actor:




- ( _yℓ_ _−_ _Q_ ( _sℓ, aℓ|φ_ )) [2] ;


_∀ℓ_



_∇θJ ≈_ [1]

_Nb_




- _∇aQ_ ( _s, a|φ_ ) _|a_ = _µ_ ( _s_ ) _∇θµ_ ( _s|θ_ );


_∀i_




      
_st_ = _s, at_ = _a_
�����



_Q_ _[π]_ ( _st, at_ ) = E




- _∞_

- _γ_ _[k]_ _rt_ + _k_


_k_ =0



_._ (15)



16: Update target networks:


_′_ _′_
_φ_ _←_ _τφ_ + (1 _−_ _τ_ ) _φ_ ;


_′_ _′_
_θ_ _←_ _τθ_ + (1 _−_ _τ_ ) _θ_ _._


17: **end for**
18: **end for**


IV. PROPOSED CLUSTER-BASED MULTI-AGENT DDPG
SOLUTION


To enable scalable and decentralised decision-making across
multiple UAVs, we develop a cluster-based multi-agent deep
deterministic policy gradient (CMADDPG) framework in this
section, where each UAV operates as an independent DDPG
agent managing the users within its assigned cluster. By
allowing each agent to learn its own policy while considering
the interactions with other agents, CMADDPG facilitates the



Thus, the _Q_ -value _Q_ ( _st, at_ ) represents the expected cumulative reward for taking action _at_ in state _st_, considering the
agent’s future states and actions under the current policy.
During the training process, the _Q_ -value is updated in
DDPG using the follow equation:


_′_
_yt_ = _rt_ + _γQ_ _[′]_ ( _st_ +1 _, π_ _[′]_ ( _st_ +1) _|φ_ ) _,_ (16)


where _yt_ is the target _Q_ -value, the discount factor; _′_ _γ_ is
the discount factor, and _Q_ _[′]_ ( _st_ +1 _, π_ _[′]_ ( _st_ +1) _|φ_ ) is the _Q_ -value
predicted by the _target critic network_ for the next state _st_ +1
and action _π_ _[′]_ ( _st_ +1), using the _target actor network π_ _[′]_ . The
critic network is trained by minimising the loss between the



Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY MADRAS. Downloaded on September 04,2025 at 03:05:46 UTC from IEEE Xplore. Restrictions apply.

© 2025 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,

but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.


This article has been accepted for publication in IEEE Internet of Things Journal. This is the author's version which has not been fully edited and

content may change prior to final publication. Citation information: DOI 10.1109/JIOT.2025.3598597



learning of coordinated behaviours in shared environments. In
our design, the centralised training and distributed execution
(CTDE) mechanism is adopted, where agents have access to
global information during training but make decisions based
only on their local observations during execution. The proposed CMADDPG framework enables UAV agents to optimise
task offloading and bandwidth allocation decisions collaboratively, enhancing the system’s reliability and scalability in
dynamic SAGIN scenarios. The following subsections provide
a detailed development of the proposed CMADDPG solution,
including the design of the its components, the training and
execution procedures under the CTDE framework, and the
overall algorithm.


_A. Key Components of The CMADDPG Framework_


Fig. 2. A diagram of the proposed CMADDPG approach for solving the
problem (10).


The diagram in Figure 2 illustrates the core components
and training flow of the CMADDPG approach, applied to task
offloading in a SAGIN environment. Each UAV (agent _m_ )
operates with a unique actor-critic pair, enabling decentralised
policy learning while leveraging centralised critics and shared
experiences for coordinated task offloading decisions.
_1) Environment and observations:_ The environment represents the overall system state **s** = _{s_ 1 _, s_ 2 _, . . ., sM_ _}_, where
each _sm_ corresponds to the local observation of agent _m_ . At
each time step, agent _m_ uses its actor network _µθm_ to map its
observation _sm_ to an action _am_, expressed as


_am_ = _µθm_ ( _sm_ ) + _O,_ (18)


where _O_ represents exploration noise added during training. The joint action **a** = _{a_ 1 _, a_ 2 _, . . ., aM_ _}_ affects the environment, resulting in a new state _s_ _[′]_ and rewards _r_ =
_{r_ 1 _, r_ 2 _, . . ., rM_ _}_ for each agent.
_2) Decentralised actor networks:_ Each agent _m_ has an
actor network _µθm_ that maps its own observations _sm_ to
actions _am_, forming a policy _πm_ = _µθm_ ( _sm_ ) that aims
to maximise its cumulative expected return. The objective
function for the actor network of agent _m_ is formulated as


_J_ ( _θm_ ) = E _s,a∼R_ [ _Qφm_ ( _s, a_ )] _,_ (19)



7


where _Qφm_ is the centralised critic for agent _m_ that estimates
the expected return of taking action _am_ while considering
the actions of other agents in the joint action space **a** =
_{a_ 1 _, a_ 2 _, . . ., aM_ _}_ .
To improve the actor policy _µθm_, agent _m_ performs gradient
ascent on _J_ ( _θm_ ), updating _θm_ to maximise the expected
return, given by


_∇θm_ _J_ ( _θm_ )



where the chain rule is applied to calculate the gradient
of _J_ ( _θm_ ) with respect to _θm_ . The first term, _∇θmµθm_ ( _sm_ ),
computes the sensitivity of the action _am_ with respect to
changes in the actor parameters _θm_, while the second term,
_∇amQφm_ ( _s, a_ ), assesses how changes in _am_ impact the expected return.
The update rule for the actor network parameters _θm_ is then
given by

_θm ←_ _θm_ + _αθm∇θmJ_ ( _θm_ ) _,_ (21)


where _αθm_ is the learning rate for the actor. This update
encourages each agent _m_ to choose actions that maximise its
expected cumulative return based on the feedback from the
centralised critic.
Although each agent’s actor is decentralised and optimises
only its own policy, the use of centralised critics enables
indirect coordination by incorporating the effects of the actions
of other agents. This structure promotes cooperation in multiagent settings where agents’ actions are interdependent, such
as in task offloading scenarios where agents must balance
individual and global objectives.
_3) Centralised critic networks:_ Each agent _m_ has a centralised critic network _Qφm_ that estimates the Q-value of the
state-action pair ( _s, a_ ) by considering the joint actions of all
agents, **a** = _{a_ 1 _, a_ 2 _, . . ., aM_ _}_, as well as the global state _s_ .
The objective of the critic network for agent _m_ is to minimise
the temporal difference (TD) error, which is defined as the
discrepancy between the current Q-value estimate and the
target Q-value.
The target Q-value for agent _m_, denoted by _ym_, is computed
using the reward _rm_ received by agent _m_ and the Q-value from
the target critic network _Qφ′m_, which is updated more slowly
to stabilise learning. The target for each agent _m_ is given by


_ym_ = _rm_ + _γQφ′m_ [(] _[s][′][, a][′]_ [)] _[,]_ (22)


where _s_ _[′]_ is the next state, _γ_ is the discount factor, and
_a_ _[′]_ = _{µθ_ 1 _[′]_ [(] _[s]_ 1 _[′]_ [)] _[, . . ., µ][θ]_ _M_ _[′]_ [(] _[s]_ _M_ _[′]_ [)] _[}]_ [ is the set of actions chosen]
by each agent’s target actor network _µθm′_ [for the next state] _[ s][′]_ [.]
This formulation allows each critic to anticipate the long-term
returns considering future actions of all agents.
The critic network parameters _φm_ are updated by minimising the critic loss _L_ ( _φm_ ), which is defined as the mean squared
error between the current Q-value and the target Q-value as
follows:


             _L_ ( _φm_ ) = E( _s,a,r,s′_ ) _∼R_ ( _Qφm_ ( _s, a_ ) _−_ _ym_ ) [2][�] _._ (23)


Minimising this loss function adjusts _φm_ to reduce the TD




_,_ (20)



= E _s∼R_




_∇θm_ _µθm_ ( _sm_ ) _∇am_ _Qφm_ ( _s, a_ )��� _am_ = _µθm_ ( _sm_ )



Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY MADRAS. Downloaded on September 04,2025 at 03:05:46 UTC from IEEE Xplore. Restrictions apply.

© 2025 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,

but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.


This article has been accepted for publication in IEEE Internet of Things Journal. This is the author's version which has not been fully edited and

content may change prior to final publication. Citation information: DOI 10.1109/JIOT.2025.3598597



error, thereby improving the accuracy of _Qφm_ in predicting
the expected cumulative reward for each joint state-action pair
( _s, a_ ).
The update rule for the critic parameters _φm_ is derived from
the gradient of the loss function _L_ ( _φm_ ) with respect to _φm_,
given by

_φm ←_ _φm −_ _αφm∇φmL_ ( _φm_ ) _,_ (24)


where _αφm_ is the learning rate for the critic. This gradientbased update ensures that the critic’s Q-value estimates more
closely align with the target values, enhancing the critic’s
ability to assess the expected return for various action combinations in multi-agent settings.
_4) Target networks for stability:_ Each actor and critic
network has a corresponding target network, denoted as _µθm′_
and _Qφ′m_ for agent _m_ . These target networks are slowly
updated versions of the main networks to ensure stability
during training by providing fixed targets. The target updates
are performed using soft updates, where the target parameters
_θm_ _[′]_ [and] _[ φ]_ _m_ _[′]_ [are updated towards the main parameters] _[ θ][m]_ [and]
_φm_ as follows:



_θm_ _[′]_ _[←]_ _[τθ][m]_ [+ (1] _[ −]_ _[τ]_ [)] _[θ]_ _m_ _[′]_ _[,]_

_φ_ _[′]_ _m_ _[←]_ _[τφ][m]_ [+ (1] _[ −]_ _[τ]_ [)] _[φ][′]_ _m_ _[,]_



(25)


(26)



8


shared experiences. Each UAV (agent _m_ ) independently optimises its task offloading strategy while indirectly considering
other agents’ actions through the centralised critic. By integrating these components, CMADDPG enables efficient and
coordinated decision-making in multi-agent systems with interdependent objectives. The overall algorithm is summarised
as the Algorithm 2.


_C. Complexity Discussion_


The computational complexity of single-agent DDPG is
dominated by one actor and one critic network update per
training step: using a mini-batch of size _B_, the time complexity is _O_ - _B_ ( _F_ actor + _F_ critic)� _,_ where _F_ actor and _F_ critic denote
the cost of a forward-backward pass through the actor and
critic, respectively. The memory footprint includes the network parameters and the replay buffer, i.e., _O_ - _Pθ_ + _Pφ_ +
_B_ ( _S_ + _A_ )� _,_ with _Pθ_ and _Pφ_ the parameter counts and _S_,
_A_ the state/action dimensions. In contrast, MADDPG maintains _M_ actor–critic pairs and each critic consumes the joint
state–action of all agents, so its per-update time complexity
grows to _O_ - _B M_ ( _F_ actor+ _F_ critic _[∗]_ [)] - _,_ where _F_ critic _[∗]_ [accounts for the]
enlarged - _M S_ + _M A_ - input, and its memory cost increases to
_O_ - _M_ ( _Pθ_ + _Pφ_ )+ _B M_ ( _S_ + _A_ )� _._ Hence, whereas DDPG scales
linearly with network size, MADDPG incurs an approximately
quadratic increase in compute with respect to _M_, trading
efficiency for the ability to learn coordinated multi-agent
policies.


V. SIMULATION RESULTS AND DISCUSSIONS


_A. Parameter Settings_


For simulations, we consider a system model that consists
of _M_ = 3 UAVs, _N_ = _{_ 15 _,_ 21 _,_ 24 _}_ GUs. We assume
that the assignments of UAVs and GUs are conducted in
advance, with each UAV serving the same number of GUs,
e.g., each UAV serves 5 GUs within its coverage area for
the scenario where there are _N_ = 15 GUs. The simulations
are conducted in a Python environment, ultilising well-known
packages such as pytorch, gymnasium, pettingzoo

[31], pandas, and matplotlib to implement the proposed solutions and visualise numerical results.
For training the DDPG model, we set the learning rate of
the actor to 10 _[−]_ [5] while the learning rate of critic is 10 _[−]_ [4] .
The discount factor is set to _γ_ = 0 _._ 99 and the factor for target
network update is _τ_ = 0 _._ 05. The batch size for sampling in
the training is set to 256 and the maximum size of the reply
buffer is 10 [6] . Other communication and computing parameters
are provided in Table II.


_B. Numerical Results_


_1) Training performance of Algorithm 1:_ The training
performance of the proposed Algorithm 1 is displayed in
Fig. 3, where the episode reward is plotted against the training
episodes for two different scenarios _N_ = 15 GUs and _N_ = 21
GUs, both with _M_ = 3 UAVs. The results show that in
both scenarios, the algorithm demonstrates an upward trend
in rewards as the training progresses, indicating successful



where _τ_ is the soft update rate.
_5) Shared replay buffer:_ A shared replay buffer _R_ stores
the experiences ( **s, a, r, s’** ) from all agents, which can be
expressed as


( **s, a, r, s’** ) = _{_ ( _s_ 1 _, a_ 1 _, r_ 1 _, s_ _[′]_ 1 [)] _[, . . .,]_ [ (] _[s][M]_ _[, a][M]_ _[, r][M]_ _[, s]_ _M_ _[′]_ [)] _[}][.]_

(27)
By training on experiences from the joint action space, each
agent’s critic can learn about interactions with other agents,
improving the capacity to cooperate and coordinate in complex
tasks like task offloading.


_B. Training Flow and Proposed CMADDPG Algorithm_


For each agent _m_, the target Q-value in the critic network
is computed as


_ym_ = _rm_ + _γQφ′m_ ( _s_ _[′]_ _, a_ _[′]_ ) _,_ (28)

where _a_ _[′]_ = _{µθ_ 1 _[′]_ [(] _[s]_ 1 _[′]_ [)] _[, . . ., µ][θ]_ _M_ _[′]_ [(] _[s]_ _M_ _[′]_ [)] _[}]_ [ is the set of next actions]
from the target actors. The critic loss is minimised to match
the Q-value with this target, expressed as


             _L_ ( _φm_ ) = E( _s,a,r,s′_ ) _∼R_ ( _Qφm_ ( _s, a_ ) _−_ _ym_ ) [2][�] _._ (29)


Each agent _m_ updates its actor by maximising the expected
Q-value over its action as follows:



1
_∇θm_ _J_ ( _θm_ ) _≈_ _|R|_



_s_ - _∈R_ _∇θm_ _µθm_ ( _sm_ ) _∇am_ _Qφm_ ( _s, a_ )��� _am_ = _µθm_ ( _sm_ ) _[.]_


(30)



This gradient update encourages each actor to select actions
that maximise the estimated Q-value, balancing local rewards
with cooperative behaviour.
Overall, this CMADDPG framework combines decentralised actor policies with centralised critic structures and



Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY MADRAS. Downloaded on September 04,2025 at 03:05:46 UTC from IEEE Xplore. Restrictions apply.

© 2025 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,

but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.


This article has been accepted for publication in IEEE Internet of Things Journal. This is the author's version which has not been fully edited and

content may change prior to final publication. Citation information: DOI 10.1109/JIOT.2025.3598597



9



**Algorithm 2** : Proposed CMADDPG Algorithm for Solving
Problem (10).

1: **Initialisation:**
2: Initialise actor network _µθm_ and critic network _Qφm_ for
each agent _m_ with parameters _θm_ and _φm_, respectively.
3: Initialise target networks _µθm′_ and _Qφ′m_ for each agent _m_
with parameters _θm_ _[′]_ _[←]_ _[θ][m]_ [and] _[ φ][′]_ _m_ _[←]_ _[φ][m]_ [.]
4: Initialise a shared replay buffer _R_ to store experiences
from all agents.

5: **for** each episode **do**
6: Reset the environment to obtain the initial state **s** =
_{s_ 1 _, s_ 2 _, . . ., sM_ _}_, where _M_ is the number of agents.
7: **for** each agent _m_ **do**
8: Initialise episode reward _Rm_ = 0.
9: **end for**



10: **for** each time step _t_ **do**
11: **for** each agent _m_ **do**
12: Select an action _am_ = _µθm_ ( _sm_ ) + _O_, where _O_ is
noise for exploration.
13: **end for**
14: Execute actions _{a_ 1 _, a_ 2 _, . . ., aM_ _}_ in the environment.

15: Observe the next state _s_ _[′]_, rewards _{r_ 1 _, r_ 2 _, . . ., rM_ _}_,
and done signals.
16: Store the transition ( **s, a, r, s’** ) in the replay buffer
_R_ .
17: Update _s ←_ _s_ _[′]_ .
18: **if** update step (every _K_ steps) **then**

19: Sample a mini-batch of experiences ( **s, a, r, s’** )
from the replay buffer _R_ .
20: **for** each agent _m_ **do**
21: _**Critic Update:**_
22: Compute the target action for each agent _j_ as
_a_ _[′]_ _j_ [=] _[ µ][θ]_ _j_ _[′]_ [(] _[s]_ _j_ _[′]_ [)][.]
23: Compute the target Q-value _ym_ = _rm_ +
_γQφ′m_ ( _s_ _[′]_ _, a_ _[′]_ ), where _a_ _[′]_ = _{a_ _[′]_ 1 _[, . . ., a][′]_ _M_ _[}]_ [.]
24: Update the critic by minimizing the loss:
_L_ ( _φm_ ) = ( _Qφm_ ( _s, a_ ) _−_ _ym_ ) [2] .

25: _**Actor Update:**_
26: Update the actor by maximizing
the expected Q-value: _∇θm_ _J_ _≈_
1         _|R|_ _s∈R_ _[∇][θ][m]_ _[µ][θ][m]_ [(] _[s][m]_ [)] _[∇][a][m]_ _[Q][φ][m]_ [(] _[s, a]_ [)] _[|][a]_ _m_ [=] _[µ]_ _θm_ [(] _[s]_ _m_ [)][.]

27: **end for**
28: **for** each agent _m_ **do**
29: Update target networks:
_θm_ _[′]_ _[←]_ _[τθ][m]_ [+ (1] _[ −]_ _[τ]_ [)] _[θ]_ _m_ _[′]_ [,]
_φ_ _[′]_ _m_ _[←]_ _[τφ][m]_ [+ (1] _[ −]_ _[τ]_ [)] _[φ][′]_ _m_ [.]

30: **end for**
31: **end if**
32: **end for**
33: **end for**


learning. For _N_ = 15 GUs, the algorithm converges faster,
reaching a stable reward by around episode 100, with minimal
fluctuations. In contrast, the case with _N_ = 21 GUs exhibits
a slower convergence rate, stabilising around episode 250.
This difference in convergence speed can be attributed to the
increased complexity of managing more users, which adds to


|Parameters|Value|
|---|---|
|Distance from UAVs to GU<br>Satellites’ altitude<br>SRF model<br>Noise spectral density<br>Path-loss exponent<br>Rician K-factor<br>Task size<br>Required CPU cycles of ta<br>Delay tolerance<br>Maximum energy consump<br>Energy consumption param<br>Transmission power of GU<br>Transmission power of UA<br>Maximum bandwidth for e<br>Maximum bandwidth for th|s<br>_di,j ∼U_(400_,_ 500) m<br>780 km<br>(_ω, δ, ϵ_) = (5_e−_4_,_ 0_._063_,_ 2)<br>_N_0 =_ −_174 dBm/Hz<br>_α_ = 2<br>_K_ = 5<br>_Si ∼U_(100_,_ 500) KB<br>    sks<br>_Ci ∼U_(800_,_ 1200) megacycles.<br>_Di ∼U_(2_,_ 5) s<br>  tion of GU<br>_E_max<br>_i_<br>_∼U_(1_,_ 1_._5) J<br>  eter<br>_θ_ = 10_−_27 Watt.s3/cycle3<br>_Pi_ = 20 dBm<br>   V<br>_Pj_ = 37 dBm<br>   ach UAV<br>_B_max<br>_j_<br>= 20 MHz<br>   e satellite<br>_B_SAT = 100 MHz.|


|Col1|Col2|Col3|
|---|---|---|
||||
||||
||||
||||
||~~N = 15 GU~~<br>N = 21 GU|~~   s~~<br>   s|



Fig. 3. Training performance over time of Algorithm 1 for the scenarios of
_M_ = 3 UAVs with _N_ = _{_ 15 _,_ 21 _}_ GUs.


the challenge of reliable task offloading. However, in both
cases, the algorithm achieves stable and consistent rewards
as the training progresses, highlighting its robustness and
effectiveness in handling varying numbers of ground users.
The shaded regions around the curves represent the standard
deviation, showing that the variability in performance decreases as the number of episodes increases, further indicating
stable learning outcomes.

_2) Training performance of the multi-agent approach:_
Fig. 4 illustrates the training performance of the proposed
multi-agent approach (Algorithm 2) for the scenario with _M_ =
3 UAVs and _N_ = 21 GUs. Initially, all agents exhibit unstable
learning behaviours, with sharp drops in reward, especially
within the first 200 episodes. These fluctuations reflect the
increased task complexity and decision conflicts arising from
a denser network environment, where more users compete for
limited communication and computation resources. Notably,
UAV 0 and UAV 1 experience multiple periods of degraded
performance between episodes 200 and 600, likely due to
difficulties in policy coordination or local optima in the
shared environment. In contrast, UAV 2 shows a relatively



TABLE II
SIMULATION PARAMETERS [28], [29], [32].







Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY MADRAS. Downloaded on September 04,2025 at 03:05:46 UTC from IEEE Xplore. Restrictions apply.

© 2025 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,

but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.


10





This article has been accepted for publication in IEEE Internet of Things Journal. This is the author's version which has not been fully edited and

content may change prior to final publication. Citation information: DOI 10.1109/JIOT.2025.3598597






|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|||||||||
|||||||||
|||||||||
|||||||||
||||||~~M =~~<br>M =<br>M =|<br>  4<br>  6||





|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
||||||~~UAV_0~~<br>|
|~~0~~<br>~~200~~<br>~~400~~<br>~~600~~<br>~~800~~<br>~~100~~<br>Episode Index<br>UAV_1<br>~~UAV_2~~|~~0~~<br>~~200~~<br>~~400~~<br>~~600~~<br>~~800~~<br>~~100~~<br>Episode Index<br>UAV_1<br>~~UAV_2~~|~~0~~<br>~~200~~<br>~~400~~<br>~~600~~<br>~~800~~<br>~~100~~<br>Episode Index<br>UAV_1<br>~~UAV_2~~|~~0~~<br>~~200~~<br>~~400~~<br>~~600~~<br>~~800~~<br>~~100~~<br>Episode Index<br>UAV_1<br>~~UAV_2~~|~~0~~<br>~~200~~<br>~~400~~<br>~~600~~<br>~~800~~<br>~~100~~<br>Episode Index<br>UAV_1<br>~~UAV_2~~|~~0~~<br>~~200~~<br>~~400~~<br>~~600~~<br>~~800~~<br>~~100~~<br>Episode Index<br>UAV_1<br>~~UAV_2~~|


Fig. 4. Training performance over time of Algorithm 2 for the scenarios of
_M_ = 3 UAVs with _N_ = 21 GUs.
















|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||~~a=1e-0~~<br>a=5e~~-~~0<br>|~~, c=0.00~~<br>5, c=0.00<br>|<br> 5<br>||
|~~0~~<br>~~200~~<br>~~400~~<br>~~600~~<br>~~800~~<br><br>~~a=1e-05, c=0.00~~<br>a=5e~~-~~05, c=0.00|~~0~~<br>~~200~~<br>~~400~~<br>~~600~~<br>~~800~~<br><br>~~a=1e-05, c=0.00~~<br>a=5e~~-~~05, c=0.00|~~0~~<br>~~200~~<br>~~400~~<br>~~600~~<br>~~800~~<br><br>~~a=1e-05, c=0.00~~<br>a=5e~~-~~05, c=0.00|~~0~~<br>~~200~~<br>~~400~~<br>~~600~~<br>~~800~~<br><br>~~a=1e-05, c=0.00~~<br>a=5e~~-~~05, c=0.00|~~000~~<br>~~ 01~~<br> 05|~~000~~<br>~~ 01~~<br> 05|



Fig. 5. Training performance over time of Algorithm 2 for the scenarios of
_M_ = 3 UAVs with _N_ = 15 GUs with varying learning rates of actor and
critic networks.


faster recovery and smoother trajectory toward convergence.
Despite the presence of intermediate performance instability,
all agents eventually converge to consistently higher reward
values by the end of training, indicating the effectiveness of
the proposed learning framework. This result demonstrates that
the multi-agent system is capable of adapting to complex and
highly coupled environments, maintaining robust convergence
behaviour under heavier ground user loads.
_3) Training performance of Algorithm 2 with varying learn-_
_ing rates:_ Fig. 5 presents the training performance of the
proposed multi-agent approach (Algorithm 2) under different
combinations of actor and critic learning rates, where _M_ = 3
UAVs are deployed to serve _N_ = 15 ground users. Each
curve corresponds to a specific configuration, denoted by _a_
for the actor learning rate and _c_ for the critic learning rate.
The plot highlights how sensitive the learning dynamics are
to these hyperparameters. The setting with _a_ = 1e _−_ 5 and
_c_ = 0 _._ 001 achieves the best and most stable performance,
with fast convergence and minimal fluctuations. Conversely,
configurations with either too large or too small learning rates
(e.g., _a_ = 5e _−_ 5 _, c_ = 0 _._ 005 or _a_ = 1e _−_ 5 _, c_ = 0 _._ 0001)
result in slower convergence and higher instability throughout
training. These results emphasise the importance of carefully
tuning both actor and critic learning rates to ensure stable
learning and optimal policy performance in complex, multiagent environments.



Fig. 7. The effectiveness of the proposed Algorithm 1 in maximising the
reliable task offloading portions with different settings of _Bj_ [max] and maximum
CPU required by the computational tasks in the scenario of _N_ = 15 GUs.
Here, “10-OPT” represents the optimal bandwidth allocation scheme with
_Bj_ [max] = 10 MHz and “30-EBA” represents the equal bandwidth allocation
scheme with _Bj_ [max] = 30 MHz.


_4) Training performance with varying of number of UAV_
_agents:_ Fig. 6 illustrates the training performance of the
proposed multi-agent approach (Algorithm 2) under different
numbers of UAV agents, specifically _M_ = 3 _,_ 4 _,_ 6, while
serving _N_ = 24 ground users. The plot presents the average
reward per agent across training episodes and highlights the
influence of the number of cooperative agents on learning
dynamics. As observed, the scenario with _M_ = 6 consistently
achieves the highest and most stable reward, demonstrating
that greater agent coordination significantly enhances learning
efficiency and task offloading reliability. The _M_ = 4 case also
shows strong convergence behaviour, though with intermittent
fluctuations during the learning process. In comparison, the
_M_ = 3 scenario converges more slowly and stabilises at a
lower average reward, indicating reduced capacity to manage
task allocation and bandwidth distribution with fewer agents.
These results validate the scalability and effectiveness of the
proposed multi-agent framework in adapting to varying system
configurations and user demand levels.

_5) Effectiveness of the proposed solution in maximising_
_reliable task offloading:_ To demonstrate the effectiveness



Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY MADRAS. Downloaded on September 04,2025 at 03:05:46 UTC from IEEE Xplore. Restrictions apply.

© 2025 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,

but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.


This article has been accepted for publication in IEEE Internet of Things Journal. This is the author's version which has not been fully edited and

content may change prior to final publication. Citation information: DOI 10.1109/JIOT.2025.3598597






|Col1|Col2|Col3|
|---|---|---|
||||
||||
||~~11~~<br>|~~11~~<br>|



Fig. 8. The effectiveness of the proposed Algorithm 2 in minimising the
latency with varying required CPU cycles compared to baseline benchmarks.


of the proposed solution, we conducted simulations with
different settings for the UAV’s bandwidth budget and the
required CPU cycles of the tasks. The bar chart in Fig. 7
illustrates the superior performance of the proposed solution
in maximising reliable task offloading portions under various bandwidth allocation schemes and CPU requirements
for computational tasks, compared to the benchmark scheme.
The comparison is made between Max _Ci_ = 900 megacycles
and Max _Ci_ = 1200 megacycles. The results demonstrate
that the optimal allocation schemes outperform the equal
allocation strategy. For instance, the 30-OPT scheme achieves
the highest reliable offloading portion, reaching nearly 1.0
for Max _Ci_ = 900, while 30-EBA shows significantly lower
performance in both cases. This highlights the efficiency of the
proposed solution in utilising resources to improve reliable
task offloading. In addition, Fig. 7 demonstrates how the
UAV’s bandwidth budget affects the offloading process. As
shown in the figure, increasing the bandwidth budget for GUto-UAV communication significantly enhances the reliability
of task offloading, allowing a higher portion of tasks to be
fully offloaded to complete the task within the delay tolerance.
_6) Effectiveness of the proposed solution in minimising the_
_latency:_ Fig. 8 illustrates the effectiveness of the proposed
Algorithm 2 in reducing average per-GU latency under varying
maximum CPU cycle demands, compared to two baseline
approaches: Equal-Split and Random. As the required CPU
cycles increase from 1000 to 1200, the average latency for
all methods increases accordingly due to the greater computational workload. However, Algorithm 2 consistently achieves
lower latency than both baselines across all CPU demand
levels. This demonstrates its ability to allocate resources more
efficiently by jointly optimising task offloading and bandwidth
distribution, even under more demanding computational requirements. In contrast, the Equal-Split and Random strategies
exhibit higher and less stable latency, highlighting the benefits
of adaptive and learning-based decision-making in complex
edge computing scenarios.


VI. CONCLUSION


In conclusion, we have investigated the optimal design of
task offloading and bandwidth allocation for reliability-driven



11


SAGIN-enabled edge computing. The proposed system model
captures the dynamic environment of computing demands, the
energy budgets of ground users, and the computing capacity
of the satellite’s edge server. A DRL-based solution has been
developed to jointly optimise the offloaded task portions and
bandwidth allocations, thereby enhancing the reliability of the
offloading process within the system. In addition to the singleagent DRL approach, we have further extended the solution
to a multi-agent framework, enabling decentralised decisionmaking across UAVs while maintaining coordinated system
performance through a centralised training and distributed execution mechanism. The effectiveness of the proposed methods
has been demonstrated through simulation results, showing
stable training convergence and a significant improvement in
the reliability of task offloading. As part of future work, we
aim to extend the current framework by incorporating realtime UAV mobility optimisation, such as trajectory control
and adaptive positioning, which will further enhance system
efficiency, resilience, and task reliability in dynamic SAGIN
environments.


REFERENCES


[1] Y. Liu, L. Jiang, Q. Qi, K. Xie, and S. Xie, “Online computation
offloading for collaborative space/aerial-aided edge computing toward
6G system,” _IEEE Trans. Veh. Technol._, vol. 73, no. 2, pp. 2495–2505,
Feb. 2024.

[2] T. Ma, H. Zhou, B. Qian, N. Cheng, X. Shen, X. Chen, and B. Bai,
“UAV-LEO integrated backbone: A ubiquitous data collection approach
for B5G Internet of remote things networks,” _IEEE J. Sel. Areas_
_Commun._, vol. 39, no. 11, pp. 3491–3505, Nov. 2021.

[3] D. Zhou, M. Sheng, J. Li, and Z. Han, “Aerospace integrated networks
innovation for empowering 6G: A survey and future challenges,” _IEEE_
_Commun. Surveys Tuts._, vol. 25, no. 2, pp. 975–1019, Apr. 2023.

[4] Q. Chen, Z. Guo, W. Meng, S. Han, C. Li, and T. Q. S. Quek, “A
survey on resource management in joint communication and computingembedded SAGIN,” _IEEE Commun. Surveys Tuts._, 2024.

[5] T. Do-Duy, D. V. Huynh, E. Garcia-Palacios, T.-V. Cao, V. Sharma, and
T. Q. Duong, “Joint computation and communication resource allocation
for unmanned aerial vehicle NOMA systems,” in _Proc. IEEE 28th_
_Int. Workshop Comput. Aided Modeling Design Commun. Links Netw._
_(CAMAD)_, Edinburgh, United Kingdom, Nov. 2023, pp. 290–295.

[6] Y. Gong, H. Yao, and A. Nallanathan, “Intelligent sensing, communication, computation, and caching for satellite-ground integrated networks,”
_IEEE Netw._, vol. 38, no. 4, pp. 9–16, Jul. 2024.

[7] B. Shang, Y. Yi, and L. Liu, “Computing over space-air-ground integrated networks: Challenges and opportunities,” _IEEE Netw._, vol. 35,
no. 4, pp. 302–309, Aug. 2021.

[8] J. He, N. Cheng, Z. Yin, C. Zhou, H. Zhou, W. Quan, and X.-H. Lin,
“Service-oriented network resource orchestration in space-air-ground
integrated network,” _IEEE Trans. Veh. Technol._, vol. 73, no. 1, pp. 1162–
1174, Jan. 2024.

[9] D. V. Huynh, S. R. Khosravirad, S. L. Cotton, O. A. Dobre, and T. Q.
Duong, “DRL-based optimisation for task offloading in space-air-ground
integrated networks: A reliability-driven approach,” in _Proc. IEEE Int._
_Conf. Commun. (ICC)_, Montreal, Canada, Jun. 2025.

[10] F. Tang, H. Hofner, N. Kato, K. Kaneko, Y. Yamashita, and M. Hangai,
“A deep reinforcement learning-based dynamic traffic offloading in
space-air-ground integrated networks (SAGIN),” _IEEE J. Sel. Areas_
_Commun._, vol. 40, no. 1, pp. 276–289, Jan. 2022.

[11] B. Cao, J. Zhang, X. Liu, Z. Sun, W. Cao, R. M. Nowak, and
Z. Lv, “Edge–cloud resource scheduling in space–air–ground-integrated
networks for internet of vehicles,” _IEEE Internet of Things J._, vol. 9,
no. 8, pp. 5765–5772, Apr. 2022.

[12] C. Huang, G. Chen, P. Xiao, Y. Xiao, Z. Han, and J. A. Chambers, “Joint
offloading and resource allocation for hybrid cloud and edge computing
in sagins: A decision assisted hybrid action space deep reinforcement
learning approach,” _IEEE J. Sel. Areas Commun._, vol. 42, no. 5, pp.
1029–1043, May 2024.



Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY MADRAS. Downloaded on September 04,2025 at 03:05:46 UTC from IEEE Xplore. Restrictions apply.

© 2025 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,

but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.


This article has been accepted for publication in IEEE Internet of Things Journal. This is the author's version which has not been fully edited and

content may change prior to final publication. Citation information: DOI 10.1109/JIOT.2025.3598597


[13] I. Leyva-Mayorga, M. Martinez-Gost, M. Moretti, A. P´erez-Neira,
M. Angel V´azquez, P. Popovski, and B. Soret, “Satellite edge computing [´]
for real-time and very-high resolution earth observation,” _IEEE Trans._
_Commun._, vol. 71, no. 10, pp. 6180–6194, Oct. 2023.

[14] Y. Gao, Z. Yan, K. Zhao, T. de Cola, and W. Li, “Joint optimization
of server and service selection in satellite-terrestrial integrated edge
computing networks,” _IEEE Trans. Veh. Technol._, vol. 73, no. 2, pp.
2740–2754, Feb. 2024.

[15] T. Q. Duong, L. D. Nguyen, T. T. Bui, K. D. Pham, and G. K.
Karagiannidis, “Machine learning-aided real-time optimized multibeam
for 6G integrated satellite-terrestrial networks: Global coverage for
mobile services,” _IEEE Netw._, vol. 37, no. 2, pp. 86–93, Apr. 2023.

[16] Z. Song, Y. Hao, Y. Liu, and X. Sun, “Energy-efficient multiaccess edge
computing for terrestrial-satellite internet of things,” _IEEE Internet of_
_Things J._, vol. 8, no. 18, pp. 14 202–14 218, Sep. 2021.

[17] C. Ding, J.-B. Wang, H. Zhang, M. Lin, and G. Y. Li, “Joint optimization
of transmission and computation resources for satellite and high altitude
platform assisted edge computing,” _IEEE Trans. Commun._, vol. 21, no. 2,
pp. 1362–1377, Feb. 2022.

[18] F. Chai, Q. Zhang, H. Yao, X. Xin, R. Gao, and M. Guizani, “Joint
multi-task offloading and resource allocation for mobile edge computing
systems in satellite IoT,” _IEEE Trans. Veh. Technol._, vol. 72, no. 6, pp.
7783–7795, Jun. 2023.

[19] L. Zhao, D. Wu, L. Zhou, and Y. Qian, “Radio resource allocation for
integrated sensing, communication, and computation networks,” _IEEE_
_Trans. Wireless Commun._, vol. 21, no. 10, pp. 8675–8687, Oct. 2022.

[20] S. Mahboob and L. Liu, “Revolutionizing future connectivity: A contemporary survey on AI-empowered satellite-based non-terrestrial networks
in 6G,” _IEEE Commun. Surveys Tuts._, vol. 26, no. 2, pp. 1279–1321,
Apr. 2024.

[21] P. Zhang, N. Chen, S. Shen, S. Yu, N. Kumar, and C.-H. Hsu, “Aienabled space-air-ground integrated networks: Management and optimization,” _IEEE Netw._, vol. 38, no. 2, pp. 186–192, Apr. 2024.

[22] M. Kim, H. Lee, S. Hwang, M. Debbah, and I. Lee, “Cooperative
multiagent deep reinforcement learning methods for UAV-aided mobile
edge computing networks,” _IEEE Internet of Things J._, vol. 11, no. 23,
pp. 38 040–38 053, Dec. 2024.

[23] D. Xiao, W. Ting, F. Qiang, Y. Chenhui, T. Tao, W. Lu, S. Yuanming, and
C. Mingsong, “Multi-agent reinforcement learning for dynamic resource
management in 6G in-X subnetworks,” _IEEE Trans. Wireless Commun._,
vol. 22, no. 3, pp. 1900–1914, Mar. 2023.

[24] H. S. Salman, P. Y. Min, T. Y. Kyaw, S. Walid, H. Zhu, and H. C. Seon,
“Satellite-based ITS data offloading & computation in 6G networks:
A cooperative multi-agent proximal policy optimization DRL with
attention approach,” _IEEE Trans. Mobile Comput._, vol. 23, no. 5, pp.
4956–4974, May 2024.

[25] L. Zhilong, Z. Jiayi, Z. Yong, and A. Bo, “Energy-efficient multiagent reinforcement learning for UAV trajectory optimization in cellfree massive MIMO networks,” _IEEE Trans. Wireless Commun._, 2025,
doi: 10.1109/TWC.2025.3550266.

[26] W. Zhiying, S. Gang, W. Yuhui, Y. fang, and N. Dusit, “Clusterbased multi-agent task scheduling for space-air-ground integrated
networks,” _IEEE_ _Trans._ _Cogn._ _Commun._ _Netw._, 2025, doi:
10.1109/TCCN.2025.3553297.

[27] W. Liu, B. Li, W. Xie, Y. Dai, and Z. Fei, “Energy efficient computation
offloading in aerial edge networks with multi-agent cooperation,” _IEEE_
_Trans. Wireless Commun._, vol. 22, no. 9, pp. 5725–5739, Sep. 2023.

[28] M.-H. T. Nguyen, T. T. Bui, L. D. Nguyen, E. Garcia-Palacios, H.-J.
Zepernick, H. Shin, and T. Q. Duong, “Real-time optimized clustering
and caching for 6G satellite-UAV-terrestrial networks,” _IEEE Trans._
_Intell. Transp. Syst._, vol. 25, no. 3, pp. 3009–3019, Mar. 2024.

[29] D. V. Huynh, V.-D. Nguyen, S. Chatzinotas, S. R. Khosravirad, H. V.
Poor, and T. Q. Duong, “Joint communication and computation offloading for ultra-reliable and low-latency with multi-tier computing,” _IEEE_
_J. Sel. Areas Commun._, vol. 41, no. 2, pp. 521–537, Feb. 2022.

[30] R. S. Sutton and A. G. Barto, _Reinforcement Learning: An Introduction_,
2nd ed. MIT Press, Oct. 2018.

[31] J. Terry, B. Black, N. Grammel, M. Jayakumar, A. Hari, R. Sullivan,
L. S. Santos, C. Dieffendahl, C. Horsch, R. Perez-Vicente, N. Williams,
Y. Lokesh, and P. Ravi, “Pettingzoo: Gym for multi-agent reinforcement
learning,” in _Adv. Neural Inf. Process. Syst._, vol. 34, 2021, pp. 15 032–
15 043.

[32] T. Q. Duong, D. V. Huynh, Y. Li, E. Garcia-Palacios, and K. Sun, “Digital twin-enabled 6G aerial edge computing with ultra-reliable and lowlatency communications,” in _Proc. 2022 1st International Conference on_
_6G Networking (6GNet)_, Paris, France, Jul. 2022.



12



Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY MADRAS. Downloaded on September 04,2025 at 03:05:46 UTC from IEEE Xplore. Restrictions apply.

© 2025 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,

but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.


