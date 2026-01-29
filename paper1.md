IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY, VOL. 74, NO. 7, JULY 2025 11235

## Service Function Chain Dynamic Scheduling in Space-Air-Ground Integrated Networks


Ziye Jia _, Member, IEEE_, Yilu Cao, Lijun He _, Member, IEEE_, Qihui Wu _, Fellow, IEEE_,
Qiuming Zhu _, Senior Member, IEEE_, Dusit Niyato _, Fellow, IEEE_, and Zhu Han _, Fellow, IEEE_



_**Abstract**_ **—As an important component of the sixth generation**
**communication technologies, the space-air-ground integrated net-**
**work (SAGIN) attracts increasing attentions in recent years. How-**
**ever, due to the mobility and heterogeneity of the components such**
**as satellites and unmanned aerial vehicles in multi-layer SAGIN,**
**the challenges of inefficient resource allocation and management**
**complexity are aggregated. To this end, the network function vir-**
**tualization technology is introduced and can be implemented via**
**service function chains (SFCs) deployment. However, urgent unex-**
**pected tasks may bring conflicts and resource competition during**
**SFC deployment, and how to schedule the SFCs of multiple tasks**
**in SAGIN is a key issue. In this paper, we address the dynamic and**
**complexity of SAGIN by presenting a reconfigurable time extension**
**graph and further propose the dynamic SFC scheduling model.**
**Then, we formulate the SFC scheduling problem to maximize the**
**number of successful deployed SFCs within limited resources and**
**time horizons. Since the problem is in the form of integer linear**


Received 14 December 2024; accepted 12 February 2025. Date of publication
21 February 2025; date of current version 18 July 2025. This work was supported
in part by National Natural Science Foundation of China under Grant 62301251
and Grant 62201463, in part by the Natural Science Foundation of Jiangsu
Province of China under Grant BK20220883, in part by the Natural Science
Foundation on Frontier Leading Technology Basic Research Project of Jiangsu
under Grant BK20222001, in part by the National Research Foundation, SingaporeandInfocommMediaDevelopmentAuthorityunderitsFutureCommunications Research & Development Programme under Grant FCP-NTU-RG-2022010 and Grant FCP-ASTAR-TG-2022-003, in part by Singapore Ministry of
Education (MOE) under GrantRG87/22 and Grant RG24/24, in part by the NTU
Centre for Computational Technologies in Finance (NTU-CCTF), in part by the
RIE2025 Industry Alignment Fund - Industry Collaboration Projects (IAF-ICP)
(Award I2301E0026), administered by A*STAR, in part by Alibaba Group
and NTU Singapore through Alibaba-NTU Global e-Sustainability CorpLab
(ANGEL), in part by NSF under Grant ECCS-2302469, and Grant CMMI2222810, Toyota. Amazon, and in part by Japan Science and Technology Agency
(JST) Adopting Sustainable Partnerships for Innovative Research Ecosystem
(ASPIRE) under Grant JPMJAP2326. The review of this article was coordinated
by Dr. Wei Quan. _(Corresponding author: Lijun He.)_

Ziye Jia is with the College of Electronic and Information Engineering,
Nanjing University of Aeronautics and Astronautics, Nanjing 211106, China,
and also with the State Key Laboratory of ISN, Xidian University, Xian 710071,
[China (e-mail: jiaziye@nuaa.edu.cn).](mailto:jiaziye@nuaa.edu.cn)

Yilu Cao, Qihui Wu, and Qiuming Zhu are with the College of Electronic and
Information Engineering, Nanjing University of Aeronautics and Astronautics,
[Nanjing 211106, China (e-mail: caoyilu@nuaa.edu.cn; wuqihui@nuaa.edu.cn;](mailto:caoyilu@nuaa.edu.cn)
[zhuqiuming@nuaa.edu.cn).](mailto:zhuqiuming@nuaa.edu.cn)

Lijun He is with the School of Information and Control Engineering, China
[University of Mining and Technology, Xuzhou 221116, China (e-mail: lijunhe@](mailto:lijunhe@cumt.edu.cn)
[cumt.edu.cn).](mailto:lijunhe@cumt.edu.cn)

Dusit Niyato is with the School of Computer Science and Engineering,
[Nanyang Technological University, Singapore 639798 (e-mail: dniyato@ntu.](mailto:dniyato@ntu.edu.sg)
[edu.sg).](mailto:dniyato@ntu.edu.sg)

Zhu Han is with the Department of Electrical and Computer Engineering,
University of Houston, Houston, TX 77004 USA, and also with the Department
of Computer Science and Engineering, Kyung Hee University, Seoul 446-701,
[South Korea (e-mail: hanzhu22@gmail.com).](mailto:hanzhu22@gmail.com)

Digital Object Identifier 10.1109/TVT.2025.3543259



**programming and intractable to solve, we propose the algorithm**
**by incorporating deep reinforcement learning. Finally, simulation**
**results show that the proposed algorithm has better convergence**
**and performance compared to other benchmark algorithms.**


_**Index Terms**_ **—Space-air-ground integrated network (SAGIN),**
**network function virtualization, service function chain scheduling,**
**resource allocation, deep reinforcement learning.**


I. INTRODUCTION


HE space-air-ground integrated networks (SAGINs) stand
out as pivotal elements in the evolution of the sixth gen# **T**
eration communication technologies in recent years. SAGIN
can provide global services, especially for the remote areas,
which gathers significant attentions from both academia and
industry [1], [2]. SAGIN is mainly composed of satellites,
unmanned aerial vehicles (UAVs), ground stations, as well as
various users [3]. Compared with terrestrial networks, SAGIN is
a multi-layer heterogeneous network with diverse resources and
complex structure, which can support various tasks for global
coverage [4]. However, in SAGIN, satellites move periodically
with high dynamic, while UAVs move flexibly with detailed
planning. Besides, the resource capabilities of diverse nodes are
different [5]. Since the traditional satellites or aerial UAVs are
generally designed for certain types of tasks [6], the different
layers of networks are isolated and cannot share resources,
resulting in low resource utilization, large overhead, and unsatisfied services. Therefore, it is necessary to cooperate the multiple
resources in SAGIN to provide better services for the increasing
number of terrestrial users.

The network function virtualization (NFV) technology can be
introduced in SAGIN to improve the resource management efficiency. In particular, NFV decouples the network functions from
hardware devices by deploying software on general-purpose
devices instead of specialized devices [7], [8]. NFV can tackle
the differences and isolation among multiple resource nodes
in different networks to realize interconnections and resource
sharing for different tasks. Based on NFV, the implementation
of resource allocation can be deemed as service function chains
(SFCs) deployment [9]. In detail, SFC is a sequence of virtual
network functions (VNFs) that are executed in a certain order to
deliver specific network services [10]. Consequently, the SFCbased mechanism provides ideas for the resource management
in SAGIN, but the following challenges should be focused.
r In SAGIN, the key components such as satellites and

UAVs are highly dynamic, and the relative motion among



0018-9545 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and similar technologies.
Personal use is permitted, but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.


Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY MADRAS. Downloaded on September 08,2025 at 12:18:33 UTC from IEEE Xplore. Restrictions apply.


11236 IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY, VOL. 74, NO. 7, JULY 2025



different nodes leads to dynamic network topology. Therefore, it is challenging to accurately characterize the multilayer resources across time, which further aggravates the
difficulty to establish the SFC deployment model.
r With the increasing numbers and types of tasks, it is prone

to emerging unexpected demands. The urgent unexpected
tasks may bring conflicts in SFC deployment and competitions for resources, which in turn leads to task failures.
r Due to the heterogeneity of satellites and UAVs in terms

of dynamics, coverages, and resource capacity, it is challenging to efficiently allocate such heterogenous resources
to satisfy multiple SFC demands.
As such, in this paper, we focus on dealing with these challenges. Firstly, due to the highly dynamic characteristics of
SAGIN, we propose the reconfigurable time extension graph
(RTEG) to depict the multiple resources in SAGIN, by dividing
the time horizon into multiple time slots, and the network
topology is deemed as quasi-static in each time slot. In addition,
according to NFV, each task corresponds to an SFC, and the SFC
deployment and dynamic scheduling model is designed based
on RTEG. Then, the SFC scheduling problem is formulated to
maximize the number of successful deployed SFCs, i.e., tasks.
Since the problem is in the form of integer linear programming
(ILP), direct solutions are intractable due to the unacceptable
time complexity [11]. Hence, we transform the problem into a
Markov decision process (MDP) and further design the deep
reinforcement learning (DRL)-based algorithms. Finally, extensive simulations are conducted to verify the performance of the
proposed algorithms.

In summary, the main contributions of this work are summarized as follows:
r We address the dynamics and heterogeneities of SAGIN by

proposing RTEG for resource representation, and propose
the detailed model of SFC deployment and scheduling
based on RTEG for resource allocation.
r To solve the formulated problem, we transform it into

an MDP, and then propose the DRL-based algorithms,
in which the algorithm for the mutual selection of SFCs
and nodes is designed, so that SFCs can be effectively
scheduled and the resources are utilized efficiently. The
complexity of the algorithms is also analyzed.
r The feasibility and efficiency of the proposed algorithms

are evaluated through extensive simulations with correspondinganalyses,andsatisfactorysolutionsareefficiently
obtained for the SFC scheduling problem.
The rest of this paper unfolds as follows. The related work
is described in Section II. The system model is proposed in
Section III, and the problem formulation is presented in Section IV. In Section V, the algorithms are designed. Simulation
results and corresponding analyses are presented in Section VI.
Finally, conclusions are drawn in Section VII.


II. RELATED WORK


AsforthedeploymentandschedulingofSFCinterrestrialnetworks, there exist sufficient researches. For example, the authors
in [12] proposedaheuristicalgorithmwithaquantumannealerto



solve the VNF scheduling problem in virtual machines. In [13],
the authors designed a two-phased algorithm to solve the VNF
deployment and flow scheduling problems in distributed data
centers. The authors in [14] presented a deep Dyna-Q approach
to handle the SFC dynamic reconfiguration problem in the
Internet of Things (IoT) network. In [15], a game theory-based
approach to solve SFC service latency problem at the edge was
studied. Theauthorsin [16] proposedadynamicSFCembedding
scheme with matching algorithm and DRL in the industrial IoT
network. The authors in [17] optimized the VNF placement and
flowschedulinginmobilecorenetworks.However,thesemodels
and mechanisms cannot be directly applied to the multi-layer
dynamic SAGIN with high heterogeneity.

There exist some studies of SFC deployment in single layer
networks in the air such as the flying ad hoc network (FANET),
or satellite network in the space. The authors in [18] presented a
mathematicalframeworktosolvetheVNFplacementproblemin
a FANET. In [19], the authors studied a multiple service delivery
problem using SFC in the low earth orbit (LEO) satelliteterrestrial integrated network, and designed an improved response algorithm and an adaptive algorithm to achieve the
Nash equilibrium. The authors in [20] designed an IoT platform
running within software defined network (SDN)/NFV-ready infrastructures, which applied to miniaturized CubeSats. In [21],
the authors proposed a new edge-cloud architecture based on
machine learning, which studied the UAV resource utilization of
SFC embedding. [22] leveraged UAV-aided mobile-edge computing and NFV to enhance smart agriculture applications, and
it introduced the decentralized federated learning to optimize
NFV function orchestration.The authors in [23] proposed an
approach based on Asynchronous Advantage Actor-Critic to
deploy VNFs with low latency during heterogeneous bandwidth
demands. [24] investigated the orchestration of NFV chains in
satellite networks, followed by the design of a brand-and-price
algorithm combining three methods, and proposed an approximate algorithm based on the beam search. However, these works
have not considered the connectivity among multi-layers of
SAGIN.

There exist a couple of works related with the SFC or VNF
deployment in SAGIN. In [25], a novel cyber-physical system
spanning ground, air, and space was introduced, which was
supported by SDN and NFV techniques. The authors in [26]
used the federation learning algorithm to figure out the SFC
embedding problem in SAGIN, and reconfigured SFC to reduce the service blocking rate. In [27], the authors studied a
reconfigurable service provisioning framework and proposed a
heuristic greedy algorithm to solve the SFC planning problem in
SAGIN. In [10], an iterative alternating optimization algorithm
by the convex approximation is used to deal with the SFC
deployment and scheduling in SAGIN from the perspective of
network operators, so as to maximize the network profit. The
authors in [28] investigated online dynamic VNF mapping and
scheduling in SAGIN, and proposed two Tabu search-based
algorithms to obtain suboptimal solutions. In [29], the authors
constructed a service model by dividing the network slices and
proposed an SFC mapping method based on delay prediction.
However, the dynamic topology of SAGIN across time have not



Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY MADRAS. Downloaded on September 08,2025 at 12:18:33 UTC from IEEE Xplore. Restrictions apply.


JIA et al.: SERVICE FUNCTION CHAIN DYNAMIC SCHEDULING IN SPACE-AIR-GROUND INTEGRATED NETWORKS 11237


TABLE I
KEY NOTATIONS


Fig. 1. Scenario of SAGIN.




_[t]_ _i_ _[, n]_ _j_ _[t][′]_



(U2S), and satellite-to-satellite (S2S), ( _n_ _[t]_



been well considered in these works, which is a significant issue
and cannot be neglected.

As analyzed above, the SFC deployment and scheduling
issues are well studied in the terrestrial network, single aerial
UAV network, or single satellite network. However, as far as the
authors’ knowledge, the researches on SFC scheduling problem
in SAGIN is not comprehensive, and most designed algorithms
are heuristic. These algorithms can not be well adapted to
large-scale networks and complete large-scale tasks. Hence, in
this paper, we take into account the connectivity and dynamic of
the multi-layer SAGIN, propose the corresponding deployment
and scheduling model, and design the DRL-based algorithms
to cope with network structures of different scales and diverse
numbers of task requirements.


III. SYSTEM MODEL


In this section, the models of network, SFC dynamic scheduling, channel, as well as energy cost are elaborated. Key notations
are listed in Table I.


_A. Network Model_


The SAGIN scenario includes ground nodes, UAVs in the
air and LEO [1] satellites in the space. We conduct an SFC
deployment model in SAGIN, as shown in Fig. 1. In detail,
it is characterized as _G_ = ( _N_ _, L_ ), where _N_ = _Ng ∪Nu ∪Ns_
represents three types of nodes, _n_ _[t]_ _i_ _[∈N]_ [. Denote] _[ L]_ [ =] _[ L][gu][ ∪]_

_Lsg ∪Lug ∪Luu ∪Lus ∪Lss ∪Lt_ as all links between two
nodes, i.e., ground-to-UAV (G2U), satellite-to-ground (S2G),
UAV-to-ground (U2G), UAV-to-UAV (U2U), UAV-to-satellite


1Since only LEO satellites are considered in the model of this work, we use
the term “satellite” for the LEO satellite in the rest of the paper.



(U2S), and satellite-to-satellite (S2S), ( _n_ _[t]_ _i_ _[, n]_ _j_ _[t]_ [)] _[ ∈L]_ [. To repre-]

sent the multiple and heterogenous resources in SAGIN, we
design the RTEG, which divides a time horizon into a set of
_T_ time slots, _t ∈_ _T_ . _T_ is the total number of time slots. Each
time slot _t_ has the same length _τ_, which is sufficiently short
so that the link is quasi-static in the same time slot. The same
node can be regarded as various nodes in diverse time slots,
with different states and resource conditions. In addition, if
the data in node _n_ _[t]_ _i_ [cannot be transmitted in current time slot]

_t_, it is stored in the node to next time slot _t_ +1. Therefore,
link _Lt_ = _{_ ( _n_ _[t]_ _i_ _[, n][t]_ _i_ [+][1] ) _|n_ _[t]_ _i_ _[∈N][u][ ∪N][s][, t][ ∈]_ _[T]_ _[}]_ [ is introduced to]



_s_ 2 is idle, VNF _f_ 3 [processing can be carried out. However, it]

makes _f_ 3 [2] [exceed the delay requirement, i.e., SFC deployment]

and processing is not completed due to latency restrictions.




_[t]_ _i_ _[, n][t]_ _i_ [+][1]




_[t]_ _i_ [+][1] ) _|n_ _[t]_ _i_



link _Lt_ = _{_ ( _n_ _[t]_ _i_ _[, n][t]_ _i_ [1] ) _|n_ _[t]_ _i_ _[∈N][u][ ∪N][s][, t][ ∈]_ _[T]_ _[}]_ [ is introduced to]

indicate the storage of node _ni_ from time slot _t_ to _t_ +1.



_B. SFC Dynamic Scheduling Model_



Based on RTEG, we build the SFC scheduling model in
SAGIN, as shown in Fig. 2. In detail, there exist three tasks
_{r_ 1 _, r_ 2 _, r_ 3 _}_ in Fig. 2(a). VNFs are deployed on UAVs or satellites
to construct SFCs, where _F_ 1 : _{f_ 1 [1] _[}]_ [,] _[ F]_ [2][ :] _[ {][f]_ [ 1] 2 _[→]_ _[f]_ [ 2] 2 _[→]_ _[f]_ [ 3] 2 _[}]_ [ and]




[ 2] 2 _[→]_ _[f]_ [ 3] 2




[ 1] _k_ _[→]_ _[f]_ [ 2] _k_



to construct SFCs, where _F_ 1 : _{f_ 1 [1] _[}]_ [,] _[ F]_ [2][ :] _[ {][f]_ [ 1] 2 _[→]_ _[f]_ [ 2] 2 _[→]_ _[f]_ [ 3] 2 _[}]_ [ and]

_F_ 3 : _{f_ 3 [1] _[→]_ _[f]_ [ 2] 3 _[}]_ [. The SFC is expressed as] _[ F][k]_ [ :] _[ {][f]_ [ 1] _k_ _[→]_ _[f]_ [ 2] _k_ _[→]_



3 [1] _[→]_ _[f]_ [ 2] 3




[ 2] 3 _[}]_ [. The SFC is expressed as] _[ F][k]_ [ :] _[ {][f]_ [ 1] _k_



1 [1] _[}]_ [,] _[ F]_ [2][ :] _[ {][f]_ [ 1] 2




[ 1] 2 _[→]_ _[f]_ [ 2] 2



_F_ 3 : _{f_ 3 [1] _[→]_ _[f]_ [ 2] 3 _[}]_ [. The SFC is expressed as] _[ F][k]_ [ :] _[ {][f]_ [ 1] _k_ _[→]_ _[f]_ [ 2] _k_ _[→]_

_· · · →_ _fk_ _[l][k]_ _[}]_ [, where] _[ f][ m]_ _k_ represents the _m_ -th VNF in the SFC

of the _k_ -th task, and _lk_ is the number of VNFs in the SFC,
_f_ _[m]_ _[∈F][k]_ [.] _[ K]_ [ represents the number of SFCs,] _[ k][ ∈K]_ [. It is worth]



_k_ _[l][k]_ _[}]_ [, where] _[ f][ m]_ _k_



_fk_ _[m]_ _[∈F][k]_ [.] _[ K]_ [ represents the number of SFCs,] _[ k][ ∈K]_ [. It is worth]

noting that VNFs should be executed in the time order to keep
SFC sequence.

Correspondingly, the initial deployment of the three SFCs is
shown in Fig. 2(b). Due to the resource restrictions, each node
(satellite or UAV) in a time slot can only accommodate limited
VNFs. For instance, at time slot 10, VNF _f_ 3 [2] [reaches satellite] _[ s]_ [2]

from UAV _u_ 2, but at the same time, there exists another VNF _f_ 2 [3]

being processed on satellite _s_ 2. Hence, VNF _f_ 3 [2] [can only being]

stored on satellite _s_ 2, generating the waiting delay until time
slot 11. When VNF _f_ 2 [3] [processing is completed and satellite]

_s_ 2 is idle, VNF _f_ 3 [2] [processing can be carried out. However, it]



Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY MADRAS. Downloaded on September 08,2025 at 12:18:33 UTC from IEEE Xplore. Restrictions apply.


11238 IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY, VOL. 74, NO. 7, JULY 2025


Fig. 2. SFC dynamic scheduling in SAGIN based on RTEG. (a) SFC deployment in SAGIN. (b) SFC offline deployment corresponding to (a). (c) SFC online
scheduling.



Hence, we present another scheme for SFC scheduling in
Fig. 2(c). When SFC _F_ 2 transmits from satellite _s_ 1 to _s_ 2 at
time slot 9, satellite _s_ 2 does not process VNF _f_ 2 [3] [immediately,]



where _G_ 0 is the channel gain when the link distance between the
ground and UAV _d_ _[t]_ [=][ 1 m. Wherein,] _[ {][a][t]_ _[, b][t]_ _[}]_ [ and] _[ {][a][t]_ _[, b][t]_ _[}]_




_[t]_ _gu_ [=][ 1 m. Wherein,] _[ {][a][t]_




_[t]_ _u_ _[, b][t]_




_[t]_ _u_ _[}]_ [ and] _[ {][a][t]_




_[t]_ _g_ _[, b][t]_



time slot 9, satellite _s_ 2 does not process VNF _f_ 2 [immediately,]

but stores VNF _f_ [3][. When SFC] _[ F]_ [3][ is transmitted from UAV] _[ u]_ [2]



but stores VNF _f_ 2 [3][. When SFC] _[ F]_ [3][ is transmitted from UAV] _[ u]_ [2]

to satellite _s_ 2 at time slot 10, VNF _f_ 3 [2] [is handled firstly. After]



ground and UAV _d_ _[t]_ _gu_ [=][ 1 m. Wherein,] _[ {][a]_ _u_ _[t]_ _[, b][t]_ _u_ _[}]_ [ and] _[ {][a][t]_ _g_ _[, b][t]_ _g_ _[}]_

are the horizon locations of the UAV and the ground station,
respectively. Then, the signal-to-noise ratio (SNR) is



to satellite _s_ 2 at time slot 10, VNF _f_ 3 [is handled firstly. After]

completing VNF _f_ [2][, VNF] _[ f]_ [ 3] [is processed. Thus, both SFC] _[ F]_ [2]



_gu_
_σ_ [2]



3 [2][, VNF] _[ f]_ [ 3] 2



completing VNF _f_ 3 [2][, VNF] _[ f]_ [ 3] 2 [is processed. Thus, both SFC] _[ F]_ [2]

and SFC _F_ 3 satisfy the delay requirements.

It is worth noting that all tasks can be transmitted to nodes
within effective communication distance and with sufficient
resources. All nodes can effectively handle tasks within the
scope of node communication, without node failure and link
disconnection. In this case, a comprehensive SFC scheduling
scheme is executed for all tasks.


_C. Channel Model_


In Fig. 1, there exist six types of channels, including G2U,
U2U, U2S, S2S, S2G and U2G, where the channel types of U2S
and S2S are all related to line-of-sight (LoS) communication.

_1) Channel Model of G2U and U2G:_ Since the height of
UAVs is much higher than the ground, we consider that the
communication between ground stations and UAVs is LoS,
ignoring the small-scale fading and shadow [30], [31]. Hence,
the channel power gain between UAV _n_ _[t]_ _u_ [and ground station] _[ n][g]_

at time slot _t_ is



_3) Channel Model Related to Satellites:_ Following [35], the
available data rate of link ( _n_ _[t][, n][t]_ [)] _[ ∈L][us][ ∪L][ss]_ [ is expressed as]



Ψ _gu_ = _[P][ tr][G][t]_




_[t]_ _i_ _[, n][t]_ _j_




_[t]_ _j_ [)] _[ ∈L][us][ ∪L][ss]_ [ is expressed as]



= ( _[P]_ _d_ _[ tr][t]_ _gu_ _[ι]_ ) [0][2] _[,]_ (2)



0



where _P_ _[tr]_ includes _P_ _[tr]_



_g_ _[tr]_ [and] _[P][ tr]_ _u_



where _P_ includes _Pg_ [and] _[P][ tr]_ _u_ [,whichindicatethetransmission]

power from the ground and the UAV, respectively. _σ_ 0 [2] [denotes the]

White Gaussian noise power, and _ι_ 0 = _G_ 0 _/σ_ 0 [2] [is the reference]



White Gaussian noise power, and _ι_ 0 = _G_ 0 _/σ_ 0 [is the reference]

SNR [32], [33].

_2) Channel Model of U2U:_ Following [34], the path loss of
U2U is calculated as



_PLuu_ = 20log10( _d_ _[t]_ _uu_ [) +][ 20][log] 10 [(] _[f][uu]_ [)] _[ −]_ [147] _[.]_ [55] _[,]_ (3)


where _d_ _[t]_ _uu_ [indicates the distance between two UAVs, and] _[ f][uu]_ [is]

the frequency. The SNR of U2U can be expressed as



10

Ψ _uu_ = _[P][uu]_ [10] _[−]_ _[P Luu]_

_σuu_ [2]



_,_ (4)



where _Puu_ and _σuu_ [2] [denote the transmission power and noise]

power between two UAVs, respectively.



_G_ 0 _G_ 0

_G_ _[t]_ _gu_ [=] ( _d_ _[t]_ _gu_ ) [2] [=] ( _a_ _[t]_ _u −_ _ag_ ) [2] + ( _b_ _[t]_ _u −_ _bg_ ) [2] + _h_ [2] _u_




_[t]_ _i_ _[,n][t]_ _j_




_[tr]_ _ij_ _[G][re]_ _ij_



_,_ (1)



_r_ _[s]_



( _n_ _[t]_



_PijG_ _[tr]_ _ij_

_[t]_ _j_ [)][ =]



_PijGij_ _[G]_ _ij_ _[L][ij][L][l]_

(5)
( _Eb/N_ 0) _reqkBTsS_ _[,]_



Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY MADRAS. Downloaded on September 08,2025 at 12:18:33 UTC from IEEE Xplore. Restrictions apply.


JIA et al.: SERVICE FUNCTION CHAIN DYNAMIC SCHEDULING IN SPACE-AIR-GROUND INTEGRATED NETWORKS 11239



where _r_ _[s]_




_[t]_ _i_ _[,n][t]_ _j_



where _r_ ( _n_ _[t]_ _i_ _[,n][t]_ _j_ [)][ includes] _[ r]_ ( _n_ _[t]_ _i_ _[,n][t]_ _j_ [)][ and] _[ r]_ ( _n_ _[t]_ _i_ _[,n][t]_ _j_ [)][, which denote the]

available data rate of U2S and S2S, respectively. _Pij_ is the
transmission power from UAV or satellite _n_ _[t]_ _i_ [to satellite] _[ n]_ _j_ _[t]_ [.] _[ G][tr]_ _ij_

and _G_ _[re]_ _ij_ [represent the transmitting and receiving antenna gains,]

respectively. _Ll_ is the total line loss. ( _Eb/N_ 0) _req_ denotes the
required ratio of the received energy-per-bit to noise density. _kB_
is the Boltzmann constant. _Ts_ indicates the noise temperature of
the total system, and _S_ denotes the maximum slant range. Moreover, _Lij_ isrelatedtothefreespaceloss,i.e., _Lij_ = ( 4 _πS_ ~~_[t]_~~ _c_ _[f][ cen]_ [)][2][,]



the path energy cost of UAV _n_ _[t]_ _i_ [is calculated as]



( _n_ _[t]_




_[t]_ _j_ [)][ includes] _[ r]_ ( _[us]_ _n_ _[t]_ _i_



( _n_ _[t]_




_[t]_ _i_ _[,n][t]_ _j_




_[t]_ _j_ [)][ and] _[ r]_ ( _[ss]_ _n_ _[t]_ _i_



( _n_ _[t]_




_[t]_ _i_ _[,n][t]_ _j_




_[p][n][t]_ [+][1] _∥_ 2

_i_
_vnt_



+ _P_ [H] _[t]_




_[t]_ _i_ _[τ,][ ∀][n]_ _i_ _[t]_



_∥pnt_



_t_

_i_ _[−]_ _[p][n][t]_ _i_ [+][1]




_[t]_ _i_ [to satellite] _[ n]_ _j_ _[t]_




_[t]_ _j_ [.] _[ G][tr]_ _ij_



_E_ [P] _[t]_



_n_ _[t]_




_[t]_ _i_ _[,u]_ [ =] _[ P]_ [ M] _n_ _[t]_ _i_



_n_ _[t]_



_i_



_n_ _[t]_




_[t]_ _i_ _[∈N][u][, t][ ∈]_ _[T,]_ [ (10)]



_i_



where _pnt_ denotesthegeographicalpositionofUAV _n_ _[t]_ _i_ [.Besides,]

the total communication energy cost of UAVs is



_, ∀n_ _[t]_ _i_ _[∈N][u][, t][ ∈]_ _[T,]_ (11)



_P_ _[tr][t]_



_n_ _[t]_



( _n_ _[t]_




_[t]_ _i_ _[,n][t]_ _j_




_[t]_ _j_ [)] _[δ][k]_







_ij_ ~~_[t]_~~ _c_ _[f][ cen]_ [)][2][,]



_E_ [O] _[t]_




_[t]_ _i_ _[,u]_ [ =]



_n_ _[t]_ _j_ _[∈N]_



_n_ _[t]_






_k∈K_




_[t]_ _j_ [)]




_[t]_ _i_ _[z]_ ( _[k]_ _n_

_r_ _[u]_



( _n_ _[t]_




_[t]_ _i_ _[,n][t]_ _j_



where _S_ _[t]_



_ij_ _[t]_ [is the maximum slant range in time slot] _[ t]_ [.] _[ f][ cen]_ _ij_



in which _P_ _[tr][t]_




_[t]_ _i_ [represents the transmitted power of UAV] _[ n]_ _i_ _[t]_ [.]



where _Sij_ [is the maximum slant range in time slot] _[ t]_ [.] _[ f][ cen]_ _ij_ refers

to the centering frequency.

The S2G channel is affected by atmospheric precipitation, so
that the meteorological satellites are used to predict the S2G
channel state [36], [37]. Accordingly, the SNR of S2G is



( _n_ _[t]_ _i_ _[, n][t]_ _j_ [)][; otherwise] _[ z]_ ( _[k]_ _n_ _[t]_ _i_ _[,n][t]_ _j_ [)][ =][ 0.] _[ δ][k]_ [ indicates the data amount]

of SFC _Fk_ . Hence, the total energy cost can be expressed as



_i_ [.]



_n_ _[t]_



Binary variable _z_ _[k]_




_[t]_ _i_ _[,n][t]_ _j_



( _n_ _[t]_




_[t]_ _j_ [)][ =][ 1 denotes SFC] _[ F][k]_ [ is deployed on link]



( _n_ _[t]_




_[t]_ _i_ _[, n][t]_ _j_




_[t]_ _j_ [)][; otherwise] _[ z]_ ( _[k]_



( _n_ _[t]_




_[t]_ _i_ _[,n][t]_ _j_



Ψ _sg_ = _[P][sg][G]_ _sg_ _[tr]_




_[tr]_ _sg_ _[G][re]_ _sg_



_,_ (6)



_sg_ _[G]_ _sg_ _[L][ij][L][r]_

_N_ 0 _Bsg_



_n_ _[t]_




_[t]_ _i_ _[,u]_ [ +] _[ E]_ _n_ [O] _[t]_ _i_



_E_ _[total][t]_



_n_ _[t]_




_[total][t]_ _i_ _[,u]_ [ =] _[ E]_ _n_ [P] _[t]_ _i_



_n_ _[t]_




_[t]_ _i_ _[,u][,][ ∀][n]_ _i_ _[t]_




_[t]_ _i_ _[∈N][u][, t][ ∈]_ _[T.]_ (12)



where _Psg_ represents the transmission power. _G_ _[tr]_ _sg_



where _Psg_ represents the transmission power. _Gsg_ [denotes the]

transmitter antenna gain of satellites, and _G_ _[re]_ _sg_ [indicates the]

receiver antenna gain of the ground station. _Bsg_ represents
the bandwidth of S2G. _Lr_ is the rain attenuation and can be
acquired from ITU-R P.618-12, i.e., _Lr_ = _Le · γR_ _[t]_ [, where] _[ L][e]_ [ is]



_n_ _[t]_



_n_ _[t]_



acquired from ITU-R P.618-12, i.e., _Lr_ = _Le · γR_ [, where] _[ L][e]_ [ is]

the slant-path length [38], and _γR_ _[t]_ [denotes the attenuation per]

kilometer in time slot _t_ .

According to Shannon formula, the maximum data rate of
G2U, U2U, U2G and S2G can be calculated as



_2) Energy Cost of Satellites:_ The energy consumption of
satellites is primarily associated with the data transmission and
reception. _En_ _[re][t]_ _i_ _[,s]_ [ denotes the energy cost of a data receiver,]

while _En_ _[tr][t]_ _i_ _[,s]_ [ indicates the transmitter energy cost of satellites.]

Therefore, we have: _∀n_ _[t]_ _[∈N][s][, t][ ∈]_ _[T,]_




_[t]_ _i_ _[∈N][s][, t][ ∈]_ _[T,]_



( _n_ _[t]_




_[t]_ _j_ _[,n][t]_ _i_



( _n_ _[t]_




_[t]_ _j_ _[,n][t]_ _i_



+



_P_ _[re]_



_n_ _[t]_ _j_ _[∈N][u]_



_P_ _[re]_



_us_ _[re][z]_ ( _[k]_



_ss_ _[re][z]_ ( _[k]_ _n_




_[t]_ _i_ [)] _[δ][k]_




_[t]_ _i_ [)] _[δ][k]_



⎞


~~⎠~~ _,_



_E_ _[re][t]_




_[t]_ _i_ _[,s]_ [ =]






_k∈K_



⎛

⎝ [�]







_n_ _[t]_




_[t]_ _i_ [)]




_[t]_ _i_ [)]



_n_ _[t]_ _j_ _[∈N][s]_



_r_ _[us][t]_



( _n_ _[t]_




_[t]_ _j_ _[,n][t]_ _i_



_r_ _[ss]_



( _n_ _[t]_




_[t]_ _j_ _[,n][t]_ _i_



(13)
and



⎛



_r_ ( _nt_




_[t]_ _j_ [)][ =] _[B]_ [log][2][(][1][ + Ψ)] _[,][ ∀]_ [(] _[n]_ _i_ _[t]_



_t_

_i_ _[,n][t]_ _j_




_[t]_ _i_ _[, n][t]_ _j_




_[t]_ _j_ [)] _[ ∈L \ {L][us]_ _[∪L][ss]_ _[∪L][t][}][,]_



( _n_ _[t]_




_[t]_ _i_ _[,n][t]_ _j_



( _n_ _[t]_




_[t]_ _i_ _[,n][t]_ _j_



+



_P_ _[tr]_



_P_ _[tr]_



_ss_ _[tr][z]_ ( _[k]_



_n_ _[t]_ _j_ _[∈N][s]_



_sg_ _[tr][z]_ ( _[k]_



(7)
where _r_ ( _nt_ _[,n][t]_ [)][ =] _[ r][gu][t]_ _[t]_ _[ ∪]_ _[r]_ ( _[uu][t]_ _[t]_ [)] _[ ∪]_ _[r][ug][t]_ _[t]_ _[ ∪]_ _[r][sg][t]_ _[t]_ [, and]




_[t]_ _j_ [)] _[δ][k]_




_[t]_ _j_ [)] _[δ][k]_



⎞


~~⎠~~ _,_






_k∈K_



⎝ [�]








_[t]_ _j_ [)][ =] _[ r]_ ( _[gu]_ _n_ _[t]_



where _r_ ( _nti_ _[,n][t]_ _j_ [)][ =] _[ r]_ ( _n_ _[t]_ _i_ _[,n][t]_ _j_ [)] _[ ∪]_ _[r]_ ( _n_ _[t]_ _i_ _[,n][t]_ _j_ [)] _[ ∪]_ _[r]_ ( _n_ _[t]_ _i_ _[,n][t]_ _j_ [)] _[ ∪]_ _[r]_ ( _n_ _[t]_ _i_ _[,n][t]_ _j_ [)][, and]

_B_ = _Bgu ∪_ _Buu ∪_ _Bug ∪_ _Bsg_ indicates the bandwidth of different channels. Ψ = Ψ _gu ∪_ Ψ _uu ∪_ Ψ _ug ∪_ Ψ _sg_ denotes the SNR of
G2U, U2U, U2G and S2G, respectively.



_E_ _[tr][t]_



_n_ _[t]_




_[t]_ _i_ _[,s]_ [ =]



_ss_ _[tr]_ [and] _[ P][ tr]_ _sg_



_t_

_i_ _[,n][t]_ _j_




_[t]_ _i_ _[,n][t]_ _j_




_[t]_ _j_ [)] _[ ∪]_ _[r]_ ( _[ug]_ _n_ _[t]_ _i_



( _n_ _[t]_




_[t]_ _i_ _[,n][t]_ _j_



( _n_ _[t]_




_[t]_ _j_ [)] _[ ∪]_ _[r]_ ( _[uu]_ _n_ _[t]_ _i_



( _n_ _[t]_




_[t]_ _i_ _[,n][t]_ _j_




_[t]_ _j_ [)] _[ ∪]_ _[r]_ ( _[sg]_ _n_ _[t]_ _i_



( _n_ _[t]_




_[t]_ _i_ _[,n][t]_ _j_




_[t]_ _j_ [)]




_[t]_ _j_ [)]



_n_ _[t]_ _j_ _[∈N][g]_



_r_ _[ss]_



( _n_ _[t]_




_[t]_ _i_ _[,n][t]_ _j_



_r_ ~~_[sg]_~~ _[t]_



( _n_ _[t]_




_[t]_ _i_ _[,n][t]_ _j_



_us_ _[re]_ [and] _[ P][ re]_ _ss_



_D. Energy Cost Model_


_1) Energy Cost of UAVs:_ UAVs primarily consume energy
during hovering, moving, and communication [39]. The moving
power is expressed as



(14)
where _Pus_ _[re]_ [and] _[ P][ re]_ _ss_ [denote the received power of U2S and S2S,]

respectively. _Pss_ _[tr]_ [and] _[ P][ tr]_ _sg_ [represent the transmitted power of S2S]

and S2G, respectively. Hence, the total energy consumption of
satellites is



_n_ _[t]_




_[t]_ _i_ _[,s]_ [ indicates the general operation energy cost.]



_E_ _[total][t]_




_[total][t]_ _i_ _[,s]_ [ =] _[ E]_ _n_ _[re][t]_ _i_



_n_ _[t]_




_[t]_ _i_ _[,s]_ [ +] _[ E]_ _n_ _[tr][t]_ _i_



_n_ _[t]_



_n_ _[t]_




_[t]_ _i_ _[,s]_ [ +] _[ E]_ _n_ _[op][t]_ _i_



_n_ _[t]_




_[t]_ _i_ _[,s][,][ ∀][n]_ _i_ _[t]_




_[t]_ _i_ _[∈N][s][, t][ ∈]_ _[T,]_ (15)



in which _E_ _[op][t]_



_P_ [M] _[t]_



_n_ _[t]_




_[t]_ _i_ [)] _[,][ ∀][n]_ _i_ _[t]_




_[t]_ _i_ [=]



_i_
_v_ [max] _[t]_



_vnt_



_n_ _[t]_



IV. PROBLEM FORMULATION



( _P_ [max] _[t]_




[max] _[t]_ _i_ _−_ _Pn_ [H] _[t]_ _i_



_A. Constraints_



_n_ _[t]_



_n_ _[t]_




_[t]_ _i_ _[∈N][u][, t][ ∈]_ _[T,]_ (8)



_i_



_ti_ [denotes the moving speed of UAV] _[ n]_ _i_ _[t]_



in which _vnt_




_[t]_ _i_ [, and] _[ v]_ _n_ [max] _[t]_



_n_ _[t]_



the maximum speed.in which _vnti_ [denotes the moving speed of UAV] _P_ [max] _[t]_ represents the power at the UAV’s _[ n]_ _i_ [, and] _[ v]_ _n_ _[t]_ _i_ is



_t_

_i_ _[,f][ m]_ _k_




_[t]_ _i_ represents the power at the UAV’s



_1) Deployment Constraints:_ We introduce binary variable
_xnt_ _[,f][ m]_ [=][ 1 to indicate VNF] _[ f][ m]_ _k_ [in SFC] _[ F][k]_ [ is deployed on node]



_xnti_ _[,f][ m]_ _k_ [=][ 1 to indicate VNF] _[ f][ m]_ _k_ [in SFC] _[ F][k]_ [ is deployed on node]

_n_ _[t]_ [; otherwise] _[ x]_ _[t]_ _[ m]_ [=][ 0. Each VNF can only be deployed on]



_n_ _[t]_ _i_ [; otherwise] _[ x]_ _n_ _[t]_ _i_ _[,f][ m]_ _k_ [=][ 0. Each VNF can only be deployed on]

one node, i.e.,



_i_ [; otherwise] _[ x]_ _n_ _[t]_




_[ m]_ _k_ [=][ 1 to indicate VNF] _[ f][ m]_ _k_



_n_ _[t]_ _i_

[H]

_n_ _[t]_



maximum speed, and _P_ [H] _[t]_




_[t]_ _i_ [indicates the hovering power, i.e.,]




_[t]_ _i_ _[,f][ m]_ _k_



_, ∀n_ _[t]_ _i_ _[∈N][u][, t][ ∈]_ _[T,]_ (9)



_, ∀n_ _[t]_








_[ m]_ _k_ [=][ 1] _[,][ ∀][f][ m]_ _k_




_[ m]_ _k_ _[∈F][k][.]_ (16)



_P_ [H] _[t]_



_n_ _[t]_ _i_ _[∈N][u][∪N][s]_



_xnt_



_t_

_i_ _[,f][ m]_ _k_




_[t]_ _i_ [= Θ]







- [(] _[M][n]_ _i_ _[t]_



_μ_ [2]



_n_ _[t]_




_[t]_ _i_ [)][3]



_n_ _[t]_




_[t]_ _i_ _[ν][n]_ _i_ _[t]_



_i_



where Θ =




~~�~~



where Θ = _g_ [3] _/_ (2 _πϑ_ ) represents the environmental parame
ter. _g_ is the earth gravity acceleration, and _ϑ_ indicates the air
density. _Mnt_ [denotes the mass of UAV] _[ n]_ _i_ _[t]_ [,] _[ μ]_ _n_ _[t]_ [is the radius, and]



_ti_ [denotes the mass of UAV] _[ n]_ _i_ _[t]_



_i_ [. Therefore,]



_i_ [,] _[ μ]_ _n_ _[t]_



density. _Mnti_ [denotes the mass of UAV] _[ n]_ _i_ [,] _[ μ]_ _n_ _[t]_ _i_ [is the radius, and]

_νnt_ [represents the number of propellers in UAV] _[ n]_ _i_ _[t]_ [. Therefore,]



Correspondingly, a finite number of different VNFs can be
deployed on one node at the same time. Besides, VNF _fk_ _[m]_

can be deployed only when SFC _Fk_ passes through node _n_ _[t]_ _i_ [.]

Consequently, we have




_[t]_ _i_ _[,][ ∀][f][ m]_ _k_



_ti_ [represents the number of propellers in UAV] _[ n]_ _i_ _[t]_




_[ m]_ _k_ _[≤]_ _[y]_ _n_ _[k]_



_n_ _[t]_



_xnt_



_t_

_i_ _[,f][ m]_ _k_




_[ m]_ _k_ _[∈F][k][, n]_ _i_ _[t]_




_[t]_ _i_ _[∈N]_ _[,]_ (17)



Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY MADRAS. Downloaded on September 08,2025 at 12:18:33 UTC from IEEE Xplore. Restrictions apply.


11240 IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY, VOL. 74, NO. 7, JULY 2025



where binary variable _y_ _[k]_



where binary variable _yn_ _[k][t]_ _i_ [=][ 1 denotes that SFC] _[ F][k]_ [ passes]

through node _n_ _[t]_ [; otherwise] _[ y]_ _[t]_ [=][ 0.]



cannot exceed the storage capacity _Unt_



_ϱk,_ ( _nt_



_n_ _[t]_



through node _n_ _[t]_ _i_ [; otherwise] _[ y]_ _n_ _[t]_ _i_ [=][ 0.]

In addition, we introduce binary variable _Ik_ to denote whether
all VNFs of SFC _Fk_ are deployed successfully, i.e.,



cannot exceed the storage capacity _Unti_ [, i.e.,]




_i_ [; otherwise] _[ y]_ _n_ _[t]_




_[t]_ _i_ _[, n][t]_ _i_ [+][1]




_[t]_ _i_ [1] ) _∈L, t ∈_ _T._ (23)




_[t]_ _i_ [+][1] ) _[δ][k][ ≤]_ _[U][n][t]_ _i_




_[t]_ _i_ _[,][ ∀]_ [(] _[n]_ _i_ _[t]_



_k∈K_



_t_

_i_ _[,n]_ _i_ _[t]_ [+][1]




  1 _,_ if







_xnt_



_∀k ∈K._ (18)



SFCs with varying data amount transmit in diverse links, the
channel capacity restriction should be satisfied:




_[ m]_ _k_ [=] _[ l][k][,]_



_Ik_ =



⎧
⎨


⎩



_n_ _[t]_ _i_ _[∈N][u][∪N][s]_

0 _,_ otherwise _,_






_k∈K_



_z_ _[k]_



_fk_ _[m][∈F][k]_



_t_

_i_ _[,f][ m]_ _k_



( _n_ _[t]_




_[t]_ _i_ _[,n][t]_ _j_




_[t]_ _j_ [)] _[τ,][ ∀]_ [(] _[n]_ _i_ _[t]_




_[t]_ _i_ _[, n][t]_ _j_




_[t]_ _j_ [)] _[ ∈L \ L][t][, t][ ∈]_ _[T.]_ (24)




_[t]_ _j_ [)] _[δ][k][ ≤]_ _[r]_ [(] _[n]_ _i_ _[t]_




_[t]_ _i_ _[,n][t]_ _j_



_2) Sequence Constraints for SFC:_ The VNFs of SFC _Fk_ :
_{f_ [1] _[→]_ _[f]_ [ 2] _[→· · · →]_ _[f][ l][k]_ _[}]_ [ must be deployed sequentially][ [40]][:]



Furthermore, the total energy consumption cannot exceed the
energy capacity _E_ [max] _[t]_ [, i.e.,]



_n_ _[t]_




_[t]_ _i_ [, i.e.,]




[ 2] _k_ _[→· · · →]_ _[f][ l]_ _k_ _[k]_



_k_ [1] _[→]_ _[f]_ [ 2] _k_




_[ l]_ _k_ _[k]_ _[}]_ [ must be deployed sequentially][ [40]][:]



_xnt_



_t_

_i_ _[,f][ m]_ _k_








_[ m]_ _k_ _[σ][f][ m]_ _k_



_, ∀fk_ _[m]_ _[∈F][k][,]_ (19)



_n_ _[t]_




_[t]_ _i_ _[,][ ∀][n]_ _i_ _[t]_



_E_ _[c]_




_[t]_ _i_ [+] _[ E]_ _n_ _[total][t]_ _i_



_mk_ +1 _−_ _tfk_ _[m]_






_t∈T_



where _E_ _[total][t]_



_tf m_ +1



_k_ _[m]_ _[≥]_



_n_ _[t]_




_[total][t]_ _i_ _≤_ _En_ [max] _[t]_ _i_



_n_ _[t]_




_[t]_ _i_ _[∈N][u]_ _[∪N][s][, t][ ∈]_ _[T,]_ (25)



_n_ _[t]_ _i_ _[∈N][u][∪N][s]_



_n_ _[t]_



_k_ _k_

_ϕnt_



_n_ _[t]_



_i_



_k_ _[m]_ [denotes the time slot when VNF] _[ f][ m]_ _k_



where _tf_ _[m]_



where _tfk_ _[m]_ [denotes the time slot when VNF] _[ f][ m]_ _k_ [starts processing.]

Let _σfk_ _[m]_ [(bit) represent the computing resource required by VNF]

_fk_ _[m]_ [, and] _[ ϕ][n][t]_ _i_ [(bit/s) indicate the computation ability of node] _[ n]_ _i_ _[t]_ [.]

_3) Flow Constraints:_ The SFC deployment must satisfy the
flow conservation constraints:
⎧ ⎪⎪⎪⎪⎪⎪⎨ �(( _nn_ _[t]_ _o_ _[t]_ _i_ _[,n]_ _,n_ _[t]_ _j_ _[t]_ _j_ [)][)] _[∈L\L][∈L\L][t][t][ z][ z]_ ( _[k]_ ( _[k]_ _nn_ _[t]_ _i_ _[t]_ _o_ _[,n]_ _,n_ _[t]_ _j_ _[t]_ _j_ [)][)][ +][ =][�][ 1] _[,]_ ( _[ ∀]_ _n_ _[t]_ _j_ _[k][−][ ∈K]_ [1] _,n_ _[t]_ _j_ [)] _[∈L][, n][t][−]_ _o_ _[t]_ [1] _[ z][∈N]_ ( _[k]_ _n_ _[t][−][, t]_ [1] _,n_ _[ ∈][t]_ [)] _[T,]_ (20



where _En_ _[t]_ _i_ denotes the transmission energy cost. Besides, the

energy cost for computation is expressed as







_fk_ _[m][∈F][k]_




_[ m]_ _k_ _[σ][f][ m]_ _k_




_[t]_ _i_ _[∈N][u]_ _[∪N][s][, t][ ∈]_ _[T,]_




_[t]_ _i_ [(bit/s) indicate the computation ability of node] _[ n]_ _i_ _[t]_ [.]



_E_ _[c]_




_[t]_ _i_ [=]




_[ m]_ _k_ _[e][c][,][ ∀][n]_ _i_ _[t]_






_k∈K_



_xnt_



_t_

_i_ _[,f][ m]_ _k_



_k_ [, and] _[ ϕ][n][t]_ _i_



_n_ _[t]_








_[t]_ _j_ [)] _[∈L\L][t][ z]_ ( _[k]_




_[c]_ _u_ [and] _[ e]_ _s_ _[c]_




_[t]_ _j_ [)][ =][ 1] _[,][ ∀][k][ ∈K][, n][t]_



�( _n_ _[t]_ _o,n_ _[t]_ _j_ [)] _[∈L\L][t][ z]_ ( _[k]_ _n_ _[t]_ _o,n_ _[t]_ _j_ [)][ =][�][ 1] _[,][ ∀][k][ ∈K][, n]_ _o_ _[t]_ _[∈N]_ _[, t][ ∈]_ _[T,]_ (20a)



_n_ _[c]_ _[t]_ [ includes] _[ E]_ _n_ _[c]_



_n_ _[c]_ _[t]_ _,u_ [and] _[ E]_ _n_ _[c]_



( _n_ _[t]_ _o,n_ _[t]_ _j_



( _n_ _[t]_ _o,n_ _[t]_ _j_




_[t]_ _i_ _[,n][t]_ _j_

( _[k]_ _n_ _[t]_




_[t]_ _j_ _[−]_ [1] _,n_ _[t]_ _j_




_[t]_ _j_ [)] _[∈L][t][−]_ [1] _[ z]_ ( _[k]_



( _n_ _[t][−]_ [1]



( _n_ _[t]_




_[t]_ _i_ _[,n][t]_ _j_




_[t]_ _j_ [)] _[∈L\L][t][ z]_ ( _[k]_



( _n_ _[t]_



_j_

_[t]_ _j_ [)][ +][�] (



_j_

_[t]_ _i_ [)][ +][�]



( _n_ _[t][−]_ [1]




_[t]_ _j_ _[−]_ [1] _,n_ _[t]_ _j_




_[t]_ _j_ _[,n][t]_ _i_




_[t]_ _j_ _[,n]_ _j_ _[t]_ [+][1]




_[t]_ _i_ [)] _[∈L\L][t][ z]_ ( _[k]_




_[t]_ _j_ [+][1] ) _∈Lt_ _[z]_ ( _[k]_



( _n_ _[t]_



( _n_ _[t]_




_[t]_ _j_ _[,n][t]_ _j_ [+][1]



⎪⎪⎪⎪⎪⎪⎩



( _ni_ _[,n]_ _j_ [)] _[∈L\L][t]_ ( _ni_ _[,n]_ _j_ [)] ( _nj_ _,nj_ [)] _[∈L][t][−]_ [1] ( _n_ _[t]_ _j_ [1] _,n_ _[t]_ _j_ [)]

= [�] _[t]_ _[t]_ _[k]_ _[t]_ _[t]_ [�] _[t]_ _[t]_ [+][1] _[z][k]_ _[t]_ [+][1]



( _n_ _[t]_




_[t]_ _i_ _[,n][t]_ _d_



( _n_ _[t]_




_[t]_ _j_ _[,n][t]_ _i_




_[t]_ _j_ [+][1] ) _[,]_



_∀k ∈K, n_ _[t]_




- _∀k ∈K, n_ _[t]_ _j_ _[∈N]_ _[, t][ ∈{]_ [2] _[, . . .,][ T−]_ [1] _[}][,]_ (20b)



(26)
where _e_ _[c]_ includes _e_ _[c]_ _u_ [and] _[ e]_ _s_ _[c]_ [, which are the energy consumption]

of per unit computing resource on a UAV and a satellite node,
respectively. _En_ _[c]_ _[t]_ [ includes] _[ E]_ _n_ _[c]_ _[t]_ _,u_ [and] _[ E]_ _n_ _[c]_ _[t]_ _,s_ [, which denote the]

computing energy cost of UAVs and satellites, respectively.

_5) Delay Constraints:_ The total time cost of SFC _Fk_ deployment cannot exceed the maximum tolerable delay _D_ [max], i.e.,



_k_, i.e.,




_[t]_ _i_ _[,n][t]_ _d_



_t_ _[f]_




_[f]_ _k_ [+] _[ t]_ _k_ _[tr]_




_[tr]_ _k_ [+]




_[t]_ _d_ [)] _[∈L\L][t][ z]_ ( _[k]_



( _n_ _[t]_




_[t]_ _d_ [)][ =][ 1] _[,][ ∀][k][ ∈K][, n]_ _d_ _[t]_




_[t]_ _d_ _[∈N]_ _[, t][ ∈]_ _[T,]_ (20c)







_ϱk,_ ( _nt_



_t_

_i_ _[,n]_ _i_ _[t]_ [+][1]




_[t]_ _i_ [+][1] ) _[≤]_ _[D]_ _k_ [max]



_k_ [max] _, ∀k ∈K,_ (27)



_i_ ) _∈Lt_



where _n_ _[t]_




_[t]_ _o_ [and] _[ n]_ _d_ _[t]_



where



where _no_ [and] _[ n]_ _d_ [denote the original node and the destination]

node of the SFC deployment, respectively. Eqs. (20a), (20b) and
(20c) represent the flow constraints at the start, intermediate, and
end nodes, respectively. Binary variable _z_ _[k]_ _[t]_ _[t]_ [ 1 denotes]



( _n_ _[t]_




_[t]_ _i_ _[,n]_ _i_ _[t]_ [+][1]







_xnt_



_t_

_i_ _[,f][ m]_ _k_




_[ m]_ _k_ _[σ][f][ m]_ _k_



_, ∀k ∈K,_ (28)







_t_ _[f]_ _k_ [=]



_i_



( _n_ _[t]_




_[t]_ _i_ _[,n][t]_ _j_




_[t]_ _j_ [)][ =][ 1 denotes]



_k_ _k_

_ϕnt_



_fk_ _[m][∈F][k]_



SFC _Fk_ is deployed on link ( _n_ _[t]_ _i_




_[t]_ _i_ _[, n][t]_ _j_




_[t]_ _j_ [)][; otherwise] _[ z]_ ( _[k]_



( _n_ _[t]_




_[t]_ _i_ _[,n][t]_ _j_



which denotes the VNF processing delay for SFC _Fk_, and _t_ _[tr]_ _k_

indicates the transmission delay for SFC _Fk_ :



_n_ _[t]_ _i_ _[∈N][u][∪N][s]_



_n_ _[t]_




_[t]_ _j_ [)][ =][ 0.]



The binary variable _z_ _[k]_




_[t]_ _j_ _[−]_ [1] _,n_ _[t]_ _j_




_[t]_ _j_ _[,n][t]_ _i_ [+][1]



( _n_ _[t][−]_ [1]




_[t]_ _j_ [)][ =][ 0 if] _[ t]_ [ =][ 1, and] _[ z]_ ( _[k]_



( _n_ _[t]_




_[t]_ _i_ [+][1] ) [=][ 0]



if _t_ = _T_ .




_[t]_ _i_ _[,n][t]_ _j_



_, ∀k ∈K._ (29)



_z_ _[k]_



( _n_ _[t]_



For SFC _Fk_, only one of three situations can occur in time
slot _t_, including deployment on a node, transmission on a link,
or storage on a node. Therefore, we have: _∀k ∈K, t ∈_ _T,_




_[t]_ _j_ [)] _[δ][k]_



_t_ _[tr]_ _k_ [=]








_[t]_ _j_ [)]



( _n_ _[t]_




_[t]_ _i_ _[,n][t]_ _j_




_[t]_ _j_ [)] _[∈L\L][t]_



_r_ ( _nt_



_t_

_i_ _[,n][t]_ _j_







_z_ _[k]_




_[t]_ _i_ _[,n][t]_ _j_











_t_

_i_ _[,n][t]_ _i_ [+][1]



_B. Optimization Objective_



The objective is to maximize the number of successful deployed SFCs, i.e.,



_xnt_




_[ m]_ _k_ [+]



( _n_ _[t]_




_[t]_ _i_ [+][1] ) [=][ 1] _[,]_



_t_

_i_ _[,f][ m]_ _k_




_[t]_ _j_ [)][ +]



_ϱk,_ ( _nt_



_n_ _[t]_ _i_ _[∈N][u][∪N][s]_



( _n_ _[t]_




_[t]_ _i_ _[,n][t]_ _j_




_[t]_ _j_ [)] _[∈L\L][t]_



( _n_ _[t]_




_[t]_ _i_ _[,n]_ _i_ _[t]_ [+][1]



_i_ ) _∈Lt_



(21)
where _ϱ_ _[k]_ _[t]_ [+][1] [=][ 1 indicates SFC] _[ F][k]_ [ is stored on node] _[ n][t]_ [from]



_P_ 0 : max
_**X**_ _,_ _**Y**_ _,_ _**Z**_ _,_ _**V**_ _,_ _**I**_







_Ik_




_[t]_ _i_ _[,n]_ _i_ _[t]_ [+][1]



_i_ [from]



( _n_ _[t]_




_[t]_ _i_ [+][1] ) [=][ 1 indicates SFC] _[ F][k]_ [ is stored on node] _[ n]_ _i_ _[t]_



time slot _t_ to _t_ +1; otherwise _ϱ_ _[k]_




_[t]_ _i_ _[,n]_ _i_ _[t]_ [+][1]



time slot _t_ to _t_ +1; otherwise _ϱ_ _[k]_

( _n_ _[t]_ _i_ _[,n]_ _i_ _[t]_ [+][1] ) [=][ 0.]

_4) Resource Constraints:_ The total computing resource for
SFCs on the deployed nodes cannot exceed the computation
capacity, i.e.,



s.t. (16) _,_ (17) _,_ (19) _−_ (25) _,_ (27) _,_



_k∈K_



( _n_ _[t]_




_[t]_ _i_ _[,n][t]_ _j_



_xnt_




_[ m]_ _k_ _[, y]_ _n_ _[k]_



_n_ _[t]_



_t_

_i_ _[,f][ m]_ _k_




_[t]_ _i_ _[, z]_ ( _[k]_



( _n_ _[t]_




_[t]_ _j_ [)] _[, ϱ]_ ( _[k]_



( _n_ _[t]_




_[t]_ _i_ _[,n]_ _i_ _[t]_ [+][1]



(30)

_[t]_ _i_ [+][1] ) _[, I][k][ ∈{]_ [0] _[,]_ [ 1] _[}][,]_




_[ m]_ _k_ _[,][ ∀][k][ ∈K][, n]_ _i_ _[t]_






_k∈K_








_[t]_ _i_ _[∈N][u]_ _[∪N][s][, t][ ∈]_ _[T,]_ (22)



where _**X**_ = _{xnt_



where _**X**_ = _{xnti_ _[,f][ m]_ _k_ _[,][ ∀][k][ ∈K][, n]_ _i_ _[t]_ _[∈N][u][ ∪N][s][, t][ ∈]_ _[T]_ _[}]_ [,] _**Y**_ =

_{y_ _[k][t]_ _[,][ ∀][k][ ∈K][, n][t]_ _[∈N]_ _[, t][ ∈]_ _[T]_ _[}]_ [,] _**[ Z]**_ [ =] _[{][z][k]_ _[t]_ _[t]_ _[,][ ∀][k]_ _[∈K][,]_ [ (] _[n][t][, n][t]_ [)]



_n_ _[t]_




_[t]_ _i_ _[,][ ∀][k][ ∈K][, n]_ _i_ _[t]_



_t_

_i_ _[,f][ m]_ _k_



_xnt_




_[ m]_ _k_ _[σ][f][ m]_ _k_




_[t]_ _i_ _[,][ ∀][n]_ _i_ _[t]_




_[t]_ _i_ _[,n][t]_ _j_



_t_

_i_ _[,f][ m]_ _k_




_[ m]_ _k_ _[≤]_ _[C][n]_ _i_ _[t]_




_[t]_ _i_ _[∈N]_ _[, t][ ∈]_ _[T]_ _[}]_ [,] _**[ Z]**_ [ =] _[{][z]_ ( _[k]_



( _n_ _[t]_




_[t]_ _j_ [)] _[,][ ∀][k]_ _[∈K][,]_ [ (] _[n]_ _i_ _[t]_




_[t]_ _i_ _[, n][t]_ _i_ [+][1]




_[t]_ _i_ _[, n][t]_ _j_




_[t]_ _j_ [)]



_fk_ _[m][∈F][k]_



_∈L \ Lt, t ∈_ _T_ _}_, _**V**_ = _{ϱ_ _[k]_




_[t]_ _i_ _[,n]_ _i_ _[t]_ [+][1]




_[t]_ _i_ [1] ) _∈Lt,_




_[t]_ _i_ [+][1] ) _[,][ ∀][k][ ∈K][,]_ [ (] _[n]_ _i_ _[t]_



_ti_ [includes] _[ C]_ _n_ _[u][t]_ _i_



( _n_ _[t]_



where _Cnt_



_n_ _[t]_




_[t]_ _i_ [and] _[ C]_ _n_ _[s]_



where _Cnti_ [includes] _[ C]_ _n_ _[t]_ _i_ [and] _[ C]_ _n_ _[t]_ _i_ [, which denote the resource]

capacity of UAVs and satellites, respectively.

When SFC _Fk_ waits to be processed on node _n_ _[t]_ _i_ [, it consumes]

the storage resources of node _n_ _[t]_ [. Hence, the total stored data]



_n_ _[t]_



When SFC _Fk_ waits to be processed on node _n_ _[t]_ _i_



_i_ [. Hence, the total stored data]



_t ∈_ _T_ _}_, and _**I**_ = _{Ik, ∀k ∈K}_ . It is noted that _P_ 0 is an ILP
problem, which is difficult to solve within limited time complexity [41]. Hence, in the following section, we design efficient
algorithms based on DRL.



Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY MADRAS. Downloaded on September 08,2025 at 12:18:33 UTC from IEEE Xplore. Restrictions apply.


JIA et al.: SERVICE FUNCTION CHAIN DYNAMIC SCHEDULING IN SPACE-AIR-GROUND INTEGRATED NETWORKS 11241


_[̸]_

_[̸]_



V. ALGORITHM DESIGN


_A. MDP Transformation_


As the above discussions, the original problem _P_ 0 is intractable to directly solve. Hence, to cope with the dynamically
changing and complex problem, we propose an algorithm based
on DRL. Besides, the mutual selection of SFCs with nodes
and links in SAGIN (MSSNL-SAGIN) is included, and so
the algorithm is termed as DRL-MSSNL-SAGIN. Firstly, we
consider transforming the SFC scheduling problem as an MDP,
which consists of five tuples _⟨S, A, P, R, γ⟩_, where _S_ is the
set of system states, _A_ denotes the action set, _P_ represents the
state transition probability, _R_ denotes the reward function, and _γ_
indicates the discount factor. The tuples are detailed as follows.

_1) System State:_ We capture the system state at the beginning
ofeachtimeslot _t_ .State _s_ _[t]_ _k_ [isdividedintotwoparts:oneisrelated]

to SFC _Fk_, and another is related to nodes in SAGIN, i.e.,


_[̸]_

_[̸]_



scheduled optimally and efficiently. For each SFC, minimizing
the ineffective time consumption, such as waiting time on the
busy node, can help improve the optimization objective. Therefore, when an SFC takes an action, the immediate reward is set
as


_[̸]_

_[̸]_



_R_ _[t]_


_[̸]_

_[̸]_



_k_ _[t]_ [=] _[ c]_ [0] _[−]_ _[c]_ [1] _[∗]_ _[t][c]_ _k_


_[̸]_

_[̸]_




_[c]_ _k_ [(] _[t]_ [)] _[ −]_ _[c]_ [2] _[∗]_ _[t][w]_ _k_


_[̸]_

_[̸]_




_[w]_ _k_ [(] _[t]_ [)] _[,]_ (34)


_[̸]_

_[̸]_




_[t]_ _k_ [=] _[ {]_ _**[F]**_ _[ t]_ _k_


_[̸]_

_[̸]_



_s_ _[t]_


_[̸]_

_[̸]_



_k_ _[,]_ _**[ N]**_ _[ t][}][,]_ (31)


_[̸]_

_[̸]_



where _**F**_ _[t]_


_[̸]_

_[̸]_



_k_ _[t]_ _[,][ C]_ _k_ _[t]_


_[̸]_

_[̸]_




_[t]_ _k_ [=] _[ {][k, v][t]_


_[̸]_

_[̸]_



where _**F**_ _[t]_ _k_ [=] _[ {][k, v]_ _k_ _[t]_ _[,][ C]_ _k_ _[t]_ _[}]_ [ includes the index] _[ k]_ [ of SFC] _[ F][k]_ [, the]

state _v_ _[t]_ [of the VNF being processed in SFC] _[ F][k]_ [, and the node]


_[̸]_

_[̸]_



state _vk_ _[t]_ [of the VNF being processed in SFC] _[ F][k]_ [, and the node]

_C_ _[t]_ [selected by SFC] _[ F][k]_ [ in the previous time slot. Besides,] _**[ N]**_ _[ t]_ [ =]


_[̸]_

_[̸]_



_Ck_ _[t]_ [selected by SFC] _[ F][k]_ [ in the previous time slot. Besides,] _**[ N]**_ _[ t]_ [ =]

_{η_ _[t][, η][t][, . . ., η][t]_ _[}]_ [ represents the resource occupancy of all nodes,]


_[̸]_

_[̸]_



1 _[t][, η]_ 2 _[t]_


_[̸]_

_[̸]_



2 _[t][, . . ., η]_ _I_ _[t]_


_[̸]_

_[̸]_



_{η_ 1 _[t][, η]_ 2 _[t][, . . ., η]_ _I_ _[t]_ _[}]_ [ represents the resource occupancy of all nodes,]

where _I_ denotes the total number of nodes. _ηi_ _[t]_ [indicates the]

amount of resources already occupied on node _n_ _[t]_ _i_ [. Moreover,]

state _v_ _[t]_ [can be further expressed as]


_[̸]_

_[̸]_



_k_ [can be further expressed as]


_[̸]_

_[̸]_



0 _,_ if VNF _fk_ is being transmitted on _L_ in _t,_
1 _,_ if VNF _fk_ is being processed on _N_ in _t,_
2 _,_ if VNF _fk_ is stored and waiting in _t._


_[̸]_

_[̸]_



_vk_ _[t]_ [=]


_[̸]_

_[̸]_



⎧
⎪⎨


⎪⎩


_[̸]_

_[̸]_



(32)


_[̸]_

_[̸]_



_2) Action:_ Each SFC needs to select an effective node in a
time slot if it wants to be successfully deployed and processed
on the node. Therefore, action _a_ _[t]_ _k_ [is set as the node selected]

by SFC _Fk_ in current time slot _t_ . Besides, the whole action set
contains all SFCs, i.e., _A_ _[t]_ = _{a_ _[t][, a][t]_ _[, . . ., a][t]_ _[}]_ [, where] _[ K]_ [ is the]


_[̸]_

_[̸]_




_[t]_ 1 _[, a][t]_ 2


_[̸]_

_[̸]_




_[t]_ 2 _[, . . ., a][t]_ _K_


_[̸]_

_[̸]_



contains all SFCs, i.e., _A_ _[t]_ = _{a_ _[t]_ 1 _[, a][t]_ 2 _[, . . ., a][t]_ _K_ _[}]_ [, where] _[ K]_ [ is the]

total number of SFCs.

_3) State Transition:_ By selecting different actions, the state
of SFC changes accordingly. Firstly, the pending VNF state
_vk_ _[∗][t]_ [+][1] of the next time slot is obtained, and then the determined

VNF state _vk_ _[t]_ [+][1] is acquired through the selection of nodes. The

pending VNF state _v_ _[∗][t]_ [+][1] is defined as follows:


_[̸]_

_[̸]_



_k_ is defined as follows:


_[̸]_

_[̸]_




_[̸]_

_[̸]_


2 _,_ if _vk_ [=][ 0] _[, t]_ _k_ [(] _[t]_ [)] _[ <]_ [=][ 1] _[, a]_ _k_ [=] _[ C]_ _k_ _[,]_

2 _,_ if _vk_ _[t]_ [=][ 1] _[, t]_ _k_ _[p]_ [(] _[t]_ [)] _[ <]_ [=][ 1] _[, a]_ _k_ _[t]_ [=] _[ C]_ _k_ _[t]_ _[,]_

2 _,_ if _v_ _[t]_ [=][ 2] _[, a][t]_ [=] _[ C][t]_ _[,]_




_[̸]_

_[̸]_

_k_ _[t]_ [=][ 1] _[, t]_ _k_ _[p]_

_k_ _[t]_ [=][ 0] _[, t]_ _k_ _[c]_

_k_ _[t]_ [=][ 1] _[, t]_ _k_ _[p]_

_k_ _[t]_ [=][ 2] _[, a][t]_




_[̸]_

_[̸]_


_[t]_ _k_ [=] _[ C]_ _k_ _[t]_




_[̸]_

_[̸]_


_k_ _[,]_



0 _,_ if _v_ _[t]_

_[̸]_

_[̸]_



_k_ _[t]_ [=][ 0] _[, t]_ _k_ _[c]_

_k_ _[t]_ [=][ 0] _[, t]_ _k_ _[c]_ _[̸]_

_[̸]_



0 _,_ if _vk_ _[t]_ [=][ 0] _[, t]_ _k_ _[c]_ [(] _[t]_ [)] _[ >]_ [ 1] _[,]_

0 _,_ if _v_ _[t]_ [=][ 0] _[, t][c]_ [(] _[t]_ [)] _[ <]_ [=] _[̸]_

_[̸]_




_[t]_ _k_ _[̸]_ [=] _[ C]_ _k_ _[t]_

_[̸]_




_[c]_ _k_ [(] _[t]_ [)] _[ <]_ [=][ 1] _[, a][t]_ _[̸]_

_[̸]_



0 _,_ if _vk_ [=][ 0] _[, t]_ _k_ [(] _[t]_ [)] _[ <]_ [=][ 1] _[, a]_ _k_ _[̸]_ [=] _[ C]_ _k_ _[,]_

0 _,_ if _v_ _[t]_ [=][ 1 or 2] _[, a][t]_ _[̸]_ [=] _[ C][t]_ _[,]_




_[̸]_

_[̸]_

_vi_ _[∗][t]_ [+][1] =



⎧
⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎨ _[̸]_ _[̸]_

⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎩




_[̸]_

0 _,_ if _vk_ [=][ 1 or 2] _[, a]_ _k_ _[̸]_ [=] _[ C]_ _k_ _[,]_

1 _,_ if _v_ _[t]_ [=][ 1] _[, t][p]_ [(] _[t]_ [)] _[ >]_ [ 1] _[, a][t]_




_[̸]_

_[̸]_

1 _,_ if _vk_ [=][ 1] _[, t]_ _k_ [(] _[t]_ [)] _[ >]_ [ 1] _[, a]_ _k_ [=] _[ C]_ _k_ _[,]_

2 _,_ if _v_ _[t]_ [=][ 0] _[, t][c]_ [(] _[t]_ [)] _[ <]_ [=][ 1] _[, a][t]_ [=] _[ C]_




_[̸]_

_k_ _[t]_ [=][ 1 or 2] _[, a][t]_ _[̸]_




_[̸]_

_[̸]_

_[p]_ _k_ [(] _[t]_ [)] _[ >]_ [ 1] _[, a]_ _k_ _[t]_




_[̸]_

_[̸]_

_[t]_ _k_ [=] _[ C]_ _k_ _[t]_




_[̸]_

_[t]_ _k_ _[̸]_ [=] _[ C]_ _k_ _[t]_



where _t_ _[w]_ _k_ [(] _[t]_ [)][ represents the waiting time at the node selected]

by SFC _Fk_ . Constants _c_ 0, _c_ 1 and _c_ 2 are weighting coefficients,
which are used to adjust the reward value, the weight of transmission time and waiting time consumption, respectively, to ensure
that the reward remains within a fixed range.


_B. DRL-Based Algorithm_


With the states, actions, state transitions, and rewards, the
optimal scheduling policy _π_ _[∗]_ can be obtained by the reinforcement learning (RL) algorithm to maximize rewards over
time [42], [43]. In Q-learning, the optimal policy is obtained by
continuous learning. During the learning process, the Q-value
table is updated iteratively, i.e.,


_Q_ ( _s_ _[t]_ _, A_ _[t]_ ) _←_ _Q_ ( _s_ _[t]_ _, A_ _[t]_ ) + _α_ [ _R_ ( _s_ _[t]_ _, A_ _[t]_ )


+ _γ_ max (35)

_A_ _[t]_ [+][1] _[Q]_ [(] _[s][t]_ [+][1] _[, A][t]_ [+][1][)] _[ −]_ _[Q]_ [(] _[s][t][, A][t]_ [)]] _[,]_


where _α_ denotes the learning rate, and _γ ∈_ [0 _,_ 1] is the discount
factor representing the attenuation value of rewards. If _γ_ is closer
to 1, it is sensitive to future rewards. _Q_ ( _s_ _[t]_ _, A_ _[t]_ ) indicates the
expected reward for the state-action pair ( _s_ _[t]_ _, A_ _[t]_ ), which can
express the probability of taking action _A_ _[t]_ at state _s_ _[t]_ . If the Qvalue table is able to converge to its optimal Q* after sufficiently
large episodes, the optimal policy is obtained as


_π_ _[∗]_ = arg max (36)

_A_ _[t][ Q][∗]_ [(] _[s][t][, A][t]_ [)] _[.]_


As a basic method of RL, Q-learning performs well in small
states and action spaces. However, in this paper, the scale of
states and spaces is quite large, so it is intractable to build the
Q-value table. Since the introduction of deep neural networks
(DNNs) into the framework of Q-learning can deal with the
large-scale problem, deep Q network (DQN) is an effective
method. However, DQN cannot always guarantee the convergence. The estimated Q-values may fluctuate continuously during training and even fail to converge to the optimal solution.
Hence, double deep Q network (DDQN) is a better mechanism

_[̸]_

for the SFC scheduling problem.

_[̸]_ To be specific, the state value is used as the input for DNNs,

and all the action values are output. Then, the action with the
maximum value is directly selected as the next action. Moreover,
there exist two types of networks in DDQN: the online network
and target network, and DDQN can stabilize the overall performance by these two networks. In addition, the parameters of
the online network are completely copied to the target network
at intervals for update. Such delayed updates can ensure the
training stability for the Q network. The weights _θ_ _[−]_ of the target
network is fixed during the iteration while the weights _θ_ of the
online network are updated. By constantly updating _θ_, the loss
function _L_ ( _θ_ ) is minimized, so as to gradually reach the optimal




_[̸]_

_[̸]_

(33)




_[̸]_

_[̸]_


_[c]_ _k_ [(] _[t]_ [)] _[ <]_ [=][ 1] _[, a][t]_

_[p]_ _k_ [(] _[t]_ [)] _[ <]_ [=][ 1] _[, a]_ _k_ _[t]_




_[̸]_

_[̸]_


_[t]_ _k_ [=] _[ C]_ _k_ _[t]_

_[t]_ _k_ [=] _[ C]_ _k_ _[t]_




_[̸]_

_[̸]_


where _t_ _[c]_




_[̸]_

_[̸]_


where _t_ _[c]_ _k_ [(] _[t]_ [)][ denotes the transmission time over the channel,]

which is consumed by SFC _Fk_ after selecting a certain action.
_t_ _[p]_ [(] _[t]_ [)][ indicates the remaining processing time for VNF] _[ f][k]_ [ cur-]




_[̸]_

_[̸]_


_tk_ [(] _[t]_ [)][ indicates the remaining processing time for VNF] _[ f][k]_ [ cur-]

rently being processed of SFC _Fk_ .

_4) Reward:_ Since the optimization objective is to deploy as
many SFCs as possible within limited time, SFCs need to be




_[̸]_

_[̸]_


Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY MADRAS. Downloaded on September 08,2025 at 12:18:33 UTC from IEEE Xplore. Restrictions apply.


11242 IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY, VOL. 74, NO. 7, JULY 2025


Fig. 3. The training process of DRL-MSSNL-SAGIN algorithm.



solution. Most steps of DDQN and DQN are similar, but DQN
always selects the maximum output value of the target network,
while DDQN firstly obtains the action with the maximum output
valuefromtheonlinenetwork,andthenacquirestheoutputvalue
of the target network corresponding to this action. Then, the loss
function _L_ ( _θ_ ) is formulated as





_s_ _[t]_ _, a_ _[t]_ ; _θ_



_,_ (37)


 


��2 [�]



_L_ ( _θ_ ) = E _st,at,R_ ( _st,at_ ) _,s′t_


and



��

_y_ _[DDQN]_ _−_ _Q_



the target Q-value. After that, the loss function is calculated by
the Q-value achieved from the online network and the target
Q-value. The parameters of the online network are updated by
the back-propagation algorithm, and periodically copied to the
target network to maintain the stability. It is notable that the
training process of DDQN in this paper is an offline mechanism.

Due to the dynamic nature of SAGIN, the distances between
nodes change according to time slots, which affect the task deployments of the current time slot. The current SFC deployment
scheme may not be optimal in the next time slot, which is a shortsighted local optimal situation. DDQN can cope with the dynamic nature of SAGIN, adjust the deployment situation at any
time, and propose different SFC deployment schemes according
to different network conditions. The Actor-Critic algorithm or
Proximal Policy Optimization algorithm is more suitable for
the scene of continuous action space. If these algorithms are
applied to the discrete action space, it may cause inefficient
calculation. Instead, it increases the computation amount and
cost. The action space is simple and discrete. Hence, DDQN is
suitable for dealing with discrete action spaces.

Algorithm 1 provides the detailed DRL-MSSNL-SAGIN procedures for determining the optimal scheduling policy. The
movement trajectory of UAVs are clearly known, and the trajectories of satellites are regular. In the whole training process,
we consider the position relationships and states of UAV nodes
and satellite nodes in different time slots. At the beginning of
the training, the relevant values of the online network, target
network and replay memory _D_, as well as the system state are
initialized (lines 1-4). Each SFC selects an action by the _ϵ_ -greedy
policy at the beginning of each time slot in each episode (line
6). Then, SFCs obtain the next states based on actions, which
includes the VNF states of SFCs, and the node states (lines 7-8).



_y_ _[DDQN]_ = _R_ ( _s_ _[t]_ _, a_ _[t]_ ) + _γQ_







_s_ _[′][t]_ _,_ arg max

_a_ _[′][t]_ _∈A_ _[t][Q]_ [(] _[s][′][t][, a][′][t]_ [;] _[ θ]_ [);] _[ θ][−]_



_,_



(38)
where E[ _·_ ] is the expectation operator, _γ_ denotes the discount
rate, and _θ_ _[−]_ indicates the weights of a target network. The action
_a_ _[t]_, as mentioned above, can be obtained from the online network
_Q_ ( _s_ _[t]_ _, a_ _[t]_ ; _θ_ ) with the _ϵ_ -greedy policy by the DNN.

In DDQN, the experience replay memory _D_ is used to cope
with the instability of learning. In detail, after passing through
DNN, a new experience ( _s_ _[t]_ _, a_ _[t]_ _, R_ ( _s_ _[t]_ _, a_ _[t]_ ) _, s_ _[′][t]_ ) is obtained, and
the transformed experience is put into _D_ . In this way, small
batches of experience from _D_ are sampled uniformly and randomly to train the neural network. Random sampling reduces the
correlation between the training samples, and thus local minima
can be avoided during the training process.

The specific training process of DDQN is shown in Fig. 3.
By interacting with the environment, actions and rewards can be
obtained, according to the state information. The information are
put into the experience replay memory, and a batch is randomly
sampled to train the neural network. Then, the best action for
the next state is selected by the online network, and the target
network is used to evaluate the Q-value of this action to obtain



Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY MADRAS. Downloaded on September 08,2025 at 12:18:33 UTC from IEEE Xplore. Restrictions apply.


JIA et al.: SERVICE FUNCTION CHAIN DYNAMIC SCHEDULING IN SPACE-AIR-GROUND INTEGRATED NETWORKS 11243



**Algorithm 1:** DRL-MSSNL-SAGIN Algorithm.



**Input:** State _s_ _[t]_, all nodes _n ∈Nu ∪Ns_, and set _K_ .
**Output:** The number of successful deployed SFCs.
1: **Initialization:**
2: Initialize the replay memory _D_, DDQN network
parameter _θ_, online network _Q_ ( _s_ _[t]_ _, a_ _[t]_ ; _θ_ ), and the
target network _Q_ ( _s_ _[t]_ _, a_ _[t]_ ; _θ_ _[−]_ ) with _θ_ _[−]_ = _θ_ .
3: **for** each episode **do**
4: Initialize state _s_ [0] and the node state.
5: **while** step _t < T_ **do**
6: Each _Fk_ chooses _a_ _[t]_ using _ϵ_ -greedy policy.
7: Each _Fk_ obtains the next pending state based on
_a_ _[t]_ .
8: Update _t_ _[c]_ [(] _[t]_ [)][,] _[ t][p]_ [(] _[t]_ [)][, and the serial number of the]



**Algorithm 2:** Algorithm for VNF State Transition.

**Input:** VNF state _v_ _[t]_, action _a_ _[t]_, next pending VNF state _v_ _[∗][t]_,



and all nodes _n ∈Nu ∪Ns_ .
**Output:** Next VNF state _v_ _[t]_ [+][1] .
1: Categorize actions into two types: one corresponding
to UAV _Nu_ and another corresponding to satellite _Ns_ .
2: **if** the node is only selected by one SFC _Fk_ **then**
3: VNF state _v_ _[t]_ [+][1] _←_ 1.



3: VNF state _vk_ _[t]_ [1] _←_ 1.

4: **else if** the node is selected by more than one SFC **then**
5: **if** the pending state _v_ _[∗][t]_ [+][1] = 1 **then**




_[c]_ _k_ [(] _[t]_ [)][,] _[ t][p]_ _k_



8: Update _t_ _[c]_ _k_ [(] _[t]_ [)][,] _[ t]_ _k_ [(] _[t]_ [)][, and the serial number of the]

currently processing VNF.
9: Obtain _v_ _[t]_ [+][1] according to Algorithm 2.
10: Calculate the reward _R_ _[t]_ [of] _[ F][k]_ [ according to][ (][34]



10: Calculate the reward _Rk_ _[t]_ [of] _[ F][k]_ [ according to][ (][34][)][.]

11: Each _Fk_ stores transition ( _s_ _[t]_ _k_ _[, a][t]_ _k_ _[, R]_ _k_ _[t]_ _[, s][t]_ _k_ [+][1] ).



5: **if** the pending state _vk_ _[t]_ [1] = 1 **then**

6: The node still accepts the current SFC _Fk_, and the
remaining computational resources of the node is
updated.
7: **end if**
8: Sort the remaining SFCs that select the node
according to the amount of data in an ascending
sequence.
9: Select SFCs according to the remaining computing
resources of the node.
10: Update the next state of SFC _v_ _[t]_ [+][1] = 1.




_[t]_ _k_ _[, a][t]_




_[t]_ _k_ _[, R][t]_



_k_ _[t]_ _[, s][t]_ _k_ [+][1]



11: Each _Fk_ stores transition ( _s_ _[t]_ _k_ _[, a][t]_ _k_ _[, R]_ _k_ _[t]_ _[, s][t]_ _k_ [1] ).

12: Each _Fk_ samples a batch of transitions from _D_
randomly.
13: Calculate _y_ _[DDQN]_ according to (38).
14: Update parameter _θ_ _[−]_ according to (37).
15: **if** all SFCs finish deployments **then**
16: Break.
17: **end if**
18: **end while**
19: **end for**



10: Update the next state of SFC _vk_ _[t]_ [1] = 1.

11: **end if**



The next VNF states of SFCs is updated by Algorithm 2 (line 9).
Then, the results on the status of SFC deployment completion are
obtained (line 10). The neural network model is then optimized
by the resulting reward (lines 11-14). If all SFCs are successfully
deployed, the training round ends; otherwise the loop continues
to the maximum value of steps (lines 15-16).

Moreover, Algorithm 2 picks SFCs that selecting the same
nodes in order to determine the next states of SFCs. In detail,
whether the node is selected by more than one SFC is assessed
firstly. If there is only one SFC, the next VNF state _vk_ _[t]_ [+][1] is set

as 1 (lines 2-3). Otherwise, it proceeds to the next judgement. If
the pending VNF state _vk_ _[∗][t]_ [+][1] is 1, the state keeps unchanged and

the node resource utilization is updated (lines 5-6). Then, the
remaining SFCs that choose the same node are arranged based
on the data size of SFCs in the ascending order and selected
partly according to the remaining node resources (lines 8-9).


_C. Complexity Analysis_



Assuming that the width of the _i_ -th layer of the neural network
is _Wi_ and there exist _M_ layers in total, the computational
complexity of forward propagation in Algorithm 1 is _O_ ( _S · A ·_

- _M_ _−_ 1

_i_ =1 _[W][i][W][i]_ [+][1][)][, where] _[ S]_ [ and] _[ A]_ [ are the numbers of elements in]
a state and an action, respectively. Besides, the complexity of Algorithm 2 is related to the number of nodes _I_ . There are _K_ SFCs
that need to be trained, and the total number of episodes and
steps are _D_ and _P_, respectively. Hence, the total computational
complexity is _O_ ( _D · P ·_ ( _K · S · A ·_ [�] _[M]_ _i_ = _[−]_ 1 [1] _[W][i][W][i]_ [+][1][ +] _[ I]_ [))][.]



TABLE II
PARAMETER SETTING


VI. SIMULATION RESULTS


_A. Simulation Setups_


In this section, we conduct simulations for the SFC deployment in SAGIN using Python. The experimental hardware
environment is based on the Intel(R) Core(TM) i9-10940X CPU,
DDR4 64 GB RAM, and GeForce RTX 3090 24GB*2 GPU. The
specific parameters used in the simulation are listed in Table II.
The scenario is set up with 30 UAVs and 2 satellites. Among
them, the UAVs are randomly arranged in a circle with a radius
of 400 m and the distance between any two UAVs cannot be less
than 20 m for safety consideration. The position information
of the satellites is selected from the Starlink G6-35 near the
coordinates of 32 _[◦]_ N, 119 _[◦]_ E on February 23, 2024 around 11:45



Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY MADRAS. Downloaded on September 08,2025 at 12:18:33 UTC from IEEE Xplore. Restrictions apply.


11244 IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY, VOL. 74, NO. 7, JULY 2025


UTC. Compared with UAVs, satellites are more like auxiliary
functional nodes, which can carry a larger amount of data. When
UAVs are unable to carry the deployment and processing of
tasks, tasks are uploaded to the satellites. Therefore, the number
of satellites set is small. The number of SFCs is set as 200, with
2 or 3 VNFs to be processed in each SFC, and the data volume is

[500 Mbit, 4,000 Mbit]. In RTEG, it is assumed that both UAVs
and satellites are kept relatively stationary in a time slot, and the
length of the time slot is set as 5 seconds.

During the simulation of DRL, the neural network structure
consists of an input layer (the number of the state), three hidden
layers (64, 32 and 32 neurons, respectively) and an output
layer (the number of actions). The ReLU function is set as
the activation function and the model parameters are updated
using the Adam optimizer. In addition, we set the learning rate
as 0.001, the discount factor as 0.9, and the _ϵ_ -greedy strategy
in action selection is chosen linearly within [0, 0.9]. During
optimizing the network model, an experience replay memory
with capacity of 500 samples is set, and the selected batch size
is 8. Then, a total of 3,000 episodes are performed, and the upper
limit of step is set as 100, which is the maximum number of time
slots in an episode.


_B. Simulation Results_



_1) Training Effect Under Different Learning Parameters:_
The DRL-MSSNL-SAGIN algorithm is evaluated by different
learning rates, as shown in Fig. 4. It is observed that various
learning rates have different performances on the convergence
of the algorithm as well as the results. At the beginning of the
training, the rewards and results are unsatisfactory. With the
increment of episodes, there is a significant rise in rewards, and
the convergence speed is accelerated. The number of SFCs completing the deployment in limited time also gradually increases.
When the learning rate is 0.001, the rewards and results are
superior to other learning rates. If the learning rate is greater
or less than 0.001, it shows different degrees of disadvantage.
Among them, when the learning rate is 0.05, the results are the
worst and do not converge to a more stable result, which means
that a larger learning rate may lead to a local optimum rather
than a global optimum. In addition, too small learning rate may
cause DRL to be trapped in the local optimal solution, and can
not jump out to find the global optimal solution. Due to the
small step size, DRL may only explore around the local optimal
solution in a small amplitude, and cannot conduct a broader
search. Taking into account the actual implementation of the
algorithm, the learning rate of 0.001 is selected.

Fig. 5 shows the evaluation of the algorithm with different
optimization strategies. It can be seen from Fig. 5(a) that the
convergence speed of Adam optimizer is fast and it converges
smoothly, while the rewards of Adadelta optimizer continue to
grow and cannot converge quickly enough. Fig. 5(b) shows the
completionoftheoptimizationobjectiveanditisobviousthatthe
optimization results of these optimizers are different, in which
the Adam optimizer relatively performs good. Thus, the Adam
optimizer is chosen for the following simulations.



Fig. 4. The training effect under different learning rates. (a) Reward _versus_
episode. (b) The number of completed tasks _versus_ episode.


Different DNN structural layers also have impacts on the
training situation of DRL, as shown in Fig. 6. It is noted that
when there is only one or two hidden layers, the convergence
is slow. As the number of hidden layers increases, the training
performs faster convergence. In addition, the results of three and
four layers are similar, but four layers increase the training time.
Therefore, to efficiently carry out the simulation, we leverage
the three DNN structure layers (64, 32, 32).

Fig. 7 shows the performance comparison of different SFC
numbers, while other values are fixed. It is observed that with
the increasing number of SFC, the completed number of SFCs
is decreasing and fluctuating. Especially when the number of
SFCs reaches 400 in Fig. 7(a), it is difficult to have a stable
convergence. It is accounted that the number of nodes and the
corresponding resources are limited. The affordability of the
whole network for various SFC numbers at different UAV sizes
is compared in Fig. 7(b). It is observed that when the number
of SFCs is small, increasing the number of UAVs does not
significantly impact the number of successful deployed SFCs.
However, by combining Fig. 7(a) and (b), if the SFC number



Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY MADRAS. Downloaded on September 08,2025 at 12:18:33 UTC from IEEE Xplore. Restrictions apply.


JIA et al.: SERVICE FUNCTION CHAIN DYNAMIC SCHEDULING IN SPACE-AIR-GROUND INTEGRATED NETWORKS 11245


Fig. 6. The effect of convergence rate under different DNN layers.


Fig. 5. Distinctions between different optimizers. (a) Reward _versus_ episode.
(b) The number of completed tasks _versus_ episode.


reaches 400, the number of UAVs greatly impacts the outcome.
When the UAV quantity grows, the number of successfully
deployed SFCs also increases rapidly. It verifies that the SFC
scheduling results are dependent on the scale of networks.



_2) Comparison of Different Algorithms:_ We compare the
proposed DRL-MSSNL-SAGIN algorithm for the SFC scheduling problem with Q-learning, Sarsa, and DQN, as shown in
Fig. 8. It is observed in Fig. 8(a) that the convergence speed
of four algorithms is similar, and when the number of SFC is
small, the results are also similar. Moreover, in Fig. 8(b), when
the number of SFCs and UAVs increases, i.e., when the scales
of networks and tasks increase, the results of Q-learning and
Sarsa are unsatisfactory, since only a part of SFCs successfully
deployed in limited time. Besides, there is a big gap between
the results of Q-learning, Sarsa and DRL-MSSNL-SAGIN. Furthermore, Q-learning has the worst results, and Sarsa performs
only slightly better than Q-learning. In addition, it is evident
from Fig. 8(c) that the node resource utilization of Q-learning
and Sarsa does not raise significantly with the increment of
the SFC number. Hence, these two algorithms are completely
unsuitable for large-scale network. On the contrary, the results



Fig. 7. Performance comparison of different SFC numbers. (a) Reward _versus_
episode. (b) The number of completed tasks _versus_ the number of UAVs.


of DRL-MSSNL-SAGIN and DQN are obviously better than
Q-learning and Sarsa, and the proposed algorithm is slightly
better than DQN, which is consistent with the characteristics
of DDQN and DQN. When the number of SFCs increases,
the resource utilization of nodes also increases significantly.
Especially when the number of SFCs is increased to 400, the



Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY MADRAS. Downloaded on September 08,2025 at 12:18:33 UTC from IEEE Xplore. Restrictions apply.


11246 IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY, VOL. 74, NO. 7, JULY 2025


VII. CONCLUSION


In this paper, we considered the highly dynamic characteristics of SAGIN and investigated the SFC deployment and
scheduling model by proposing RTEG. The SFC scheduling
problem was formulated with the objective of maximizing the
number of successfully deployed SFCs in a finite time horizon, with the considerations of channel conditions, energy constraints, deployment limitations, and multiple resource capacities. To tackle this problem, we reformulated it as an MDP
and proposed the DRL-based algorithms. Besides, we designed
the algorithm for VNF state transition to achieve the efficient
SFC scheduling. Via simulations, we analyzed the influences
of various parameters on the SFC scheduling problem, and
selected the appropriate parameters to obtain better optimization
results. Simulations also showed that the proposed algorithm
outperformed other benchmark algorithms, with respect to fast
convergence, better optimization results, and efficient resource
utilization.


REFERENCES


[1] J. Liu, Y. Shi, Z. M. Fadlullah, and N. Kato, “Space-air-ground inte
grated network: A survey,” _IEEE Commun. Surveys Tut._, vol. 20, no. 4,
pp. 2714–2741, Fourth Quarter 2018.

[2] N. Cheng et al., “6G service-oriented space-air-ground integrated network:

A survey,” _Chin. J. Aeronaut._, vol. 35, no. 9, pp. 1–18, Sep. 2022.

[3] H. Cui et al., “Space-air-Ground integrated network (SAGIN) for 6G:

Requirements, architecture and challenges,” _China Commun._, vol. 19,
no. 2, pp. 90–108, Feb. 2022.

[4] C. Huang, G. Chen, P. Xiao, Y. Xiao, Z. Han, and J. A. Chambers, “Joint

offloading and resource allocation for hybrid cloud and edge computing in
SAGINs:Adecisionassistedhybridactionspacedeepreinforcementlearning approach,” _IEEE J. Sel. Areas Commun._, vol. 42, no. 5, pp. 1029–1043,
May 2024.

[5] Z. Yin, N. Cheng, T. H. Luan, Y. Song, and W. Wang, “DT-assisted multi
point symbiotic security in space-air-ground integrated networks,” _IEEE_
_Trans. Inf. Forensics Security_, vol. 18, pp. 5721–5734, 2023.

[6] N. Kato et al., “Optimizing space-air-ground integrated networks by arti
ficial intelligence,” _IEEE Wireless Commun._, vol. 26, no. 4, pp. 140–147,
Aug. 2019.

[7] J. G. Herrera and J. F. Botero, “Resource allocation in NFV: A comprehen
sive survey,” _IEEE Trans. Netw. Serv. Manag._, vol. 13, no. 3, pp. 518–532,
Sep. 2016.

[8] J. Ordonez-Lucena, P. Ameigeiras, D. Lopez, J. J. Ramos-Munoz, J. Lorca,

and J.Folgueira,“Network slicing for 5Gwith SDN/NFV: Concepts,architectures, and challenges,” _IEEE Commun. Mag._, vol. 55, no. 5, pp. 80–87,
May 2017.

[9] Y. Cao, Z. Jia, C. Dong, Y. Wang, J. You, and Q. Wu, “SFC deployment

in space-air-ground integrated networks based on matching game,” in
_Proc. IEEE Conf. Comput. Commun. Workshop_, Hoboken, NJ, USA, 2023,
pp. 1–6.

[10] J. He et al., “Service-oriented network resource orchestration in space
air-ground integrated network,” _IEEE Trans. Veh. Technol._, vol. 73, no. 1,
pp. 1162–1174, Jan. 2024.

[11] V. V. Vazirani, _Approximation Algorithms_ . Berlin, Germany: Springer,



Fig. 8. Comparison of various RL algorithms. (a) Reward _versus_ episode.
(b) The number of completed tasks _versus_ scale of SFCs and UAVs. (c) The
utilization ratio of nodes _versus_ the number of SFCs.


optimization result of DRL-MSSNL-SAGIN is more than twice
of the results of Q-learning and Sarsa, and the resource utilization is approximately 15% greater than Sarsa and 20% greater
than Q-learning. Therefore, the proposed DRL-MSSNL-SAGIN
algorithm has excellent performance for large-scale networks.



2001.

[12] W. Xuan, Z. Zhao, L. Fan, and Z. Han, “Minimizing delay in network

function visualization with quantum computing,” in _Proc. IEEE 18th Int._
_Conf. Mobile Ad Hoc Smart Syst._, Denver, CO, 2021, pp. 108–116.

[13] L. Gu, J. Hu, D. Zeng, S. Guo, and H. Jin, “Service function chain deploy
ment and network flow scheduling in geo-distributed data centers,” _IEEE_
_Trans. Netw. Sci. Eng._, vol. 7, no. 4, pp. 2587–2597, Fourth Quarter 2020.

[14] Y. Liu, Y. Lu, X. Li, Z. Yao, and D. Zhao, “On dynamic service function

chain reconfiguration in IoT networks,” _IEEE Internet Things J._, vol. 7,
no. 11, pp. 10969–10984, Nov. 2020.

[15] A. Abouaomar, S. Cherkaoui, Z. Mlika, and A. Kobbane, “Service func
tion chaining in MEC: A mean-field game and reinforcement learning
approach,” _IEEE Syst. J._, vol. 16, no. 4, pp. 5357–5368, Dec. 2022.



Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY MADRAS. Downloaded on September 08,2025 at 12:18:33 UTC from IEEE Xplore. Restrictions apply.


JIA et al.: SERVICE FUNCTION CHAIN DYNAMIC SCHEDULING IN SPACE-AIR-GROUND INTEGRATED NETWORKS 11247




[16] H. Chen, S. Wang, G. Li, L. Nie, X. Wang, and Z. Ning, “Distributed

orchestration of service function chains for edge intelligence in the industrial Internet of Things,” _IEEE Trans. Ind. Inform._, vol. 18, no. 9,
pp. 6244–6254, Sep. 2022.

[17] B. Ren, S. Gu, D. Guo, G. Tang, and X. Lin, “Joint optimization of VNF

placement and flow scheduling in mobile core network,” _IEEE Trans._
_Cloud Comput._, vol. 10, no. 3, pp. 1900–1912, Third Quarter 2022.

[18] G. Colajanni, P. Daniele, L. Galluccio, C. Grasso, and G. Schembra, “Ser
vice chain placement optimization in 5G FANET-based network edge,”
_IEEE Commun. Mag._, vol. 60, no. 11, pp. 60–65, Nov. 2022.

[19] X. Qin, T. Ma, Z. Tang, X. Zhang, H. Zhou, and L. Zhao, “Service-aware

resource orchestration in ultra-dense LEO satellite-terrestrial integrated
6G: A service function chain approach,” _IEEE Trans. Wireless Commun._,
vol. 22, no. 9, pp. 6003–6017, Sep. 2023.

[20] G. Araniti, G. Genovese, A. Iera, A. Molinaro, and S. Pizzi, “Virtual
izing nanosatellites in SDN/NFV enabled ground segments to enhance
service orchestration,” in _Proc. IEEE Glob. Commun. Conf._, Waikoloa,
HI, Dec. 2019, pp. 1–6.

[21] N. T. Kien, V. Hoang Anh, V. D. Phong, N. Ngoc Minh, and N. H. Thanh,

“Machine learning-based service function chain over UAVs: Resource
profiling and framework,” in _Proc. IEEE 31st Int. Telecommun. Netw. Appl._
_Conf._, Sydney, Australia, 2021, pp. 127–133.

[22] M. Akbari, A. Syed, W. S. Kennedy, and M. Erol-Kantarci, “Constrained

federated learning for AoI-limited SFC in UAV-aided MEC for smart agriculture,” _IEEE Trans. Mach. Learn. Commun. Netw._, vol. 1, pp. 277–295,
2023.

[23] J. Jia and J. Hua, “Dynamic SFC placement with parallelized VNFs in

data center networks: A DRL-based approach,” _ICT Exp._, vol. 10, no. 1,
pp. 104–110, Feb. 2024.

[24] Z. Jia, M. Sheng, J. Li, D. Zhou, and Z. Han, “VNF-Based service pro
vision in software defined LEO satellite networks,” _IEEE Trans. Wireless_
_Commun._, vol. 20, no. 9, pp. 6139–6153, Sep. 2021.

[25] I. F. Akyildiz and A. Kak, “The Internet of Space Things/CubeSats:

A ubiquitous cyber-physical system for the connected world,” _Comput._
_Netw._, vol. 150, pp. 134–149, Feb. 2019.

[26] P. Zhang, Y. Zhang, N. Kumar, and M. Guizani, “Dynamic SFC embedding

algorithm assisted by federated learning in space-air-ground-integrated
network resource allocation scenario,” _IEEE Internet Things J._, vol. 10,
no. 11, pp. 9308–9318, Jun. 2023.

[27] G. Wang, S. Zhou, S. Zhang, Z. Niu, and X. Shen, “SFC-Based service

provisioning for reconfigurable space-air-ground integrated networks,”
_IEEE J. Sel. Areas Commun._, vol. 38, no. 7, pp. 1478–1489, Jul. 2020.

[28] J. Li, W. Shi, H. Wu, S. Zhang, and X. Shen, “Cost-aware dynamic

SFC mapping and scheduling in SDN/NFV-Enabled space-air-groundintegrated networks for internet of vehicles,” _IEEE Internet Things J._,
vol. 9, no. 8, pp. 5824–5838, Apr. 2022.

[29] P. Zhang, P. Yang, N. Kumar, and M. Guizani, “Space-air-Ground inte
grated network resource allocation based on service function chain,” _IEEE_
_Trans. Veh. Technol_, vol. 71, no. 7, pp. 7730–7738, Jul. 2022.

[30] Y. Zeng, R. Zhang, and T. J. Lim, “Throughput maximization for UAV
Enabled mobile relaying systems,” _IEEE Trans. Commun._, vol. 64, no. 12,
pp. 4983–4996, Dec. 2016.

[31] Z. Jia, Q. Wu, C. Dong, C. Yuen, and Z. Han, “Hierarchical aerial

computing for Internet of Things via cooperation of HAPs and UAVs,”
_IEEE Internet Things J._, vol. 10, no. 7, pp. 5676–5688, Apr. 2023.

[32] J. Zhang et al., “Stochastic computation offloading and trajectory schedul
ing for UAV-Assisted mobile edge computing,” _IEEE Internet Things J._,
vol. 6, no. 2, pp. 3688–3699, Apr. 2019.

[33] Y. Liu, K. Xiong, Q. Ni, P. Fan, and K. B. Letaief, “UAV-Assisted

wireless powered cooperative mobile edge computing: Joint offloading,
CPU control, and trajectory optimization,” _IEEE Internet Things J._, vol. 7,
no. 4, pp. 2777–2790, Apr. 2020.

[34] A. A. Khuwaja, Y. Chen, N. Zhao, M.-S. Alouini, and P. Dobbins, “A

survey of channel modeling for UAV communications,” _IEEE Commun._
_Surveys Tuts._, vol. 20, no. 4, pp. 2804–2821, Fourth Quarter 2018.

[35] A. Golkar and I. Lluch i Cruz, “The federated satellite systems paradigm:

Concept and business case evaluation,” _Acta Astronautica_, vol. 111,
pp. 230–248, Jun. 2015.

[36] Z. Jia, M. Sheng, J. Li, and Z. Han, “Toward data collection and trans
mission in 6G space-air-ground integrated networks: Cooperative HAP
and LEO satellite schemes,” _IEEE Internet Things J._, vol. 9, no. 13,
pp. 10516–10528, Jul. 2022.

[37] D. Zhou, M. Sheng, R. Liu, Y. Wang, and J. Li, “Channel-aware mission

scheduling in broadband data relay satellite networks,” _IEEE J. Sel. Areas_
_Commun._, vol. 36, no. 5, pp. 1052–1064, May 2018.




[38] _Propagation Data and Prediction Methods Required for the Design of_

_Earth-Space Telecommunication Systems_, document P.618-12 Rec. ITUR, 2015.

[39] A. A. Al-Habob, O. A. Dobre, S. Muhaidat, and H. V. Poor, “Energy
efficient information placement and delivery using UAVs,” _IEEE Internet_
_Things J._, vol. 10, no. 1, pp. 357–366, Jan. 2023.

[40] J. Li, W. Shi, N. Zhang, and X. Shen, “Delay-aware VNF scheduling: A

reinforcement learning approach with variable action set,” _IEEE Trans._
_Cogn. Commun. Netw._, vol. 7, no. 1, pp. 304–318, Mar. 2021.

[41] L. A. Wolsey and G. L. Nemhauser, _Integer and Combinatorial Optimiza-_

_tion_ . Hoboken, NJ, USA: Wiley, 1999, vol. 55.

[42] J. Filar and K. Vrieze, _Competitive Markov Decision Processes_ . Berlin,

Germany: Springer Science & Business Media, 2012.

[43] K. Arulkumaran, M. P. Deisenroth, M. Brundage, and A. A. Bharath,

“Deepreinforcementlearning:Abriefsurvey,” _IEEESignalProcess.Mag._,
vol. 34, no. 6, pp. 26–38, Nov. 2017.


**Ziye Jia** (Member, IEEE) received the B.E., M.S.,
and Ph.D. degrees in communication and information systems from Xidian University, Xi’an, China,
in 2012, 2015, and 2021, respectively. From 2018
to 2020, she was a Visiting Ph.D. Student with the
Department of Electrical and Computer Engineering, University of Houston, Houston, TX, USA. She
is currently an Associate Professor with the Key
Laboratory of Dynamic Cognitive System of Electromagnetic Spectrum Space, Ministry of Industry
and Information Technology, Nanjing University of
Aeronautics and Astronautics, Nanjing, China. Her current research interests
include space-air-ground networks, aerial access networks, UAV networking,
resource optimization, and machine learning.


**Yilu Cao** was a postgraduate student with the College
of Electronic and Information Engineering, Nanjing
University of Aeronautics and Astronautics, Nanjing, China. Her research interests include space-airgroundintegratednetworks,networkfunctionvirtualization techniques, deep reinforcement learning, and
game theory.


**Lijun He** (Member, IEEE) received the B.S. degree in
electronic information science and technology from
Anqing Normal University, Anhui, China, in 2013,
and the Ph.D. degree in military communications
from the State Key Laboratory of ISN, Xidian University, Xi’an, China, in 2020. From 2018 to 2019,
he was with the University of Toronto, Toronto, ON,
Canada, as a Visiting Scholar funded by China Scholarship Council. From 2020 to 2022, he was a PostDoctoral Researcher with the School of Software,
Northwestern Polytechnical University, where he was
an Associate Professor with the School of Software, from 2022 to 2024. He is
currently an Associate Professor with the School of Information and Control
Engineering, China University of Mining and Technology, Xuzhou, China. His
current research interests include routing, scheduling, resource allocation, and
satellite communications.



Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY MADRAS. Downloaded on September 08,2025 at 12:18:33 UTC from IEEE Xplore. Restrictions apply.


11248 IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY, VOL. 74, NO. 7, JULY 2025



**Qihui Wu** (Fellow, IEEE) received the B.S. degree in
communications engineering and the M.S. and Ph.D.
degrees in communications and information systems
from the Institute of Communications Engineering,
Nanjing, China, in 1994, 1997, and 2000, respectively. From 2003 to 2005, he was a Post-Doctoral Research Associate with Southeast University, Nanjing.
From 2005 to 2007, he was an Associate Professor
with the College of Communications Engineering,
PLA University of Science and Technology, Nanjing,
where he was a Full Professor, from 2008 to 2016.
From March 2011 to September 2011, he was an Advanced Visiting Scholar
with the Stevens Institute of Technology, Hoboken, NJ, USA. Since May
2016, he was a Full Professor with the College of Electronic and Information
Engineering, Nanjing University of Aeronautics and Astronautics, Nanjing.
His current research interests include wireless communications and statistical
signal processing, with an emphasis on system design of software defined radio,
cognitive radio, and smart radio.


**Qiuming Zhu** (Senior Member, IEEE) received the
B.S. degree in electronic engineering from Nanjing
University of Aeronautics and Astronautics (NUAA),
Nanjing, China, in 2002 and the M.S. and Ph.D. degrees in communication and information system from
NUAA in 2005 and 2012, respectively. Since 2021,
he is a Professor in the Department of Electronic Information Engineering, NUAA. From 2016 to 2017,
and from 2018 to 2023, he was also an Academic
Visitor with Heriot-Watt University, Edinburgh, U.
K. He has authored or coauthored more than 160
articles in refereed journals and conference proceedings. He holds more than 50
China and international patents. His current research interests include channel
sounding, modeling, and emulation for the fifth/sixth generation (5G/6G) mobile
communication and unmanned aerial vehicles (UAV) communication systems,
3D spectrum mapping and environment awareness.


**Dusit Niyato** (Fellow, IEEE) received the B.Eng.
degree from the King Mongkuts Institute of Technology Ladkrabang, Bangkok, Thailand and the Ph.D.
degree in Electrical and Computer Engineering from
the University of Manitoba, Canada. He is currently
a Professor with the College of Computing and Data
Science, Nanyang Technological University, Singapore. His research interests are in the areas of mobile
generative AI, edge intelligence, quantum computing
and networking, and incentive mechanism design.



**Zhu Han** (Fellow, IEEE) received the B.S. degree
in electronic engineering from Tsinghua University,
Beijing, China, in 1997, and the M.S. and Ph.D.
degrees in electrical and computer engineering from
the University of Maryland, College Park, MD, USA
in 1999 and 2003, respectively. From 2000 to 2002,
he was an R&D Engineer of JDSU, Germantown,
Maryland. From 2003 to 2006, he was a Research
Associate at the University of Maryland. From 2006
to 2008, he was an Assistant Professor with Boise
State University, Idaho. He is currently a John and Rebecca Moores Professor in the Electrical and Computer Engineering Department
as well as in the Computer Science Department at the University of Houston,
Texas. Dr. Han’s main research targets on the novel game-theory related concepts
criticaltoenablingefficientanddistributiveuseofwirelessnetworkswithlimited
resources. His other research interests include wireless resource allocation and
management, wireless communications and networking, quantum computing,
data science, smart grid, carbon neutralization, security and privacy. Dr. Han was
the recepient of NSF Career Award in 2010, the Fred W. Ellersick Prize of the
IEEE Communication Society in 2011, the EURASIP Best Paper Award for the
Journal on Advances in Signal Processing in 2015, IEEE Leonard G. Abraham
Prize in the field of Communications Systems (best paper award in IEEE JSAC)
in 2016, IEEE Vehicular Technology Society 2022 Best Land Transportation
Paper Award, and several best paper awards in IEEE conferences. Dr. Han was
an IEEE Communications Society Distinguished Lecturer from 2015 to 2018
and ACM Distinguished Speaker from 2022 to 2025, AAAS fellow since 2019,
and ACM Fellow since 2024. Dr. Han is a 1% highly cited Researcher since 2017
according to Web of Science. Dr. Han is also the winner of the 2021 IEEE Kiyo
Tomiyasu Award (an IEEE Field Award), for outstanding early to mid-career
contributions to technologies holding the promise of innovative applications,
with the following citation: “for contributions to game theory and distributed
management of autonomous communication networks.”



Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY MADRAS. Downloaded on September 08,2025 at 12:18:33 UTC from IEEE Xplore. Restrictions apply.


