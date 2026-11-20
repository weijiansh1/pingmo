**飞行品质、飞控基础策略、特权训练与迭代蒸馏：文献梳理**

检索日期：2026-10-23。范围以固定翼飞行品质和低层飞控为主，同时覆盖四旋翼教师—学生方法；把 LLM/VLM 航空导航作为单独分支。共收录 32 篇核心文献。采用关键词检索和参考文献向后追溯，以论文、作者页面、正式会议/期刊和机构仓储为依据。这是一份有选择的研究梳理，不是按照数据库全量筛选流程完成的系统综述。

最直接的连接是：**以飞行品质定义控制目标，在仿真中训练具有额外状态/动力学知识的教师，再通过学生自身采样的蒸馏训练得到可部署策略，最后独立验证跨工况、跨机型或仿真到实机的表现。** 现有论文分别覆盖这条链的不同部分。本次未找到足够证据证明“多轮蒸馏会持续提高固定翼飞行品质”，也未找到一篇已经把这些环节全部完成并充分验证的论文。后一句是本次检索范围内的结论，不是对整个领域不存在相关工作的断言。

“飞控大模型”需按实际含义选文献。若指跨机型通用控制策略，最贴近的是 [RAPTOR](https://arxiv.org/abs/2509.11481)；它最终只有 2084 个参数，不能按模型规模称作 LLM 式大模型。若指语言、视觉语言基础模型，则应读 [OpenFly](https://arxiv.org/abs/2502.18041) 和[航空学报综述](https://hkxb.buaa.edu.cn/CN/10.7527/S1000-6893.2026.33148)，但这些工作的任务指标不能替代低层控制评价。

下表把各关键词放在同一条技术链上。它是本报告的归纳，不是某一篇论文的原图。

| 关键词 | 实际解决的问题 | 训练/部署边界 | 优先参考 |
|---|---|---|---|
| 飞行品质 | 飞机在规定任务下的响应、稳定性和可操纵性；有人驾驶还涉及补偿与工作负荷 | 回报、轨迹误差和飞行员评分不是同一指标 | F01、F02 |
| 教师模型 | 提供较好的动作、轨迹、分布或潜变量监督 | 可以是 MPC/规划器、RL 策略或多工况专家集，不必是大网络 | P01、P02、G01 |
| 特权训练 | 使用部署时缺失但训练时可得的信息 | 要逐项说明完整状态、动力学参数、环境地图等信息是否仅训练可用 | D02、P04、P06 |
| 学生模型 | 用机载可用信息完成闭环控制 | 可能需要历史记忆才能处理部分可观测性；模型更小不是唯一目标 | G01、G02、P03 |
| 蒸馏 | 把教师知识转入学生表示或结构 | 动作回归、分布匹配、轨迹模仿和潜变量匹配不是同一损失 | D01、D03、D05 |
| 多轮蒸馏 | 解决数据分布、模型容量或教师改进问题 | 必须说明每轮变的是数据、学生、教师还是环境 | D04、D06、D07、D10 |
| 迁移学习 | 将已有能力用于不同动力学、传感器或任务 | 区分零梯度部署、历史内适应、少样本微调与重新训练 | F03、G01、G05、T01、T02 |
| 强化学习 | 从闭环交互与奖励中优化行为 | 可训练教师，也可直接训练学生；并非所有特权学习都使用 RL | F02、F03、G04、D08、D09 |

最值得优先读的八篇及原因如下。若你的研究对象最终确定为固定翼，先读前两篇，再用后六篇形成方法背景。

1. **F02 Jansen & van Kampen，2025**：直接把 SAC 飞控与纵向飞行品质联系起来。需要注意，本次只读到正式摘要、PDF 首页及部分引言，后续正文端点超时；没有替其推断未核验的评价细节。
2. **F03 Bøhn et al.，2024**：固定翼实飞与 sim-to-real，能帮助判断传感器、执行器延迟和模拟器设计的重要性。
3. **G01 RAPTOR，2026**：多教师 RL、循环学生、跨机型适应和嵌入式部署的直接组合。
4. **G02 Extreme Adaptation，2025**：动力学潜变量、历史推断与低层电机控制的直接相关工作。
5. **P03 Precise Aggressive Aerial Maneuvers，2026**：RL 教师到视觉学生的完整飞行实例。
6. **D02 Unifying Distillation and Privileged Information，2016**：解释为何“教师比学生知道得多”和“教师比学生大”是两个不同维度。
7. **D04 DAgger，2011**：说明学生闭环采样对减少分布偏移的作用。
8. **D05 Distilling Policy Distillation，2019**：确定蒸馏的采样分布、损失与 RL 奖励如何结合；尤其适合设计多轮训练。

关于飞行品质，建议在读论文时分清三个层次。第一层是指令或轨迹跟踪，例如角速度误差、超调和调节时间。第二层是飞行品质相关动态特性，例如短周期响应、阻尼、带宽/延迟及适用条件下的 CAP。第三层是规定任务中的飞行员补偿和工作负荷。[Cooper–Harper 原报告](https://ntrs.nasa.gov/citations/19690013177)和 [NASA 飞行品质经验报告](https://ntrs.nasa.gov/api/citations/20090025299/downloads/20090025299.pdf)可以帮助理解第三层。对自主无人机，不能直接把不存在飞行员参与的控制回报当作 Cooper–Harper 评分。

[MIL-STD-1797 的官方目录页](https://quicksearch.dla.mil/qaDocDetails.aspx?ident_number=70037)可用作标准检索入口，其对象是有人驾驶航空器。本报告没有核验标准正文数值，不提供可直接套用的阈值。固定翼、四旋翼以及不同任务阶段应明确各自评价模型和适用范围。

“特权信息”至少有三种用法，做相关工作时应分开写：

- **特权教师策略**：教师掌握学生部署时没有的状态或环境信息，再监督学生。Deep Drone Acrobatics、Learning by Cheating 属于这一主线。
- **非对称 actor–critic**：critic 可看完整状态，actor 始终只看部署观测；可以不经过独立教师—学生蒸馏。Pinto 和 Learning to Fly in Seconds 提供这一方向的依据。
- **特权参数/潜变量训练**：训练时知道动力学或环境参数，部署时由历史估计相应潜变量。RMA、Extreme Adaptation 是重要参考；RAPTOR 则通过不同动力学上的专用教师和循环学生实现适应，不能简单描述成教师都把质量、惯量显式作为输入。

教师的额外信息也有可恢复性边界。如果两个真实状态产生相同的学生可用观测历史，却要求不同的最优动作，单纯增加蒸馏轮数无法消除这种歧义。这是对部分可观测控制的分析，不是上述某篇论文已经证明所有飞控场景都会发生此问题。研究中需要检查历史长度、传感器配置和教师使用的信息，而不是只比较网络层数。[广义蒸馏](https://arxiv.org/abs/1511.03643)、[RMA](https://arxiv.org/abs/2107.04034)和 [RAPTOR 的方法推导](https://arxiv.org/html/2509.11481v2)是理解这一边界的入口。

“多轮蒸馏”建议明确为下列某一种。

| 形式 | 每轮究竟改变什么 | 代表论文 | 对飞控研究的意义和限制 |
|---|---|---|---|
| 多代自蒸馏 | 新学生学习上一代模型，之后成为下一代教师 | BAN，D06 | 可研究同容量模型的训练收益；原论文报告收益递减，没有飞控品质随代数单调提高的结论 |
| 多级容量压缩 | 大教师→中间助手→小学生 | TAKD，D07 | 对大教师与嵌入式小模型的容量差有针对性；不自动解决信息缺失 |
| 交互式反复蒸馏 | 当前学生采样→专家标注→更新学生 | DAgger，D04；RAPTOR，G01 | 与闭环控制关联最直接。须说明是否保留历史聚合数据，不能把全部 on-policy 变体都等同于原始 DAgger |
| 教师与学生共同改进 | 专家规划改进→学生模仿→学生辅助下一轮专家 | Expert Iteration，D10 | 可启发规划/RL 教师迭代，但需要适合连续飞行动力学的具体实现 |
| 蒸馏与环境 RL 同时进行 | 奖励更新加教师监督，监督权重可调整 | Kickstarting，D09；Distral，D08 | 可以研究超越教师和目标域适应；两者的训练结构也不同，不能仅统称“反复蒸馏” |
| 多教师合并 | 多个专用教师监督同一个学生 | Policy Distillation，D03；RAPTOR，G01 | 解决多任务/多动力学知识合并；教师数量与蒸馏代数是独立变量 |

对于飞控，建议首先研究“学生实际闭环状态上的交互式蒸馏”。依据是原始 Policy Distillation 明确讨论了教师轨迹与 DAgger 的差异，RAPTOR 在短暂教师热身后主要采用当前学生采样，2026 年狭缝飞行论文也采用当前策略数据的迭代模仿。这个阅读顺序是本报告的判断，不代表在你的飞行器上已经验证它优于所有替代方案。[Rusu](https://arxiv.org/html/1511.06295)、[RAPTOR](https://arxiv.org/html/2509.11481v2)、[Wu et al.](https://arxiv.org/html/2604.05828v1)

论文实际的 Related Work / Introduction 引用关系如下。表中“启发”是本报告的分析；“正文比较”指已读到的原文关系。不同版本会重排参考文献编号，因此以下以题名与作者定位。

| 已查论文与位置 | 原文追溯或比较的工作 | 原文定位与对本课题的启发 |
|---|---|---|
| [RAPTOR，引言与 Meta-Imitation Learning](https://arxiv.org/html/2509.11481v2) | One Net to Rule Them All；Extreme Adaptation；Learning by Cheating；Learning quadrupedal locomotion over challenging terrain；RMA；A-NC；DAgger | 正文比较当前状态策略与历史适应策略，也比较低层角速度控制与完整位置到电机控制。其 on-policy 学习与原始 DAgger 的历史数据聚合不同。研究跨机型策略必须同时对齐记忆、控制层级和教师数量。 |
| [Bøhn，引言](https://arxiv.org/html/2111.04153) | 作者 2019 年 PPO 固定翼工作；已有固定翼特定着陆/风洞 RL；ADP、PID/LQR、MPC；sim-to-real 方法 | 从仿真姿态控制推进到无约束室外实飞，强调延迟和随机化。原文指出 MPC 对照在目标与硬件上有差异，不应自行宣称其已在统一条件下全面超过 MPC。 |
| [Deep Drone Acrobatics，§II Related Work](https://arxiv.org/html/2006.05768) | Lupashin 的迭代特技控制；Mellinger 的机动；Abbeel 的直升机示范学习；机载视觉敏捷飞行；MPC 专家模仿 | 区分仅解决控制且依赖外部准确状态的方案，与处理机载感知的策略。教师可由最优控制生成，贡献涉及感知与迁移，不能简化成“更大的教师网络”。 |
| [Learning High-Speed Flight in the Wild，引言、图 6 与补充分类](https://rpg.ifi.uzh.ch/docs/Loquercio21_Science.pdf) | 分模块感知/建图/规划；感知延迟；学习感知—动作与特权专家方法 | 全图规划专家监督局部感知学生，学生预测轨迹后仍有 MPC 跟踪。引用它时应写清知识转移发生在规划层，避免和直接电机策略混为一谈。 |
| [Extreme Adaptation，§I-A Related Work](https://arxiv.org/html/2409.12949) | MRAC、L1、MPC、自适应 INDI；学习控制；从模型控制器模仿；四足机器人跨环境适应 | 将主要挑战落在跨四旋翼的执行器和动力学变化。跨机型效果应与经典自适应、历史适应器以及单一随机化策略比较。 |
| [DATT，Related Work 与实验/附录](https://arxiv.org/html/2310.09053) | 非线性 MPC；L1 自适应；RMA；域随机化与教师—学生适应 | 将 RMA 式学习适应器作为真实基线，观察到仿真和实机排序改变。这里的负面结果是保留经典扰动估计基线的直接理由。 |
| [Policy Distillation，§2 Previous Work](https://arxiv.org/html/1511.06295) | Buciluă 的 Model Compression；Hinton 蒸馏；深度 RL；多任务学习；规划器监督；DAgger | 策略蒸馏具有多条前史，不应只追溯模型压缩。尤其指出自身教师轨迹方案与 DAgger 学习者采样的区别。 |
| [Unifying Distillation and Privileged Information，框架与 Comparison to prior work](https://arxiv.org/html/1511.03643) | Hinton 的蒸馏；Vapnik/Vashist 与 Vapnik/Izmailov 的 LUPI；SVM+ | 提供教师/学生信息表示不同的统一视角。容量压缩和特权信息迁移可以同时存在，也可以各自独立。 |
| [Born Again Neural Networks，引言与 Related Work](https://arxiv.org/html/1805.04770) | Born-Again Trees；Model Compression；Hinton；浅层/深层知识转移 | 研究同容量学生和多代训练，贡献不只是压缩。它的实验领域和收益递减现象都应在引用中保留。 |
| [Teacher Assistant，Related Work](https://arxiv.org/html/1902.03393) | 模型压缩；输出及中间层蒸馏；多模型/互学习；蒸馏理论 | 作者定位是固定教师/学生大小下用中间容量网络改善转移。它不能单独支撑“反复采样改善闭环分布”的论点。 |
| [Learning to Fly in Seconds，§III Related Works](https://arxiv.org/html/2311.13081) | Flightmare；不同输出层级的 RL 飞控；端到端电机控制；PPO 与 TD3 | 引用速度和控制效果时要对齐仿真复杂度、动作层级、样本量与训练硬件。把只输出角速度的策略与直接电机策略放在同一张表会误导比较。 |
| [2026 年狭缝飞行论文，引言与 Policy Distillation](https://arxiv.org/html/2604.05828v1) | 窄缝轨迹规划与机载感知；传感动作学习；Policy Distillation；Learning by Cheating；DAgger | 将现有蒸馏框架与适合狭窄解空间的 RL 初始化结合。其贡献包含任务约束、探索和迁移设计，不应全部归因于“蒸馏轮数”。 |
| [何海洋等，参考文献表](https://bhxb.buaa.edu.cn/bhzk/cn/article/doi/10.13700/j.bh.1001-5965.2024.0075) | Bøhn 2019/2024；示教知识辅助无人机 RL；固定翼六自由度 RL；传统鲁棒/容错控制 | 可沿中文文献继续扩展。本文只核对到其参考文献表，未把这些二级文献都当作已读原文或独立实证。 |

F02 是与飞行品质最直接的工作，但本次没有读全其 Related Work/参考文献。已见引言将问题连接到增益调度、ADP/自适应评价器和增量方法；以上没有进一步列出未经核验的具体引用。写正式论文前，应补读全文及其 2024 年学位论文，确认所用飞行品质判据、评价包线、非线性辨识方法和实验层级。

若要写自己的 Related Work，可按四段组织，而不是把九个关键词各写一节。

第一段讨论**飞行品质约束下的学习飞控**：由经典任务/响应评价引入 RL 固定翼控制，再区分 Jansen 的飞行品质分析与 Bøhn 的实飞迁移证据。第二段讨论**特权信息到部署观测的转移**：由广义蒸馏、非对称 actor–critic 和 Learning by Cheating 引出飞行特技、野外飞行及狭缝穿越。第三段讨论**历史适应与跨机型统一策略**：以 RMA、Extreme Adaptation、DATT、RAPTOR 对照隐式参数推断、经典扰动估计和多教师整合。第四段讨论**迭代蒸馏**：先用 DAgger/Czarnecki 说明闭环分布，再区分 BAN、TAKD、Distral/Kickstarting 和 Expert Iteration 的不同迭代对象。

下面是一段可改写进论文的中文草稿。它只概述已有工作，没有预设你的方法已经获得性能提升。

> 学习飞控研究已从仿真中的指令跟踪扩展至真实固定翼和四旋翼平台。Bøhn 等通过固定翼实飞研究了强化学习姿态控制的仿真迁移，而 Jansen 与 van Kampen 将强化学习控制器置于纵向飞行品质框架中评价。另一方面，特权学习利用仿真中的完整状态或环境信息训练专家，再将行为迁移至依赖部署观测的策略；Deep Drone Acrobatics、Learning High-Speed Flight in the Wild 以及近期的狭缝机动学习展示了这一思路在不同飞行任务中的应用。为进一步适应未知动力学，RMA 和 Extreme Adaptation 使用交互历史估计控制相关潜变量，RAPTOR 则将大量专用强化学习教师整合为循环学生策略。闭环蒸馏还需要处理学习者诱导的状态分布变化，DAgger 与 Distilling Policy Distillation 为此提供了方法与理论依据。上述文献分别支持学习控制、特权知识转移和跨系统适应，但它们的验证对象并不相同；迭代蒸馏对固定翼飞行品质的影响仍需在明确的任务、包线和资源预算下进行直接比较。

上述草稿的引用依次对应 F03、F02、P01–P03、P06、G02、G01、D04、D05；正式稿应按你的引用格式插入文末条目。

如果把它发展为课题，较清楚的待检验问题是：**在相同机载观测、学生容量与训练资源下，多工况教师加交互式蒸馏，能否提高未见固定翼工况中的飞行品质达标比例，并减少违约或失稳？** 这是由文献推导的研究问题，不是本次已验证的效果。只把“大模型、特权训练、多轮蒸馏、迁移”串联起来，还不足以形成独立贡献。

建议的比较设计如下，也是本报告的研究建议。

| 要回答的问题 | 应比较的方法 | 需要固定的条件 |
|---|---|---|
| 教师监督是否有贡献 | 学生直接 RL；非对称 actor–critic；单次教师蒸馏 | 机载观测、学生容量、交互预算、评估工况 |
| 多轮有无额外收益 | 单次蒸馏；学生采样的交互式蒸馏；增加普通训练步数的等预算基线 | 总模拟步数、专家查询数、总计算量，至少逐项报告 |
| 多教师是否必要 | 单个随机化教师；多个工况教师；直接随机化学生 | 工况覆盖、教师总训练预算与学生结构 |
| 历史记忆是否有贡献 | 当前状态策略；历史窗口策略；循环策略；经典参数/扰动估计器 | 动作层级、延迟、传感器质量 |
| 迁移是否成立 | 训练分布、未见参数组合、范围外测试、仿真/实机分别报告 | 开发与测试分离；不得将测试结果用于反复选择最好轮数 |

评价除跟踪误差外，还应报告适用的飞行品质判据、舵面/电机饱和与变化率、闭环振荡、控制延迟、计算资源及失败比例。跨机型任务必须同时说明动作归一化、指令层级和传感器配置。多次随机种子、预先声明的完整测试工况以及失败结果都应保留，避免把平均误差下降当作全部控制性质改善。

可复用的检索词包括：

- `("handling qualities" OR "flying qualities") AND ("reinforcement learning" OR "neural flight control")`
- `("fixed-wing" OR aircraft) AND ("policy distillation" OR "teacher student" OR "privileged information")`
- `(quadrotor OR "flight control") AND ("foundation policy" OR "cross-platform" OR "rapid adaptation")`
- `("on-policy distillation" OR "dataset aggregation" OR "iterative distillation") AND (control OR robotics)`
- `("teacher assistant" OR "born again" OR "expert iteration") AND (distillation OR "reinforcement learning")`
- `飞行品质 强化学习`、`固定翼 特权学习 教师 学生`、`飞控 多教师 蒸馏`、`飞行器 动力学 迁移学习`

版本与阅读边界：RAPTOR、Extreme Adaptation、Bøhn、航空 RL 综述、OpenFly 的预印本与正式发表时间不同，表中尽量采用核实的正式年份；仅确认预印本的条目保留预印本身份。矩阵中的 read_scope 记录本次实际阅读范围。没有把全部 32 篇都描述为已精读全文，也没有把检索到的参考文献自动当成已读原始论文。


以下是全部 32 篇的文献矩阵。每条给出原始来源链接、核心内容、证据对象和本次阅读范围。


**飞行品质与固定翼强化学习**

| 编号与文献 | 核心内容 | 验证范围 | 本次阅读范围 |
|---|---|---|---|
| F01 · [The Use of Pilot Rating in the Evaluation of Aircraft Handling Qualities](https://ntrs.nasa.gov/citations/19690013177) · 1969 · NASA TN D-5153 | Cooper–Harper 评价的基础文献，把完成规定任务的性能与飞行员补偿/工作负荷联系起来。用于明确飞行品质的评价对象，不能把自主策略的回报直接换算为飞行员评分。 | 评价方法与技术报告 | NASA 书目信息及摘要；定义另与 NASA 2009 年经验报告交叉核对 |
| F02 · [Longitudinal Handling Qualities Evaluation for Soft Actor-Critic Deep Reinforcement Learning Flight Control](https://repository.tudelft.nl/record/uuid:5078b54b-5ea7-4ec4-afce-0e8964361fb8) · 2025 · AIAA SCITECH 2025 Forum | 以 SAC 设计喷气机俯仰角速度指令控制系统，将学习控制器放回纵向飞行品质框架评价。摘要报告在所测试的非标称飞行条件、重心偏移和传感器偏差下满足 Level 1；应把结论限制在论文评价条件内。 | 论文报告的纵向闭环评价；本次未核清全部实验条件，不列为已核验实飞 | 正式发表信息、摘要、PDF 首页及部分引言；后续正文读取超时 |
| F03 · [Data-Efficient Deep Reinforcement Learning for Attitude Control of Fixed-Wing UAVs: Field Experiments](https://arxiv.org/abs/2111.04153) · 2024 · IEEE Transactions on Neural Networks and Learning Systems | SAC 固定翼姿态控制，从仿真迁移至 Skywalker X8 实飞，与 ArduPlane PID 比较。执行器延迟和动力学随机化对迁移很关键。作者报告实飞性能相当；不能改写为全面优于 PID。 | 仿真、室外固定翼实飞、PID 对照与线性分析 | 正文引言、相关工作、方法与实验描述；期刊元数据经 PubMed 核对 |
| F04 · [基于深度强化学习的固定翼无人机纵向控制](https://bhxb.buaa.edu.cn/bhzk/cn/article/doi/10.13700/j.bh.1001-5965.2024.0075) · 2026 · 北京航空航天大学学报 52(4):1306–1315 | 以多时刻速度、俯仰角误差等训练 DDPG，输出升降舵与推力。比较指令变化和参数摄动下的 PID；摘要仅称部分情况下性能更好。可作为中文固定翼端到端控制入口。 | 飞行仿真与 PID 对照；不是实飞或飞行品质评级证据 | 期刊摘要、图表与全部 22 条参考文献 |
| F05 · [Neural Network-based Flight Control Systems: Present and Future](https://arxiv.org/abs/2206.05596) · 2022 · Annual Reviews in Control 53:97–137 | 梳理神经网络飞控的模型方法、直接/间接自适应、近似动态规划及 RL，并讨论闭环稳定性条件。适合补足飞控背景，避免只从深度学习视角讨论控制器。 | 综述，不构成一项新的实飞效果验证 | 作者摘要及正式出版信息 |
| F06 · [A Survey on Reinforcement Learning in Aviation Applications](https://arxiv.org/abs/2211.02147) · 2024 · Engineering Applications of Artificial Intelligence 136:108911 | 航空 RL 的总体检索入口，区分规划、运行决策与实时控制。适合扩展检索，不宜把其中航空调度或决策任务的结果作为低层飞控证据。 | 航空 RL 综述 | 摘要与期刊元数据 |

F02 版本/使用说明：2024 年 H. Jansen 学位论文与此会议论文同一研究脉络，不作为两项独立证据重复计数


F03 版本/使用说明：2021 年预印本、2023 年 DOI/早期出版标识、2024 年卷期；按正式卷期列年。文中约 3 分钟指模拟交互数据量，不是仅用 3 分钟实飞完成全部训练


F04 版本/使用说明：DOI 中的 2024 不是正式卷期年份；正式刊期为 2026


F06 版本/使用说明：预印本始于 2022，期刊正式发表为 2024


**通用飞控策略与跨机型适应**

| 编号与文献 | 核心内容 | 验证范围 | 本次阅读范围 |
|---|---|---|---|
| G01 · [RAPTOR: A Foundation Policy for Quadrotor Control](https://arxiv.org/abs/2509.11481) · 2026 · Science Robotics 11(114):eaec1481 | 为 1000 个采样四旋翼分别训练 RL 教师，再经 Meta-Imitation Learning 得到含 GRU 的学生。2084 参数策略在 10 种真实四旋翼上验证，覆盖 32 g–2.4 kg。学生靠观测—动作历史适应未知动力学；基础策略不等于参数庞大的语言模型。 | 跨机型实飞、仿真、教师数量与学生容量分析；未验证固定翼飞行品质 | arXiv v2 正文、引言引用链、训练算法与补充推导；正式期刊信息取作者代码仓库 |
| G02 · [A Learning-Based Quadcopter Controller With Extreme Adaptation](https://arxiv.org/abs/2409.12949) · 2025 · IEEE Transactions on Robotics 41:3948–3964 | 把 RL 与模仿学习结合，利用传感器—动作历史推断动力学潜变量，以适应质量、尺寸和执行器变化。对 RAPTOR 是直接相关的先行工作。它接收总推力与机体角速度指令，比较时需对齐控制层级。 | 仿真及不同四旋翼实飞；低层控制与适应 | 正文 Related Work、方法及实验摘要；正式卷期与 DOI 核对 |
| G03 · [One Net to Rule Them All: Domain Randomization in Quadcopter Racing Across Different Platforms](https://arxiv.org/abs/2504.21586) · 2025 · arXiv:2504.21586 | 以动力学随机化训练只依赖当前状态的统一赛车控制策略，在两种真实四旋翼上测试。论文比较随机化幅度，揭示速度与鲁棒性之间的取舍，是检验是否需要历史记忆和教师蒸馏的重要基线。 | 两平台实飞与随机化强度比较 | 作者摘要及 RAPTOR 对该工作的具体比较 |
| G04 · [Learning to Fly in Seconds](https://arxiv.org/abs/2311.13081) · 2024 · IEEE Robotics and Automation Letters 9(7):6336–6343 | 以快速动力学仿真、非对称 actor–critic 与 TD3 训练直接输出电机指令的策略，并迁移到 Crazyflie。Related Works 明确区分位置、速度、总推力/角速度与电机控制层级；是 RAPTOR 的训练基础。 | 仿真训练及真实四旋翼轨迹跟踪对照 | 正文 Related Works、方法与训练/迁移说明 |
| G05 · [DATT: Deep Adaptive Trajectory Tracking for Quadrotor Control](https://proceedings.mlr.press/v229/huang23a.html) · 2023 · Conference on Robot Learning, PMLR 229:326–340 | 组合 RL 轨迹跟踪和 L1 自适应扰动估计。其对照中，学习式 RMA 适应器在仿真表现较好，却在真实无人机上逊于 L1 方案，作者将其与 sim-to-real 偏移联系起来。说明特权潜变量学习不应被默认视为最佳适应方案。 | 真实四旋翼、有风与扰动条件、MPC/非线性控制/RMA 对照 | 正文 Related Work、实验与附录关于 DATT-RMA 的比较 |

G01 版本/使用说明：2025 年首个预印本、2026 年正式论文；1000 个教师是多教师，不是 1000 代蒸馏


G02 版本/使用说明：2024 年预印本，2025 年期刊发表


G03 版本/使用说明：本次按已核实的预印本版本引用，未确认后续正式出版


G04 版本/使用说明：2023 年预印本、2024 年正式期刊；训练用时依赖所报告的硬件和任务


**特权教师与可部署学生**

| 编号与文献 | 核心内容 | 验证范围 | 本次阅读范围 |
|---|---|---|---|
| P01 · [Deep Drone Acrobatics](https://arxiv.org/abs/2006.05768) · 2020 · Robotics: Science and Systems | 由拥有特权状态的最优控制器提供示范，学生根据机载传感历史和参考轨迹输出控制。以感知抽象与迭代模仿支持仿真到真实特技飞行。教师是最优控制器，不应被写成大语言模型或必然由 RL 训练的网络。 | 仿真和真实四旋翼特技飞行 | 全文可读，重点阅读 §II Related Work、教师/学生定义与 DAGGER 训练 |
| P02 · [Learning High-Speed Flight in the Wild](https://matthias.pw/publication/high_speed) · 2021 · Science Robotics 6(59):eabg5810 | 掌握完整环境地图和飞行状态的规划专家生成轨迹监督；学生由深度、速度、姿态及目标方向预测候选轨迹，再交给 MPC 跟踪。验证森林等未见环境中的实飞迁移，重点是感知—规划，不是直接蒸馏电机控制器。 | 多种真实环境高速飞行及基线/消融 | 作者公开 PDF，重点阅读引言、方法图 6、特权专家和学生学习 |
| P03 · [Precise Aggressive Aerial Maneuvers with Sensorimotor Policies](https://arxiv.org/abs/2604.05828) · 2026 · Science Robotics | 先用低维 oracle 观测训练 RL 教师，以规划轨迹辅助初始化探索，再把行为蒸馏到使用视觉/本体感知历史的学生。采用当前学生采样的迭代模仿方式，实飞验证狭缝穿越。与用户关键词中的 RL—特权训练—学生—迁移联系直接。 | 真实狭缝穿越、硬件在环比较及域随机化消融 | arXiv 正文引言、训练与蒸馏章节；期刊页面方法和实验 |
| P04 · [Asymmetric Actor Critic for Image-Based Robot Learning](https://arxiv.org/abs/1710.06542) · 2017 · arXiv:1710.06542 | 训练时让 critic 使用完整状态，actor 使用部署可得的图像。它利用特权信息改善训练，但不要求另训练一个教师 actor 再蒸馏；应作为两阶段教师—学生方案的独立基线。 | 机器人图像控制方法；不是固定翼证据 | 原作者摘要；与 Learning to Fly in Seconds 的使用方式交叉核对 |
| P05 · [Learning by Cheating](https://proceedings.mlr.press/v100/chen20a.html) · 2020 · Conference on Robot Learning, PMLR 100:66–75 | 先从易用的特权环境表示学习驾驶策略，再训练视觉学生。是将“决策学习”与“从真实传感器获得决策能力”分开的代表工作；飞控文献常沿用这一思路。 | 自动驾驶实验；方法迁移参照，不能当作航空验证 | PMLR 原文摘要与 RAPTOR 对该工作的引用说明 |
| P06 · [RMA: Rapid Motor Adaptation for Legged Robots](https://arxiv.org/abs/2107.04034) · 2021 · Robotics: Science and Systems | 训练中编码环境/动力学特权参数，部署时以状态—动作历史预测所需潜变量并驱动策略。在四足机器人上验证快速适应，是飞控历史适应器的重要方法来源。 | 四足机器人实物实验；不能外推为无人机或固定翼效果 | 正文方法与相关文献；结合 DATT 的跨领域反例 |
| P07 · [A-NC: Adaptive Neural Control with implicit online inference of privileged parameters](https://proceedings.mlr.press/v283/paluch25a.html) · 2025 · Learning for Dynamics and Control, PMLR 283:987–998 | 从知道不可测参数的 NMPC 生成监督，训练 GRU 根据历史隐式适应。比较角度传感器偏移、杆长与质量变化，为“历史是否能恢复教师的特权信息”提供较清晰的实验模型。 | 仿真和物理倒立摆；不是飞行器证据 | PMLR 原文摘要与 RAPTOR 的直接引用 |

P03 版本/使用说明：预印本与正式期刊参考文献编号不同；以下引用链只按标题对应


P04 版本/使用说明：此条明确按 2017 年预印本记录


P05 版本/使用说明：会议届次为 CoRL 2019，论文集出版年为 2020


**蒸馏、迭代蒸馏与强化学习结合**

| 编号与文献 | 核心内容 | 验证范围 | 本次阅读范围 |
|---|---|---|---|
| D01 · [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) · 2015 · arXiv:1503.02531 | 知识蒸馏基础：使用教师输出的软分布训练学生。可用于压缩，也可传递集成模型行为；分类温度缩放形式迁移到连续动作时需要重定义合适的策略匹配损失。 | 监督学习蒸馏；不是闭环飞控保证 | 原论文摘要，并沿 Policy Distillation、BAN、Generalized Distillation 的引用追溯 |
| D02 · [Unifying Distillation and Privileged Information](https://arxiv.org/abs/1511.03643) · 2016 · International Conference on Learning Representations | 统一知识蒸馏与 LUPI：教师与学生可以使用不同信息表示，教师未必只靠参数量取胜。适合为仿真真值教师、有限观测学生提供概念框架，但不能恢复部署观测中根本不存在的信息。 | 理论分析与合成/实际数据实验；不含飞行验证 | 正文框架、Comparison to prior work 与参考文献 |
| D03 · [Policy Distillation](https://arxiv.org/abs/1511.06295) · 2016 · International Conference on Learning Representations | 将 RL 策略迁移和压缩到学生，并把多个任务教师合并为一个策略。Previous Work 把方法定位于模型压缩、深度 RL、多任务和模仿学习的交叉处，是多飞行工况专家合并的基础参照。 | Atari 单任务/多任务实验；不是连续飞控验证 | 正文 §2 Previous Work、单任务/多任务及在线蒸馏描述 |
| D04 · [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html) · 2011 · AISTATS, PMLR 15:627–635 | DAgger 让学习者访问状态，再向专家询问标签并聚合数据，处理闭环模仿中的分布偏移。对飞控，“多轮”更应关注不断收集学生自身状态，而不只是重复训练同一批教师轨迹。 | 序列决策理论与实验；保证有条件 | PMLR 摘要、可访问正文及飞控论文中的具体使用 |
| D05 · [Distilling Policy Distillation](https://proceedings.mlr.press/v89/czarnecki19a.html) · 2019 · AISTATS, PMLR 89:1331–1340 | 分析谁来采样轨迹、如何匹配策略以及如何加入环境奖励；某些直观组合并不对应所声称的梯度目标，可能振荡。应先定义采样分布与更新规则，再谈迭代蒸馏效果。 | 理论与随机 MDP 实验；不提供任意神经飞控的收敛保证 | 公开 PDF 引言、方法分类及相关理论陈述 |
| D06 · [Born Again Neural Networks](https://proceedings.mlr.press/v80/furlanello18a.html) · 2018 · ICML, PMLR 80:1607–1616 | 同容量学生重新初始化后学习前一代教师，可继续多代；视觉和语言任务中报告改善，但也指出收益递减。最直接对应“学生再成为教师”的多代蒸馏，不能据此推断飞控品质随代数单调上升。 | 分类和语言建模实验；不是飞控闭环实验 | 正文引言、Related Work 与多代训练描述 |
| D07 · [Improved Knowledge Distillation via Teacher Assistant](https://ojs.aaai.org/index.php/AAAI/article/view/5963) · 2020 · AAAI Conference on Artificial Intelligence | 当教师和学生容量差距较大时，引入中间容量的教师助手进行分级蒸馏。对应大教师→中模型→机载小模型的路线；解决的是容量差距，不自动解决传感器缺失和闭环分布偏移。 | 监督学习压缩对照实验 | 正文 Related Work、动机与教师助手方案；AAAI 元数据 |
| D08 · [Distral: Robust Multitask Reinforcement Learning](https://arxiv.org/abs/1707.04175) · 2017 · Advances in Neural Information Processing Systems | 各任务策略学习并向共享策略蒸馏，任务策略又受共享策略的 KL 正则引导。适合研究多工况之间的知识共享及负迁移；它是联合多任务学习，而不是单向多代压缩。 | 多任务 RL 基准实验 | 作者摘要及可访问正文 |
| D09 · [Kickstarting Deep Reinforcement Learning](https://arxiv.org/abs/1803.03835) · 2018 · arXiv:1803.03835 | 用现有教师加速学生 RL，学生同时从环境奖励学习，教师约束可随训练调整。适合以传统控制器或旧任务策略启动新飞控学习，并研究学生能否在目标域进一步改进。 | RL 基准；对航空效果需另做验证 | 摘要、正文关于 teacher–student 与 RL 的结合 |
| D10 · [Thinking Fast and Slow with Deep Learning and Tree Search](https://arxiv.org/abs/1705.08439) · 2017 · Advances in Neural Information Processing Systems | Expert Iteration 交替进行专家搜索改进与学生模仿，再用学生提升下一轮搜索。对应教师也不断变强的迭代改进框架；若用于飞控，必须另设计连续动力学下的可行专家。 | Hex 棋类实验；仅作方法参照 | 原论文摘要及版本说明 |

D02 版本/使用说明：2015 年预印本，ICLR 2016 正式论文


D03 版本/使用说明：2015 年预印本，2016 年会议版本


D07 版本/使用说明：2019 年预印本，AAAI 2020


**迁移学习与大模型边界**

| 编号与文献 | 核心内容 | 验证范围 | 本次阅读范围 |
|---|---|---|---|
| T01 · [Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) · 2018 · IEEE International Conference on Robotics and Automation | 训练时随机化动力学参数，降低策略对单一模拟器的依赖。飞控可据此覆盖质量、惯量、执行器延迟等变化，但随机化分布需要与真实系统匹配。 | 机器人 sim-to-real；不是固定翼实验 | 原作者摘要及 RAPTOR 引用条目 |
| T02 · [Transfer Learning in Deep Reinforcement Learning: A Survey](https://pmc.ncbi.nlm.nih.gov/articles/PMC11018366/) · 2023 · IEEE Transactions on Pattern Analysis and Machine Intelligence | 按知识类型和迁移机制整理深度 RL 迁移，包括策略蒸馏、表示和模型迁移。用于区分同机换任务、跨机型、仿真到实机和少样本微调，并检查负迁移问题。 | 综述 | 作者公开稿摘要及策略蒸馏段落 |
| L01 · [OpenFly: A Comprehensive Platform for Aerial Vision-Language Navigation](https://arxiv.org/abs/2502.18041) · 2026 · International Conference on Learning Representations | 提供航空视觉语言导航的数据生成、基准和导航模型，把图像与语言指令用于导航决策。适用于 LLM/VLM/VLA 意义的“大模型”分支；导航成功率不能替代姿态控制稳定性或固定翼飞行品质评价。 | 航空 VLN 数据与基准；与低层飞控任务不同 | 作者 arXiv 正文摘要及任务定义；会议版题名由公开论文记录交叉核对 |
| L02 · [大型基础模型赋能下无人飞行器智能化：进展、应用与展望](https://hkxb.buaa.edu.cn/CN/10.7527/S1000-6893.2026.33148) · 2026 · 航空学报 47(15):333148 | 从感知、规划、控制、交互梳理 LLM/VLM 等基础模型用于无人飞行器的方式，并覆盖导航、主动搜索、语义配送和集群协同。适合中文综述的宏观入口，不能替代具体控制器的实验依据。 | 综述，含 151 条参考文献入口 | 期刊中英文摘要、元数据和参考文献表 |

T01 版本/使用说明：2017 年预印本，ICRA 2018


L01 版本/使用说明：2025 年预印本旧题名为 A Versatile Toolchain and Large-scale Benchmark for Aerial Vision-Language Navigation；2026 版改题。为简洁 BibTeX 作者以 and others 截断


可编辑数据为 references.json；Excel 可直接打开 literature_matrix.csv；LaTeX 引用使用 references.bib。缓存为本地个人阅读材料，报告的可分享内容是原创摘要及来源链接。
