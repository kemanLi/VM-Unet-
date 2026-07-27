# CS-DSFR-VSS 方法设计与公式备忘

> 用途：保存第三处改进的论文方法、公式、实现约束和消融设计，供后续写作与编码核对。
> 状态：研究设计稿，尚未完成代码实现和实验验证。论文中只能保留最终实际实现且经过消融验证的部分。

## 1. 模块名称与定位

建议名称：**跨尺度方向一致性慢–快状态路由视觉状态空间块**（Cross-Scale Direction-Consistent Slow–Fast State Routing Visual State Space Block，**CS-DSFR-VSS**）。

“慢–快状态”不是将状态下标硬切成两组，而是根据学习后的状态转移参数和输入相关离散步长，估计各状态模态的连续有效时间尺度，再进行方向条件软路由。

本模块的核心不是给 Mamba 首次增加状态选择能力。原始选择性 SSM 已通过输入相关的 \(B_t\)、\(C_t\) 和 \(\Delta_t\) 进行内容选择。本文增加的是以下显式联系：

\[
\text{血管方向可靠性}
\longleftrightarrow
\text{状态时间尺度}
\longleftrightarrow
\text{SS2D 扫描方向}.
\]

## 2. 原始 VSS 与 SS2D

给定第 \(l\) 个尺度的输入特征：

\[
X^l\in\mathbb{R}^{B\times H_l\times W_l\times C_l},
\tag{1}
\]

原始 VSS 首先执行归一化和输入投影：

\[
[U^l,Z^l]
=\operatorname{Split}\!\left(
W_{\mathrm{in}}^l\operatorname{LN}(X^l)
\right),
\tag{2}
\]

其中，\(U^l\) 为扫描主分支，\(Z^l\) 为原始输出门控分支。主分支经过深度卷积和激活：

\[
V^l=\operatorname{SiLU}\!\left(
\operatorname{DWConv}_{3\times3}(U^l)
\right).
\tag{3}
\]

Cross-Scan 将二维特征转换为四条一维序列：

\[
\{V_d^l\}_{d\in\mathcal D}
=\operatorname{CrossScan}(V^l),
\qquad
\mathcal D=\{h^+,h^-,v^+,v^-\},
\tag{4}
\]

分别对应行优先正向、行优先反向、列优先正向和列优先反向扫描。它们不是严格沿四个空间直线方向追踪血管。

第 \(d\) 路由输入内容生成选择性状态空间参数：

\[
[\delta_{d,t}^l,B_{d,t}^l,C_{d,t}^l]
=W_{x,d}^lV_{d,t}^l,
\tag{5}
\]

\[
\Delta_{d,t}^l
=\operatorname{Softplus}\!\left(
W_{\Delta,d}^l\delta_{d,t}^l+b_{\Delta,d}^l
\right),
\tag{6}
\]

\[
A_{d,c,n}^l=-\exp(A_{\log,d,c,n}^l).
\tag{7}
\]

状态递推可写为：

\[
h_{d,c,n,t}^l
=\bar A_{d,c,n,t}^lh_{d,c,n,t-1}^l
+\bar B_{d,c,n,t}^lV_{d,c,t}^l,
\tag{8}
\]

\[
Y_{d,c,t}^l
=\sum_{n=1}^{N}C_{d,n,t}^lh_{d,c,n,t}^l
+D_{d,c}^lV_{d,c,t}^l,
\tag{9}
\]

其中：

\[
\bar A_{d,c,n,t}^l
=\exp\!\left(\Delta_{d,c,t}^lA_{d,c,n}^l\right).
\tag{10}
\]

项目中 \(N=d_{\mathrm{state}}=16\)。因 \(A<0\) 且 \(\Delta>0\)，不同状态模态具有不同衰减速度，但不存在代码预先标记的 `slow_state` 或 `fast_state` 分支。

原始 SS2D 将恢复到二维坐标的四路输出固定相加：

\[
Y_{\mathrm{sum}}^l=\sum_{d\in\mathcal D}\widehat Y_d^l.
\tag{11}
\]

## 3. 共享多方向 Sobel 几何先验

设输入图像的绿色通道为 \(I_G\)，多方向固定卷积核组为：

\[
\mathcal K=\{K_1,K_2,\ldots,K_K\}.
\tag{12}
\]

第 \(k\) 个方向响应为：

\[
R_k=K_k*I_G.
\tag{13}
\]

### 3.1 原有幅值引导

原有 Sobel 引导使用多方向均方根能量：

\[
G=\operatorname{Clip}\!\left(
\frac{\sqrt{\varepsilon+\frac{2}{K}\sum_{k=1}^{K}R_k^2}}{q},
0,1
\right),
\tag{14}
\]

并在对应尺度执行：

\[
\widetilde F^l=F^l\odot(1+\lambda_GG^l).
\tag{15}
\]

该路径回答“哪里存在需要增强的明显结构”，但平方求和后不再保留方向。

### 3.2 连续无向方向和方向明确度

对各方向响应的局部能量进行平滑：

\[
E_k^l=\mathcal G_\sigma*(R_k^l)^2.
\tag{16}
\]

使用双角度方向矩：

\[
m_x^l=\frac{\sum_{k=1}^{K}E_k^l\cos(2\phi_k)}
{\sum_{k=1}^{K}E_k^l+\varepsilon},
\tag{17}
\]

\[
m_y^l=\frac{\sum_{k=1}^{K}E_k^l\sin(2\phi_k)}
{\sum_{k=1}^{K}E_k^l+\varepsilon}.
\tag{18}
\]

梯度方向及其明确度为：

\[
\theta_g^l=\frac{1}{2}\operatorname{atan2}(m_y^l,m_x^l),
\tag{19}
\]

\[
\kappa^l=\sqrt{(m_x^l)^2+(m_y^l)^2}.
\tag{20}
\]

若 \(\theta_g^l\) 表示梯度法线，则血管切线方向为：

\[
\theta^l=\theta_g^l+\frac{\pi}{2}.
\tag{21}
\]

双角度表示满足 \(\theta\equiv\theta+\pi\)，符合血管没有天然箭头的无向轴属性。实现前必须用合成水平线、垂直线和斜线校准现有 Sobel 核角度约定。

## 4. 跨尺度方向一致性

定义归一化双角度轴向向量：

\[
v^l=[\cos(2\theta^l),\sin(2\theta^l)].
\tag{22}
\]

不能直接池化角度；应对置信度加权方向向量或方向能量进行平均池化：

\[
\widehat v^{l-1}
=\operatorname{Norm}\!\left(
\operatorname{AvgPool}(\kappa^{l-1}v^{l-1})
\right),
\tag{23}
\]

\[
\widehat\kappa^{l-1}=\operatorname{AvgPool}(\kappa^{l-1}).
\tag{24}
\]

跨尺度方向一致性定义为：

\[
c^l=\kappa^l\widehat\kappa^{l-1}
\frac{1+\langle v^l,\widehat v^{l-1}\rangle}{2},
\qquad c^l\in[0,1].
\tag{25}
\]

最浅层可令：

\[
c^0=\kappa^0.
\tag{26}
\]

## 5. 特征条件可靠性门

Sobel 同样会响应视盘、病灶、FOV 边缘和噪声，因此从原始 VSS 的 \(Z^l\) 门控分支生成轻量内容可靠性：

\[
Q^l=\sigma\!\left(
W_Q^l\operatorname{SiLU}(Z^l)+b_Q^l
\right),
\qquad Q^l\in[0,1]^{B\times H_l\times W_l\times1}.
\tag{27}
\]

最终几何可靠性为：

\[
\mathcal R^l=c^l\odot Q^l.
\tag{28}
\]

若暂不使用跨尺度一致性，则：

\[
\mathcal R^l=\kappa^l\odot Q^l.
\tag{29}
\]

若未对 \(Q^l\) 单独监督，论文中应称其为“特征条件可靠性门”，不能称为经过标定的血管概率或语义置信度。

## 6. 扫描轴方向对齐

水平和垂直扫描轴的对齐系数为：

\[
g_h^l=\cos^2(\theta^l),
\qquad
g_v^l=\sin^2(\theta^l).
\tag{30}
\]

映射到四路扫描：

\[
g_{h^+}^l=g_{h^-}^l=g_h^l,
\qquad
g_{v^+}^l=g_{v^-}^l=g_v^l.
\tag{31}
\]

多方向 Sobel 用于稳健估计连续方向；最终投影到两个轴，是因为现有 SS2D 的四路序列只有行、列两个不同的空间轴及其逆序，而不是新模块只使用两方向 Sobel。

## 7. 连续状态时间尺度

由于当前 SS2D 的 \(C_t\) 在同一方向内没有内部通道维，先对各内部通道的状态衰减率进行聚合：

\[
\tau_{d,n,t}^l=
\left[
\varepsilon+
\frac{1}{C_l'}\sum_{c=1}^{C_l'}
\Delta_{d,c,t}^l|A_{d,c,n}^l|
\right]^{-1}.
\tag{32}
\]

在状态维度上标准化对数时间尺度：

\[
s_{d,n,t}^l=
\frac{
\operatorname{sg}(\log\tau_{d,n,t}^l)-\mu_{d,t}^l
}{
\sigma_{d,t}^l+\varepsilon
}.
\tag{33}
\]

其中，\(\operatorname{sg}\) 为停止梯度；\(s>0\) 表示相对慢状态，\(s<0\) 表示相对快状态。停止路由分支对时间尺度统计的梯度，不影响 \(A\) 和 \(\Delta\) 在原始选择性扫描路径中的正常训练。

## 8. 方向条件慢–快状态路由

定义方向极性：

\[
r_{d,t}^l=2g_{d,t}^l-1,
\qquad r_{d,t}^l\in[-1,1].
\tag{34}
\]

沿血管的扫描轴有 \(r>0\)，偏向慢状态；正交扫描轴有 \(r<0\)，偏向快状态；对角方向有 \(r\approx0\)，不施加强慢偏置。

状态软路由权重为：

\[
a_{d,n,t}^l=N\cdot
\operatorname{Softmax}_{n}\!\left(
\gamma_l r_{d,t}^ls_{d,n,t}^l
\right),
\tag{35}
\]

并满足：

\[
\frac{1}{N}\sum_{n=1}^{N}a_{d,n,t}^l=1.
\tag{36}
\]

因此状态路由重新分配状态间的相对贡献，而不直接改变其平均尺度。

## 9. 对状态读出参数的残差调制

第一版只调制 \(C_t\)，保持 \(A\)、\(B_t\) 和 \(\Delta_t\) 的原始递推机制：

\[
\widetilde C_{d,n,t}^l
=C_{d,n,t}^l
\left[
1+\alpha_l\mathcal R_t^l(a_{d,n,t}^l-1)
\right].
\tag{37}
\]

相应输出为：

\[
\widetilde Y_{d,c,t}^l
=\sum_{n=1}^{N}\widetilde C_{d,n,t}^lh_{d,c,n,t}^l
+D_{d,c}^lV_{d,c,t}^l.
\tag{38}
\]

选择 \(C_t\) 的原因：它控制当前位置如何读取已形成的状态；修改 \(B_t\) 会影响后续位置的状态写入，修改 \(\Delta_t\) 会同时改变记忆时长和输入注入，风险更高。

将 \(\alpha_l\) 初始化为 0，可保证：

\[
\alpha_l=0\quad\Rightarrow\quad\widetilde C=C.
\tag{39}
\]

当 \(\mathcal R\approx0\) 时，模块也自动回退到原始内容选择。

## 10. 基线保持的四路自适应融合

状态扫描完成并恢复二维坐标后，保留原始固定和：

\[
Y_{\mathrm{sum}}^l=\sum_{d\in\mathcal D}\widehat Y_d^l.
\tag{40}
\]

构造四路几何先验：

\[
P_{h^+}^l=P_{h^-}^l
=\frac{1-\mathcal R^l}{4}
+\frac{\mathcal R^lg_h^l}{2},
\tag{41}
\]

\[
P_{v^+}^l=P_{v^-}^l
=\frac{1-\mathcal R^l}{4}
+\frac{\mathcal R^lg_v^l}{2}.
\tag{42}
\]

从扫描前特征生成四路内容 logit：

\[
L^l=W_{\mathrm{dir}}^lV^l+b_{\mathrm{dir}}^l.
\tag{43}
\]

几何与内容联合融合权重为：

\[
\pi_d^l=\operatorname{Softmax}_{d}\!\left[
L_d^l+\eta_l\log(P_d^l+\varepsilon)
\right].
\tag{44}
\]

自适应输出为：

\[
Y_{\mathrm{adapt}}^l
=4\sum_{d\in\mathcal D}\pi_d^l\widehat Y_d^l.
\tag{45}
\]

系数 4 保证均匀权重时数值尺度与原始固定相加一致。最终采用残差式融合：

\[
Y_{\mathrm{fuse}}^l
=Y_{\mathrm{sum}}^l
+\rho_l(Y_{\mathrm{adapt}}^l-Y_{\mathrm{sum}}^l).
\tag{46}
\]

将 \(\rho_l\) 初始化为 0，可保证：

\[
\rho_l=0\quad\Rightarrow\quad
Y_{\mathrm{fuse}}^l=Y_{\mathrm{sum}}^l.
\tag{47}
\]

## 11. 模块输出

保持原始 \(Z\) 分支门控和输出投影：

\[
O^l=W_{\mathrm{out}}^l\left[
\operatorname{LN}(Y_{\mathrm{fuse}}^l)
\odot\operatorname{SiLU}(Z^l)
\right],
\tag{48}
\]

\[
X_{\mathrm{out}}^l
=X^l+\operatorname{DropPath}(O^l).
\tag{49}
\]

当 \(\alpha_l=0\) 且 \(\rho_l=0\) 时，完整模块严格退化为原始 VSS。

## 12. 与原有 Sobel 引导的关系

\[
\{R_k\}_{k=1}^{K}
\longrightarrow
\begin{cases}
G,&\text{边缘幅值：用于原有外部特征增强};\\
\theta,\kappa,c,&\text{方向和可靠性：用于 VSS 内部路由}.
\end{cases}
\tag{50}
\]

二者共用一个 Sobel 响应源，但回答不同问题：

- 幅值引导：哪些位置需要增强；
- 状态方向路由：该位置更应读取哪些时间尺度和哪些扫描方向的结果。

## 13. 建议的论文创新表述

> 与仅改变扫描路径或在 VSS 外部增加卷积注意力的方法不同，本文从状态空间动力学角度出发，根据学习到的状态转移参数 \(A\) 和输入相关离散步长 \(\Delta_t\) 估计状态的连续时间尺度，并利用共享多方向 Sobel 响应提取的血管轴向先验，对状态读出参数 \(C_t\) 进行方向条件慢–快软路由。同时，采用基线保持的四路自适应融合调节不同扫描序列的贡献。该方法不改变 SS2D 的四路扫描拓扑和状态递推过程，并通过零初始化残差调制保持与原始 VSS 的兼容性。

不得写成“原始 Mamba 没有使用快慢状态”或“SS2D 严格沿血管方向扫描”。

## 14. 建议的实现边界

第一版核心实现：

1. 共享多方向响应及方向/可靠性金字塔；
2. 特征条件可靠性门；
3. 基于 \(A,\Delta\) 的时间尺度统计；
4. 只调制 \(C_t\)；
5. 基线保持的四路融合；
6. 所有新增强度零初始化或近零初始化；
7. 不修改 \(A\)、\(B_t\)、\(\Delta_t\) 和 Cross-Scan 路径。

后续扩展消融：

- 是否加入跨尺度一致性；
- 是否加入特征条件可靠性门；
- 仅状态路由；
- 仅方向融合；
- 状态路由与方向融合同时启用；
- 只替换浅层、只替换深层、编码器替换、全网络替换；
- 后续再分别研究 \(B_t\) 或 \(\Delta_t\) 调制，不与首版核心实验混在一起。

## 15. 推荐的模型级消融矩阵

| 编号 | 高分辨率模块 | 外部 Sobel 幅值引导 | CS-DSFR-VSS | 目的 |
|---|---:|---:|---:|---|
| A0 | 否 | 否 | 否 | 原始 VM-UNet |
| A1 | 是 | 否 | 否 | 第一处改进 |
| A2 | 是 | 是 | 否 | 已完成两处改进的基线 |
| A3 | 是 | 否 | 是 | 验证新 VSS 的独立贡献 |
| A4 | 是 | 是 | 是 | 最终完整模型 |

CS-DSFR-VSS 内部消融建议：

| 编号 | 状态路由 | 方向融合 | 内容可靠性 | 跨尺度一致性 |
|---|---:|---:|---:|---:|
| B0 | 否 | 否 | 否 | 否 |
| B1 | 是 | 否 | 否 | 否 |
| B2 | 否 | 是 | 否 | 否 |
| B3 | 是 | 是 | 否 | 否 |
| B4 | 是 | 是 | 是 | 否 |
| B5 | 是 | 是 | 是 | 是 |

其中 A3 要求新 VSS 能在不启用外部 Sobel 幅值门 \(G\) 的情况下，仍从共享方向响应获得 \(\theta,\kappa,c\)。因此代码中必须将“计算 Sobel 几何先验”和“应用外部幅值门”设计成两个独立开关。

## 16. 编码前待确认

1. 正式采用的 Sobel 核版本：`s3_d2`、`s3_d4`、`s5_d4` 或 `s5_d8`；
2. 首版是否完整实现状态路由、方向融合、内容可靠性和跨尺度一致性，还是先实现最小核心版本；
3. 新 VSS 替换范围：全 15 个、只替换编码器、或指定尺度；
4. \(Q\) 是否只由最终分割损失训练，首版建议不增加辅助标签监督；
5. 是否从已有两处改进的 checkpoint 继续训练，还是所有消融统一从相同 VMamba 预训练初始化；
6. 对照实验需要独立模型名，还是同一模型名下使用配置开关；建议模型级变体独立命名，模块内部消融使用显式配置；
7. 是否要求旧 checkpoint 在严格加载模式下继续兼容；建议保留兼容加载并明确报告新增参数和未加载参数。
