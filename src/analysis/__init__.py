"""事后可解释性分析模块。

融合两篇文献方法：
- SVEA（Tripathi et al., IEEE BigData 2020）：以「状态特征块」为玩家的合作博弈，
  特征函数 = hinge-loss 线性 ERM 训练误差的边际改善，Shapley 值误差分摊
  e_j = c̃/n − φ_j，普适阈值 0 挑选支配性特征子集（全局 FSS）。
- DeepSHAP 族局部归因（Fernando et al., SIGIR'19）：Expected Gradients
  （模型内省反向传播，对参照/背景 MC）与 KernelSHAP（模型无关代理）双估计器，
  参照敏感性 + top-k 重叠/recall 验证。

模块只做纯数值/梯度计算，不含任何训练或 immutable 物理文件改动。
"""
