# Sobel算子筛选实验

该目录只负责第一次Sobel算子筛选。核心模型实现位于
`models/vmunet/sobel_guidance/`和`models/vmunet/*_sobel.py`，baseline与现有
高分辨率模型不依赖本目录。

## 锁定条件

- 基础结构：`vmunet_highres` + 方案二guided skip
- 初始化：scratch
- 引导强度：lambda=1.0
- 随机种子：42
- q统计：未增强训练集576x576图像、安全FOV内0.99分位数
- 训练：增强后的192x192 Patch实时计算Sobel
- 验证/测试：576x576整图计算一次，再按滑窗位置裁剪，不缩放

## 五种比较

| 名称 | 模板 | 方向 |
|---|---:|---|
| `s3_d2` | 3x3 | 0, 90 |
| `s3_d4` | 3x3 | 0, 45, 90, 135 |
| `s5_d2` | 5x5 | 0, 90 |
| `s5_d4` | 5x5 | 0, 45, 90, 135 |
| `s5_d8` | 5x5 | 0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5 |

## 运行

先计算两个数据集的q统计：

```bash
python experiments/sobel_operator_ablation/compute_q_stats.py
```

运行单项：

```bash
python experiments/sobel_operator_ablation/run_one.py \
  --dataset DRIVE --operator s5_d8
```

顺序运行全部十项：

```bash
bash experiments/sobel_operator_ablation/run_all.sh /root/autodl-tmp/VM-UNet
```

该脚本先执行核心单元测试、重新计算q和一次完整CUDA前向/反向检查，然后显式
顺序执行DRIVE与STARE各5项实验。结果统一写入
`results/sobel_operator_ablation/`。

## 固定最终算子

确定最佳算子后只执行一次，例如：

```bash
python experiments/sobel_operator_ablation/select_operator.py --operator s5_d8
```

它会生成`configs/sobel_selected.json`。后续baseline、高分辨率、Sobel及新模块
消融只读取这一个最终配置，不再遍历五种算子。

后续实验读取方式：

```bash
python train.py --dataset DRIVE --model vmunet_highres_sobel \
  --sobel-selected-config configs/sobel_selected.json \
  --run-tag selected_sobel
```
