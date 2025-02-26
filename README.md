# Swin Transformer and ViT for Rice Leaf Disease Classification

This repository contains implementations of Swin Transformer and Vision Transformer (ViT) for classifying rice leaf diseases into 10 categories.（该存储库包含用于将水稻叶片病害分为10类的Swin Transform和Vision Transform（ViT）的实现）

## Files
- `swin_transformer.py`: Swin Transformer model implementation.（- `swin_transformer.py`: Swin Transformer 模型实现）
- `vit.py`: Vision Transformer (ViT-B/16) model implementation.（- `vit.py`: Vision Transformer (ViT-B/16) 模型实现）
- `train.py`: Training and testing script for both models.（- `train.py`：两种模型的训练和测试脚本）

## Dataset
- Uses the dataset from `/kaggle/input/rice-plant-leaf-disease-classification-v1i-folder`（在kaggle上面运行数据集路径使用此路径）
- Uses the dataset from './data'（在windows上面运行数据集路径使用此路径）

## Requirements
- See `requirements.txt` for dependencies.（有关依赖项，请参阅“requirements.txt”）

## Usage
1. Install dependencies: `pip install -r requirements.txt`（安装依赖项：`pip install -r requirements.txt`）
2. Run the training script: `python train.py`（运行训练脚本：`python train.py`）
3. 使用的数据集的格式是文件夹格式，如果需要加载其它格式类型的数据集，只需要修改加载数据集的方法，其它保持不变

## 对比
## 1. 可视化图表分析:
## Swin Transformer
损失曲线 (Training and Validation Loss):
训练损失（蓝色）和验证损失（红色）在最初几个 epoch（约 0-5）迅速下降，从 1.0 降至 0.2-0.3。
在 5-10 个 epoch 后，损失趋于平稳，训练损失略低于验证损失，但波动较小。
损失在 10-35 个 epoch 期间保持稳定，验证损失略有波动，但未显著上升，表明模型未过拟合。

准确率曲线 (Training and Validation Accuracy):
训练准确率（蓝色）和验证准确率（红色）在最初几个 epoch 迅速上升，从 0.6 升至 0.9 以上。
在 10-15 个 epoch 后，准确率趋于平稳，验证准确率在 0.93-0.94 之间波动，最终达到 94.35%（最佳验证准确率）。
训练准确率略高于验证准确率，但差距较小（约 0.01-0.02），表明泛化能力较好。

## Vision Transformer (ViT-B/16)
损失曲线 (Training and Validation Loss):
训练损失（蓝色）和验证损失（红色）在 0-5 个 epoch 迅速下降，与 Swin-T 类似，从 1.0 降至 0.2-0.3。
验证损失在 5-15 个 epoch 后稍微波动，但总体保持稳定，低于 Swin-T 的验证损失（更平滑）。
训练损失和验证损失的差距较小，表明模型泛化较好，未明显过拟合。

准确率曲线 (Training and Validation Accuracy):
训练准确率（蓝色）和验证准确率（红色）在 0-10 个 epoch 迅速上升，从 0.6 升至 0.9 以上。
在 10-20 个 epoch 后，验证准确率稳定在 0.93-0.94 之间，最终达到 94.35%（最佳验证准确率）。
训练准确率略高于验证准确率，波动较小，测试准确率（94.65%）略高于 Swin-T（94.43%）。

## 2. 性能对比分析（结合数据和图表）:
## 准确率和稳定性
验证准确率一致: Swin-T 和 ViT-B/16 均达到 94.35%，表明两者的优化能力在验证集上几乎相同。

测试准确率微差: ViT-B/16 (94.65%) 略高于 Swin-T (94.43%)，差距为 0.22%，但在统计意义上几乎可以忽略（可能受随机性或数据划分影响）。

## 收敛速度:
Swin-T: 在 10-15 个 epoch 内接近最佳验证准确率（94.35%），收敛较快。

ViT-B/16: 在 15-20 个 epoch 内稳定在 94.35%，收敛稍慢，但验证损失和准确率波动更小，显示更平滑的收敛过程。

## 过拟合风险:
两者的训练损失和验证损失/准确率差距均较小，Swin-T 的训练准确率略高于验证准确率（约 0.01-0.02），ViT-B/16 差距更小（约 0.005-0.01），表明两者泛化能力相当。

## 训练效率
Swin-T:
损失和准确率曲线显示快速收敛（10-15 epoch），表明窗口自注意力机制和参数少（28M）使其训练效率更高。
验证损失和准确率波动较小，但验证损失在后期略有上升（图中可见轻微波动）。

ViT-B/16:
损失和准确率曲线收敛稍慢（15-20 epoch），但验证损失和准确率更平滑，波动更小，可能得益于更大的参数量（86M）和全局自注意力对特征的全面建模。
图表显示 ViT 的训练损失和验证损失差距更小，泛化更稳定。

## 3. 模型特性对比:
Swin Transformer (Swin-T, 小型模型)
参数量: 约 28M
结构:
窗口自注意力（7x7），层次设计（Patch Merging）。
初始 4x4 patch，逐步减少 token（56x56 -> 7x7）。
计算复杂度: O(N * window_size²)，较低。

Vision Transformer (ViT-B/16, 中等模型)
参数量: 约 86M
结构:
全局自注意力，固定 197 个 token（16x16 patch + CLS）。
计算复杂度: O(N²)，较高。

## 4. 异同点总结:
相同点
任务目标: 两者用于水稻叶病 10 类分类，输入 224x224 图像。
训练流程: 使用相同优化器（Adam, lr=0.0001）、损失函数（CrossEntropyLoss）、早停（patience=10）和数据预处理。
性能接近: 验证准确率均为 94.35%，测试准确率相差仅 0.22%（94.43% vs 94.65%）。
收敛趋势: 两者在 10-20 个 epoch 内收敛，损失和准确率曲线呈快速下降/上升后趋于平稳。

## 不同点
## 规模：
Swin Transformer (Swin-T)：小型 (28M 参数)，参数量较少，适合资源受限环境。

Vision Transformer (ViT-B/16)：中等 (86M 参数)，参数量较大，具有更强的表达能力。
## 嵌入维度：
Swin Transformer (Swin-T)：初始嵌入维度为 96，通过 Patch Merging 逐步增加至 768，层次化设计。

Vision Transformer (ViT-B/16)：嵌入维度固定为 768，全程保持不变。
## 注意力机制：
Swin Transformer (Swin-T)：使用窗口自注意力 (7x7)，限制注意力范围，降低计算复杂度。

Vision Transformer (ViT-B/16)：使用全局自注意力，对所有 197 个 token 进行全连接计算。
## 层次结构：
Swin Transformer (Swin-T)：具有层次结构，通过 4 个阶段（Patch Merging）动态减少 token 数量（从 56x56 到 7x7）。

Vision Transformer (ViT-B/16)：无层次结构，token 数量固定为 197（16x16 patch + CLS token）。
## 计算复杂度：
Swin Transformer (Swin-T)：计算复杂度为 O(N * 49)，窗口自注意力效率高，N 为 token 数量。

Vision Transformer (ViT-B/16)：计算复杂度为 O(197²)，全局自注意力计算量大。
## 参数效率：
Swin Transformer (Swin-T)：参数效率高，28M 参数在中小数据集上表现优异，减少冗余。

Vision Transformer (ViT-B/16)：参数效率中等，86M 参数容量较大，但部分参数可能冗余。
## 收敛速度：
Swin Transformer (Swin-T)：收敛较快，在 10-15 个 epoch 内接近最佳性能。

Vision Transformer (ViT-B/16)：收敛稍慢，在 15-20 个 epoch 内稳定性能。
## 曲线波动：
Swin Transformer (Swin-T)：验证损失和准确率曲线略有波动，可能因参数少对数据噪声更敏感。

Vision Transformer (ViT-B/16)：验证损失和准确率曲线更平滑，表明更大参数量和全局建模能力使模型更稳定。


