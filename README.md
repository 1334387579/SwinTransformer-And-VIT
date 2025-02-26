# Swin Transformer and ViT for Rice Leaf Disease Classification

This repository contains implementations of Swin Transformer and Vision Transformer (ViT) for classifying rice leaf diseases into 10 categories.
该存储库包含用于将水稻叶片病害分为10类的Swin Transform和Vision Transform（ViT）的实现。


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
