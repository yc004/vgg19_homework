# Intel Image Classification (MMPretrain + VGG19)

## 项目简介

- 使用 OpenMMLab MMPretrain 配置基于 VGG19 的 6 类 Intel Image Classification 任务
- 提供脚本完成数据下载、训练、日志/预测可视化
- 训练输出 (日志、权重、可视化) 默认保存在 `work_dirs/vgg19_intel/`

## 快速开始

1. 创建并激活环境 (示例使用 conda)：

```bash
conda create -n intelcls python=3.11 -y
conda activate intelcls
pip install -r requirements.txt
```

> `requirements.txt` 基于导出列表，若 `pip` 遇到 CUDA/torch 安装问题，可先用 conda 安装匹配的 CUDA 与 torch，再用 pip 补齐 `mmengine`/`mmpretrain` 等。

1. 下载并整理数据：

```bash
python download_data.py
```

数据将放到 `data/intel_image/train` 与 `data/intel_image/val` 目录，类目录名与官方一致。

1. 开始训练：

```bash
python train.py
```

- 使用 `configs/vgg19_intel.py` 配置，默认 5 个 epoch、batch size 32、SGD 优化器
- 日志与权重会写到 `work_dirs/vgg19_intel/<timestamp>/`

1. 可视化结果：

- 训练曲线：

```bash
python visualize_results.py
# 输出 training_results.png
```

- 随机验证样本预测可视化：

```bash
python visualize_predictions.py
# 默认使用 work_dirs/vgg19_intel/epoch_4.pth，不存在则尝试 epoch_5.pth
# 输出 prediction_results.png
```

脚本会优先尝试 GPU，不可用时回退到 CPU。

## 数据说明

- 数据集：Intel Image Classification (buildings, forest, glacier, mountain, sea, street)
- 目录结构示例：

```
data/intel_image/
  train/buildings/...jpg
  train/forest/...jpg
  ...
  val/sea/...jpg
```

## 主要文件

- `configs/vgg19_intel.py`：模型、数据管线与优化器配置
- `train.py`：加载配置并启动训练 Runner
- `download_data.py`：从 GitHub 拉取并整理数据集至本地结构
- `visualize_results.py`：读取日志生成 loss / accuracy 曲线并保存为图
- `visualize_predictions.py`：载入权重对验证集样本推理并绘图
