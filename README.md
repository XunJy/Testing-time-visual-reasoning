# Testing-Time Visual Reasoning

这是一个面向测试时视觉推理的可复现实验仓库。仓库不绑定某一个方法或某一个模型，
而是把研究对象明确拆成两个维度：

```text
实验 = 方法（method） + 模型（model） + 数据集（dataset）
```

例如，`FuDD + CLIP`、`FuDD + SigCLIP` 和 `FuDD + SigCLIP 2` 是三个独立实验。
它们可以复用同一份 FuDD 算法，但必须使用各自的模型后端、配置、缓存和结果目录。
后续的裁剪或训练残差头同样必须与具体模型组合，不能只记录“crop”或
“residual head”而省略基线模型。

## 实验矩阵

| ID | 方法 | 模型 | 数据集 | 状态 |
|---|---|---|---|---|
| 01 | FuDD | OpenAI CLIP ViT-L/14@336px | CUB-200-2011 | **已完整复现** |
| 02 | FuDD | SigCLIP | CUB-200-2011 | 计划中 |
| 03 | FuDD | SigCLIP 2 | CUB-200-2011 | 计划中 |
| 04 | FuDD | EVA02-CLIP-L/14@336 | CUB-200-2011 | **已完整运行：+1.4498 pp** |
| 05 | 监督残差分类头 | OpenAI CLIP ViT-L/14@336px | CUB-200-2011 | **探索性完成：87.1074%** |
| 06 | 类别无关特征适配器 | OpenAI CLIP ViT-L/14@336px | BirdMix-v1 → CUB | 预注册实验进行中 |
| 07 | 类别无关特征适配器 | OpenAI CLIP ViT-L/14@336px | BirdMix-v2（6 源）→ CUB | 数据缓存进行中 |

完整注册表和命名规则见 [`experiments/README.md`](experiments/README.md)。未来可按同一
格式增加 `crop + model`、`residual head + model` 或其他组合。

## 已完成：FuDD + CLIP

[FuDD（Follow-Up Differential Descriptions）](https://github.com/BatsResearch/fudd)
在 CUB-200-2011 官方测试集上的首个独立复现已经完成：

| 来源 | 单模板 Top-1 | FuDD k=10 Top-1 | 变化 |
|---|---:|---:|---:|
| ICLR 2024 论文 | 63.48% | 65.90% | +2.42 pp |
| 本项目完整运行 | **63.3586%** | **65.7404%** | **+2.3818 pp** |

本项目结果对应 5,794 张测试图像：baseline 正确 3,671 张，FuDD 正确 3,809 张；
错→对 404 张，对→错 266 张，净增 138 张。32 个样本的批量实现与作者式逐样本
reference 预测全部一致。

机器可读结果保存在
[`experiments/01_fudd_clip_cub/runs/20260714T185902.445729Z-full-3b975c99f4/`](experiments/01_fudd_clip_cub/runs/20260714T185902.445729Z-full-3b975c99f4/)。
该 run 是不可修改的历史产物；本次通用目录重构不会改写它或伪装成一次新 GPU 运行。

## 项目结构

```text
.
├── src/ttvr/
│   ├── data/
│   │   ├── cub.py                 # 共享 CUB 下载、划分和类别校验
│   │   ├── bird_manifest.py       # 多鸟类数据集的统一 manifest
│   │   ├── birdnet.py             # BirdNET 图片与许可锁定
│   │   ├── inat2021.py            # iNaturalist Mini Aves
│   │   ├── big_bird.py            # Big Bird bbox crop 数据管线
│   │   ├── visual_wetlandbirds.py # WetlandBirds 视频 crop 数据管线
│   │   ├── usgs_aerial_avian.py   # USGS 航拍鸟类 crop
│   │   └── nm_uas_waterfowl.py    # NM UAS 水鸟 crop
│   ├── models/
│   │   ├── base.py                # 模型后端统一接口和特征容器
│   │   ├── cached.py              # 双编码器共享缓存和聚合
│   │   ├── clip.py                # OpenAI CLIP 后端
│   │   └── open_clip.py           # 锁定 checkpoint 的 OpenCLIP/EVA 后端
│   ├── methods/
│   │   ├── fudd/
│   │       ├── config.py          # FuDD 组合配置
│   │       ├── prompts.py         # 差异描述与 Top-k 候选提示
│   │       └── evaluation.py      # 模型无关的 FuDD 重排
│   │   ├── residual_head/         # CUB 监督线性/残差分类头
│   │   └── feature_adapter/       # 跨数据集类别无关残差适配器
│   └── metrics.py                 # 共享准确率、排序和转移指标
│
├── scripts/
│   ├── fudd/
│       ├── run_clip_cub.py        # FuDD + OpenAI CLIP + CUB
│       └── run_eva02_clip_cub.py  # FuDD + EVA02-CLIP + CUB
│   ├── residual_head/             # 实验 05 的缓存和训练入口
│   └── feature_adapter/           # 鸟类数据准备、分片缓存和研究入口
│
├── experiments/
│   ├── README.md                  # 方法+模型+数据集注册表
│   ├── 01_fudd_clip_cub/          # 已完成
│   ├── 02_fudd_sigclip_cub/       # 计划中
│   ├── 03_fudd_sigclip2_cub/      # 计划中
│   ├── 04_fudd_eva02_clip_cub/    # 已完成：FuDD +1.4498 pp
│   ├── 05_residual_head_clip_cub/ # 探索性监督基线
│   ├── 06_feature_adapter_clip_multi_bird/ # BirdMix-v1
│   ├── 07_feature_adapter_clip_birdmix_v2_cub/ # 六源 BirdMix-v2
│   └── _template/                 # 新组合模板
│
├── tests/
│   ├── data/
│   ├── models/
│   ├── methods/fudd/
│   └── test_metrics.py
│
├── data/fudd_official/            # 上游来源与哈希；JSON 在运行时下载
├── pyproject.toml
└── README.md
```

### 两个共享轴

`src/ttvr/methods/` 只定义“如何推理或适配”。例如 FuDD 负责 Top-k 候选、差异描述
聚合和候选内重排，不负责选择 CLIP 或 SigCLIP。

`src/ttvr/models/` 只定义“如何把图像和文本编码成可比较特征”。所有后端应实现
`VisionLanguageBackend` 接口。这样新增 SigCLIP 或 SigCLIP 2 时，不需要复制 FuDD
逻辑；新增裁剪方法时，也可以复用同一模型后端。

## 运行 FuDD + CLIP

推荐 Python 3.10–3.12：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip "setuptools>=69,<81" wheel
python -m pip install --no-build-isolation -e ".[openai-clip,dev]"
python -m pytest -q
```

OpenAI CLIP 固定在上游提交 `a1d071733d7111c9c014f024669f959182114e33`。该旧包的
构建脚本仍依赖 `pkg_resources`，因此这里把构建工具锁在兼容区间，并关闭该依赖的
隔离构建。`openai-clip` 是独立 extra，不会被 EVA02 或其他模型实验自动安装。

先运行 smoke，再运行全部 5,794 张：

```bash
python scripts/fudd/run_clip_cub.py --max-samples 32
python scripts/fudd/run_clip_cub.py
```

正式 CLI 每次创建新的 UTC 时间戳与源码摘要 run 目录，不覆盖旧结果。
CLI 是本项目唯一的实验入口，算法实现位于 `src/ttvr/`。

## 运行 FuDD + EVA02-CLIP

EVA02 实验使用同一份 FuDD 方法代码，但采用独立模型后端、缓存和 run 目录：

```bash
python -m pip install -e ".[eva02,dev]"
python scripts/fudd/run_eva02_clip_cub.py --max-samples 32
python scripts/fudd/run_eva02_clip_cub.py
```

模型、HF revision、safetensors SHA-256、tokenizer、336px 预处理及 FP16/FP32
边界均已锁定。完整 5,794 张实验的 baseline 为 69.9344%，FuDD 为 71.3842%，
提升 +1.4498 个百分点。协议、统计检验和逐图产物见
[`experiments/04_fudd_eva02_clip_cub/README.md`](experiments/04_fudd_eva02_clip_cub/README.md)。

## 跨鸟类数据集训练

实验 07 不用 CUB 训练残差分类器，而是在六个外部鸟类来源上训练同一个
`768 -> 128 -> 768` 类别无关特征适配器，再一次性测试未见过的 CUB 物种。
当前来源为 iNaturalist Mini Aves、BirdNET、Big Bird、Visual WetlandBirds、
USGS aerial avian 和 NM UAS waterfowl。所有来源先统一到锁定的 BirdNET/AviList
物种标识；与 CUB 重合的物种、训练文本及跨源精确重复图像都会在训练任务构造时
排除。

图像特征按 2,048 行原子分片并保存到 Google Drive。Colab 被回收后会核验并复用
已经完成的分片，而不是重新计算整个数据集。完整协议、Drive 路径和三 seed
运行入口见
[`experiments/07_feature_adapter_clip_birdmix_v2_cub/README.md`](experiments/07_feature_adapter_clip_birdmix_v2_cub/README.md)。

## Python API

```python
from ttvr import FuDDConfig, run_clip_cub_experiment

config = FuDDConfig(
    data_root="data",
    prompt_root="data/fudd_official",
    cache_dir=".cache/fudd_clip_cub",
    model_name="ViT-L/14@336px",
    precision="fp32",
    top_k=10,
    seed=2026,
)

report = run_clip_cub_experiment(config)
print(report.to_dict())
```

未来的 SigCLIP runner 会构造 SigCLIP backend，然后调用同一个 FuDD `evaluate_cub`，
而不是复制 `methods/fudd/evaluation.py`。

## 数据与公开发布边界

- CUB-200-2011 图像、压缩包和模型权重不提交到 Git。
- FuDD 官方 CUB 描述来自作者固定提交
  `32264231fec047eb0bbbf59bfdbc8e6d208a096b`。上游没有明确许可证，因此公开仓库
  不二次分发 JSON；运行时从作者仓库下载并验证 SHA-256。
- `Old Patch/` 是本地冻结历史，约 753 MB，不参与安装、导入或公开推送。
- 每个新实验必须保留配置、环境、逐样本预测和 checksum；训练式残差头还必须记录
  训练/验证划分、checkpoint、seed 和父 baseline。

## 新实验约定

新组合使用 `NN_<method>_<model>_<dataset>` 命名。测试集运行前必须先锁定：

- 方法、模型和精确 checkpoint；
- 数据划分、输入分辨率和预处理；
- prompt、归一化、候选构造和融合公式；
- 是否使用训练数据，以及验证集如何调参；
- seed、precision、主指标和验收标准。

计划或旧工程数字不能写成新组合的结果。扩展实验必须创建新的 run，不得覆盖
FuDD + CLIP positive control。
