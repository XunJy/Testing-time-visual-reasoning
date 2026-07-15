# Testing-Time Visual Reasoning

一个把测试时视觉推理拆成可复现、可比较实验的研究仓库。每项实验都由三个轴唯一确定：

```text
实验 = 方法（method） + 模型（model） + 数据集（dataset）
```

方法实现只保留一份，位于 `src/ttvr/methods/`；模型编码后端位于
`src/ttvr/models/`；具体组合的锁定协议、配置和不可变结果位于
`experiments/NN_<method>_<model>_<dataset>/`。正式实验使用 Python CLI，不依赖
notebook。

## 当前结论

| ID | 实验 | 状态 | 主要结果 |
|---:|---|---|---|
| [01](experiments/01_fudd_clip_cub/) | FuDD + OpenAI CLIP ViT-L/14@336 + CUB | **完成** | Top-1 `63.3586% → 65.7404%`，**+2.3818 pp** |
| [02](experiments/02_fudd_siglip_cub/) | FuDD + SigLIP + CUB | 计划中 | 尚未实现或运行 |
| [03](experiments/03_fudd_siglip2_cub/) | FuDD + SigLIP 2 + CUB | 计划中 | 尚未实现或运行 |
| [04](experiments/04_fudd_eva02_clip_cub/) | FuDD + EVA02-CLIP-L/14@336 + CUB | **完成** | Top-1 `69.9344% → 71.3842%`，**+1.4498 pp** |
| [05](experiments/05_residual_head_clip_cub/) | 监督线性/残差头 + OpenAI CLIP + CUB | **探索性完成** | linear `86.9348%`，residual `87.1074%`；两者差异不显著 |
| [06](experiments/06_feature_adapter_clip_multi_bird/) | BirdMix-v1 特征适配器 + OpenAI CLIP → CUB | **冻结、未完成** | 已预注册，无正式运行；由独立的 07 扩展来源 |
| [07](experiments/07_feature_adapter_clip_birdmix_v2_cub/) | BirdMix-v2 六源特征适配器 + OpenAI CLIP → CUB | **完成，未支持正迁移** | 3-seed mean `62.4554%`，相对 `63.3586%` 基线为 **−0.9032 pp** |

完整注册表、命名约定和 artifact 合同见
[`experiments/README.md`](experiments/README.md)。

这些结果支持三个不同层次的结论：

1. FuDD 的官方差异描述重排在本项目的两次锁定运行中均改善了 OpenAI CLIP 和
   EVA02-CLIP 基线。
2. 使用 CUB 训练集的监督适配能大幅提高准确率，但实验 05 没有证明“残差先验”优于
   普通线性探针；它也不是零样本测试时方法。
3. 不使用 CUB 图像拟合、仅从六个外部鸟类来源学习的类别无关残差映射没有迁移增益；
   三个随机种子的点估计都为负，均未通过预注册的严格迁移标准。

不能把这三种协议的绝对准确率直接混为同一排行榜。

## 仓库结构

```text
.
├── src/ttvr/                         # 可复用 Python 包
│   ├── data/                         # CUB 与鸟类数据 manifest/校验
│   ├── methods/
│   │   ├── fudd/                     # 模型无关的 FuDD 重排
│   │   ├── residual_head/            # 监督线性/残差分类头
│   │   └── feature_adapter/          # 类别无关残差特征映射
│   ├── models/                       # CLIP/OpenCLIP 后端与缓存接口
│   ├── experiments/                  # 通用 artifact 写入工具
│   ├── gpu_lock.py                   # 正式 GPU 任务互斥锁
│   └── metrics.py                    # Top-k、转移与配对统计
├── scripts/
│   ├── fudd/                         # 实验 01/04 与配对分析入口
│   ├── residual_head/                # 实验 05 缓存与训练入口
│   └── feature_adapter/              # 实验 06/07 数据、训练、汇总入口
├── experiments/                      # 方法 + 模型 + 数据集注册表
│   ├── 01_fudd_clip_cub/             # 冻结的完整 FuDD/CLIP 运行
│   ├── 02_fudd_siglip_cub/           # 计划实验
│   ├── 03_fudd_siglip2_cub/          # 计划实验
│   ├── 04_fudd_eva02_clip_cub/       # 冻结的完整 EVA02 运行
│   ├── 05_residual_head_clip_cub/    # 监督适配探索
│   ├── 06_feature_adapter_clip_multi_bird/ # 冻结的未完成 v1 协议
│   ├── 07_feature_adapter_clip_birdmix_v2_cub/ # 完成的六源 v2 研究
│   └── _template/                    # 新实验模板
├── tests/                            # data/models/methods/统计测试
├── docs/                             # 当前文档入口与历史归档
├── data/fudd_official/README.md      # 上游 FuDD 资产来源和许可边界
├── AGENTS.md                         # 本仓库维护规则
└── pyproject.toml
```

冻结实验目录不会因结构整理而改名或覆盖。实验 06 的旧 ID 已进入配置和验证代码，
因此即使名称不如 v2 统一也保留原样。实验 02/03 尚无运行产物，本次已纠正为官方模型
名称 `SigLIP` / `SigLIP 2`。

## 安装与测试

推荐 Python 3.10–3.12：

```bash
git clone git@github.com:XunJy/Testing-time-visual-reasoning.git
cd "Testing-time-visual-reasoning"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip "setuptools>=69,<81" wheel
python -m pip install -e ".[dev]"
python -m pytest -q
```

OpenAI CLIP 固定在提交
`a1d071733d7111c9c014f024669f959182114e33`。需要运行实验 01、05 或 07 时安装：

```bash
python -m pip install --no-build-isolation -e ".[openai-clip,dev]"
```

EVA02 实验使用独立依赖：

```bash
python -m pip install -e ".[eva02,dev]"
```

## 复现实验

### FuDD + OpenAI CLIP

先运行 32 张 smoke，再运行完整 CUB test：

```bash
python scripts/fudd/run_clip_cub.py --max-samples 32
python scripts/fudd/run_clip_cub.py
```

正式结果和全部 5,794 条预测位于
[`experiments/01_fudd_clip_cub/runs/20260714T185902.445729Z-full-3b975c99f4/`](experiments/01_fudd_clip_cub/runs/20260714T185902.445729Z-full-3b975c99f4/)。

### FuDD + EVA02-CLIP

```bash
python scripts/fudd/run_eva02_clip_cub.py --max-samples 32
python scripts/fudd/run_eva02_clip_cub.py
```

模型 revision、checkpoint SHA、预处理、精度边界和配对统计见
[`experiments/04_fudd_eva02_clip_cub/README.md`](experiments/04_fudd_eva02_clip_cub/README.md)。

### 监督线性/残差头

```bash
PYTHONPATH=src python scripts/residual_head/cache_clip_cub_features.py
PYTHONPATH=src python scripts/residual_head/run_clip_cub.py
```

该实验使用 CUB train 拟合，因此只作为监督适配基线。完整解释见
[`experiments/05_residual_head_clip_cub/README.md`](experiments/05_residual_head_clip_cub/README.md)。

### BirdMix-v2 外部鸟类迁移

实验 07 涉及六个来源、九个特征缓存、三随机种子和独立 fail-closed 汇总器。不要从
根 README 复制零散命令；以它的
[`README`](experiments/07_feature_adapter_clip_birdmix_v2_cub/README.md)、
锁定配置和 schema 为准。Git 中保留了正式
[`summary.json`](experiments/07_feature_adapter_clip_birdmix_v2_cub/analysis/20260715T155506.258376Z-summary-aa87ae1873/summary.json)
及 checksum。完整大体积运行、预测和 checkpoint 位于实验 README 链接的 Google
Drive 文件夹，需要相应 Drive 权限。

## 实验与 artifact 合同

- 新实验命名为 `NN_<method>_<model>_<dataset>`。
- 方法实现不得为每个模型复制；模型后端不得嵌入某一个方法。
- 每次正式运行创建新的 UTC 时间戳目录，绝不覆盖旧结果。
- 完成的推理实验应保留配置、环境、逐样本预测、聚合结果、生命周期状态和 SHA-256。
- 训练实验还必须保留训练/验证划分、seed、选择规则、checkpoint 身份和父基线。
- 小型可公开 artifact 放入 Git；大型数据、缓存和权重外置时，README 必须记录位置、
  权限边界、大小或 checksum。
- 计划实验、smoke 数字和历史项目数字不能冒充当前正式结果。

## 数据与发布边界

- CUB-200-2011 图片、压缩包和基础模型权重不提交 Git。
- FuDD CUB 描述固定到作者提交
  `32264231fec047eb0bbbf59bfdbc8e6d208a096b`。上游仓库未声明这些 JSON 的许可，
  因此本仓库只保存来源与哈希，运行时下载并校验，不二次分发。
- `.cache/`、本地数据、构建目录和历史工作区均被忽略，不是 GitHub 项目结构的一部分。
- 当前仓库尚未声明项目级开源许可证；公开可读不等于自动授予复用权。上游资产继续
  服从各自许可。

## 文档

- [`experiments/README.md`](experiments/README.md)：正式实验注册表。
- [`docs/README.md`](docs/README.md)：文档入口和历史归档说明。
- 每个实验自己的 `README.md`：唯一可信的协议、运行状态和结果解释。
