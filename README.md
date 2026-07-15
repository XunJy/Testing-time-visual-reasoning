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
│   │   └── cub.py                 # 共享 CUB 下载、划分和类别校验
│   ├── models/
│   │   ├── base.py                # 模型后端统一接口和特征容器
│   │   └── clip.py                # OpenAI CLIP 后端
│   ├── methods/
│   │   └── fudd/
│   │       ├── config.py          # FuDD 组合配置
│   │       ├── prompts.py         # 差异描述与 Top-k 候选提示
│   │       └── evaluation.py      # 模型无关的 FuDD 重排
│   └── metrics.py                 # 共享准确率、排序和转移指标
│
├── scripts/
│   └── fudd/
│       ├── run_clip_cub.py        # FuDD + CLIP + CUB 正式入口
│       └── build_clip_cub_notebook.py
│
├── experiments/
│   ├── README.md                  # 方法+模型+数据集注册表
│   ├── 01_fudd_clip_cub/          # 已完成
│   ├── 02_fudd_sigclip_cub/       # 计划中
│   ├── 03_fudd_sigclip2_cub/      # 计划中
│   └── _template/                 # 新组合模板
│
├── tests/
│   ├── data/
│   ├── models/
│   ├── methods/fudd/
│   └── test_metrics.py
│
├── data/fudd_official/            # 上游来源与哈希；JSON 在运行时下载
├── notebooks/fudd/                # 可选界面，不维护另一份算法
├── docs/
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
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
```

先运行 smoke，再运行全部 5,794 张：

```bash
python scripts/fudd/run_clip_cub.py --max-samples 32
python scripts/fudd/run_clip_cub.py
```

正式 CLI 每次创建新的 UTC 时间戳与源码摘要 run 目录，不覆盖旧结果。长时间 GPU
任务建议通过 `tmux` 运行。远程 Colab 连接和结果回传流程见
[`docs/REMOTE_GPU_CONNECTION.md`](docs/REMOTE_GPU_CONNECTION.md)。

可选 Notebook 由同一份源码生成：

```bash
python scripts/fudd/build_clip_cub_notebook.py
```

Notebook 只是界面；正式算法仍位于 `src/ttvr/`。

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
