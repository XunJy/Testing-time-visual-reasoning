# FuDD 两次实验失败图片清单

定义：`degraded` 表示 baseline 正确、FuDD 错误；`both_wrong` 表示同一模型的
baseline 和 FuDD 都错误。所有计数均来自完整 5,794 张 CUB 测试集。

| 模型 | FuDD 改错 | baseline/FuDD 都错 | FuDD 恢复 | 两者都对 |
|---|---:|---:|---:|---:|
| openai_clip | 266 | 1,719 | 404 | 3,405 |
| eva02_clip | 173 | 1,485 | 257 | 3,879 |

## 跨模型重合

| 类别 | 两模型交集 | 仅第一个模型 | 仅第二个模型 | Jaccard |
|---|---:|---:|---:|---:|
| degraded | 30 | 236 | 143 | 7.3350% |
| both_wrong | 1,090 | 629 | 395 | 51.5610% |

四种配置全部错误的图片共 **1,090** 张；两个模型的 baseline 都错为 1,363 张，两个模型的 FuDD 都错为 1,283 张。

## 清单

- [`openai_clip_degraded.csv`](openai_clip_degraded.csv)
- [`openai_clip_both_wrong.csv`](openai_clip_both_wrong.csv)
- [`eva02_clip_degraded.csv`](eva02_clip_degraded.csv)
- [`eva02_clip_both_wrong.csv`](eva02_clip_both_wrong.csv)
- [`degraded_in_both_models.csv`](degraded_in_both_models.csv)
- [`both_wrong_in_both_models.csv`](both_wrong_in_both_models.csv)
- [`cross_model_categories.csv`](cross_model_categories.csv)
- [`summary.json`](summary.json)

CSV 同时给出图片相对路径、真实类别、baseline/FuDD Top-1 类别，以及真实类别
在两个 Top-10 排名中的位置。原始 run 未被修改。
