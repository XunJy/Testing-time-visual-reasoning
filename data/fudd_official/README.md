# FuDD 官方 CUB 提示资产

本目录记录 FuDD 作者公开仓库中与 CUB-200-2011 直接相关的预生成文本资产，
不包含 CUB 图像、标签文件或模型权重。本地运行时可以保存经校验的原始文件；公开
Git 仓库只跟踪本说明，不二次分发上游 JSON。

## 来源与版本

- 上游仓库：<https://github.com/BatsResearch/fudd>
- 固定提交：`32264231fec047eb0bbbf59bfdbc8e6d208a096b`
- 原始路径：`differential_descriptions/cub_class_names.json`
- 原始路径：`differential_descriptions/cub_prompt_pairs.json`
- 获取日期：2026-07-14

论文说明这些差异描述由 `gpt-3.5-turbo-0301` 生成。缓存包含 200 个类别及
19,900 个无序类别对（即 `200 choose 2`）；每个类别对包含 3 至 5 组差异描述。
本项目直接读取缓存，因此运行复现实验不需要再次调用 LLM API。

正式 CLI 会在文件缺失时从上述固定提交下载，并在读取前验证 SHA-256。也可以手动
执行：

```bash
python -c "from ttvr import download_official_prompts; download_official_prompts('data/fudd_official')"
```

## 完整性

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `cub_class_names.json` | 4,334 | `01eff4094773d49b92210f56efb9cab13860c0cdaf9953ff1e0ef082cf226723` |
| `cub_prompt_pairs.json` | 21,985,463 | `cc4300eaaf7c7bf46e515839ebe03abe827e86dafc3f49f63ea97a6c4c035237` |

可在仓库根目录验证：

```bash
shasum -a 256 data/fudd_official/cub_*.json
```

## 许可与使用边界

截至上述固定提交，上游仓库根目录未提供 `LICENSE` 或 `COPYING` 文件。因此，
上游资产的许可状态**未被明确声明**；本地下载的副本不构成额外授权，相关权利
仍归原作者或权利人所有。因此这两份 JSON 已被 `.gitignore` 排除。使用者应从作者
固定提交获取；在重新分发或用于其他场景前，应自行确认授权条件，必要时联系 FuDD
作者。

CUB-200-2011 数据集有独立的使用条款，请从
<https://www.vision.caltech.edu/datasets/cub_200_2011/> 获取，不要把下载后的数据集
提交到本仓库。Caltech 页面说明其不拥有原始图片版权，图片使用限于非商业研究和
教育用途。
