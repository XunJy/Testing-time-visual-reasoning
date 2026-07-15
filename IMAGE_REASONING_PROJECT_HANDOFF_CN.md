# 图像推理项目交接说明

> **2026-07-14 新实验线更新：** 本文主体保留迁移前项目的历史背景与 SigLIP 2
> 结论，不作为当前目录结构说明。当前 canonical workspace 为
> `/Users/xunj/Desktop/Testing-Time Visual Reasoning`。根目录的新实现已经通过直接 CLI
> 完整复现 FuDD × OpenAI CLIP：CUB 官方测试集 5,794 张，单模板
> `63.3586%`，FuDD k=10 `65.7404%`，提升 `+2.3818 pp`。新代码、运行方法和
> 验收说明见 [`README.md`](README.md)，机器可读结果见
> [`experiments/01_fudd_clip_cub/`](experiments/01_fudd_clip_cub/)。以下旧结果和路径
> 继续作为历史记录保留，未被本次运行覆盖。

## 1. 项目背景

原始题目是 **Testing-Time Visual Reasoning**。我将其具体化为：在不重新训练视觉主干的情况下，让模型在第一次分类后，通过生成文字描述、比较相似类别、观察裁剪区域或重新组合提示词等方式获得额外证据，再进行第二次判断，从而提高细粒度零样本图像分类性能。

当前主要任务是 CUB-200-2011 鸟类细粒度分类，共 200 个类别 [4]。核心模型是 SigLIP 2 [3]，主要强基线为 SO400M、384 像素输入和五个通用文本模板。早期方法的原始对照模型为 CLIP [1]，SigLIP 2 则建立在 SigLIP 的 sigmoid 图文预训练目标之上 [2]。

## 2. 统一实验协议

- 方法筛选：从 CUB 官方训练集固定抽取 1,000 张平衡验证图像，每类 5 张，随机种子为 2026。
- 最终评估：未用于调参的官方测试集，共 5,794 张图像。
- 严格零样本设置中，不使用 CUB 训练图像、标签、属性或边界框来拟合分类器。
- Ground-truth 边界框只作为诊断上限，不属于严格零样本结果。
- CAPA 是单独的训练式适配实验，不应归入纯测试时零样本方法。

## 3. 基线结果

| 方法 | Top-1 | Top-5 |
|---|---:|---:|
| SigLIP 2 Base-224，五个通用提示词 | 65.46% | 90.46% |
| SigLIP 2 SO400M-384，五个通用提示词 | **79.53%** | **96.77%** |
| SO400M 全图与 GT 边界框分数平均（诊断） | 80.13% | 97.08% |

从 Base-224 换成 SO400M-384 带来了 14.07 个百分点的稳定提升，是所有实验中唯一的大幅提升。GT 边界框只增加 0.60 个百分点，说明强模型仍能从更准确的局部区域中获益，但剩余空间已经很小。

## 4. 已完成的方法与结果

### 4.1 LLM 生成类别视觉描述

参考 CuPL 的 LLM 类别描述思路 [5]，使用 Qwen3-8B [12] 为每种鸟生成八类固定描述：轮廓、喙、头部/眼睛、喉部/胸部、背部、翅膀、尾部和腿。然后将这些描述编码为文本特征，与图像重新计算相似度。

- 强基线：80.1%（固定验证集）
- 八槽描述均匀融合：78.8%
- 八槽描述加 AutoCLIP：79.2%
- 将描述换成错误类别但保持槽位匹配的控制组：78.9%
- 正确描述相对错误描述仅高 0.3 个百分点，95% bootstrap CI 为 [-1.4, +2.0]

因此，该实验没有证明 Qwen 描述提供了可靠的类别语义增益。人工检查还发现部分描述虽然听起来合理，但存在事实错误。正确描述与错误类别描述的控制实验也受到 Ma 等人关于“真实描述语义”与“无语义提示词集成效应”区分方法的启发 [7]。

### 4.2 FuDD 相似类别差异描述

使用 FuDD 官方代码和完整的 19,900 对 CUB 差异描述缓存 [6]。流程是：第一轮先得到 Top-10 候选类别，再使用专门描述这些相似类别差异的文本进行第二轮分类。

- 在原论文的 OpenAI CLIP ViT-L/14@336 设置中，成功复现 63.36% → 65.74%，提升 2.38 个百分点；论文报告为 2.42。
- 将相同描述和两阶段公式迁移到 SigLIP 2 SO400M-384：
  - 固定验证集：70.80% → 73.10%
  - 官方测试集：72.73% → 72.54%，变化 -0.19 个百分点
  - 配对 95% CI 为 [-1.12, +0.76]，因此应解释为**没有显著改善**，而不是显著下降。

这里的 SigLIP 2 FuDD 基线使用单个模板，不能直接与五模板强基线 79.53% 比较。尚未完成从五模板强基线出发的公平 FuDD 融合实验。

### 4.3 固定裁剪与多视图测试时增强

对同一图像生成多个固定裁剪或放大视图，再通过平均或启发式选择合并分类结果。

- Base-224：65.3% → 67.1%，提升 1.8 个百分点。
- SO400M-384：80.1% → 79.8%，下降 0.3 个百分点。
- SO400M 固定裁剪均值：78.8%，下降 1.3 个百分点。
- 熵梯度 ROI 在 200 张图像筛选中：76.0% → 73.5%。其平均 GT-box IoU 仅为 0.183，并出现 51/200 个空 ROI。

这说明裁剪能帮助分辨率较低的弱模型，但对强模型，错误裁剪会丢掉上下文或关键部位，融合后反而覆盖原本正确的全图预测。

注意：这里测试的是 **fixed-crop test-time augmentation**，不是完整复现 TPT [8]。真正的 TPT 还包括大量随机视图、置信度筛选以及对可学习提示词进行测试时熵最小化更新，因此论文中不能写成“TPT 在 SigLIP 2 上失败”。

### 4.4 AutoCLIP 自动提示词加权

参考 AutoCLIP [9]，让每张图像自动调整五个通用提示词的权重，而不是简单平均。

- 相对于强归一化模板集成基线：80.1% → 79.9%。
- 但相对于严格匹配的 uniform-score 聚合基线：79.7% → 79.9%，实际增加 0.2 个百分点。

变化很小且没有统计可靠性。当前只有五个非常相似的提示词，而 AutoCLIP 原论文通常使用数量更多、差异更大的提示集合，所以不能笼统声称 AutoCLIP 已被完整否定。

### 4.5 CAPA 部位—属性残差模块

训练一个约 89.5 万参数的小型残差分支，使鸟的局部部位特征与正确属性描述接近，并与只修改一个属性的错误描述拉远。该设计受 CLIP-Adapter 的残差适配器 [10] 和 ViLLA 的细粒度区域—属性对齐 [11] 启发，但 CAPA 本身是本项目设计的实验模块，并不是已发表论文中的现成方法。SigLIP 2 主干基本冻结，且此阶段没有使用 Qwen 或其他 LLM。

- 在 40 个未见类别上，正确属性排序相对打乱控制提高 13.54 个百分点，证明模块确实学到了局部语义。
- 40-way 全局 SigLIP 2 基线：95.94%。
- 加入残差类名分支：94.10%。
- 即使使用 held-out 类别官方属性构造的特权 oracle 分支：95.05%。
- 200-way：基线 79.72%，类名融合 77.45%，oracle 融合 77.08%。

结论是：局部分支并非完全没有学习，但它比强全局分支弱；固定融合使它破坏的正确结果多于修复的错误。

### 4.6 Qwen3-VL 直接分类与重排序

还测试了完全用 Qwen3-VL [13] 代替 SigLIP 2，以及让它依据自己生成的线索对候选类别重排序。

- 200 张筛选图像中，Qwen3-VL 直接分类约 35.5%，对应 SigLIP 2 为 76%。
- 50 张重排序筛选中，Qwen 为 42%，基线为 76%；只修复 2 个结果，却破坏 19 个原本正确的结果。

该模型在开放式视觉问答中可以生成合理文本，但在封闭的 200 类细粒度分类及严格类名映射中不够可靠。

## 5. 当前最合理的总体解释

目前不能证明“SigLIP 2 已经内化了所有推理”，也不能声称“所有测试时推理方法在强模型上都无效”。更严谨的结论是：

> 在当前 CUB 协议下，测试的额外证据能够修复一部分错误，但没有为最强的五模板 SigLIP 2 基线带来统计可靠的净提升。随着基线增强，可修复的错误更少、更难；固定启用且未经校准的辅助分支会破坏更多原本正确的预测。

可以用下面的关系解释净变化：

\[
\Delta=(1-A)r-Ac
\]

其中，\(A\) 是基线准确率，\(r\) 是修复原错误的概率，\(c\) 是破坏原正确结果的概率。基线为 80% 时，辅助方法必须在错误样本上具有约四倍于正确样本破坏率的修复能力，才可能产生正收益。

已有数据与此一致：

- Base 裁剪：修复率约 11.8%，破坏率约 3.5%，最终 +1.8 pp。
- SO400M 裁剪：修复率约 8.5%，破坏率约 2.5%，最终 -0.3 pp。
- FuDD：修复 390 个错误，同时破坏 401 个正确结果，最终 -0.19 pp。

FuDD 的 oracle 二选一诊断可达到约 79.46%，明显高于其单模板首轮 72.73%。这表明第二轮证据并非完全没有互补信息，真正的瓶颈很可能是**何时启用、相信哪一个答案，以及如何校准两轮分数**。

## 6. 可形成的研究问题

建议将论文问题从“测试时推理能否提高准确率”改为：

> **Why do CLIP-era test-time reasoning methods fail to provide reliable gains for a stronger vision-language model?**

可以拆成三个可检验问题：

1. 辅助描述或视图是否包含能够修复错误的互补信息？
2. 性能失败主要来自证据本身质量差，还是来自融合、校准和门控失败？
3. 随着同一模型家族的基线增强，辅助推理的净收益是否系统性减小？

## 7. 最值得继续的实验

1. 直接利用现有预测，统一报告每个方法的 wrong→right、right→wrong、修复率、破坏率、margin 和配对置信区间。
2. 完成公平的 FuDD-5T：由五模板 79.53% 强基线产生 Top-10，再比较直接替换、校准融合、仅低 margin 启用、打乱描述和仅类名控制；所有阈值只在固定验证集确定，之后锁定并运行官方测试集一次。
3. 使用 80 个多样化提示词忠实测试 AutoCLIP，并在 CLIP 上运行相同协议作为 positive control。
4. 如果要论证“模型越强，增益越小”，应在 SigLIP 2 Base、Large、SO400M 上用相同分辨率和相同协议测试；否则只能称为当前 checkpoint 和 CUB 上的结果。
5. 如果不运行官方 TPT，应始终把现有裁剪实验称为 fixed-crop TTA。

## 8. 重要的表述边界

可以说：

- 当前测试的方法没有为最强 SigLIP 2 基线带来可靠净提升。
- FuDD 在原 CLIP 设置中的提升已复现，但迁移到 SigLIP 2 的 matched 结果为统计不显著的 null。
- 辅助证据存在纠错能力，但固定融合和缺少可靠门控导致净收益消失。

不要说：

- 所有 testing-time reasoning 在 SigLIP 2 上都必然无效或有害。
- 已经证明 SigLIP 2 “内化了推理”。
- 已完整复现并否定 TPT、CuPL 或 AutoCLIP。
- 单模板 FuDD 的 72.54% 比五模板基线 79.53% 低 7 个点，因此 FuDD 严重伤害性能。

## 9. 代码与结果位置

完整图像项目位于：

`/Users/xunj/Desktop/Reasoning/`

主要总结文件：

`/Users/xunj/Desktop/Reasoning/docs/CUB_ZERO_SHOT_RESULTS.md`

重要结果目录：

- `experiments/01_cub_zero_shot_baselines/`
- `experiments/02_fixed_crop_tta/`
- `experiments/03_autoclip_prompt_weighting/`
- `experiments/04_entropy_gradient_crop/`
- `experiments/05_qwen3_class_semantics/`
- `experiments/09_capa_siglip2_attribute_training/`

## 10. 给下一位助手的工作要求

请先读取上述总结文件和对应 metrics，不要重新猜测已有结果，也不要直接宣称项目失败。下一步应优先完成现有预测的错误转移分析和公平 FuDD-5T 门控实验，并始终区分“统计上无显著提升”“明确下降”和“尚未按原论文协议完整测试”这三种情况。

## 11. 参考文献

[1] Radford, A., Kim, J. W., Hallacy, C., et al. (2021). **Learning Transferable Visual Models From Natural Language Supervision.** *Proceedings of the 38th International Conference on Machine Learning (ICML)*, 8748–8763. [Paper](https://proceedings.mlr.press/v139/radford21a.html)

[2] Zhai, X., Mustafa, B., Kolesnikov, A., & Beyer, L. (2023). **Sigmoid Loss for Language Image Pre-Training.** *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 11975–11986. [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Zhai_Sigmoid_Loss_for_Language_Image_Pre-Training_ICCV_2023_paper.html)

[3] Tschannen, M., Gritsenko, A., Wang, X., et al. (2025). **SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features.** arXiv:2502.14786. [Paper](https://arxiv.org/abs/2502.14786) · [Official implementation](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md)

[4] Wah, C., Branson, S., Welinder, P., Perona, P., & Belongie, S. (2011). **The Caltech-UCSD Birds-200-2011 Dataset.** California Institute of Technology, Technical Report CNS-TR-2011-001. [Dataset and report](https://www.vision.caltech.edu/datasets/cub_200_2011/)

[5] Pratt, S., Covert, I., Liu, R., & Farhadi, A. (2023). **What Does a Platypus Look Like? Generating Customized Prompts for Zero-Shot Image Classification.** *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 15691–15701. [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Pratt_What_Does_a_Platypus_Look_Like_Generating_Customized_Prompts_for_ICCV_2023_paper.html) · [Code](https://github.com/sarahpratt/CuPL)

[6] Esfandiarpoor, R., & Bach, S. H. (2024). **Follow-Up Differential Descriptions: Language Models Resolve Ambiguities for Image Classification.** *International Conference on Learning Representations (ICLR)*. [Paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/17eca81974d2fd34f458916e0bfd820d-Abstract-Conference.html)

[7] Ma, P., Rietdorf, L., Kotovenko, D., Hu, V. T., & Ommer, B. (2025). **Does VLM Classification Benefit from LLM Description Semantics?** *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI-25)*. [Paper](https://ojs.aaai.org/index.php/AAAI/article/download/32638/34793) · [arXiv](https://arxiv.org/abs/2412.11917)

[8] Shu, M., Nie, W., Huang, D.-A., Yu, Z., Goldstein, T., Anandkumar, A., & Xiao, C. (2022). **Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models.** *Advances in Neural Information Processing Systems 35 (NeurIPS)*. [Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/5bf2b802e24106064dc547ae9283bb0c-Abstract-Conference.html)

[9] Metzen, J. H., Saranrittichai, P., & Mummadi, C. K. (2023). **AutoCLIP: Auto-tuning Zero-Shot Classifiers for Vision-Language Models.** arXiv:2309.16414. [Paper](https://arxiv.org/abs/2309.16414)

[10] Gao, P., Geng, S., Zhang, R., et al. (2021). **CLIP-Adapter: Better Vision-Language Models with Feature Adapters.** arXiv:2110.04544. [Paper](https://arxiv.org/abs/2110.04544)

[11] Varma, M., Delbrouck, J.-B., Hooper, S., Chaudhari, A., & Langlotz, C. (2023). **ViLLA: Fine-Grained Vision-Language Representation Learning from Real-World Data.** arXiv:2308.11194. [Paper](https://arxiv.org/abs/2308.11194)

[12] Yang, A., Li, A., Yang, B., et al. (2025). **Qwen3 Technical Report.** arXiv:2505.09388. [Paper](https://arxiv.org/abs/2505.09388) · [Official repository](https://github.com/QwenLM/Qwen3)

[13] Bai, S., Cai, Y., Chen, R., et al. (2025). **Qwen3-VL Technical Report.** arXiv:2511.21631. [Paper](https://arxiv.org/abs/2511.21631) · [Official repository](https://github.com/QwenLM/Qwen3-VL)
