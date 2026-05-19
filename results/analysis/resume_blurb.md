# PhysVLM 简历项目描述草稿

## A 版：具身智能算法

**基于 PhysVLM 的机器人物理可达性理解复现与评估**

- 复现 CVPR 2025 PhysVLM，搭建 RGB 图像 + S-P Map 的多模态推理流水线，用于判断机械臂对桌面物体的物理可达性。
- 基于 EQA-phys simulator 生成 CR5、FR5、PANDA、UR5 四类机械臂评测数据，共 1,600 条可达性问答样本，并实现不依赖官方 server 的离线评测脚本。
- 在四机器人 benchmark 上取得 79.94% 总准确率，其中 FR5 达到 83.75%、PANDA 达到 80.25%；进一步按问题类型统计 direct-pick 与 reachable-space 两类任务表现。
- 设计 S-P Map 消融评测协议，对比 aligned S-P Map、black depth placeholder 与 mismatched S-P Map，用于验证空间物理编码对 reachability reasoning 的贡献。
- 增加 PyTorch 推理 profiling 小闭环，统计端到端 latency、tokens/sec 与 CUDA peak memory，为后续部署优化提供 baseline。
- 进行错误分析，发现模型主要存在 Yes-bias：321 个错误中 319 个为 false positive，说明模型更易将不可达物体误判为可达，为后续 S-P Map 输入质量和判别阈值优化提供依据。
- 技术栈：PyTorch, Transformers, Hugging Face, VLM, S-P Map, PyBullet, Colab GPU。

## B 版：LLM Agent / 多模态推理

**多模态大模型的具身物理推理复现与误差分析**

- 复现 PhysVLM 的多模态推理流程，将视觉输入与机器人空间物理信息编码为 RGB + S-P Map 输入，验证 VLM 在 embodied QA 场景中的物理推理能力。
- 改写官方推理流程，构建 standalone inference 与 offline evaluation 脚本，绕开官方缺失的 server 启动入口并修复 checkpoint 加载兼容问题。
- 在 1,600 条 EQA-phys 仿真问答上完成自动化评测，输出按机器人、问题类型、Yes/No 混淆矩阵和对象级错误率划分的分析表。
- 实现 S-P Map ablation evaluator，支持 normal / black / mismatch 三种 depth mode，并自动生成消融指标表与 Markdown 报告。
- 构建 profiling 脚本记录单样本推理延迟、吞吐与显存峰值，明确 PyTorch baseline 与后续 ONNX/TensorRT 优化边界。
- 通过错误案例分析定位模型偏向肯定回答的问题，为后续接入通用 VLM baseline、prompt calibration 或 OOD 机器人泛化实验奠定基础。
- 技术栈：Python, PyTorch, Transformers, Hugging Face, Multimodal LLM, Automated Evaluation。

## README 结果段短版

Reproduced PhysVLM on the EQA-phys simulator benchmark with four robot platforms. The standalone evaluator processed 1,600 reachability QA examples and achieved 79.94% overall accuracy. Per-robot accuracy was 81.25% on CR5, 83.75% on FR5, 80.25% on PANDA, and 74.50% on UR5. Error analysis showed a strong Yes-bias: 319 of 321 errors were false positives, suggesting that the model tends to overestimate physical reachability in ambiguous scenes. S-P Map ablation showed that replacing S-P Maps with black placeholders reduced accuracy from 79.94% to 77.44%.
