# RTX 5090 × InfiniteTalk 推理优化方案

> 目标：RTX 5090 32GB 单卡，LightX2V 四步蒸馏推理，视频最长边约 1080；先获得最大吞吐，再根据固定质量门槛逐项加回必要计算。

---

## 一、速度优先策略与质量边界

本项目必须使用 LightX2V，否则原始 40 步推理速度无法满足要求。因此这里不从最保守配置开始，而采用“最快档 → 质量回退档”的顺序：

| 档位 | 配置 | 用途 |
|---|---|---|
| S0 极限速度 | LightX2V 4 步、text CFG 1、audio CFG 1、BF16 FlashAttention | 首轮性能基线 |
| S1 口型增强 | S0 + audio CFG 2 | S0 口型或说话人归属不达标时使用 |
| S2 严格复现 | S1 + `--deterministic` | 仅在必须逐 seed 稳定复现时使用 |

所有档位共同保持：

- `sample_steps=4`
- `sample_shift=2`
- `sample_text_guide_scale=1`
- BF16 DiT 权重
- FlashAttention 全局注意力
- 不降低输出分辨率和帧数
- 不使用 FP8/INT8 权重量化
- 默认不使用 SageAttention、TeaCache、滑窗注意力或稀疏注意力替换

S0 先保留图像生成主干的分辨率、四步采样、BF16 权重和完整注意力，只关闭额外 audio CFG 分支。它可能影响口型同步强度，因此必须通过质量门槛；不达标时只回退到 S1，不应直接增加采样步数或降低分辨率。

### 分辨率说明

`infinitetalk-720` 是约 `960×960` 总像素面积的宽高比 bucket，并不是最长边限制。16:9 附近通常会选择约 `1280×704`，超宽画幅可能更长。如果“最长边 1080”是硬性输出规格，需要另行增加最大边约束；如果只是输入素材最长边为 1080，则无需修改 bucket。

---

## 二、CFG 前向次数：LightX2V 最关键的速度开关

当前代码的真实分支数为：

| text CFG | audio CFG | 每步 DiT 前向次数 | 说明 |
|---:|---:|---:|---|
| `1` | `1` | **1 次** | 极限吞吐；没有额外 audio CFG 分支 |
| `1` | `2` | **2 次** | 推荐质量配置；音频/口型引导更强 |
| 非 `1` | 任意值 | **3 次** | cond + drop-text + uncond，不适合 LightX2V |

必须保持 `sample_text_guide_scale=1`。`audio CFG=1` 与 `audio CFG=2` 的速度不是相同的：后者会增加一次完整 14B DiT 前向。

生产调优从 `audio CFG=1` 开始。对同一输入、同一 seed 检查口型同步、闭口准确性、停顿和多人说话切换；通过则直接采用 S0，失败才切换 `audio CFG=2`。这样质量成本只支付在确实需要的任务上。

---

## 三、已落地的质量等价优化

### 1. 删除热循环中的 CUDA 全局清缓存

原代码在每次 DiT 前向后调用 `torch.cuda.empty_cache()` 和 `torch.cuda.ipc_collect()`；参考注意力图内部还会在 40 个 Transformer block 中反复清缓存。这些调用破坏 CUDA allocator 复用并造成频繁同步。

已完成：

- 删除去噪循环内每次 CFG 前向后的 `torch_gc()`。
- 删除参考注意力图计算内部的 `torch_gc()`。
- 删除无必要的 chunk 内清缓存与同步。
- 只在 T5、CLIP、DiT 确实被卸载、下一阶段需要回收显存时保留 `empty_cache()`。
- 删除无意义的 `ipc_collect()`。

这部分不改变任何模型运算。

### 2. 缓存时间步无关的条件投影

原实现会在每个采样步、每个 CFG 分支重复计算 T5 文本投影、CLIP 图像投影、AudioProjModel 音频投影，以及人像 mask 到 token mask 的转换。这些结果在同一个视频 chunk 的四个去噪步中保持不变，现在每个条件分支只计算一次并复用。

对于 LightX2V 常用的 `text CFG=1, audio CFG=2`，cond 与 null-audio 分支还会共享文本、CLIP 和空间 mask 投影，只分别计算不同的音频条件。

### 3. RoPE 频率常驻 CUDA，并缓存三维 RoPE 网格

原 `self.freqs` 不是注册 buffer，可能在每层 Q/K 中重复 CPU→GPU 搬运；三维 RoPE 网格也会在每个 block、每个 CFG 分支、每个采样步重新执行 complex128 拼接。

已完成：

- 将 `freqs` 注册为非持久 buffer，使其随模型移动并在显存管理模式下常驻 CUDA。
- 按 `(T,H,W,device)` 缓存完整三维 RoPE 网格。
- 保留原始 float64/complex128 RoPE 计算精度，不采用 float32 近似。

### 4. 明确控制注意力后端

旧代码只要检测到 SageAttention 已安装就自动启用。SageAttention 会量化注意力中的部分计算，不能归入严格无损路径。

现在默认使用 `--attention_backend flash`，SageAttention 只能通过 `--attention_backend sage` 显式启用。质量优先生产配置禁止使用 `sage`。

### 5. 推理模式与性能测量

- 将核心推理上下文改为 `torch.inference_mode()`。
- 默认启用 `torch.backends.cudnn.benchmark`，固定尺寸下由 cuDNN 选择最快 kernel。
- 新增 `--deterministic` 用于严格复现。
- 新增 `--profile`，记录总 CUDA 时间、每帧耗时、峰值 allocated/reserved 显存。

---

## 四、RTX 5090 显存策略：WanVideoWrapper 风格块交换

14B DiT 的 BF16 权重约占 28GB，1080 级视频的激活、RoPE 缓存、latent 和 VAE 还需要额外空间，因此采用 WanVideoWrapper 的完整 transformer block 交换机制。

推荐起点：

```text
blocks_to_swap=20
prefetch_blocks=1
block_swap_non_blocking=false
```

`blocks_to_swap=N` 表示把 DiT 最后的 N 个 blocks 放在 CPU；前面的 blocks、embedding、adapter 和 head 常驻 GPU。推理进入一个交换 block 前，把它整体搬到 GPU，并在 CUDA stream 上预取后续 block；执行完当前 DiT forward 后，释放当前及预取 block。这里交换的是完整 block，不是逐个 Linear/Norm 的动态包装。

调优流程：

1. 先固定 `blocks_to_swap=20`、`prefetch_blocks=1`，用 `--profile` 建立速度和峰值显存基线。
2. 若显存有余量，依次尝试 `16`、`12`、`8`；数值越小，GPU 常驻 block 越多，通常越快。
3. 若 OOM，依次尝试 `24`、`28`；数值越大，CPU↔GPU 搬运越多，通常越慢。
4. `prefetch_blocks=1` 是速度/显存的起点；只有 PCIe 搬运明显成为瓶颈时才测试 `2`。
5. `block_swap_non_blocking=true` 仅在确认 CPU 权重存储已适合异步拷贝后测试；否则先保持 false，避免异步搬运收益不稳定。

旧的 `num_persistent_param_in_dit` 是逐模块 VRAM manager 的兼容路径，不要与 `blocks_to_swap` 同时使用。块交换模式当前要求单卡，不与 USP/FSDP 混用。

不建议在严格质量路径中用 FP8/INT8 来换取全模型常驻。

---

## 五、推荐命令

### S0 极限速度：LightX2V 四步 + audio CFG 1

```powershell
python generate_infinitetalk.py `
    --ckpt_dir weights/Wan2.1-I2V-14B-480P `
    --wav2vec_dir weights/chinese-wav2vec2-base `
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors `
    --lora_dir weights/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors `
    --lora_scale 1.0 `
    --input_json examples/single_example_image.json `
    --size infinitetalk-720 `
    --sample_steps 4 `
    --sample_text_guide_scale 1 `
    --sample_audio_guide_scale 1 `
    --blocks_to_swap 25 `
    --prefetch_blocks 1 `
    --attention_backend sage `
    --offload_model true `
    --profile `
    --mode streaming `
    --motion_frame 9 `
    --save_file infinitetalk_lightx2v_5090_fast
```

### S1 口型质量回退：audio CFG 2

仅将下面一项改为：

```powershell
--sample_audio_guide_scale 2
```

该配置每步执行两次 DiT。只有 S0 的口型同步、闭口准确性或多人说话人归属不达标时才启用。

### SageAttention 实验档

将 S0 命令中的 `--attention_backend flash` 替换为 `sage`。RTX 5090 使用 SageAttention 需要 CUDA 12.8 或更高版本。推荐安装 SageAttention 2.2.0：

```powershell
pip install sageattention==2.2.0 --no-build-isolation
```

SageAttention 会量化注意力计算，因此这是近似加速档，必须与 S0 FlashAttention 输出做口型、身份和时序一致性 A/B。当前暂不叠加 `torch.compile`，先单独测量 SageAttention 的实际收益。

### 不建议作为第一回退项

不要先增加采样步数、切换 40 步基础模型、降低分辨率或改变 motion frame。这些改动的成本或影响范围都大于单独恢复 audio CFG。质量回退顺序应为：

```text
audio CFG 1 → audio CFG 2 → deterministic（仅复现需求）
```

---

## 六、5090 基准测试矩阵

所有测试必须固定输入、seed、输出帧数和后台环境，并至少运行两次；首次运行包含 kernel 预热，不纳入最终对比。

| 组别 | audio CFG | block swap | attention | 目的 |
|---|---:|---:|---|---|
| A | 1 | 20 blocks / prefetch 1 | Flash | 极限速度基线及质量首检 |
| B | 1 | 16 blocks / prefetch 1 | Flash | 测试更高 GPU 常驻量 |
| C | 1 | 24 blocks / prefetch 1 | Flash | 测试更低显存档 |
| D | 2 | A/B 中最优配置 | Flash | S0 不达标时的口型质量回退 |

记录以下指标：

- `PROFILE total`
- `sec/frame`
- `peak_allocated`
- `peak_reserved`
- 首帧主体一致性
- 口型同步和闭口准确性
- 长视频 chunk 接缝
- 多人场景说话人归属

当前开发机是 RTX 4060 Laptop，且缺少项目完整 PyTorch/权重运行环境，因此本文不填写虚假的 RTX 5090 加速倍数。最终数据必须由目标 5090 使用 `--profile` 实测。

---

## 七、商业级加速方案状态矩阵

下面按“先无损工程优化、再进入近似计算”的顺序记录商业部署路线。`[x]` 表示当前项目已经有代码或可用开关；`[ ]` 表示尚未实现或尚未完成目标 5090 的验收。

### 已支持或已落地

- [x] LightX2V 四步蒸馏推理
- [x] BF16 DiT 推理
- [x] FlashAttention 后端（当前以 FlashAttention 2 为 5090 稳定基线）
- [x] 条件投影缓存：文本、CLIP、audio 和静态 mask
- [x] RoPE buffer 常驻 CUDA 及三维 RoPE 网格缓存
- [x] `torch.inference_mode()`
- [x] 固定尺寸下的 cuDNN benchmark
- [x] 完整 DiT block CPU↔GPU 交换
- [x] CUDA stream 预取后续 block，`prefetch_blocks=1`
- [x] `--profile` 性能和峰值显存统计
- [x] `audio CFG=1/2` 速度优先与质量回退档

### 商业级但尚未完成

- [ ] pinned host memory 权重池，配合真正异步 H2D/D2H 搬运
- [ ] block swap double-buffer 和 PCIe 搬运/计算流水线调度
- [ ] 每个固定分辨率/帧数 bucket 的 CUDA Graph capture
- [ ] 仅编译 DiT block 内部的 `torch.compile`
- [ ] CFG batch fusion：一次 batch=2 完成 cond/uncond
- [ ] 跨 timestep 的 cross-attention K/V 缓存
- [ ] 融合 QKV、RMSNorm、MLP 等 DiT 专用 kernel
- [ ] TensorRT 或 AOTInductor 固定 shape engine
- [ ] 多卡 tensor parallel / pipeline parallel
- [ ] VAE 编解码与下一 chunk 的流水线重叠

### 需要画质 A/B 的近似加速

- [ ] SageAttention 3；暂不考虑，属于 4-bit/FP4 近似注意力路径，当前项目的 `sage` 入口也不是 SageAttention 3 专用实现
- [ ] FP8 权重或 FP8 activation
- [ ] INT8 weight-only / SmoothQuant
- [ ] token merge、时空稀疏 attention、跳步缓存
- [ ] 进一步的蒸馏、少步采样或结构化剪枝

注意：RTX 5090 不应把官方 FlashAttention 3 作为默认后端。FA3 的官方实现主要针对 Hopper；5090 的稳定基线应使用 FlashAttention 2 或 PyTorch SDPA。SageAttention、FP8/INT8 和稀疏注意力均不属于严格无损路径。

---

## 八、不采用或暂缓的方案

| 方案 | 结论 | 原因 |
|---|---|---|
| TeaCache | 不采用 | 四步采样没有有效缓存窗口，且跳步会改变结果 |
| SageAttention 3 | 暂不考虑 | 4-bit/FP4 近似注意力，画质风险高于当前无损优先路线 |
| FP8/INT8 DiT | 严格质量路径不采用 | 改变权重数值 |
| RoPE float32 | 不采用 | 虽可能更快，但不保证数值等价 |
| Attention Sliding Window | 不采用 | 这里指注意力滑动窗口；不是 InfiniteTalk 已有的长视频 chunk 滑动窗口，模型按全局注意力训练，推理替换会改变分布 |
| FastWan/VSA | 暂缓 | InfiniteTalk 音频模块不兼容，需要重新训练/微调 |
| Batch CFG | 暂缓 | 1080 级 batch=2 激活显存过高，5090 32GB 风险大 |

---

## 九、已修改文件

- `wan/multitalk.py`：移除热路径清缓存、缓存 CFG 条件、启用 inference mode 和 cuDNN benchmark，接入 WanVideoWrapper 风格块交换。
- `wan/modules/multitalk_model.py`：Flash/Sage 显式选择、RoPE buffer/网格缓存、静态条件预计算及 block swap 调度入口。
- `src/vram_management/block_swap.py`：完整 DiT block 的 CPU↔GPU 交换与 CUDA stream 预取。
- `wan/utils/multitalk_utils.py`：移除参考注意力图内部 CUDA 清缓存。
- `generate_infinitetalk.py`：新增 `--attention_backend`、`--deterministic`、`--profile`、`--blocks_to_swap` 及块交换参数。

代码已通过 Python 语法编译与 `git diff --check`。RTX 5090 的实际耗时和画质验收仍需在目标机器完成。
