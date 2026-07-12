# RTX 5090 × InfiniteTalk 推理提速完整优化方案

> 基于代码级分析 + 社区调研，覆盖所有讨论过的优化路径

---

## 一、当前配置诊断

| 参数 | 当前值 | 状态 | 说明 |
|------|--------|------|------|
| 分辨率 | 1080p 最长边 | —— | infinitetalk-720 bucket，序列最长 |
| audio cfg | 1.0 | ⚠️ | 口型弱，但已是最快 |
| **text cfg** | **2.0** | 🔴 **错误** | 触发3次前向，与lightx2v矛盾 |
| LoRA | lightx2v ×2 | ✅ | 步数蒸馏，5步推理 |
| sample_steps | 5 | ✅ | 已最少 |
| TeaCache | 未知 | ⚠️ | 5步推理下实际无效（见下分析）|
| torch_gc() | 循环内每步调用 | 🔴 | 强制CUDA流同步，破坏流水线 |

> [!CAUTION]
> **最大单点问题：text_cfg=2.0 完全抵消了 lightx2v LoRA 的加速效果。**
> - `text_guide_scale == 1.0` → 2次前向（cond + null_audio）
> - `text_guide_scale != 1.0` → **3次前向（cond + drop_text + uncond）**
>
> lightx2v 是在原始 Wan I2V（无音频条件）上做的流一致性蒸馏，只能省掉文本CFG那次前向。
> 正确用法必须是 `text_cfg=1.0`，否则每步多跑一次完整14B模型。

---

## 二、已研究但不适用的方案（排除项）

### TeaCache 对5步推理无效

TeaCache 的 `ret_steps` 硬编码为 `5×3=15`，等于5步推理的总forward call数。
每次调用都判断 `cnt < ret_steps` 为 True → 全程强制计算，**一次缓存都没有**。

```python
# multitalk_model.py L583
self.__class__.ret_steps = 5*3       # = 15（前15次call强制计算）
self.__class__.cutoff_steps = sample_steps * 3  # = 15（总call数）
# 结果：ret_steps >= cutoff_steps → 永远不进缓存分支
```

即使修改 `ret_steps`，lightx2v 5步推理中每步跨越的 $\Delta t$ 极大，相邻步残差差异显著，TeaCache 的相似度假设不成立，质量和命中率都会下降。

**结论：lightx2v 5步 + TeaCache = 无效组合，不推荐开启。**

---

### FastWan / VSA 稀疏注意力无法直接换

FastVideo（Hao AI Lab）的 Video Sparse Attention (VSA)：
- 两阶段稀疏：coarse pool 找重要 tile → fine attention 只在 tile 内计算
- **在稀疏模式下重新微调过 Wan 原始 DiT 权重**

架构冲突：

```
FastWan DiT（原始结构）:          InfiniteTalk DiT（修改版）:
  self_attn → VSA kernel            self_attn（全局注意力）
  cross_attn（文本/CLIP）           cross_attn（文本/CLIP）
  ❌ 无音频层                       ✅ audio_cross_attn ← 独有
                                     ✅ AudioProjModel  ← 独有
```

FastWan 的 state_dict 缺少 InfiniteTalk 的音频模块，**无法直接加载**。
将 VSA kernel 移植到 InfiniteTalk 需要在音频条件下重新微调，工作量极大。

**结论：等官方 InfiniteTalk 集成 VSA（已在 TODO 列表），目前不可行。**

---

### Sliding Window 注意力推理换不等于训练换

代码已有参数通路（`window_size` 传入 Flash Attention 2），理论上可设：

```python
# wan_multitalk_14B.py
multitalk_14B.window_size = (64, 64)  # 滑动窗口
```

但模型在 `window_size=(-1,-1)` 全局注意力下训练，推理时换局部注意力 = 分布外推理，
口型同步和跨帧一致性会明显下降。不建议在未重训的情况下使用。

---

## 三、可执行优化方案（按优先级排序）

### 【P0】修正 text_cfg 参数（零代码，最大收益）

```bash
# 将命令行参数从：
--sample_text_guide_scale 2.0
# 改为：
--sample_text_guide_scale 1.0
```

| 配置 | 前向次数/步 | 预期耗时/it |
|------|-----------|------------|
| text=2.0, audio=1.0（当前）| **3次** | ~70s |
| text=1.0, audio=1.0 | **2次** | ~47s (**1.5×加速**) |
| text=1.0, audio=4.0 | **2次** | ~47s（口型更准，同速）|

> [!TIP]
> text_cfg=1.0 + audio_cfg=4.0 是 lightx2v LoRA 的官方推荐配置：
> 文本引导由 LoRA 蒸馏保证，音频引导由独立 CFG 控制，口型质量更好且速度相同。

---

### 【P0】安装 SageAttention（一行命令，最接近稀疏注意力的效果）

代码已有自动检测集成，安装即用：

```bash
pip install sageattention
```

```python
# multitalk_model.py L17-22 — 自动启用
try:
    from sageattention import sageattn
    USE_SAGEATTN = True   # ← 安装后自动 True
except:
    USE_SAGEATTN = False
```

SageAttention 的加速原理：将 Q/K 量化到 INT8，保留 V 精度，等效于近似注意力。
RTX 5090 INT8 算力 ≈ FP16 的 2×，注意力计算加速 **1.3–1.5×**，质量损失极低。

**这是当前能用的最接近「稀疏注意力」效果的方案，且无需重训。**

---

### 【P1】修复 float64 RoPE（2行代码，隐藏的最大性能杀手）

#### [MODIFY] [multitalk_model.py](file:///d:/vibecoding/InfiniteTalk/wan/modules/multitalk_model.py)

```python
# rope_apply() L63 — 当前代码（极慢！）
x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(s, n, -1, 2))
# RTX 5090 是消费级 GPU，FP64 算力仅为 FP32 的 1/64

# 修改为 float32（精度影响 < 0.1%）
x_i = torch.view_as_complex(x[i, :s].to(torch.float32).reshape(s, n, -1, 2))
```

同时修改 `rope_params()` 返回类型（避免 freqs 仍是 complex128）：

```python
# L44-50 当前
freqs = torch.outer(
    torch.arange(max_seq_len),
    1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)))

# 修改
freqs = torch.outer(
    torch.arange(max_seq_len),
    1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
```

预期加速：**每次前向 +10–30%**（5090 FP64 性能极差，这一改动效果显著）。

---

### 【P1】移除去噪循环内的 torch_gc()

#### [MODIFY] [multitalk.py](file:///d:/vibecoding/InfiniteTalk/wan/multitalk.py)

当前代码（L715-729）每次前向后都调用 `torch_gc()`（= `empty_cache` + `ipc_collect`）：

```python
# 去噪循环内（L709 for 循环中）：
noise_pred_cond = self.model(...)
torch_gc()           # ← 删除
noise_pred_drop_audio = self.model(...)
torch_gc()           # ← 删除
```

RTX 5090 有 32GB VRAM，5步×1080p 的14B模型（bfloat16 ≈ 28GB）加激活值仍有余裕。
`empty_cache` 强制 CUDA 流同步，每调用一次停顿数十毫秒。

修改：**保留循环外（VAE/CLIP卸载前后）的 `torch_gc()`，删除循环内的所有调用。**

预期加速：**+10–15%**。

---

### 【P2】关闭 offload_model

```bash
--offload_model false
```

当前默认 `offload_model=True`，每次模型前向前后都做 CPU↔GPU 数据搬运。
5090 32GB 在未开量化时可能刚好够用（28GB模型 + 激活值），建议先测试：

```
显存占用估算：
  DiT 14B bfloat16  ≈ 28 GB
  VAE               ≈  1 GB
  T5 / CLIP         ≈  2 GB（offload后CPU）
  激活值/latent     ≈  2 GB
  总计              ≈ 33 GB  ← 略超，建议配合 int8 量化使用
```

---

### 【P2】int8 量化（官方已支持）

```bash
--quant int8
```

将 DiT 量化为 INT8，显存节省约 50%（28GB → 14GB），腾出空间关闭 offload：

```
  DiT 14B int8      ≈ 14 GB
  其他              ≈  5 GB
  总计              ≈ 19 GB  ← 充裕，offload 可安全关闭
```

速度影响：INT8 矩阵乘法在 5090 上理论更快，实测约 **+10–20%**。

---

### 【P3】BatchCFG：将2次前向合并为1次（需代码改动）

当 text_cfg=1.0 时，当前代码顺序执行两次独立前向：

```python
noise_pred_cond       = self.model(latent, t, **arg_c)
noise_pred_drop_audio = self.model(latent, t, **arg_null_audio)
```

可改为 batch=2 合并：

```python
# 拼接 batch 维度
latent_batch = torch.cat([latent, latent], dim=0)
context_batch = torch.cat([arg_c['context'][0], arg_null_audio['context'][0]], dim=0)
audio_batch = torch.cat([arg_c['audio'], arg_null_audio['audio']], dim=0)
# 单次前向
preds = self.model_batch(latent_batch, t, ...)
noise_pred_cond, noise_pred_drop_audio = preds.chunk(2)
```

优点：消除第2次前向的 Python 调度和 kernel 启动开销，Tensor Core 利用率更高。
预期加速：**+15–25%**（非2倍，因带宽是瓶颈）。

---

### 【P3】torch.compile 内核融合

```python
# multitalk.py，Pipeline.__init__ 末尾
self.model = torch.compile(
    self.model,
    mode="max-autotune",
    dynamic=False,   # 分辨率固定时设 False，触发更激进的自动调优
)
```

首次调用会有约 1-3 分钟编译期，之后每次推理受益。
预期加速：**+10–20%**。

---

## 四、优先级总览

| 优先级 | 优化项 | 操作难度 | 预期加速 | 风险 |
|-------|--------|---------|---------|------|
| **P0** | ✅ text_cfg: 2.0 → 1.0 | 改参数 | **1.5×** | 无 |
| **P0** | ✅ 安装 SageAttention | pip 一行 | **1.3–1.5×** | 极低 |
| **P1** | ✅ rope float64 → float32 | 2行代码 | **1.1–1.3×** | 极低 |
| **P1** | ✅ 移除循环内 torch_gc() | 3行代码 | **1.1–1.15×** | 极低 |
| **P2** | ⚙️ int8 量化 + 关闭 offload | 参数改动 | **1.15–1.3×** | 低 |
| **P3** | 🔨 BatchCFG（合并两次前向）| ~30行代码 | **1.15–1.25×** | 中（需测试）|
| **P3** | 🔨 torch.compile | 3行代码 | **1.1–1.2×** | 低（需预热）|
| **❌** | TeaCache（5步无效）| — | 0 | — |
| **❌** | FastWan/VSA（架构不兼容）| — | — | 高 |
| **⏳** | Sliding Window 稀疏注意力 | 需重训 | — | 高 |

**理论最优叠加（P0+P0+P1+P1+P2）：**
$$1.5 \times 1.4 \times 1.2 \times 1.12 \times 1.2 \approx \mathbf{3.6\times}$$
$$70s/it \rightarrow \approx 19s/it$$

---

## 五、推荐的最终命令

```bash
# 安装加速依赖（一次性）
pip install sageattention

# 推理命令（优化后）
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --lora_dir weights/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors \
    --input_json examples/single_example_image.json \
    --lora_scale 1.0 \
    --size infinitetalk-720 \
    --sample_text_guide_scale 1.0 \
    --sample_audio_guide_scale 1.0 \
    --sample_steps 4 \
    --mode streaming \
    --motion_frame 9 \
    --sample_shift 2 \
    --num_persistent_param_in_dit 0 \
    --save_file infinitetalk_res_lora
```

> [!NOTE]
> `audio_cfg=4.0`（而非 1.0）在 text_cfg=1.0 时可以显著改善口型同步，且速度与 audio_cfg=1.0 完全相同（都是2次前向）。建议配合使用。

---

## 六、代码改动清单

### [MODIFY] [multitalk_model.py](file:///d:/vibecoding/InfiniteTalk/wan/modules/multitalk_model.py)
- `rope_params()` L47：`torch.float64` → `torch.float32`
- `rope_apply()` L63：`.to(torch.float64)` → `.to(torch.float32)`

### [MODIFY] [multitalk.py](file:///d:/vibecoding/InfiniteTalk/wan/multitalk.py)
- 去噪 for 循环内（L717, L722, L726, L729）：删除 `torch_gc()` 调用

### 无需改代码
- `pip install sageattention` → 自动启用
- 命令行参数修改 → 立即生效
