# 小规模 Table 3：真机结果与 dsdSim 对照

2026-09-27，NYULAB（4× RTX A6000，只用空闲卡，最多 3 张），vLLM 0.27.1。计划与实现细节见 [table3-plan.md](table3-plan.md)。

## 配置

| | |
|---|---|
| target | Llama-3.1-8B-Instruct，bf16，1 张 A6000，eager，max_num_seqs 64，prefix caching 关闭（远程 EAGLE3 需要，三种方法统一） |
| draft | Vanilla / SpecDec++：Llama-3.2-1B-Instruct（独立 vLLM 进程）；EAGLE3：yuhuili/EAGLE3-LLaMA3.1-Instruct-8B 远程 head（target 把 aux hidden states 经 ZMQ 发过去） |
| N 个 drafter | 2 张 draft 卡轮转摆放：N=1/2 每卡 ≤1 个（T1）；N=4 每卡 2 个、N=7 每卡 4 个（T2，共卡） |
| 网络 | 注入 RTT 10 ms（单向 5 ms，无抖动），ZMQ；传输层实测额外开销 p50 ≈ 1 ms |
| 负载 | dsdSim `gsm8k_trace_master` 的长度，开环 Poisson，总速率 = 1 req/s × N，前 min(400, 120N) 个请求；chat 模板，prompt 两两不同，输出严格 `target_tokens`（ignore_eos），greedy |
| 方法 | Vanilla γ=4；SpecDec++ = dsdSim `SpecPPGammaPolicy` 逐行移植（min 2 / max 8 / 阈值 0.7，每请求选一次 γ，逐位置接受率用 γ=8 标定 run 实测，实际选出 γ∈{4, 8}）；EAGLE3 链式 γ=4 |

## 门禁（全部通过）

| 门禁 | 结果 |
|---|---|
| Step A 无损（fp32，在线负载，spec vs 无 draft baseline） | 40/40 逐 token 相同；记账（每轮 accepted 由下一轮确认长度推出）全过 |
| Step D 远程 EAGLE3 vs vLLM 原生 `method=eagle3` | 平均接受长度 2.98494 vs 2.98501（相对差 2e-5） |
| Step D 无损（fp32） | 30/30 逐 token 相同 |
| K8 hidden-state 对齐（本地 Qwen3-1.7B，强制 chunked prefill） | 与 HF 参考逐位置 cos ≥ 0.999，错一位时 ≤ 0.74 |
| Table 3 全部 12 个 run | 记账全过 |

（早一轮 fp32 门禁出现过 1/40 在第 101 token 分叉；参考前向给出的 top-2 logit 差 1e-4（相对 4e-6），是在线负载下 batch 组成不同导致的 fp32 近似平局，门禁现在会自动做这项检查。）

## 真机 Table 3

dsdSim 口径（与仿真器可直接比：TTFT 到首个有接受 token 的轮结束，TPOT = Σ轮时间/Σaccepted（不含 bonus），吞吐 = 完成数/时间跨度）：

| Method | Metric | N=1 | N=2 | N=4 | N=7 |
|---|---|---|---|---|---|
| Vanilla SD | TTFT (ms) | 132.2 | 131.1 | 189.0 | 409.9 |
|  | TPOT (ms) | 20.2 | 19.6 | 28.1 | 58.1 |
|  | Thr. (req/s) | 0.86 | 1.86 | 3.69 | 5.97 |
|  | accept ratio | 0.646 | 0.641 | 0.642 | 0.639 |
| SpecDec++ | TTFT (ms) | 156.6 | 165.6 | 273.4 | 3772.7 |
|  | TPOT (ms) | 19.0 | 19.7 | 33.5 | 75.7 |
|  | Thr. (req/s) | 0.86 | 1.86 | 3.65 | 5.18 |
|  | accept ratio | 0.544 | 0.549 | 0.552 | 0.552 |
| EAGLE3 | TTFT (ms) | 112.6 | 119.3 | 163.1 | 361.6 |
|  | TPOT (ms) | 18.0 | 19.2 | 26.6 | 56.5 |
|  | Thr. (req/s) | 0.86 | 1.86 | 3.70 | 6.07 |
|  | accept ratio | 0.500 | 0.495 | 0.493 | 0.496 |

标准口径（TTFT = 首 token − 到达；TPOT = 每请求 (末−首)/(n−1) 的均值；吞吐去掉头尾 10%；平均接受长度含 bonus）：

| Method | Metric | N=1 | N=2 | N=4 | N=7 |
|---|---|---|---|---|---|
| Vanilla SD | TTFT (ms) | 93.1 | 93.3 | 142.1 | 307.1 |
|  | TPOT (ms) | 18.1 | 17.7 | 25.2 | 52.3 |
|  | Thr. (req/s) | 0.82 | 1.95 | 3.71 | 5.55 |
|  | mean accept len | 3.57 | 3.54 | 3.55 | 3.53 |
| SpecDec++ | TTFT (ms) | 114.0 | 122.2 | 208.9 | 3622.4 |
|  | TPOT (ms) | 18.5 | 19.3 | 32.5 | 73.7 |
|  | Thr. (req/s) | 0.82 | 1.95 | 3.59 | 4.56 |
|  | mean accept len | 4.27 | 4.25 | 4.22 | 4.22 |
| EAGLE3 | TTFT (ms) | 80.3 | 84.6 | 120.0 | 264.5 |
|  | TPOT (ms) | 16.3 | 17.5 | 24.3 | 51.7 |
|  | Thr. (req/s) | 0.82 | 1.95 | 3.74 | 5.65 |
|  | mean accept len | 3.00 | 2.98 | 2.97 | 2.98 |

读法：

- 吞吐在 N ≤ 4 时等于供给负载；N=7（7 req/s）超过 target 容量，三种方法都饱和，天花板 EAGLE3 6.07 > Vanilla 5.97 > SpecDec++ 5.18。
- 排序 **EAGLE3 > Vanilla > SpecDec++**。与 paper Table 3（SpecDec++ 优于 Vanilla）相反：按实测的逐位置接受率，SpecDec++ 规则对长 prompt 选到 γ=8，每步 draft 与 verify 都更贵；负载低时 TPOT 与 Vanilla 持平，接近饱和时先崩（N=7 TTFT 3.8 s，请求在 vLLM 的 `max_num_seqs` 准入队列里排队）。
- EAGLE3 每轮接受得比 1B drafter 少（L̄ 2.98 vs 3.57），但 head 只有一层、draft 便宜（每轮均值 ≈ 13 ms vs 1B 的 ≈ 39 ms），整体最快。
- N=4/7 是 T2（多个 drafter 进程共一张卡），draft 延迟因共卡争用变为 3–4 倍（N=7 时最慢 drafter p50：Vanilla 86 ms，SpecDec++ 166 ms）。这是 3 卡限制下的硬件事实，不能当作「每个 drafter 独占一卡」的结论。

## dsdSim 对照

sim 用真机标定（`scripts/final_compare.sh`）：target verify/prefill 与 draft 延迟由 `sim/calibrate_from_run.py` 从真机 `steps.jsonl` 拟合（verify = 23.6 + 0.17·B + 0.084·Σ(γ+1) ms，R² 0.80；1B draft = 11.3 + 3.2·γ + 0.25·γ·(B−1) ms，R² 0.84；EAGLE3 单独拟合），接受率用真机实测的逐位置 profile（EAGLE3 用自己的），lockstep 模式（我们的 target step 阻塞等 draft），轮数按 γ/L̄ 对齐，prompt 长度用真机实际长度。EAGLE3 对照的是 sim 的链式代理（sim 自带的 EAGLE 模型在首个零接受轮就结束整个请求，不可用）。

按档位标定（N 各自的 draft 延迟，体现共卡），`real / sim (err)`：

| Method | Metric | N=1 | N=2 | N=4 | N=7 |
|---|---|---|---|---|---|
| Vanilla SD | TTFT (ms) | 132.2 / 188.4 (+43%) | 131.1 / 193.3 (+47%) | 189.0 / 262.8 (+39%) | 409.9 / 364.6 (-11%) |
|  | TPOT (ms) | 20.2 / 18.7 (-8%) | 19.6 / 19.0 (-3%) | 28.1 / 26.7 (-5%) | 58.1 / 39.2 (-32%) |
|  | Thr. (req/s) | 0.9 / 0.9 (+0%) | 1.9 / 1.9 (+0%) | 3.7 / 3.7 (+0%) | 6.0 / 6.1 (+3%) |
| SpecDec++ | TTFT (ms) | 156.6 / 227.3 (+45%) | 165.6 / 239.8 (+45%) | 273.4 / 348.3 (+27%) | 3772.7 / 411.9 (-89%) |
|  | TPOT (ms) | 19.0 / 17.9 (-6%) | 19.7 / 18.5 (-6%) | 33.5 / 29.0 (-14%) | 75.7 / 34.8 (-54%) |
|  | Thr. (req/s) | 0.9 / 0.9 (-0%) | 1.9 / 1.9 (-0%) | 3.7 / 3.6 (-0%) | 5.2 / 6.1 (+18%) |
| EAGLE3 | TTFT (ms) | 112.6 / 151.6 (+35%) | 119.3 / 167.6 (+40%) | 163.1 / 201.5 (+24%) | 361.6 / 265.1 (-27%) |
|  | TPOT (ms) | 18.0 / 16.7 (-7%) | 19.2 / 17.5 (-9%) | 26.6 / 22.4 (-16%) | 56.5 / 30.8 (-46%) |
|  | Thr. (req/s) | 0.9 / 0.9 (-0%) | 1.9 / 1.9 (-0%) | 3.7 / 3.7 (+0%) | 6.1 / 6.3 (+3%) |

MAPE 19.5%（36 格，中位数 11.1%）；draft 延迟按全部 run 合并标定时 MAPE 22.2%（中位数 15.5%）。

结论：

1. **未饱和区（N ≤ 4）sim 很准**：吞吐一致，TPOT 误差 −16%…−3%（9 格里 7 格在 ±9% 内）。
2. **TTFT sim 系统性偏高 25–47%**：结构差异而非标定问题——sim 的控制环由 drafter 驱动，首 token 前多一次网络往返和 draft；我们（与 vLLM 一致）由 target 驱动，prefill 在 target 本地完成。
3. **饱和区（N=7）sim 严重低估**：TPOT −32…−54%，SpecDec++ TTFT −89%，吞吐天花板高估最多 18%。两个缺口：(a) sim 没有 `max_num_seqs` 准入上限，真机请求在 waiting 队列排队；(b) 每卡 4 个 drafter 进程的争用不是 draft 延迟的线性函数。这正是 paper 自称「组件标定、非端到端验证」所留下的空白，也是它 Fig.5–8 在饱和点附近结论（JSQ 反转、吞吐天花板）最需要真机复核的地方。

## n:m（2 target : 2 drafter，3 张卡）

2 个 target 各占一卡，2 个 1B drafter 共用第三张卡（每个 target 都能扇出到两个 drafter），Vanilla γ=4，RTT 10 ms，router 按到达时间派发，总负载 8 req/s，400 请求：

| 路由 | 分流 t0/t1 | TTFT mean / p99 (ms) | TPOT mean / p99 (ms) | Thr. (req/s) | L̄ | 记账 |
|---|---|---|---|---|---|---|
| RR | 200 / 200 | 147.6 / 232.4 | 27.2 / 37.7 | 6.82 | 3.55 | ✅ |
| JSQ | 199 / 201 | 154.3 / 237.6 | 27.6 / 39.0 | 6.82 | 3.53 | ✅ |

这个负载低于饱和，两种策略几乎一样，符合 paper「策略差异出现在 target 饱和附近」的叙述；要复现 JSQ/RR 交叉需要按 ρ 扫到饱和（plan.md V3），不在本次范围内。

实现上踩到的坑：第一版 JSQ 在每次派发前同步问一遍各 target 的队列深度，而 target 只在两个 engine step 之间回复，等于每个请求在 router 上多等一个 step——同样负载下 TTFT 均值 434 ms、p99 1.2 s（RR 146 ms）。现在 target 每个 step 后主动推送深度，router 用「快照 + 快照之后已派出的请求」，不再阻塞（`dsdreal/router/`）。

## 复现

```bash
# remote（NYULAB）：门禁 + Table 3，一键、等空闲卡、可续跑
nohup scripts/run_all_nyulab.sh > runs/pipeline.log 2>&1 &
# 本地：拉回结果、标定 sim、出对照表
scripts/final_compare.sh runs/table3_final
```
