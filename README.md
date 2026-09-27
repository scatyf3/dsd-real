# dsd-real

分布式投机解码（disaggregated speculative decoding）的真实系统实现，用来端到端验证 DSD-Sim 的仿真结论。

路线是 **插 vLLM，不 fork**：draft 和 target 各跑一个独立的 vLLM 引擎，靠 vLLM 原生的
`speculative_config.method = "custom_class"` 把一个远程 drafter 塞进 target 的投机路径。
Stage 0 的七个探针在 pin 住的平台上全绿，Stage 2 的 fused 和 distributed 两条腿都已经在真 GPU 上验过。

平台：**vLLM 0.27.1 / 8×L40S (sm_89, 48GB)**，pin 见 [requirements/gpu.txt](requirements/gpu.txt)。

---

## 架构

```
   Draft vLLM Server                Draft Client                        Target vLLM Server
 ┌────────────────────┐      ┌──────────────────────┐              ┌──────────────────────────┐
 │ VllmDraftEngine    │      │ ┌─────┐┌────┐┌─────┐ │  ┌────────┐  │ ┌────────────┐ ┌───────┐ │
 │  1. Propose: γ tok │◄────►│ │ IPC ││ZMQ ││HTTP │ │◄►│Routing │◄►│ │Draft Runner│ │Target │ │
 │  2. Rollback: 按   │      │ └─────┘└────┘└─────┘ │  │ policy │  │ │ [custom_   │ │Runner │ │
 │     confirmed 前缀 │      │   一个 DraftClient    │  └────────┘  │ │  class]    │ │       │ │
 │     重跑 generate  │      │        契约           │  RoundRobin  │ └────────────┘ └───────┘ │
 └────────────────────┘      └──────────────────────┘              │   RemoteDraftProposer    │
   engine_vllm.py ✅          transport.py 三种都 ✅                 │   .propose() 每 step 一次 │
   engine_replay.py ✅        server.py / http_server.py ✅          └──────────────────────────┘
   engine_echo.py   ✅        policy/routing.py ✅                        proposer.py ✅
```

控制循环在 **target** 一侧：每个 engine step，vLLM 自己调用
`RemoteDraftProposer.propose()`，propose 里同步向各 drafter 扇出、等齐、返回逐请求变长的
draft token 列表。

**三条契约（由代码和测试固定，不是约定俗成）：**

- `propose()` 跑在 engine step 内部，**没有 event loop** → 整条 `DraftClient` 路径是阻塞的。
- `speculate()` 一次吃整个 batch，**不是**逐请求 API。墙钟必须是 `max(RTT_i)` 而不是
  `Σ RTT_i`；逐请求 API 会悄悄把 N 个 draft 节点串行化，正好毁掉要测的东西。
- accept/reject 反馈**不需要单独通道**：`propose()` 入参的 `num_tokens_no_spec[i]` 就是 slot i
  的确认长度，跟上一轮的差就是这一轮接受了几个（含 bonus token）。
  `configure(stats=[])` 把它直接导出来，不用从时序斜率反推。

请求身份目前靠 slot 位置恢复（`num_tokens_no_spec` 只会在同一请求占用该 slot 期间增长），
因为协议层还没有 request id。

---

## 传输层：三种实现，一个契约

`context.md` 里写的「从 zmq 到 http，先包装成抽象层」就是这一层。三种实现都在
[proto/transport.py](dsdreal/proto/transport.py) 后面，`make_client(kind, ...)` 按名字选：

| | `InProcessDraftClient` | `ZmqDraftClient` | `HttpDraftClient` |
|---|---|---|---|
| 线 | 直接函数调用 | DEALER/ROUTER + msgpack | HTTP/1.1 + JSON |
| 服务端 | 无 | [`DraftServer`](dsdreal/draft/server.py) | [`HttpDraftServer`](dsdreal/draft/http_server.py) |
| 并发方式 | 不适用（串行，延迟按 max 记账） | 一个 DEALER 管道化多条消息，send-all → recv-all | 按 drafter 分组，**每 drafter 一个 POST**，线程池并发 |
| 每 step 往返数 | 0 | 每 drafter 1 次（管道化） | 每 drafter 1 次 |
| 错误传播 | 直接抛 | `ERR ` 帧 → 客户端 re-raise | HTTP 400 + `{"error"}` → 客户端 re-raise |
| 状态 | ✅ Stage 2a GPU 验过 | ✅ Stage 2b GPU 验过 | ✅ loopback 测试通过，**未上 GPU** |

**这不是"二选一"，是两条都留着。** 判据是 `SpeculateReply.net_excess_ms`
（= 实测 RTT − 注入延迟 − drafter 计算），也就是序列化 + 排队的真实成本：
HTTP 便于调试（curl 就能打、抓包能读、换语言不用改协议），ZMQ 开销更低。
两边 loopback 上都断言 `net_excess_ms < 20ms`，真链路上哪个更划算要用这个数说话，不是靠假设。

[tests/test_transport_parity.py](tests/test_transport_parity.py) 把这件事钉死：同一串
propose/rollback 回合分别走三种传输，**token 必须逐个相同** —— 传输只能改变成本，不能改变结果。

### HTTP 侧两个不是可选项的设置

不设的话测到的是自己的配置错误，不是网络：

- **`protocol_version = "HTTP/1.1"`** —— stdlib 默认 1.0，每个响应后关连接，于是**每个 step 都付一次
  TCP 握手**，白白多一个 RTT，还会记进 `net_excess_ms`。
- **`disable_nagle_algorithm = True`** —— 否则几十字节的小响应要等 40ms ACK 定时器。
  客户端侧 `http.client` 自己已经设了 `TCP_NODELAY`，服务端侧没有。

还有一个是结构性的：**一个 POST 带一个 drafter 这一步的全部请求**。ZMQ 靠一个 DEALER
管道化多条消息拿到并发，HTTP/1.1 没有可用的 pipelining，逐请求 POST 会把同一 drafter 的
N 个序列串行成 N 次往返。按 drafter 打包后，往返数只和 drafter 数有关，和 batch 大小无关 ——
这条由服务端计数器 `speculate_posts` 断言，**不是**用计时断言：loopback 上八次多余往返不到 1ms，
计时根本看不出来，只会在真链路上才暴露。

---

## 已验证（真 GPU）

| 阶段 | 内容 | 结果 |
|---|---|---|
| **Stage 0** | K1–K7 七个 hook 探针在 0.27.1 上重跑 | 全绿，`runs/stage0_probes.json` |
| **Stage 0** | `custom_class` + `ReplayDraftEngine` 端到端回放 | 输出与 greedy baseline **逐 token 相同** |
| **Stage 2a** | fused：真 draft 模型 + `InProcessDraftClient`，Qwen3-1.7B + Qwen3-0.6B，fp32，γ∈{1,2,4,8} | 全部**逐 token 等于**非投机 baseline |
| **Stage 2b** | distributed：同样两个模型，draft 在独立进程独立 GPU，真 ZMQ socket，扫 RTT | **TPOT 随 RTT 线性，R²=0.98，L̄=2.88** |
| **Stage 2b** | 接受长度 fused vs ZMQ 对拍 | 完全一致，`rel_diff_of_means=0.0` |
| **Stage 3 起步** | N-drafter 扇出，2 个真 ZMQ drafter + RoundRobin | 5/5 分流，token 与 baseline 完全一致 |

Stage 2b 的线性度是 K2「propose() 在关键路径上」的运行时复查 —— 跨两个真实进程和一条真
socket 之后它仍然成立。

脚本在 [scripts/](scripts/)（`verify_*.py`）+ 对应的 `slurm_*.sbatch`。

---

## drafter 的状态机与回滚

slide 上写的「rollback: prefix + verified + bonus 重 prefill」，落地成三步：

1. **丢掉上一轮的投机尾巴** —— 提议完 γ 个 token 后 drafter 比确认前缀多 γ 个位置。
2. **追加 target 的权威 token** —— 它**不等于**自己提议过的 token：最后一个是 target 的
   bonus / 纠正 token，drafter 从没见过。
3. **用绝对的 `num_confirmed` 校验**，不一致就**大声报错**而不是从错误前缀继续 draft。
   用绝对值而非增量，还让协议在重试下幂等（HTTP 客户端的 keep-alive 断线重试正靠这条）。

[`VllmDraftEngine`](dsdreal/draft/engine_vllm.py) 用的就是 slide 上那个办法：每轮对完整确认前缀
**无状态地重跑 `generate()`**，靠 `enable_prefix_caching`（公开且稳定的引擎 flag）让重复前缀便宜，
不碰任何 vLLM 内部。正确性完全建立在确定性上：fp32 + `enforce_eager` + greedy，
同一前缀必然产生同一续写，命不命中 prefix cache 都一样。

代价是每轮多一次 prefill —— 这不是忠实的 SD 回滚。更"高效"的做法（draft 侧常驻、可暂停、
能回滚的 session）要拿版本敏感的内部实现重写一遍 target 自己的 KV 回滚调度，而真 drafter
每次被拒还是得回到重新提交这条路上。

---

## Stage 0 的几条约束

**探针全绿 = 「插 vLLM 不 fork」成立。** 完整的 0.21.0 首轮记录见
[docs/stage0-findings.md](docs/stage0-findings.md)（注意：那份文档写的是最初在 WSL / RTX 5070 Ti /
vLLM 0.21.0 上的结果，平台已经换到 0.27.1 / L40S 并重跑，最新数据以
`runs/stage0_probes.json` 为准）。

换版本时踩到的两处 API 漂移，已在探针里修掉：`NgramProposer.propose()` 多了一个前置的
`num_speculative_tokens` 参数；`Scheduler.schedule()` 多了 `throttle_prefills`。

仍然成立的约束：

1. **`draft_probs` 恒为 `None`**（23/23）→ 是 **token-match 接受、T=0 无损**，不是 Leviathan
   随机拒绝采样。slide 上那句 "vLLM V1 sd support greedy only?" —— 在这条路径上是的。
2. **`RejectionSampler` 有第二个调用点**（显存 profiling 期的 `_dummy_sampler_run`，随机 logits）。
   L2 tap 必须有 `live` 门闩，否则 acceptance 数据被污染且**不报错**（788 vs 288）。
3. **0.27.1 原生支持 `custom_class`**，所以 0.21.0 时代设想的 `NgramProposer` 子类 shim 不需要了；
   `plugin.py` 里那个 `GPUModelRunner.__init__` monkeypatch 也已经删掉 —— 它会无条件覆盖
   `self.drafter`，把所有非 custom_class 的运行一起弄坏。

---

## 快速开始

```bash
source scripts/env.sh          # 任何入口都要先 source

# 1. 纯 CPU：传输层 + proposer + 路由的全部测试，无 GPU、无模型、秒级
pytest tests -q                # 64 passed

# 2. 需要 GPU
python scripts/verify_vllm_hooks.py            # Stage 0 七探针（--probe k1 可单跑）
python scripts/verify_replay_proposer.py       # custom_class 回放，端到端无损
python scripts/verify_fused_real_draft.py      # Stage 2a
python scripts/verify_distributed_rtt_sweep.py # Stage 2b（需先起 draft server）
sbatch scripts/slurm_fused_real_draft.sbatch   # 集群上批量跑
```

`requirements/base.txt` 刻意不含 torch/vLLM：传输层和它的测试**永远不该需要 GPU 栈**，
这个分界线就是靠那个文件守住的。GPU 依赖在 `requirements/gpu.txt` / `pip install -e .[gpu]`。

### 起一个 draft server

```bash
# ZMQ（Stage 2b 用的就是这条）
python scripts/run_draft_server.py --drafter-id d0 --bind tcp://0.0.0.0:5560
# HTTP（同一个引擎，换条线）
python scripts/run_draft_server.py --transport http --drafter-id d0 --bind http://0.0.0.0:5560
```

target 侧在构造 `LLM` **之前**把 client 交给 proposer：

```python
from dsdreal.proto.transport import make_client
from dsdreal.target import proposer

proposer.configure(make_client("http", addresses={"d0": "http://node1:5560"}))
#                              "zmq",  addresses={"d0": "tcp://node1:5560"}   # 换这一行即可

target = LLM(
    model=..., dtype="float32", enforce_eager=True,
    speculative_config={
        "method": "custom_class",
        "model": "dsdreal.target.proposer.RemoteDraftProposer",
        "num_speculative_tokens": GAMMA,
    },
)
```

`configure()` 吃任意 `DraftClient`，所以换传输不需要改 proposer 一行代码。

---

## 目录

```
dsdreal/
├── proto/      messages.py（msgspec，JSON 和 msgpack 都能出）
│               transport.py（DraftClient + InProcess / Zmq / Http 三种实现）
├── draft/      base.py  engine_echo.py（无 GPU）  engine_vllm.py（真模型）
│               engine_replay.py（回放）  server.py（ZMQ）  http_server.py（HTTP）
├── target/     proposer.py（custom_class）  plugin.py（entry point）
├── policy/     routing.py（RoundRobin）
├── netem/      base.py  null.py  sleep.py（RTT/jitter/延迟矩阵）
├── router/ metrics/ config/ workload/ experiments/   ← 空包，Stage 3 起填
scripts/        env.sh  verify_*.py  slurm_*.sbatch  run_draft_server.py
tests/          test_transport.py  test_zmq_transport.py  test_http_transport.py
                test_transport_parity.py  test_remote_draft_proposer.py  test_routing.py
docs/           stage0-findings.md
plan.md         完整设计：架构、指标对齐、分阶段、终止判据
```

## 下一步（Stage 3）

多 draft 对 1 target：`router/` 前端、更多 routing 策略（Random / JSQ）、
`metrics/` 收集与导出。指标对齐是最大的风险点，`plan.md` 第五节已经把每个指标的定义和
对齐陷阱写死了 —— 尤其 TTFT 必须从 router 入口打戳（vLLM 自己的 `arrival_time`
从 engine-core 准入起算，会漏掉 routing 和排队），以及 acceptance ratio
不含 bonus token（vLLM 的 `mean_acceptance_length` 含）。

HTTP 传输目前只在 loopback 上验过，**上真 GPU 跑一遍 RTT 扫描**、和 ZMQ 对比
`net_excess_ms`，是 Stage 3 顺带要做的事。
