# 超导 QPU 适配说明

当前没有公司 SDK、端点或凭证。下面是本项目定义的参考协议，不是猜测某家供应商已经实现的 API。可直接用公司 SDK 实现 `CircuitBackend`，也可部署翻译此协议的服务。

## 本地执行

默认 `auto` 使用 QB 原生电路/编译与 Torch CUDA/MPS 态演化，无 GPU 时使用 QB 原生 CPU。`qb` 和 `qb-noisy` 分别为原生态向量与密度矩阵噪声路径。超导参考配置在 `configs/superconducting.reference.json`：它是示例线形拓扑和示例误差率，`verified_company_hardware=false`。真实设备必须替换这些数据。

## 配置

复制 `configs/company.example.json`，填入明确提供的服务地址；令牌通过其指定的环境变量读取，不写入配置或测量文件。

```bash
QAMG_COMPANY_CONFIG=/absolute/path/company.json uv run qamg generate --spec examples/drive.json --encoding compact_binary --backend company --output artifacts/runs/hardware-attempt
```

`CompanyBackend` 有能力查询、编译、提交、轮询、结果和超时取消。它不会把连接成功本身视为硬件验证；来源标为 `hardware_declared` 或 `remote_simulator`。

## 能力查询

`GET /capabilities`

```json
{
  "name":"provided-company-device-name",
  "is_simulator":false,
  "max_qubits":12,
  "max_shots":4096,
  "basis_gates":["rz","sx","x","cx"],
  "coupling_map":[[0,1],[1,0]],
  "bit_order":"qubit0_left",
  "calibration_id":"provided-calibration-identifier"
}
```

这里的名称、比特数、边和校准 ID 都需要由实际服务提供。示例耦合仅示意格式。最终编码可能只有 5 个逻辑比特，但路由后分配的物理寄存器可以更大；返回计数字符串的宽度应对应测量经典寄存器，而不是空闲物理比特。

## 任务生命周期

1. `POST /jobs`：JSON 包含 `qasm3`、`shots`、唯一 `client_request_id`；返回 `job_id`。
2. `GET /jobs/{id}`：返回 `status`，取 `queued/running/completed/failed/cancelled`。未知状态明确报错。
3. `GET /jobs/{id}/result`：完成后读取计数及可用的设备计时。
4. `POST /jobs/{id}/cancel`：请求取消，超时时尽力调用。

```json
{
  "bit_order":"qubit0_left",
  "counts":{"010":60,"110":40},
  "timings":{"queue_s":0.2,"quantum_execution_s":0.003,"readout_s":0.004}
}
```

`counts` 只能含合法等宽二进制串与正整数，计数总和必须等于请求 shots。计时字段未知时省略或设 `null`；禁止填入零冒充测得时间。设备上报时间可能包含重叠阶段，不将其简单相加伪造总体时间。客户端单独记录完整往返时间。

## 真机验证清单

- 对齐公司 SDK 的门约定、QASM 支持、比特顺序、任务状态与计数格式。
- 获取实际拓扑、校准时间、shots/任务预算、重试与取消规则。
- 在授权范围执行基态、叠加态、纠缠及结构电路，保存提供商任务凭证和原始返回值。
- 在相同原问题、目标阈值与质量约束下运行基准，计入编译、队列、网络、读出和经典反馈。
- 对冷启动与可摊销热启动分别报告，必要时提供 provider 审核的硬件来源证据。

契约测试的 mock 明确叫 `contract-mock-not-a-qpu`，只证明适配状态机和解析逻辑，不证明已经完成以上硬件验证。

## 离线实验包与有额度的执行器

已准备 `artifacts/hardware-bundle-qb/`。结构任务为 30 个留出输入 × 5 次重复，每次 256 shots，使用已有训练集角度、深度 1、零在线参数优化；另有基态 `100`、叠加态及 Bell 的 Z/X/Y 测量各 5 次，共 25 次基础检查。总上限 175 个任务、44,800 shots，无重试。逻辑电路最多 5 个量子比特；原问题仍有 6/8/10 个独热结构变量。

0.3 源码快照对应的新包为 `artifacts/hardware-bundle-qb-v2/`，已通过离线完整性校验，任务与预算保持相同。旧包保留原样；实际执行应使用与预期源码对应的包。图片功能复验没有产生额外 QPU 提交。

任务包包含 QPY、可阅读的 QASM、完整原问题与代价阈值、数据划分、角度训练来源、源码快照和逐文件 SHA-256。采用 QuantumBridge 自身实现的 QPY/QASM 序列化保存逻辑电路，应用不依赖其他量子 SDK；只有取得设备能力后，才按实际门集和拓扑编译。这里没有假定公司使用示例线形板卡。基础分布检查用于发现端序、门或读出接口问题，不能独立证明物理硬件来源或认证纠缠。

可重复准备与离线校验：

```bash
uv run --locked python scripts/prepare_hardware.py --corpus artifacts/benchmarks/registered-v2/corpus.json --transfer artifacts/benchmarks/registered-v2/transfer.json --output artifacts/hardware-bundle-new
uv run --locked python scripts/run_hardware.py --bundle artifacts/hardware-bundle-new
```

两条命令都不连接远程端点。当前实际只执行过这种离线校验以及明确标注的 mock 契约测试。

未来拿到兼容端点、凭证及相应调用授权后，执行命令为：

```bash
QAMG_COMPANY_CONFIG=/absolute/path/company.json uv run --locked python scripts/run_hardware.py --bundle artifacts/hardware-bundle-qb --execute --output artifacts/hardware-execution-new --max-jobs 175 --max-shots 44800
```

额度必须覆盖整个预登记任务包；执行器实际仍以任务包上限限制调用。它会拒绝源码、依赖锁或资产哈希变化，要求重新生成版本。种子用于经典求解与电路编译；真机测量随机性不由这些种子控制。

执行器先读取能力并保存校准标识，再运行基础检查；任一分布检查未通过时，不提交结构任务。通过后按同一问题、阈值与五种经典基线执行随机顺序基准。`submissions.jsonl` 在网络提交之前 fsync 保存请求 ID、shots 和原生电路哈希，收到回复后立即保存提供商任务 ID，原生电路另存。丢失提交回复、轮询或结果报错都会停止后续提交；已有记录和失败保留，程序不会自动重试或续跑。需向提供商核对请求 ID、任务状态和计费后，才规划新一轮。

`failure-inclusive.json` 纳入所有登记运行。失败按登记时限计费，明确是分析口径而非实测命中时间；同时报告成功率及实际返回差距。编译、网络、排队、采样、读出、解码、几何和摊销训练分开保存，未知设备计时保持 null。执行结果保持 `hardware_validated=false`，需要独立核对实际设备来源、校准与任务凭证后再做真机统计；接口返回 `is_simulator=false` 本身不够。

历史 `hardware-bundle-v1` 属于旧 SDK 阶段，哈希保护保留原样；当前 QB 运行器必须使用重新准备的 QB 任务包。
