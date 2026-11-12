# 相干一号：费曼 · 超导量子计算机参考设计

资料核对：2026-11-12。此名称由用户指定，不是 Google 或 IBM 的产品型号。

画廊第一例替换了原来的夹持机构，仍保留 12 个公开示例。当前模型有 **940 个命名部件实例**，按总成、子总成、零件三级浏览。**这不是已核验的 940 件实机 BOM；“至少 500 件完全真实”尚未达到实机证据要求。** 页面、逐件清单、结构文件及 ZIP 同时保留这个状态。

## 参考依据

| 来源 | 支持内容 | 不支持的结论 |
|---|---|---|
| [Google Quantum AI · Quantum computer](https://quantumai.google/quantumcomputer) | 超导处理器、低温环境、定制测控电子学与软件的系统架构 | 本模型是 Google 某台设备的实物复制 |
| [IBM Quantum · Hardware](https://www.ibm.com/quantum/hardware) | 真空馈通、脉冲管、HEMT、超导同轴、磁屏蔽、量子极限放大器及 QPU 类别 | 本模型的尺寸、数量、装配位置由 IBM 核准 |
| [Bardin et al. (2020) · Microwave engineers](https://research.google/pubs/quantum-computing-an-introduction-for-microwave-engineers/) | 微波驱动、读出与控制电子学的功能关系 | 仪器料号、实际机柜配置或外形尺寸 |
| [Krinner et al. (2019), DOI 10.1140/epjqt/s40507-019-0072-0](https://link.springer.com/article/10.1140/epjqt/s40507-019-0072-0) | §2–3、Fig.3–6 的五级低温系统、线缆热锚、射频链路和部分实际器件型号 | Google / IBM 的完整生产 BOM 或本展示名称对应实机 |

本例以经典圆柱形低温机架为参考，不声称复刻 Google Willow 或 IBM 最新模块化系统。几何独立编写，未复制厂商 CAD、官网照片或第三方三维资产。

## 为什么采用这些部件

65 路射频配置取自 Krinner 的 Fig.3 / §3.1：25 路驱动、25 路磁通、6 路读入、5 路泵浦、4 路输出。论文另一测量语境中的 66 路，以及未来容量讨论中的 124 路，不用来补数。它也不证明本模型具有某个量子位数。

| 模型部件类别 | 建模实例数 | 计数边界 |
|---|---:|---|
| 连续同轴线缆总成 | 166 | SS 类 162、NbTi 类 4；端接随整根采购总成计一次，跨热锚不中断；分段路线由参考设计指定 |
| 铜热锚半夹块 | 260 | 65 通道 × 2 热锚 × 左右半块；器件类别参考 Fig.4(d)，数量和安装位置为设计假设 |
| 热锚压紧楔块 | 130 | 每个参考热锚一块 |
| M3 黄铜紧固螺钉 | 130 | 每个参考热锚一颗；螺钉长度和螺纹啮合未核验 |
| 低温衰减器 | 100 | 驱动 75、磁通 25；电性能类别参考论文，非实机采购凭证 |
| 红外滤波器 / 低通滤波器 | 50 / 25 | 不虚构滤波器封装内部零件；低通原文型号存在不同写法，明确待核 |
| HEMT / 环行器 / TWPA | 各 4 | 参考读出链；公开型号或缺项逐件列出，外形近似 |
| 四通道隔离器阵列 | 2 | 两个采购模组，四条电气路径不拆成四件 |
| 其余机架、冷级、屏蔽、制冷回路、QPU 封装及室温模块 | 65 | 机械与测控参考布置；密封仪器仅计包络，不凭空补内部 BOM |
| **合计** | **940** | 参考模型实例数，不是已核实实物件数 |

没有把芯片上的量子位图案、信号路径或刻字计为独立零件。每个叶节点有独立实例 ID、器件类别、来源定位、参考料号或缺项、数量依据、尺寸依据和位置依据。

展示按名义原型的 1:10 比例建模，但原型机械尺寸本身未经测量，不能乘十后当作原厂尺寸。线缆、仪器和封装采用简化外形包络，其体积不代表真实复合材料用量。金属外罩的透明度仅用于浏览；可切换恢复不透明外罩。

## 实机真实性状态

当前实机逐件核验数为 **0 / 500**。公开技术文章中的真实器件类别、有效网格、参考料号及软件中的自声明标记，都不能证明该器件真实存在于同一台设备中。

补齐实机核验需要同一台设备的授权 BOM 及版本、逐件料号/数量对应、带尺寸 CAD 或测量记录、装配坐标/连接关系，以及绑定到这些文件的实机逐件核对记录。现有代码不会因网格数超过 500 或自填 `as_built_verified=true` 就将状态升级为已核验。

`hardware-evidence.json` 报告模型件数、来源关联数、实机核验数、缺失证据及未满足的真实 500 件要求。该文件也包含在 `parts.zip` 中。`bom.csv` / `bom.json` 保留每件的证据字段；用户合并后，未合并件保留实例来源，新组合件记为几何编辑并重新计算件数，不继承实物认证。

## 生成与验证

```bash
uv run --locked python scripts/generate_showcases.py --only superconducting-computer
uv run --locked python -c "from pathlib import Path; from scripts.render_gallery_thumbnails import render; render(Path('artifacts/examples/superconducting-computer'), 'superconducting-computer')"
uv run --locked python scripts/export_gallery.py
npm --prefix web run build
./scripts/test.sh
uv run --locked python scripts/validate_showcase_details.py --output artifacts/runs/superconducting-release-audit.json
```

保留原先 10 例和夹持实例的历史数据。新增模型的首次试生成留在本机 `artifacts/runs/superconducting-computer-initial-20261112/`，随后对齐冷盘开槽与射频树，并消除夹块拼缝的设计重叠；不以首轮结果冒充最终导出。

继续使用仓库固定的 QuantumBridge SDK（`0.1.0rc1`，原生源码快照未替换），未引入其他量子执行 SDK。QB 的 3 个逻辑变量选择展示展开比例，部件和形状由参考建模代码定义。GPU / CPU 模拟并不证明物理量子加速；没有因为这个硬件展示模型存在就声称已接入实机。当前模型各件体积之和比并集体积大约 6.04%，说明名义包络存在交叠，不能当作配合已核准的装配 CAD。243 / 939 步通过所选方向的离散核对，其他件可用分级示意展开检查；碰撞采样、视觉爆炸图与连续路径认证分别报告。
