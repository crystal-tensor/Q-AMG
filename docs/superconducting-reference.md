# 相干一号：Feynman · 两张参考图约束重建

2026-11-12。名称由用户指定。当前第一例为 **852 个命名部件实例**；画廊共 12 例，其余 10 个装配参考与 1 个传动 CAD 保留。

## 图像如何使用

- **图1：外观。** 按图中的比例重建大型圆角铝型材机架、T 槽立柱、顶部仪器线束、两段银色真空桶、法兰和四个黑色提手。桌子、桌上的电脑、梯子及房间杂物不参与三维模型。
- **图2：内部。** 参考可见金色冷盘、银色支柱、两侧铜编织热连接、盘绕同轴及底部扇出和封装。未确认两张照片属于同一台设备；不把图2当作已经识别出图1隐藏结构的证据。
- 原始图片和本机神经重建试验保留在 `artifacts/runs/superconducting-photo-reference-20261112/`，不进入 Git 或公开画廊。公开元数据只保留图片角色、尺寸与 SHA-256，便于追溯。

外观图 SHA-256：`48dbcf7dad3ea3cfd5c479f9e6334a0be2dd21fea7bcff8ec0db0e0e95766805`。
内部图 SHA-256：`24ef97ccdd7587dab41ace14ce8af7a3dd6ab63280fd91bf273d3976e336ca06`。

## 建模方法与边界

先用固定版本 PartCrafter + SAM2.1 在本机 MPS 上实际试跑外观图（8 个神经部件、50 步、128 分辨率）。部件1没有形成可用表面，试验失败；原始输出、预处理、模型清单和日志均保留，没有把失败结果标为重建成功。

发布版本是另行明确标注的 **照片约束人工参数化重建**，源码为 `src/qamg/superconducting_photo.py`。它不冒充神经部件输出，亦未替换任何用户图片生成任务的失败结果。外形和可见布线按图对照；遮挡面、机械连接、螺钉位置及走线数量仍是设计推定。

模型采用名义 1:10 展示单位，没有从照片恢复实物标尺。内部金色组件竖向包络按 0.82 因子围绕顶盘拟合到外观图桶体内，这个比例选择保留在逐件证据中。48 路取样布线用于匹配图2可见密度，不声明实机实际通道数或量子位数。

| 模型类别 | 实例数 | 计数范围 |
|---|---:|---|
| 照片机架、连接角件、顶部模块及黑色线束 | 67 | 可见外观参考；顶部模块仅为仪器包络，无桌子或电脑 |
| 分节银桶、法兰、提手与桶体螺钉 | 47 | 上下两段桶体各一件，四个提手各按一根弯折件计；实体不透明 |
| 金色冷盘、银色支柱、盘缘螺钉及顶盘 | 96 | 五级冷盘；螺钉和孔位为推定 |
| 可拆内层热屏桶及底盘 | 6 | 拆开桶体时同时移至一侧 |
| 冷头、铜编织连接、波纹管和稀释回路包络 | 7 | 编织带按总成计数，不把每根金属细丝当一件 |
| 48 路参考微波布线 | 576 | 96 根连续盘绕同轴总成、192 个半夹块、96 个楔块、96 颗螺钉、96 个射频调理包络 |
| 底部扇出、连接器、封装及接口 | 53 | 24 根连续扇出线；芯片只标位置，型号和内部未识别 |
| **合计** | **852** | 参考模型实例数，非已核验实物 BOM |

连续线缆跨过热锚，不在每层冷盘处人为切段凑数；线缆端接随采购总成计一次。每个叶节点独立检查正体积连通性、导出水密、并集和对称差。真实装配、公差和材料用量不由这些几何检查证明。

## 浏览和拆解

- **整机外观：** 显示图1约束的大机架和封闭银色桶，排除桌子、电脑和杂物。
- **移开外桶：** 上下段银桶、法兰、提手和内层热屏使用独立实体位移，露出金色内部；不是降低外桶透明度。机架为放大内部暂不显示，可切回整机。
- **内部布线：** 只显示金色冷盘、支柱、编织带、盘绕线缆和底部封装，方便旋转、搜索与逐件放大。
- 三级爆炸展开和单独选择/合并继续可用。桶体移开位置是视觉示意；采样核验拆解使用原始装配姿态，不能把视图位移当作连续无碰撞路径认证。

模型 GLB 使用线性空间的金属材质，视图使用环境照明。缩略图直接渲染实际导出 GLB 的内部布线视图，没有将用户照片或生成插画冒充三维渲染。

## 来源及实机核验

用户图像支持可见外观；[Google 系统架构](https://quantumai.google/quantumcomputer)、[IBM 低温部件类别](https://www.ibm.com/quantum/hardware)、[微波工程介绍](https://research.google/pubs/quantum-computing-an-introduction-for-microwave-engineers/) 和 [Krinner 等的低温工程论文](https://link.springer.com/article/10.1140/epjqt/s40507-019-0072-0) 支持通用类别及被遮挡热锚的参考构型，不支持声称这是其中某台设备的制造 CAD。

**实机逐件核验仍为 0 / 500。** 两张照片不能提供同一设备的完整 BOM、实测尺寸、隐藏装配坐标与实机核对记录。`hardware-evidence.json`、`bom.csv`、`bom.json`、`structure.json` 和 ZIP 均保留这个范围；来源链接或自声明布尔标记不能升级为实机已核验。

量子部分仍使用仓库固定 QuantumBridge SDK。QB 测量仅选择展示展开比例；不会被描述为识别了神经部件或推断出照片遮挡内容。模拟器不证明物理量子加速。

## 复现与历史

```bash
uv run --locked python scripts/generate_showcases.py --only superconducting-computer
uv run --locked python -c "from pathlib import Path; from scripts.render_gallery_thumbnails import render; render(Path('artifacts/examples/superconducting-computer'), 'superconducting-computer')"
uv run --locked python scripts/export_gallery.py
npm --prefix web run build
./scripts/test.sh
uv run --locked python scripts/validate_showcase_details.py --output artifacts/runs/superconducting-photo-audit.json
```

之前 940 件架构参考保留于提交 `180c0050f9d5f2254c8898c73bc9bd914565781c`、原模块 `src/qamg/superconducting.py`、历史验收报告及本机快照 `artifacts/runs/superconducting-public-v1-before-photo-20261112/`。照片版与架构版的输入、几何及范围不同，不作无上下文性能比较。原夹持 CAD 的历史文件同样保留。
