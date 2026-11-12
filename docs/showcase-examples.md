# 公开资料辅助的部件装配参考

`EXAMPLES / REPRODUCIBLE` 位于页头，与产品能力说明同一行。第 2 版将原先 145 个粗略部件改为 **1,580 个命名部件**；2026-11-12 先加入 940 件架构参考，随后按用户两张图改为 852 件的“相干一号：Feynman”超导计算机参考模型并置顶，替换画廊里的夹持案例，保留其历史文件。现为 11 个参考模型（共 2,432 件）与 1 个传动 CAD，共 12 例。

| 示例 | 部件数 | 主要细化内容 |
|---|---:|---|
| 相干一号：Feynman | 852 | 图1机架与分节银色桶；图2金色冷盘、编织热连接、48 路取样盘绕同轴与扇出；[实机真实性尚未核验](superconducting-reference.md) |
| 航空发动机 | 234 | 逐片风扇/转子/静子叶片、轮盘、同轴双轴、火焰筒、喷嘴、轴承与附件 |
| 飞机 | 182 | 分段蒙皮、隔框、长桁、地板梁、翼梁翼肋、襟副翼、起落架轮端 |
| 豪华轮船 | 172 | 船壳、肋骨、水密舱壁、薄甲板、柴油发电、电力推进轴系、救生艇及吊架 |
| Tesla Model Y | 147 | 门盖与内饰、五座、密封电池包、驱动单元、四轮制动、连杆、减振与弹簧 |
| iPhone | 54 | 16 Pro 维修总成、TrueDepth、Taptic Engine、上下扬声器、USB-C、压板与紧固件 |
| Tesla Optimus | 242 | 骨架、电源、执行器、双手逐指指骨/销轴、足底结构；内部构型明确推定 |
| Falcon 9 | 129 | 逐台 Merlin、MVac、LOX/RP-1 储箱、输送管、级间、四支腿、四栅格舵、整流罩 |
| 劳力士外形/3235 功能参考 | 131 | 发条盒、走时轮系、擒纵叉、摆轮、游丝、摆陀、日历、密封和逐节表链 |
| 炼油厂 | 170 | 塔盘、降液板、换热管束/折流板、储罐、离心泵机械密封、阀体/阀杆与管段 |
| 六轴工业机械臂 | 119 | 六套电机、转子、编码器、制动、刚轮/柔轮/波发生器、输出轴承、法兰与夹爪 |

每个叶节点对应部件类别或密封采购模组。界面图案、表盘刻字、表圈纹路不计作独立零件。重复叶片、链节和塔盘是有名称的实体实例，不把分散碎块藏在一个“总成”网格内。每件检查实质正体积连通实体；空腔负体积边界不误判为碎片。

## 来源与准确性边界

原 10 例资料核对日期：2026-11-11；超导计算机：2026-11-12。技术几何独立编写；Model Y 对照照片按下述 CC BY-SA 4.0 授权保留署名，其余只链接来源，没有复制第三方 CAD。详细引用、支持范围、版本与推定范围保存在每例的 `references.json`，源码清单为 `src/qamg/engineering_references.py`。

- 手机：[Apple iPhone 16 Pro Repair Manual](https://support.apple.com/en-us/120803)。维修项目支持模组类别；主板轮廓、紧固件布置及比例为近似，不等于 Apple 官方订购清单。显示模组不拆层压结构，相机不虚构内部镜片。
- 汽车：[Tesla 2020–2024 Model Y Service Manual](https://service.tesla.com/docs/ModelY/ServiceManual/en-us/index.html)。采用双电机 AWD 与五座参考，未绑定 VIN/地区，电池包不虚构内部电芯，电机内部形状仍为通用构型。
- Model Y 外形另参考 [官方经典款尺寸](https://www.tesla.com/ownersmanual/2020_2024_modely/en_us/GUID-E47C4A6D-528E-419C-8C57-FD3864644C34.html)，按 1:25 展示长宽高、轴距及轮距。弧形车顶、溜背与门窗使用同一连续曲面分区，具体曲面依然是近似。示例卡片采用 [Elise240SX 的 2023 款 AWD 实车照片](https://commons.wikimedia.org/wiki/File:2023_Tesla_Model_Y_Long_Range_All-Wheel_Drive_in_Pearl_White_Multi-Coat,_front_right,_2024-09-25.jpg)，裁剪作者 LuvsMG481，CC BY-SA 4.0；本地为浏览器显示图的 PNG 截取，保留该授权并在页面署名。照片与当前三维渲染并排标注，照片未进入重建流程。
- 火箭：[SpaceX Falcon User’s Guide, May 2025, §2.3–2.5](https://www.spacex.com/assets/media/falcon-users-guide-2025-05-09.pdf)。九台一级发动机、单台二级发动机、四腿与四舵采用公开数量。其余内部件的形状和位置为近似；占位卫星明确标成示意载荷。
- 航空：[NASA 双转子涡扇说明](https://www.grc.nasa.gov/www/k-12/airplane/turbfan.html) 与 [FAA Airframe handbook, Chapter 1](https://www.faa.gov/documentlibrary/media/advisory_circular/ac_65-15a.pdf)。通用双转子发动机和半硬壳飞机；不是指定发动机/客机型号，叶片数、级数和机身框站为代表性取样。
- 船舶：[Wärtsilä 柴油电力推进](https://www.wartsila.com/encyclopedia/term/diesel-electric-propulsion)。支持功能架构；不是特定邮轮总布置图，舱室、发电机和肋骨数量为取样。
- 机械表：[Rolex 机芯技术说明](https://newsroom.rolex.com/watchmaking/features/movements) 与 [Datejust 41 / 3235](https://newsroom.rolex.com/watches/new-watches/datejust-41)。支持自动上链、擒纵、游丝和日历功能；齿数、轮系坐标、桥板轮廓与表链节数并非完整 3235 厂图。
- 机器人：[Tesla Optimus](https://www.tesla.com/SI) 只支持公开的系统范围。[Harmonic Drive 减速器工作原理](https://legacy.harmonicdrive.net/reference/applicationnotes/principles.php) 支持通用刚轮、柔轮、波发生器分解。**不证明 Optimus 实机采用这些关节内部设计**；其内部与工业机械臂各轴均为教学构型。
- 炼油设施：[U.S. EIA 炼油工艺说明](https://www.eia.gov/kids/energy-sources/oil/)。覆盖常压蒸馏、换热和储运单元；设备内件使用通用工程设计，非整厂 P&ID 或压力设计。

这些资料支持部件类别与功能关系，不能据此声称获得了完整真实 BOM、原厂尺寸、全部管线、连接公差、制造可行性或实物维修指导。真正按具体实物一比一建模还需要对应型号图纸、拆机测量或授权 CAD。

## 页面操作与下载

- 三级展开：总成保持子件相对位置，子总成再分开，零件级展示每个部件。
- 零件树可折叠、搜索和整组选取。“查看”只显示该总成，便于放大内部件；“只看选中部件”可恢复全部。
- “内部透视”只改变外壳的显示透明度，不修改几何。选一件可查看所属总成、材料说明与推定标识。
- `bom.csv` / `bom.json` 是**当前展示模型**的逐件清单，不是完整工厂 BOM。`parts.zip` 包含 PLY、最终层级、清单与引用。重命名/合并后这些元数据同步更新，原例不修改。
- 保存布尔合并、名称修改与图片生成需要连接本机服务；公开 Pages 可浏览、分级展开、透视、选择和下载。

## 复现与验证

```bash
uv run --locked python scripts/generate_showcases.py
uv run --locked python scripts/render_gallery_thumbnails.py
uv run --locked python scripts/export_gallery.py
npm --prefix web run build
./scripts/test.sh
uv run --locked python scripts/validate_showcase_details.py --output artifacts/runs/detail-audit.json
```

模型位于 `artifacts/examples/<slug>/`。缩略图直接对导出 GLB 使用 CPU 深度缓冲渲染。每例保留具名实体、实体有效性、并集/对称差检查、QB 原生电路计数及选中候选见证。导出前使用最大表面位移 0.0001 mm 的有界简化消除布尔薄片；验证脚本独立回读全部 PLY，检查水密、实体连通、体积、ZIP 清单、层级、实际测量计数与公开文件哈希。引用资料与布尔几何检查分别记录，几何通过不等于结构经原厂核验。

QB 仍选择横向/竖向/纵向的展开比例，各层使用同一被测参数。形状和部件清单由参考建模代码定义；不声称量子测量生成网格或实现物理量子加速。离散平移序列仍单独报告完整/部分完成，绝不将三级示意展开冒充连续无碰撞拆卸、螺纹旋出、胶粘分离或工具空间认证。
