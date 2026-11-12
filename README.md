# Q-AMG · Quantum-assisted Mechanical Generation

量子辅助机械生成、拆解与装配系统，量子部分使用公司开源 [QuantumBridge](https://github.com/crystal-tensor/QuantumBridge) 原生 SDK。

[在线页面](https://crystal-tensor.github.io/Q-AMG/) · [架构](docs/architecture.md) · [部件与识图修正](docs/part-aware-delivery.md) · [历史 QB 迁移验收](docs/qb-image-delivery.md) · [超导接入](docs/company-protocol.md)

图片路径使用 SAM2.1 large 提取主体，PartCrafter 联合生成指定数量的独立部件，再由 QB 测量选择装配展开方向。支持逐件查看、命名、展开、重新装配、布尔合并及恢复原始分件。传动装配和夹持机构保留参数化 CAD 与 STEP 导出。

部件数量是用户给定的约束。单张照片无法唯一确定背面、内部、真实尺寸和制造连接；生成部件需要与原图逐件核对。网格有效、并集一致和数量正确均不等于恢复了真实机械 BOM。

## 启动

需要 Python 3.12、uv、Node.js 24、npm 与 curl。

```bash
./scripts/demo.sh
```

打开 [本机应用](http://127.0.0.1:8765/)。首次准备约 4.9 GB 的公开模型文件，支持分段下载、重试与 SHA-256 校验。权重不进入 Git。后续推理在本机运行，上传图片不发送到外部模型 API。

默认优先使用 CUDA / Apple Metal GPU；无 GPU 使用 CPU。部件生成的标准设置为 50 步、每部件 1024 tokens；实际耗时取决于部件数和设备。图像编码使用单精度，避免本机已实测的 MPS 半精度特征偏差。

只使用参数化机械功能时可跳过图片模型：

```bash
QAMG_SKIP_IMAGE_MODEL=1 ./scripts/demo.sh
```

## 图片到部件

1. 上传 PNG、JPEG、WEBP、BMP 或 TIFF，指定部件数量与参考标尺。
2. 模型联合生成各个部件，不使用平面切块来凑数；空输出和明显碎片会报告失败。
3. 逐件选择，使用“只看选中部件”核对形状，并保存名称。
4. 使用示意爆炸、离散碰撞采样序列、重新装配和选件合并。
5. 下载 GLB、OBJ、STL、结构 JSON 和分件 ZIP。

生成失败会清空当前结果，避免把先前模型误认为本次输入的结果。原始图像、预处理、原始神经输出、失败、测量记录及修改来源均保留。命名和合并产生新运行，原运行可恢复。

当前神经部件使用区分部件的配色，未恢复照片纹理。离散碰撞采样不能认证连续路径、工具空间或真实连接。图片网格不伪造 STEP 特征历史。

透明杯等困难输入可显式选择“杯类结构辅助重建”，限定盖子、杯子、杯套三个角色：从 SAM 轮廓估计杯身锥度、杯套比例和区域颜色，结合神经外形标尺生成连续空心杯体、独立环形杯套与连通盖子近似件。QB 测量选择界面、壁厚/间隙候选及展开方向。这个模式使用旋转体结构先验；盖子提片、不可见内壁、密封与配合均不能当成照片恢复的制造真值。原神经结果与辅助重建分别保存。

```bash
uv run --locked qamg image --input /absolute/path/image.png --parts 3 --output artifacts/runs/my-image
uv run --locked qamg image --input /absolute/path/image.png --parts 3 --device cpu --backend qb --steps 30 --tokens 512 --output artifacts/runs/cpu-image
uv run --locked qamg generate --spec examples/drive.json --encoding compact_binary --output artifacts/runs/my-drive
```

较少步数和 tokens 可用于预览，仍需核对输出质量。`--resolution` 控制表面采样，无法恢复照片里不可见的工程信息。

## QB 的作用

QB 原生电路、编译、采样、噪声模拟与序列化参与结构规划。图片路径保留神经部件边界，测量结果决定导出的展开方向；合并后若使用经典布局，会明确记录其来源。

- `auto`：QB 电路 / 编译与 Torch CUDA/MPS 模拟，无 GPU 使用 QB 原生 CPU。
- `qb`：QuantumBridge 原生 StatevectorDevice。
- `qb-noisy`：原生 DensityMatrixDevice 和参考噪声，明确为模拟。
- `company`：显式配置公司的超导服务与预算后提交，见接入文档。
- `classical`：同一原问题的经典对照。

应用不通过 Qiskit、Aer 或 PennyLane 执行。测试阻断这些 SDK 后执行 QB 原生链路。

当前小规模方向子问题容易经典求解。真实量子加速与公司 QPU 尚未验证，不能把本机模拟结果当作量子加速证明。

## 验证与复现

```bash
./scripts/test.sh
uv run --locked python scripts/validate_vision_precision.py --output artifacts/new-vision-check
./scripts/benchmark.sh
```

原生采样调度器必须与权重配置一致，测试核对其更新方向及配置；另核对引导分支中的部件编号一致性、QB 采样来源、几何布尔操作和接口行为。真实神经输出与测试里的接口 stub 分别记录。

历史 TripoSR 几何切片、旧 SDK 实验、原始记录和失败保持原样。它们不代表当前部件识别质量或真实 QPU 的新测量。需要复现旧图片路径时，先运行 `scripts/prepare_models.py`，再显式选择 `--image-model triposr-geometric`。

模型版本、许可与适配见 [vendor/SOURCES.json](vendor/SOURCES.json)，下载来源和校验值见 `configs/partcrafter-files.json`、`configs/vision-model.json` 与 `models/manifest.json`。

## 公开页面

页头的 **EXAMPLES / REPRODUCIBLE** 收录 11 个细化装配参考与 1 个传动参数化 CAD 示例。11 例共 **2,520 个命名部件**，按“总成—子总成—零件”三级组织，可搜索、整组选取、单独查看子总成、透视外壳及下载零件清单。首例为 **“相干一号：费曼”超导量子计算机**：940 件参考建模，参照 Google / IBM 公开架构与低温工程论文，**尚未完成 500 件实物真实性核验**，详见 [逐件证据与范围](docs/superconducting-reference.md)。原夹持 CAD 的历史文件保留。发动机叶片、机翼翼肋、车轮制动与悬架、逐指指节、机芯轮系、塔器内件等分别建模，详见 [示例来源与复现](docs/showcase-examples.md)。

部件类别参考 Apple / Tesla 维修手册、SpaceX 用户指南、FAA / NASA 与设备厂商技术资料；形状和比例仍为独立编写的工程近似，**不是完整官方 CAD、工厂 BOM 或照片重建**。Optimus 未公开的内部构型明确标为推定，电池包与密封相机不虚构到电芯/内部镜片。页面和 `references.json` 保留每例采用的资料与覆盖边界。

GitHub Pages 展示可交互、可下载的公开示例。生成新图片、保存合并或名称需要运行本机服务，并在页面点击“连接生成服务”。浏览器可能要求允许访问本机网络。

公开资源由 `scripts/export_gallery.py` 的显式清单生成；私有图片和失败调试位于忽略的 `artifacts/runs/`，不打包进公开页面。源码与示例可公开复现，模型权重按固定版本另行下载。
