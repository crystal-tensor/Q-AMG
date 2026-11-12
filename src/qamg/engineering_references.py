"""Primary references for component classes, with explicit coverage boundaries.

No drawings, photographs or third-party CAD are redistributed. Reference facts
guide an independently authored, dimensionally approximate educational model.
"""

SOURCES = {
    "krinner-cryogenic-2019": {
        "title": "Krinner et al. · Engineering cryogenic setups for 100-qubit scale superconducting circuit systems (2019)",
        "url": "https://link.springer.com/article/10.1140/epjqt/s40507-019-0072-0",
        "supports": "§2–3 / Fig.3–6 的冷级、射频链路、铜夹块/楔块/M3 螺钉、衰减器、滤波器与放大器；非 Google/IBM 整机 BOM 或尺寸图",
    },
    "ibm-cryostat": {
        "title": "IBM Quantum · Hardware / Inside a quantum cryostat",
        "url": "https://www.ibm.com/quantum/hardware",
        "supports": "真空馈通、脉冲管、HEMT、超导同轴、磁屏蔽、量子极限放大器和 QPU 等器件类别；不证明本模型数量、坐标与尺寸",
    },
    "google-superconducting-hardware": {
        "title": "Google Quantum AI · Our quantum computer",
        "url": "https://quantumai.google/quantumcomputer",
        "supports": "超导处理器、低温系统、控制电子学与软件的系统架构；不提供同一台整机的生产 BOM",
    },
    "google-microwave-engineering": {
        "title": "Bardin et al. · Quantum Computing: An Introduction for Microwave Engineers (2020)",
        "url": "https://research.google/pubs/quantum-computing-an-introduction-for-microwave-engineers/",
        "supports": "超导量子计算的微波控制、读出与工程器件类别；不支持本模型机柜模块数量和外形尺寸",
    },
    "nasa-turbofan": {
        "title": "NASA Glenn · Turbofan engines",
        "url": "https://www.grc.nasa.gov/www/k-12/airplane/turbfan.html",
        "supports": "双转子、风扇、压气机、燃烧室、涡轮、同轴轴系与旁通气路的部件类别",
    },
    "faa-airframe": {
        "title": "FAA · Airframe handbook, AC 65-15A, Chapter 1",
        "url": "https://www.faa.gov/documentlibrary/media/advisory_circular/ac_65-15a.pdf",
        "supports": "半硬壳机身、隔框、纵向构件、翼梁、翼肋与操纵面的通用结构",
    },
    "wartsila-propulsion": {
        "title": "Wärtsilä · Diesel-electric propulsion",
        "url": "https://www.wartsila.com/encyclopedia/term/diesel-electric-propulsion",
        "supports": "邮轮柴油发电、配电与电力推进的功能总成；不支持特定邮轮布置或设备数量",
    },
    "tesla-model-y": {
        "title": "Tesla · 2020–2024 Model Y Service Manual",
        "url": "https://service.tesla.com/docs/ModelY/ServiceManual/en-us/index.html",
        "supports": "车身、门盖、座椅、悬架、制动、电池包与驱动单元的维修部件类别",
    },
    "tesla-model-y-dimensions": {
        "title": "Tesla · 2020–2024 Model Y Owner’s Manual / Dimensions",
        "url": "https://www.tesla.com/ownersmanual/2020_2024_modely/en_us/GUID-E47C4A6D-528E-419C-8C57-FD3864644C34.html",
        "supports": "经典款车长 4751、车宽 1921、车高 1624、轴距 2890、19/20 寸轮距 1636 mm；车型配置仍有差异",
    },
    "apple-16-pro": {
        "title": "Apple · iPhone 16 Pro Repair Manual",
        "url": "https://support.apple.com/en-us/120803",
        "supports": "显示屏、后玻璃、电池、相机、TrueDepth、上下扬声器、Taptic Engine、麦克风与 USB-C 等维修部件",
    },
    "tesla-optimus": {
        "title": "Tesla · Optimus / Robotics",
        "url": "https://www.tesla.com/SI",
        "supports": "双足人形机器人及视觉、感知与执行器的系统范围；未公开完整机械 BOM",
    },
    "spacex-falcon": {
        "title": "SpaceX · Falcon User’s Guide, May 2025, §2.3–2.5",
        "url": "https://www.spacex.com/assets/media/falcon-users-guide-2025-05-09.pdf",
        "supports": "一级九台 Merlin、二级单台 MVac、两种推进剂储箱、级间段、四支腿、四栅格舵及载荷整流罩",
    },
    "rolex-movements": {
        "title": "Rolex · Technical features / Movements",
        "url": "https://newsroom.rolex.com/watchmaking/features/movements",
        "supports": "自动摆陀、摆轮、游丝、擒纵与避震系统的功能部件",
    },
    "rolex-3235": {
        "title": "Rolex · Datejust 41 / Calibre 3235",
        "url": "https://newsroom.rolex.com/watches/new-watches/datejust-41",
        "supports": "3235 型自动上链、发条盒、Chronergy 擒纵、Parachrom 游丝与日历功能；非完整零件图",
    },
    "eia-refining": {
        "title": "U.S. EIA · Refining process",
        "url": "https://www.eia.gov/kids/energy-sources/oil/",
        "supports": "常压蒸馏、加热、换热、转化处理与储运的通用工艺设备类别",
    },
    "harmonic-drive": {
        "title": "Harmonic Drive · Detailed operating principle",
        "url": "https://legacy.harmonicdrive.net/reference/applicationnotes/principles.php",
        "supports": "减速器的刚轮、柔轮、波发生器；通用关节参考，不证明特定品牌使用该内部构型",
    },
}

PROFILES = {
    "superconducting-computer": {
        "variant": "相干一号：费曼 · Google / IBM 架构参考设计",
        "sources": ["google-superconducting-hardware", "ibm-cryostat", "google-microwave-engineering", "krinner-cryogenic-2019"],
        "coverage": "覆盖五级冷盘、真空与热屏、脉冲管/稀释回路、65 路参考射频线缆及热锚、衰减/滤波/放大链、QPU 封装和室温测控。",
        "inferred": "名称由用户指定，非 Google / IBM 产品。65 路取自 Krinner Fig.3 的 25 驱动、25 磁通、6 读入、5 泵浦、4 输出配置；与文中另一测量语境的 66 路不混用。夹块数量、走线、机械尺寸与位置为独立参考设计，非两家公司实机复刻。1:10 为展示比例，原型外形尺寸也未实测；密封模块不虚构内部零件，芯片图案不计件。未取得同一台设备 BOM/CAD/实机核对记录，500 件完全真实的要求尚未核实。",
        "display_scale_denominator": 10,
        "default_xray": True,
        "hardware_truth_required": True,
        "reviewed_on": "2026-11-12",
    },
    "aero-engine": {
        "variant": "通用双转子高涵道比涡扇发动机",
        "sources": ["nasa-turbofan"],
        "coverage": "覆盖风扇、双轴、压气机转静子、燃烧室、涡轮、支承及附件；级数与叶片数量为展示取样。",
        "inferred": "无指定发动机型号；气动型线、冷却孔、材料牌号、装配间隙与每级叶片数未按实机复刻。",
    },
    "airliner": {
        "variant": "通用双发窄体客机 · 半硬壳机身",
        "sources": ["faa-airframe", "nasa-turbofan"],
        "coverage": "覆盖机身隔框、蒙皮、地板梁、翼梁、翼肋、襟翼、副翼、尾翼、起落架与发动机。",
        "inferred": "非某一 Airbus/Boeing 机型；框站、铆钉、客舱座椅与管线仅为代表性布置。",
    },
    "luxury-liner": {
        "variant": "通用柴油电力推进邮轮",
        "sources": ["wartsila-propulsion"],
        "coverage": "覆盖船壳、肋骨、舱壁、甲板、柴油发电机、电机、轴系、螺旋桨、侧推与救生艇。",
        "inferred": "不对应特定船厂总布置图；房间、发电机、肋骨与救生艇数量为展示取样。",
    },
    "model-y": {
        "variant": "Model Y · 2020–2024 双电机 AWD 维修部件参考",
        "sources": ["tesla-model-y", "tesla-model-y-dimensions"],
        "coverage": "覆盖车身、四门、前后盖、五座、电池包、双驱动单元、四轮制动及悬架。电池包不虚构到电芯。",
        "inferred": "外形为经典款，并以官方外廓和轴距按 1:25 制作展示模型；曲面、焊接边界、线束与驱动内部为近似，未绑定 VIN。实车照片仅作对照，不声称照片重建。",
        "reference_dimensions_mm": {"length": 4751, "width_without_mirrors": 1921, "height": 1624,
                                    "wheelbase": 2890, "track": 1636},
        "display_scale_denominator": 25,
    },
    "iphone": {
        "variant": "iPhone 16 Pro · 维修总成与代表性紧固件",
        "sources": ["apple-16-pro"],
        "coverage": "按官方维修项目组织屏幕、后玻璃、电池、相机、TrueDepth、音频、Taptic Engine 与 USB-C。显示模组不把界面图案计作零件。",
        "inferred": "主板及小件轮廓、螺钉布置和尺寸为近似，不等于 Apple 订购清单；不展开密封相机的内部镜片。",
    },
    "optimus": {
        "variant": "Optimus 外形 / 通用机电结构推定",
        "sources": ["tesla-optimus", "harmonic-drive"],
        "coverage": "覆盖外壳、骨架、电源与控制、肩肘腕髋膝踝执行器、双手逐指指节、足部传感与传动。",
        "inferred": "Tesla 未公开完整内部机械 BOM；全部内部减速器、连杆与驱动布置为通用工程推定，不能称为实机复制。",
        "internal_basis": "engineering_inference",
    },
    "falcon": {
        "variant": "Falcon 9 · 两级运载器公开架构参考",
        "sources": ["spacex-falcon"],
        "coverage": "九台一级发动机逐台拆分，另含二级发动机、LOX/RP-1 储箱、输送管、级间结构、四支腿、四栅格舵及双瓣整流罩。",
        "inferred": "仅九台一级发动机、一台二级发动机、四支腿、四栅格舵为公开数量；发动机内部形状与内部管路为结构近似。",
        "verified_counts": {"first_stage_engine": 9, "second_stage_engine": 1, "landing_leg": 4, "grid_fin": 4},
    },
    "mechanical-watch": {
        "variant": "Datejust / 3235 功能结构参考",
        "sources": ["rolex-movements", "rolex-3235"],
        "coverage": "覆盖壳镜、密封圈、三针、日历环、发条盒、轮系、擒纵叉、摆轮、游丝、摆陀、夹板、宝石及逐节表链。",
        "inferred": "通用自动机械机芯的拓扑示意；齿数、轴位、夹板轮廓与表链节数非 Rolex 厂图或完整 3235 BOM。",
    },
    "refinery": {
        "variant": "通用炼油厂 · 常压蒸馏与换热储运单元",
        "sources": ["eia-refining"],
        "coverage": "按装置—设备—内件展开蒸馏塔、塔盘、降液管、换热器管束/折流板、储罐、离心泵、阀门与管廊。",
        "inferred": "设备内件采用通用工程构型；非整座工厂的 P&ID、真实压力设计或全部设备、法兰、焊缝清单。",
    },
    "industrial-arm": {
        "variant": "通用六轴工业机器人 · 关节减速器剖分",
        "sources": ["harmonic-drive"],
        "coverage": "六关节分别含电机、转子、制动、编码器、减速器、轴承与输出法兰，另有连杆、线缆通道和夹爪传动。",
        "inferred": "无指定机器人品牌；六关节采用统一的教学减速器构型，不代表任何厂商实机结构。",
    },
}


def reference_bundle(slug):
    profile = PROFILES[slug]
    return {**profile, "sources": [{"id": key, **SOURCES[key]} for key in profile["sources"]],
            "reviewed_on": profile.get("reviewed_on", "2026-11-11"), "geometry_origin": "independently_authored_parametric_geometry",
            "oem_cad": False, "complete_factory_bom": False, "dimensions_measured": False}
