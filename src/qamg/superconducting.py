"""相干一号：费曼 — source-linked superconducting-system reference design.

Google/IBM support the architecture, not a manufacturer BOM for this new name.
The 65 RF channels follow Krinner et al. Fig. 3 / §3.1's illustrated installed
configuration (25 drive, 25 flux, 6 read-in, 5 pump, 4 output); the paper also
reports 66 lines in another measurement context. We do not turn that difference
or the 124-line capacity estimate into extra physical parts.

Routing, mechanical dimensions, terminations and mounting quantities below are
an authored reference layout. Every instance carries that qualification. No
wire is sliced at a thermal plate merely to increase the part count. Qubit
islands, printed labels and signal paths are not independent physical parts.
"""
from __future__ import annotations

import numpy as np
import trimesh

from .mesh_parts import as_manifold
from .showcase import (
    BLUE,
    DARK,
    GOLD,
    SILVER,
    TEAL,
    WHITE,
    Model,
    beam,
    box,
    cyl,
    rounded_rect,
    tube,
    turn,
    union,
)

COPPER = [177, 113, 64]
BRASS = [197, 161, 78]
STEEL = [143, 158, 168]
PTFE = [223, 227, 220]
WIRE = [174, 139, 79]
PAPER = "krinner-cryogenic-2019"
IBM = "ibm-cryostat"
GOOGLE = "google-superconducting-hardware"
MICROWAVE = "google-microwave-engineering"
SCALE = .1
STAGES = [("RT", 154., 43.), ("50 K", 126., 39.), ("4 K", 100., 36.),
          ("Still", 75., 32.), ("CP", 50., 29.), ("MXC", 25., 26.)]


def evidence(instance, kind, locator, part_number=None, sources=None, quantity_basis="authored_reference_layout"):
    return {"instance_id": instance, "component_type": kind, "source_refs": sources or [PAPER],
            "source_locator": locator, "reference_part_number": part_number,
            "quantity_basis": quantity_basis, "dimensions_basis": "authored_nominal_display_envelope",
            "position_basis": "authored_reference_layout", "as_built_verified": False,
            "missing_evidence": ["original_as_built_bom", "dimensioned_cad", "installed_pose", "physical_inspection"]}


def curved_cable(points, radius=.10795, sections=12):
    """One continuous cable envelope across thermal anchors (not five fake cuts)."""
    points = np.asarray(points, dtype=float)
    vectors = np.diff(points, axis=0)
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    tangent = np.vstack([vectors[0], vectors[:-1]+vectors[1:], vectors[-1]])
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    angles = np.arange(sections)*2*np.pi/sections
    circle = np.column_stack([radius*np.cos(angles), radius*np.sin(angles), np.zeros(sections)])
    vertices = [circle@trimesh.geometry.align_vectors([0, 0, 1], axis)[:3, :3].T+p
                for p, axis in zip(points, tangent, strict=True)]
    vertices = np.concatenate([*vertices, [points[0], points[-1]]])
    faces = []
    for row in range(len(points)-1):
        for j in range(sections):
            a, b = row*sections+j, row*sections+(j+1) % sections
            faces.extend([[a, b, b+sections], [a, b+sections, a+sections]])
    for j in range(sections):
        a, b = j, (j+1) % sections
        faces.extend([[len(vertices)-2, b, a], [len(vertices)-1, (len(points)-1)*sections+a,
                                               (len(points)-1)*sections+b]])
    mesh = trimesh.Trimesh(vertices, faces, process=True)
    mesh.fix_normals()
    return as_manifold(mesh)


def coax_assembly(points, radius=.10795):
    """Cable and its sealed terminations count as one purchased assembly."""
    pieces = [curved_cable(points, radius)]
    for p, other in [(points[0], points[1]), (points[-1], points[-2])]:
        direction = np.asarray(other)-np.asarray(p)
        pieces.append(cyl(.29, .72, p, direction, sections=6))
        pieces.append(cyl(.22, .90, p, direction, sections=12))
    return union(*pieces)


def superconducting_computer():
    m = Model()
    from .engineering_models import bolt, pipe

    def part(label, solid, color, group, instance, kind, locator, offset=(0, 0, 0),
             part_number=None, sources=None, role="component", appearance=None, quantity_basis="authored_reference_layout"):
        if kind == "mounting_screw":
            role = "fastener"
        m.part(label, solid, color, group, offset, role=role, material={
            "copper_plate": "无氧铜 · 表面处理近似", "thermal_clamp": "无氧铜", "mounting_screw": "黄铜 M3 · 长度待核",
            "rf_cable": "UT-085 类半刚同轴线 · 复合外形包络", "superconducting_coax": "NbTi 超导同轴线 · 外形包络",
            "shield": "金属屏蔽 · 透视仅为显示方式"}.get(kind, "公开器件类别 · 材料/尺寸待原厂核对"),
            basis="source_linked_reference_design", appearance=appearance,
            evidence=evidence(instance, kind, locator, part_number, sources, quantity_basis))

    root = m.group("低温机架与真空", [0, 0, 15])
    g = m.group(root+" / 顶部与机架", [0, 0, 45])
    part("300 K 真空法兰", tube(43, 4.8, 3, (0, 0, 154)), SILVER, g, "VAC-TOP", "vacuum_flange", "IBM cryostat / Hermetic feedthrough", [0, 0, 20], sources=[IBM])
    part("设备承载底座", rounded_rect(115, 105, 6, 4, -8), DARK, g, "FRAME-BASE", "frame", "IBM cryostat / architecture", [0, 0, -25], sources=[IBM])
    for i, a in enumerate(np.linspace(0, 2*np.pi, 4, endpoint=False)):
        x, y = 54*np.cos(a+np.pi/4), 54*np.sin(a+np.pi/4)
        part(f"主承载立柱 {i+1}", cyl(1.6, 164, (x, y, 73), sections=20), STEEL, g,
             f"FRAME-POST-{i+1}", "support_post", "IBM cryostat / mounting frame", [x*.6, y*.6, 0], sources=[IBM])
    g = m.group(root+" / 可拆真空外罩", [95, 0, 0])
    proxy = {"type": "transparent_display_proxy", "rgba": [157, 178, 184, 28]}
    part("真空筒体 · 金属透视显示", tube(46.5, 46.1, 161, (0, 0, 72.5)), SILVER, g,
         "VAC-CAN", "vacuum_enclosure", "IBM cryostat / vacuum enclosure", [18, 0, 0], sources=[IBM], role="cover", appearance=proxy)
    part("真空底盖", cyl(46.5, 1.6, (0, 0, -8)), STEEL, g, "VAC-LID", "vacuum_lid", "IBM cryostat / enclosure", [20, 0, -25], sources=[IBM], role="cover", appearance=proxy)
    part("真空口法兰", tube(3.4, 2.3, 6, (44, 0, 143), [1, 0, 0]), SILVER, g, "VAC-PORT", "vacuum_port", "IBM cryostat / hermetic feedthrough", [35, 0, 0], sources=[IBM])

    root = m.group("冷级与热屏", [0, 0, -15])
    for i, (stage, z, radius) in enumerate(STAGES[1:]):
        g = m.group(root+f" / {stage} 冷级", [0, 0, 35-18*i])
        plate = tube(radius, 3.2, 1.7, (0, 0, z))
        # 88 mm LOS slot width from Krinner §3.1, at 1:10 display scale.
        for angle in [30, 90, 150, 210, 270, 330]:
            cutout = turn(box([10, 8.8, 4], (radius*.79, 0, z)), angle)
            plate = plate-cutout
        part(stage+" 冷级安装盘", plate, GOLD, g, f"STAGE-{i}-PLATE", "copper_plate", "Krinner §3.1 / Fig.3–4", [0, 0, i*3])
        upper = STAGES[i][1]
        for j, a in enumerate(np.deg2rad([30, 150, 270])):
            r = radius*.58
            part(f"{stage} 隔热支柱 {j+1}", tube(.52, .31, upper-z-2, (r*np.cos(a), r*np.sin(a), (upper+z)/2),), STEEL, g,
                 f"STAGE-{i}-SUPPORT-{j}", "support_post", "Krinner §2.1 / posts; geometry assumed", [9*np.cos(a), 9*np.sin(a), 0])
        if stage != "CP":
            height = z-4
            r = radius+1.2
            shade = tube(r, r-.32, height, (0, 0, (z+4)/2))
            part(stage+" 热辐射屏筒体", shade, COPPER if i > 1 else SILVER, g, f"SHIELD-{stage}", "shield", "Krinner §2.3", [0, 30+i*10, 0], role="cover", appearance=proxy)
            part(stage+" 热屏底盖", cyl(r, .4, (0, 0, 3.6)), COPPER if i > 1 else SILVER, g,
                 f"SHIELD-LID-{stage}", "shield", "Krinner §2.3", [0, 30+i*10, -15-i*5], role="cover", appearance=proxy)

    root = m.group("制冷与循环回路", [-65, 0, 0])
    for i, x in enumerate([-13, 13]):
        g = m.group(root+f" / 脉冲管冷头 {i+1}", [-60, (i*2-1)*22, 30])
        part(f"PT420 冷头 {i+1} 一级", cyl(3.7, 27, (x, 0, 140), sections=24), SILVER, g,
             f"PT-{i}-STAGE1", "pulse_tube", "Krinner §2.3: two Cryomech PT420", [0, 0, 16], "Cryomech PT420")
        part(f"PT420 冷头 {i+1} 二级", cyl(2.0, 27, (x, 0, 113), sections=24), STEEL, g,
             f"PT-{i}-STAGE2", "pulse_tube", "Krinner §2.3", [0, 0, -12], "Cryomech PT420")
        part(f"冷头 {i+1} 热链接", union(beam([x, 0, 101], [x, 7, 101], 1.0), beam([x, 7, 101], [x+5, 7, 101], 1.0)), COPPER, g,
             f"PT-{i}-LINK", "thermal_link", "IBM cryostat / pulse tube cooler", [0, 18, 0], sources=[IBM])
    g = m.group(root+" / 稀释单元", [-70, 0, -20])
    part("Still 蒸馏室", tube(4.7, 4.0, 9, (0, 0, 72)), COPPER, g, "DR-STILL", "still", "Krinner §3.1 / dilution unit", [0, 0, 20])
    part("混合室", tube(6.5, 5.7, 8, (0, 0, 28)), COPPER, g, "DR-MXC", "mixing_chamber", "Krinner §3.1 / center of MXC plate", [0, 0, -20])
    for i, z in enumerate([61, 49, 37]):
        # Continuous helical heat exchanger tube, one component per coil.
        a = np.linspace(0, 5*np.pi, 65)
        pts = np.column_stack([4*np.cos(a), 4*np.sin(a), z+4*a/a[-1]])
        part(f"换热器盘管 {i+1}", curved_cable(pts, .24), COPPER, g, f"DR-HX-{i}", "heat_exchanger",
             "Dilution-unit architecture; coil geometry/quantity are authored", [12+i*8, 0, -i*5])
    part("稀释回路供回气管", pipe([[0, 0, 151], [7, 0, 146], [7, 0, 74], [0, 0, 74]], .6, .15), STEEL, g,
         "DR-FEED", "helium_pipe", "Krinner §3.1 / dilution unit", [0, -20, 0])

    families = [("D", "量子位微波驱动", 25, 30), ("F", "磁通偏置", 25, 150),
                ("RI", "谐振腔读入", 6, 270), ("P", "TWPA 泵浦", 5, 270), ("RO", "读出输出", 4, 270)]
    for code, label, count, degrees in families:
        root = m.group(label, [55*np.cos(np.deg2rad(degrees)), 55*np.sin(np.deg2rad(degrees)), 0])
        angle = np.deg2rad(degrees)
        radial = np.array([np.cos(angle), np.sin(angle), 0])
        tangent = np.array([-np.sin(angle), np.cos(angle), 0])
        for channel in range(count):
            slot = channel+(6 if code == "P" else 11 if code == "RO" else 0)
            across, depth = (slot % 5-2)*1.45, (slot//5-2)*1.45
            nominal = (radial*50+tangent*across*2).tolist()
            g = m.group(root+f" / {code}-{channel+1:02d}", nominal)
            def point(index, dz=0, radial=radial, depth=depth, tangent=tangent, across=across):
                _, z, radius = STAGES[index]
                return radial*(radius*.79+depth)+tangent*across+[0, 0, z+dz]
            def put(name, solid, color, suffix, kind, locator, offset=(0, 0, 0), pn=None, quantity_basis="authored_reference_layout", code=code, channel=channel, g=g):
                part(f"{code}-{channel+1:02d} · {name}", solid, color, g, f"{code}-{channel+1:02d}-{suffix}",
                     kind, locator, offset, pn, quantity_basis=quantity_basis)

            # Continuous upper cable passes the 50 K thermal anchor. The second
            # cable passes Still without an arbitrary cut into mesh fragments.
            routes = [[point(0, -1.8), point(1, 1), point(2, 2.2)]]
            if code in {"D", "RI", "P"}:
                routes += [[point(2, -2.3), point(3, .8), point(4, 2.2)], [point(4, -2.3), point(5, 2.2)]]
            else:
                routes += [[point(2, -2.3), point(3, .8), point(4, .8), point(5, 2.2)]]
            for segment, points in enumerate(routes):
                superconducting = code == "RO" and segment > 0
                pn = "UT-085-NbTi" if superconducting else "UT-085-SS-SS" if code != "RO" else "UT-085-SS (warm output reference)"
                put(f"连续同轴线缆总成 {segment+1}", coax_assembly(points), SILVER if superconducting else WIRE,
                    f"CABLE-{segment}", "superconducting_coax" if superconducting else "rf_cable",
                    "Krinner §3.1–3.4 / Fig.3; segment routing is reference layout", tangent*(8+segment*5), pn)

            # Two-piece Cu adapter + wedge + M3 screw at the unattenuated 50 K
            # and Still anchors, following Fig.4(d)'s actual component classes.
            # Number of anchors/instances is our layout, not a claimed OEM BOM.
            for stage_index in [1, 3]:
                p = point(stage_index)
                h, w = .85, .84
                block = box([w, w, h])-cyl(.13, h+.2, sections=16)
                for sign, side in [(-1, "左"), (1, "右")]:
                    half = block ^ box([w/2, w+1, h+1], (sign*w/4, 0, 0))
                    put(STAGES[stage_index][0]+f" 铜热锚{side}半夹块", half.translate(p), COPPER,
                        f"ANCHOR-{stage_index}-{side}", "thermal_clamp", "Krinner Fig.4(d) / §3.1", tangent*sign*(4+stage_index))
                wedge = box([.3, .7, .8]).translate(p+radial*.6)
                put(STAGES[stage_index][0]+" 热锚压紧楔块", wedge, GOLD, f"WEDGE-{stage_index}", "thermal_wedge", "Krinner Fig.4(d)", radial*8)
                screw = bolt(.15, .65, p+radial*1.05, radial)
                put(STAGES[stage_index][0]+" M3 黄铜紧固螺钉", screw, BRASS, f"SCREW-{stage_index}", "mounting_screw", "Krinner §3.1 M3 brass screw; length unverified", radial*13, "M3 brass screw (length unspecified)")
            if code == "D":
                for stage_index in [2, 4, 5]:
                    p = point(stage_index)
                    attenuator = union(cyl(.43, 1.6, p, sections=6), cyl(.25, 2.5, p, sections=16))
                    put(STAGES[stage_index][0]+" 20 dB 低温衰减器", attenuator, BRASS,
                        f"ATT-{stage_index}", "attenuator", "Krinner §3.2 / 20 dB at 4 K, CP and MXC", radial*(10+stage_index*2), "20 dB cryogenic SMA attenuator (OEM PN unavailable)")
            if code == "F":
                p = point(2)
                put("4 K 低温衰减器 · 10 dB 参考", union(cyl(.43, 1.6, p, sections=6), cyl(.25, 2.5, p, sections=16)), BRASS,
                    "ATT-4K", "attenuator", "Krinner §3.3 / 10–20 dB; chosen reference is 10 dB", radial*14)
                put("MXC 低通滤波器 · 料号待核", box([1, 1, 2], point(5, -1.5)), DARK,
                    "LPF", "lowpass_filter", "Krinner §3.3 / Appendix3 give differing VLFX1300+/1350 designations", radial*18)
            if code in {"D", "F"}:
                p = point(5, -4)
                put("红外吸收滤波器", cyl(.4, 2.3, p, sections=16), COPPER,
                    "IR-FILTER", "infrared_filter", "Krinner §3.2–3.3 / custom absorptive IR filter", radial*22)
            if code == "RO":
                p = point(2, -4)
                put("4 K HEMT 低噪声放大器", box([1.5, 2.3, 1.0], p-radial*4), DARK, "HEMT", "hemt_amplifier",
                    "Krinner §3.4 / Fig.6(b)", radial*18, "LNF LNC4_8C")
                put("CP 环行器", cyl(.8, .8, point(4)-radial*5, sections=24), GOLD, "CIRC", "circulator",
                    "Krinner §3.4", radial*24, "Quinstar QCY-060400CM00")
                put("MXC TWPA 模块", box([1.6, 3.0, .8], point(5)-radial*6), BLUE, "TWPA", "quantum_limited_amplifier",
                    "Krinner §3.4 / Fig.6(a)", radial*27, "TWPA (original device PN not published)")
    # Four-channel isolator arrays are purchased modules, counted once each;
    # their four electrical paths are deliberately not counted as four parts.
    root = m.group("读出与处理器封装", [0, -55, -20])
    g = m.group(root+" / 隔离与读出", [0, -65, -25])
    for i, x in enumerate([-8, 8]):
        part(f"四通道隔离器阵列 {i+1}", box([5, 9, 2], (x, -10, 24)), GOLD, g,
             f"READOUT-ISO-ARRAY-{i}", "isolator_array", "Krinner §3.4 / Fig.6(a)", [x, -10, -i*10], "Quinstar QCC-075900XM000")
    g = m.group(root+" / QPU 封装参考", [0, -55, -65])
    part("处理器封装底座 · 几何参考", rounded_rect(16, 15, 1, 2, 16), COPPER, g, "QPU-BASE", "chip_package",
         "Google hardware / IBM quantum processors; package layout authored", [0, 0, -12], sources=[GOOGLE, IBM])
    carrier = box([12, 11, .4], (0, 0, 17.4))-box([7.5, 7.5, 1], (0, 0, 17.4))
    part("QPU 封装载板", carrier, TEAL, g, "QPU-PCB", "package_pcb", "Google hardware / quantum processor", [0, 0, 6], sources=[GOOGLE])
    part("超导量子芯片 · 未声明量子位数", box([7, 7, .16], (0, 0, 17.3)), [46, 61, 80], g,
         "QPU-DIE", "quantum_processor", "Google superconducting hardware / chip envelope only", [0, 0, 14], sources=[GOOGLE])
    part("QPU 封装上盖", rounded_rect(16, 15, 1, 1.2, 19), GOLD, g, "QPU-LID", "package_lid", "IBM cryostat / quantum processors", [0, 0, 30], sources=[IBM], role="cover", appearance=proxy)
    part("超导磁屏蔽筒体", tube(11, 10.5, 12, (0, 0, 16)), STEEL, g, "QPU-MAG-SHIELD", "magnetic_shield",
         "IBM cryostat / magnetic shield", [24, 0, 0], sources=[IBM], role="cover", appearance=proxy)
    part("磁屏蔽底盖", cyl(11, .6, (0, 0, 9.6)), STEEL, g, "QPU-MAG-LID", "magnetic_shield",
         "IBM cryostat / magnetic shield", [24, 0, -18], sources=[IBM], role="cover", appearance=proxy)

    root = m.group("室温测控与气路", [75, 0, 0])
    g = m.group(root+" / 测控机柜", [90, 0, 10])
    rack = box([30, 35, 110], (87, 0, 50))-box([27, 36, 105], (87, 1, 50))
    part("室温测控机柜框架", rack, DARK, g, "RACK-FRAME", "control_rack", "Google full-stack / control electronics", [15, 0, 0], sources=[GOOGLE])
    for i, name in enumerate(["微波本振模块", "DAC / IQ 波形模块", "ADC 读出模块", "时钟同步模块", "磁通偏置电源", "QB 控制主机"]):
        part(name+" · 仪器外形参考", box([26, 30, 12], (87, 0, 8+i*17)), SILVER if i != 5 else BLUE, g,
             f"RACK-MODULE-{i}", "control_instrument", "Google/IBM full-stack control architecture; no OEM instrument PN", [25+i*4, 0, i*4], sources=[GOOGLE, MICROWAVE])
    g = m.group(root+" / 气体循环与压缩机", [75, 45, 0])
    for i, (name, center, size) in enumerate([("氦气压缩机外壳", [66, 58, 14], [29, 24, 28]),
                                           ("气体处理单元", [96, 58, 22], [22, 19, 44]),
                                           ("涡轮分子泵外壳", [40, 58, 9], [15, 15, 18])]):
        part(name+" · 外形参考", box(size, center), WHITE, g, f"GAS-MODULE-{i}", "gas_handling",
             "Dilution refrigerator system architecture; component quantity/shape authored", [i*12, 24, 0], sources=[IBM])
    return m
