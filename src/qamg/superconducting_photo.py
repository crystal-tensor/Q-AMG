"""Photo-constrained authored reconstruction of the two supplied references.

Exterior silhouette/frame/can follow image 1, the gold wiring follows image 2.
These images are not certified views of one machine. Hidden poses, wiring
counts and nominal dimensions remain authored; this is not a neural result.
Original uploads and the separate PartCrafter attempt stay private on disk.
"""
from __future__ import annotations

from itertools import pairwise

import numpy as np

from .showcase import SILVER, Model, box, cyl, ellipsoid, orient, rounded_rect, tube, union
from .superconducting import coax_assembly, curved_cable, evidence

EXTERIOR = "user-exterior-20261112"
INTERIOR = "user-interior-20261112"
GOLD = [225, 177, 57]
PALE_GOLD = [237, 211, 140]
COPPER = [189, 116, 58]
ALUMINIUM = [185, 193, 199]
CHROME = [202, 210, 217]
BLACK = [27, 30, 35]
STAGES = [("300 K", 142., 20.3), ("50 K", 127., 19.2), ("4 K", 106., 18.),
          ("Still", 88., 16.4), ("CP", 72., 15.), ("MXC", 58., 13.4)]


def extrusion(start, end, width=2.8):
    """One connected T-slot extrusion, with grooves rather than a round post."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    length = float(np.linalg.norm(end-start))
    solid = box([width, width, length])-cyl(width*.15, length+.2, sections=12)
    for axis in [0, 1]:
        for sign in [-1, 1]:
            center = np.zeros(3)
            center[axis] = sign*width*.47
            size = [width*.22, width*.22, length+.2]
            size[axis] = width*.25
            solid -= box(size, center)
    return orient(solid, end-start).translate((start+end)/2)


def frame_ring(width, depth, radius, z):
    return (rounded_rect(width, depth, radius, 2.8, z)
            -rounded_rect(width-5.6, depth-5.6, radius-2.8, 3.1, z))


def can_handle(center, width=2.0, height=7.0):
    x, y, z = center
    # One bent grab handle, rather than three disconnected render fragments.
    return union(box([.7, .9, height], (x-width/2, y, z)),
                 box([.7, .9, height], (x+width/2, y, z)),
                 box([width+.7, .9, .7], (x, y, z+height/2-.35)))


def service_route(angle, stages, loop_spans):
    """One cable assembly with real 3D slack loops, continuous across anchors."""
    radial = np.array([np.cos(angle), np.sin(angle), 0.])
    tangent = np.array([-np.sin(angle), np.cos(angle), 0.])
    result = []
    for i, ((top, r_top), (bottom, r_bottom)) in enumerate(pairwise(stages)):
        start, end = radial*r_top+[0, 0, top], radial*r_bottom+[0, 0, bottom]
        if i == 0:
            result.append(start)
        if i in loop_spans:
            z = (top+bottom)/2
            radius = (r_top+r_bottom)/2
            result.append(radial*radius+[0, 0, z+2.0])
            result.append(radial*radius+[0, 0, z])
            for theta in np.linspace(0, 2*np.pi, 25)[1:]:
                result.append(radial*(radius+.65*theta/(2*np.pi))
                              +tangent*1.25*(1-np.cos(theta))+[0, 0, z-1.25*np.sin(theta)])
            result.append(radial*(radius+.65)+[0, 0, z-2.0])
        else:
            for t in [.33, .66]:
                result.append((1-t)*start+t*end+tangent*.3*np.sin(np.pi*t))
        result.append(end)
    return np.asarray(result)


def braided_link(start, end):
    """One purchased braided strap, not individually counted metal strands."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    pieces = []
    for sign in [-1, 1]:
        for phase in np.linspace(0, 2*np.pi, 4, endpoint=False):
            t = np.linspace(0, 1, 49)
            pts = start+(end-start)*t[:, None]
            pts[:, 0] += .65*np.sin(sign*t*5*np.pi+phase)
            pts[:, 1] += .35*np.cos(sign*t*5*np.pi+phase)
            pieces.append(curved_cable(pts, .22, sections=8))
    pieces += [box([2.4, 1.6, 1.0], start), box([2.4, 1.6, 1.0], end)]
    # The strands are joined by a narrow central web in the envelope geometry.
    pieces.append(curved_cable([start, end], .45, sections=12))
    return union(*pieces)


def superconducting_computer():
    from .engineering_models import bolt, conical_shell

    m = Model()

    def part(label, solid, color, group, instance, kind, photo=INTERIOR, offset=(0, 0, 0),
             scope="interior", reveal=None, locator=None, material=None):
        proof = evidence(instance, kind, locator or ("图1 / 外部可见轮廓" if photo == EXTERIOR else "图2 / 可见冷盘与布线"), sources=[photo])
        proof.update({"quantity_basis": "photo_density_guided_authored_sampling", "dimensions_basis": "photo_proportions_no_measured_scale",
                      "position_basis": "photo_guided_authored_pose", "view_scope": scope,
                      "reference_image_id": photo, "reveal_translation_mm": list(reveal or [0, 0, 0])})
        metal = kind not in {"can_handle", "warm_cable", "instrument_envelope", "qpu_die", "cable_harness"}
        appearance = {"type": "pbr_surface", "rgba": [*color, 255],
                      "metallic": .82 if metal else .05, "roughness": .32 if metal else .65}
        m.part(label, solid, color, group, offset, role="fastener" if kind == "mounting_screw" else "cover" if reveal else "component",
               material=material or ("金色金属表面 · 材料牌号未核" if color in [GOLD, PALE_GOLD, COPPER] else "照片外观参考 · 材料未核"),
               basis="photo_guided_authored_reference", evidence=proof, appearance=appearance)

    root = m.group("照片机架", [0, 0, 0])
    g = m.group(root+" / 铝型材承载框架", [0, 0, 0])
    for i, z in enumerate([3., 145., 187.]):
        part(f"圆角外围框梁 {i+1}", frame_ring(205, 135, 23, z), ALUMINIUM, g, f"FRAME-RING-{i}", "frame_ring", EXTERIOR, scope="frame")
    positions = [(-91, -43), (91, -43), (-91, 43), (91, 43), (-46, -48), (46, -48), (-46, 48), (46, 48)]
    for i, (x, y) in enumerate(positions):
        part(f"T 槽承载立柱 {i+1:02d}", extrusion([x, y, 4.5], [x, y, 185.5]), ALUMINIUM, g,
             f"FRAME-POST-{i}", "aluminium_extrusion", EXTERIOR, [x*.2, y*.2, 0], scope="frame")
        for j, z in enumerate([7., 142., 183.]):
            part(f"立柱 {i+1:02d} 连接角件 {j+1}", union(box([5, 1.5, 5], (x, y+1.8, z)), box([5, 5, 1.5], (x, y+3, z-1.8))), SILVER,
                 g, f"FRAME-BRACKET-{i}-{j}", "frame_bracket", EXTERIOR, scope="frame")
    for i, z in enumerate([8., 144., 184.]):
        for j, y in enumerate([-48., 48.]):
            part(f"内层横梁 {i+1}-{j+1}", extrusion([-90, y, z], [90, y, z]), ALUMINIUM, g,
                 f"FRAME-CROSS-{i}-{j}", "aluminium_extrusion", EXTERIOR, scope="frame")
    for i, x in enumerate([-46., 46.]):
        part(f"低温机安装纵梁 {i+1}", extrusion([x, -48, 145], [x, 48, 145], 4), ALUMINIUM, g,
             f"FRAME-LONG-{i}", "aluminium_extrusion", EXTERIOR, scope="frame")
        for j, y in enumerate([-48., 48.]):
            part(f"上框斜撑 {i+1}-{j+1}", extrusion([x, y, 148], [x*.43, y, 183], 2.1), ALUMINIUM, g,
                 f"FRAME-DIAGONAL-{i}-{j}", "aluminium_extrusion", EXTERIOR, scope="frame")
    g = m.group(root+" / 顶部测控与线束", [0, 0, 20])
    part("顶部设备安装盘", rounded_rect(111, 79, 3, 2, 147), SILVER, g, "TOP-DECK", "instrument_mount", EXTERIOR, scope="frame")
    for i, (x, y, size) in enumerate([(-27, -15, [18, 19, 10]), (23, 19, [21, 18, 17]), (4, -17, [15, 17, 12])]):
        part(f"图中顶部测控模块 {i+1} · 外形", box(size, (x, y, 151+size[2]/2)), BLACK if i == 0 else SILVER, g,
             f"TOP-INSTRUMENT-{i}", "instrument_envelope", EXTERIOR, scope="frame")
    for i in range(16):
        pts = [[-24+i*.7, -11, 164], [-22+i*.9, 10, 171+(i % 3)], [28+i*.2, 24, 163], [43+i*.18, 32, 148],
               [44+i*.14, 34, 125], [45+i*.12, 34, 70+(i % 4)*4]]
        part(f"顶部黑色仪器线束 {i+1:02d}", curved_cable(pts, .16, sections=8), BLACK, g,
             f"WARM-CABLE-{i}", "warm_cable", EXTERIOR, scope="frame")

    root = m.group("可拆银色桶体", [70, 65, -15])
    upper_delta, lower_delta = [59., 25., 9.], [59., 25., -25.]
    g = m.group(root+" / 上段真空筒", upper_delta)
    part("银色上段真空桶", tube(21.5, 21.15, 39, (0, 0, 119)), ALUMINIUM, g,
         "VAC-UPPER", "vacuum_can_upper", EXTERIOR, scope="outer_shell", reveal=upper_delta)
    for i, z in enumerate([99.5, 138.5]):
        part(f"上段桶连接法兰 {i+1}", tube(22.4, 20.9, 1.4, (0, 0, z)), CHROME, g,
             f"VAC-UPPER-FLANGE-{i}", "vacuum_flange", EXTERIOR, scope="outer_shell", reveal=upper_delta)
    for i, x in enumerate([-9., 9.]):
        part(f"上段黑色提手 {i+1}", can_handle((x, 20.5, 121)), BLACK, g,
             f"VAC-UPPER-HANDLE-{i}", "can_handle", EXTERIOR, scope="outer_shell", reveal=upper_delta)
    g = m.group(root+" / 下段真空筒", lower_delta)
    part("银色下段真空桶", tube(21.5, 21.15, 47.8, (0, 0, 74.7)), ALUMINIUM, g,
         "VAC-LOWER", "vacuum_can_lower", EXTERIOR, scope="outer_shell", reveal=lower_delta)
    bottom = (ellipsoid([43, 43, 5], (0, 0, 51))-ellipsoid([42.3, 42.3, 4.4], (0, 0, 51))) ^ box([50, 50, 4], (0, 0, 49))
    part("下段圆底端盖", bottom, ALUMINIUM, g, "VAC-BOTTOM", "vacuum_lid", EXTERIOR, scope="outer_shell", reveal=lower_delta)
    for i, z in enumerate([51., 98.6]):
        part(f"下段桶连接法兰 {i+1}", tube(22.4, 20.9, 1.4, (0, 0, z)), CHROME, g,
             f"VAC-LOWER-FLANGE-{i}", "vacuum_flange", EXTERIOR, scope="outer_shell", reveal=lower_delta)
    for i, x in enumerate([-12., 12.]):
        part(f"下段黑色提手 {i+1}", can_handle((x, 18.3, 80)), BLACK, g,
             f"VAC-LOWER-HANDLE-{i}", "can_handle", EXTERIOR, scope="outer_shell", reveal=lower_delta)
    g = m.group(root+" / 桶体法兰紧固", [75, 65, 0])
    for layer, (z, delta) in enumerate([(51., lower_delta), (99., upper_delta), (139., upper_delta)]):
        for i, a in enumerate(np.linspace(0, 2*np.pi, 12, endpoint=False)):
            p = [21.7*np.cos(a), 21.7*np.sin(a), z+1]
            part(f"桶体法兰 {layer+1} 螺钉 {i+1:02d}", bolt(.22, 1.2, p), CHROME, g,
                 f"VAC-BOLT-{layer}-{i}", "mounting_screw", EXTERIOR, scope="outer_shell", reveal=delta)

    root = m.group("金色冷盘与支承", [0, 0, 0])
    g = m.group(root+" / 顶部密封与悬挂", [0, 0, 18])
    part("300 K 顶部密封安装盘", tube(21.0, 2.3, 2.0, (0, 0, 142)), CHROME, g,
         "COLD-TOP-FLANGE", "vacuum_flange", EXTERIOR)
    for stage_index, (stage, z, radius) in enumerate(STAGES[1:]):
        g = m.group(root+f" / {stage} 金色冷盘", [0, 0, 10-stage_index*7])
        plate = tube(radius, 2.0, 1.15, (0, 0, z))
        # Clearance holes follow our radial cable layout; not measured OEM bores.
        for a in np.linspace(0, 2*np.pi, 24, endpoint=False):
            plate -= cyl(.48, 1.5, [(radius-2.2)*np.cos(a), (radius-2.2)*np.sin(a), z], sections=8)
        part(stage+" 镀金外观冷盘", plate, GOLD, g, f"COLD-PLATE-{stage_index}", "copper_plate")
        for j, a in enumerate(np.linspace(0, 2*np.pi, 6, endpoint=False)):
            r = radius*.72
            upper = STAGES[stage_index][1]
            center = [r*np.cos(a), r*np.sin(a), (upper+z)/2]
            part(f"{stage} 银色隔热支柱 {j+1}", tube(.48, .28, upper-z-1.2, center), CHROME, g,
                 f"COLD-POST-{stage_index}-{j}", "support_post", offset=[4*np.cos(a), 4*np.sin(a), 0])
        for j, a in enumerate(np.linspace(0, 2*np.pi, 12, endpoint=False)):
            p = [(radius-1)*np.cos(a), (radius-1)*np.sin(a), z+.8]
            part(f"{stage} 盘缘紧固螺钉 {j+1:02d}", bolt(.14, .8, p), PALE_GOLD, g,
                 f"COLD-BOLT-{stage_index}-{j}", "mounting_screw")

    root = m.group("可拆内层热屏", [80, -40, 0])
    for i, (stage, z, radius) in enumerate([STAGES[1], STAGES[2], STAGES[3]]):
        g = m.group(root+f" / {stage} 热辐射屏", [80+i*4, -35, -8])
        delta = [89.+i*3, -15., -4.+i*7]
        height = z-52
        part(stage+" 金属热屏桶", tube(radius+.65, radius+.40, height, (0, 0, (z+52)/2)), CHROME if i == 0 else GOLD, g,
             f"THERMAL-CAN-{i}", "shield", scope="thermal_shell", reveal=delta)
        part(stage+" 热屏底盘", cyl(radius+.65, .4, (0, 0, 51.7)), CHROME if i == 0 else GOLD, g,
             f"THERMAL-LID-{i}", "shield", scope="thermal_shell", reveal=delta)

    root = m.group("制冷与编织热连接", [-24, 0, 0])
    g = m.group(root+" / 双侧冷头与铜编织带", [-25, 0, 10])
    for i, x in enumerate([-10., 10.]):
        part(f"可见脉冲管冷头 {i+1}", union(cyl(1.5, 17, (x, -3, 132)), cyl(2.0, 1.1, (x, -3, 140.1))), CHROME, g,
             f"PT-COLDHEAD-{i}", "pulse_tube")
        part(f"金色铜编织热连接 {i+1}", braided_link([x, 4, 138], [x+2, 4, 127.9]), COPPER, g,
             f"THERMAL-BRAID-{i}", "braided_thermal_link", material="铜编织热连接 · 按一根总成计件")
    g = m.group(root+" / 中央稀释回路", [0, -16, 0])
    t = np.linspace(0, 1, 145)
    pts = np.column_stack([1.1*np.cos(t*18*np.pi), 1.1*np.sin(t*18*np.pi), 135-27*t])
    part("中央波纹管外形", union(curved_cable(pts, .30), cyl(.86, 28, (0, 0, 121.5))), CHROME, g,
         "DILUTION-BELLOWS", "bellows")
    part("Still 蒸馏室外壳", tube(2.7, 2.1, 5, (0, 0, 91.2)), COPPER, g, "DILUTION-STILL", "still")
    part("混合室封装外壳", tube(3.4, 2.8, 4.6, (0, 0, 60.9)), GOLD, g, "DILUTION-MXC", "mixing_chamber")

    root = m.group("金色微波布线", [0, 18, 0])
    for channel in range(48):
        angle = (channel+.3)*2*np.pi/48
        radial = np.array([np.cos(angle), np.sin(angle), 0.])
        tangent = np.array([-np.sin(angle), np.cos(angle), 0.])
        g = m.group(root+f" / RF-{channel+1:02d}", radial*15+[0, 0, 0])
        key = f"RF-{channel+1:02d}"
        routes = [service_route(angle, [(140, 17.0), (128, 17.0), (108, 15.8)], {1}),
                  service_route(angle, [(104, 15.8), (89, 14.2), (73, 12.8), (60, 11.2)], {0, 1, 2})]
        for segment, route in enumerate(routes):
            part(f"{key} 连续盘绕同轴总成 {segment+1}", coax_assembly(route, .10), PALE_GOLD if channel % 3 else CHROME, g,
                 f"{key}-COAX-{segment}", "rf_cable", offset=radial*(3+segment*3),
                 locator="图2 / 垂直线缆与应力释放圈；路径/数量按可见密度取样，连续热锚不任意切段")
        for anchor, z, radius in [(0, 127., 17.), (1, 88., 14.2)]:
            p = radial*radius+[0, 0, z]
            block = box([.8, .8, .8])-cyl(.105, 1.0, sections=12)
            for sign, side in [(-1, "左"), (1, "右")]:
                half = block ^ box([.4, 1.0, 1.0], [sign*.2, 0, 0])
                part(f"{key} 热锚 {anchor+1} {side}半夹块", half.translate(p), COPPER, g,
                     f"{key}-CLAMP-{anchor}-{side}", "thermal_clamp", offset=tangent*sign*3,
                     locator="图2 / 铜质线缆固定件；双半夹块构型参照公开低温论文，实际隐藏界面未核")
            part(f"{key} 热锚 {anchor+1} 压紧楔块", box([.3, .65, .8], p+radial*.55), GOLD, g,
                 f"{key}-WEDGE-{anchor}", "thermal_wedge", offset=radial*6)
            part(f"{key} 热锚 {anchor+1} 紧固螺钉", bolt(.14, .65, p+radial*.95, radial), CHROME, g,
                 f"{key}-BOLT-{anchor}", "mounting_screw", offset=radial*9)
        for stage, z, radius in [("4 K", 106., 15.8), ("MXC", 58., 11.2)]:
            p = radial*radius+[0, 0, z]
            part(f"{key} {stage} 射频调理器件 · 包络", union(cyl(.30, 1.6, p, sections=6), cyl(.19, 2.3, p, sections=12)), GOLD, g,
                 f"{key}-RF-MODULE-{stage}", "rf_conditioner", offset=radial*7,
                 locator="图2 / 金色射频器件外观；衰减量、滤波功能、原厂料号未从照片确认")

    root = m.group("底部扇出与芯片封装", [0, 0, -15])
    g = m.group(root+" / 金色扇出线与接插件", [0, 0, -16])
    for i in range(24):
        angle = (i+.2)*2*np.pi/24
        rad = np.array([np.cos(angle), np.sin(angle), 0.])
        target = np.array([(-1 if i < 12 else 1)*3.7, (i % 12-5.5)*.45, 47.5])
        pts = [rad*11.0+[0, 0, 57], rad*10.4+[0, 0, 53.5], rad*8+[0, 0, 50], target]
        part(f"底部封装扇出线 {i+1:02d}", coax_assembly(pts, .075), PALE_GOLD, g,
             f"QPU-FANOUT-{i}", "fanout_coax", offset=rad*6+[0, 0, -8])
        part(f"扇出连接器固定块 {i+1:02d}", box([.6, .7, .9], target), GOLD, g,
             f"QPU-CONNECTOR-{i}", "rf_connector")
    g = m.group(root+" / QPU 封装参考", [0, 0, -24])
    part("底部芯片封装托座", rounded_rect(9, 8, .7, 1.0, 46.3), COPPER, g, "QPU-BASE", "chip_package")
    part("芯片封装边框", box([8, 7, .5], (0, 0, 47))-box([5, 4, .8], (0, 0, 47)), GOLD, g, "QPU-RIM", "package_frame")
    part("超导芯片位置示意 · 未识别芯片型号", box([4.3, 3.3, .12], (0, 0, 46.9)), [43, 55, 75], g,
         "QPU-DIE", "qpu_die", locator="图2 / 底部芯片封装位置；芯片内部与量子位数未由照片确认")
    part("封装上盖 · 推定结构", rounded_rect(9, 8, .7, .65, 48), GOLD, g, "QPU-LID", "package_lid", offset=[0, 0, -8])
    # A short transition collar matches the exterior neck, not an invented desk.
    g = m.group(root+" / 底部接口", [0, 0, -10])
    part("桶底接口过渡圈", conical_shell(4.5, 6.3, 3.0, .35).translate([0, 0, 44.5]), CHROME, g,
         "BOTTOM-INTERFACE", "interface_collar", EXTERIOR)
    # Fit the image-2 chandelier proportions within image-1's taller vacuum can.
    # This is an explicit nominal-envelope choice, not a recovered hidden scale.
    for item in m.parts:
        if item.evidence["view_scope"] == "interior":
            item.solid = item.solid.scale([1, 1, .82]).translate([0, 0, 142*.18])
            item.evidence["proportion_fit"] = "interior_vertical_envelope_0.82_about_top_flange"
    return m
