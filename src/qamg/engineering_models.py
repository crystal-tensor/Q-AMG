"""Detailed, independently authored reference assemblies.

Every leaf is a named physical component or a purchased sealed module. Repeated
parts (blades, links, ribs, trays) are separate instances, not disconnected clouds
hidden in one mesh. Counts not supported by a primary reference are illustrative.
"""
from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import trimesh
from scipy.interpolate import PchipInterpolator

from .mesh_parts import as_manifold
from .showcase import (
    BLUE,
    DARK,
    GOLD,
    ORANGE,
    RED,
    SILVER,
    TEAL,
    WHITE,
    Model,
    beam,
    box,
    cyl,
    ellipsoid,
    gear,
    loft_x,
    orient,
    prism,
    rounded_rect,
    tube,
    turn,
    union,
)

COPPER = [176, 110, 68]
RUBBER = [36, 40, 43]
GLASS = [79, 120, 143]
STEEL = [119, 139, 153]
GLASS_PROXY = {"type": "transparent_display_proxy", "rgba": [131, 184, 201, 75]}


def shell_box(size, wall, center=(0, 0, 0)):
    return box(size, center)-box(np.asarray(size)-2*wall, center)


def bolt(radius, length, center=(0, 0, 0), axis=(0, 0, 1)):
    shaft = cyl(radius, length, (0, 0, -length/2), sections=16)
    head = cyl(radius*1.8, radius*1.3, sections=6)
    return orient(union(shaft, head), axis).translate(center)


def conical_shell(bottom, top, height, wall=.6):
    profile = [[bottom, 0], [top, height], [max(.1, top-wall), height],
               [max(.1, bottom-wall), 0], [bottom, 0]]
    return as_manifold(trimesh.creation.revolve(profile, sections=40))


def hollow_loft(stations, wall):
    inner = [(x, max(.15, y-wall), max(.15, z-wall), h) for x, y, z, h in stations]
    return loft_x(stations)-loft_x(inner)


def pipe(points, radius, wall=.25):
    # Explicit annular sweep avoids near-coincident sphere/cylinder booleans at
    # elbows. Every joint shares indices, including both open-end annuli.
    points = np.asarray(points, dtype=float)
    directions = np.diff(points, axis=0)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    tangents = np.vstack([directions[0], directions[:-1]+directions[1:], directions[-1]])
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    n = 32
    angles = np.arange(n)*2*np.pi/n
    rings = []
    for p, tangent in zip(points, tangents, strict=True):
        matrix = trimesh.geometry.align_vectors([0, 0, 1], tangent)[:3, :3]
        for r in [radius, max(.1, radius-wall)]:
            circle = np.column_stack([r*np.cos(angles), r*np.sin(angles), np.zeros(n)])
            rings.append(circle@matrix.T+p)
    faces = []
    for row in range(len(points)-1):
        for skin in [0, 1]:
            for j in range(n):
                a, b = (row*2+skin)*n+j, (row*2+skin)*n+(j+1) % n
                c, d = b+2*n, a+2*n
                faces.extend([[a, b, c], [a, c, d]])
    for row in [0, len(points)-1]:
        for j in range(n):
            a, b = row*2*n+j, row*2*n+(j+1) % n
            faces.extend([[a, a+n, b+n], [a, b+n, b]])
    mesh = trimesh.Trimesh(np.concatenate(rings), faces, process=False)
    mesh.fix_normals()
    return as_manifold(mesh)


def fasteners(m, group, centers, radius=1, length=3, axis=(0, 0, 1), name="连接螺钉"):
    for i, center in enumerate(centers):
        direction = np.asarray(axis, dtype=float)*(12+i*.5)
        m.part(f"{name} {i+1:02d}", bolt(radius, length, center, axis), STEEL, group,
               direction, role="fastener", material="钢 · 规格示意")


def blade_ring(m, group, x, radius, count, label, color, axial=0):
    blade = prism([[radius*.25, -1.1], [radius*.75, -radius*.12],
                   [radius, -1], [radius*.88, radius*.10], [radius*.25, 1.1]], 1.0)
    blade = turn(blade, 24, (1, 0, 0))
    for i, angle in enumerate(np.linspace(0, 2*np.pi, count, endpoint=False)):
        solid = orient(turn(blade, math.degrees(angle)), [1, 0, 0]).translate([x, 0, 0])
        m.part(f"{label} {i+1:02d}", solid, color, group,
               [axial, 22*np.cos(angle), 22*np.sin(angle)], material="叶片合金 · 型线近似")


def aero_engine():
    m = Model()
    root = m.group("进气与风扇", [80, 0, 0])
    g = m.group(root+" / 进气短舱", [90, 0, 0])
    m.part("进气唇口", tube(43, 39.8, 7, (79, 0, 0), (1, 0, 0)), SILVER, g, [20, 0, 0])
    nacelle = tube(43, 40.5, 42, (54, 0, 0), (1, 0, 0))
    for sign, side in [(1, "上"), (-1, "下")]:
        m.part(f"{side}半风扇整流罩", nacelle ^ box([50, 95, 48], (54, 0, sign*24)), WHITE, g,
               [0, 0, sign*48])
    g = m.group(root+" / 风扇转子", [116, 0, 0])
    m.part("风扇轮盘", tube(12, 4, 6, (72, 0, 0), (1, 0, 0)), SILVER, g, [6, 0, 0])
    m.part("风扇锥", orient(as_manifold(trimesh.creation.cone(radius=8, height=19)), [1, 0, 0]).translate([74, 0, 0]), SILVER, g, [35, 0, 0])
    blade_ring(m, g, 72, 38, 22, "风扇叶片", STEEL)
    g = m.group(root+" / 出口导向器", [57, 0, 0])
    m.part("出口导向器外环", tube(39, 37, 4, (56, 0, 0), (1, 0, 0)), SILVER, g)
    blade_ring(m, g, 56, 36, 16, "风扇出口导向叶片", BLUE)

    root = m.group("压气机与双轴", [0, 65, 0])
    g = m.group(root+" / 双转子轴系", [0, 38, -45])
    m.part("低压内轴", cyl(2.7, 148, (-1, 0, 0), (1, 0, 0)), SILVER, g, [-32, 0, -20])
    m.part("高压空心外轴", tube(5, 3.2, 83, (-2, 0, 0), (1, 0, 0)), STEEL, g, [0, 18, 0])
    for i, (x, radius) in enumerate([(40, 24), (26, 22), (11, 20), (-3, 18)]):
        g = m.group(root+f" / 压气机第 {i+1} 级", [25-i*18, 70+i*16, 0])
        m.part(f"第 {i+1} 级转子轮盘", tube(radius*.34, 5.3, 3.8, (x, 0, 0), (1, 0, 0)), SILVER, g)
        blade_ring(m, g, x, radius, 12, f"第 {i+1} 级转子叶片", STEEL)
        m.part(f"第 {i+1} 级静子承环", tube(radius+1, radius-.4, 3.2, (x-5, 0, 0), (1, 0, 0)), BLUE, g, [-7, 0, 0])
        blade_ring(m, g, x-5, radius*.92, 10, f"第 {i+1} 级静子导叶", BLUE, -9)
    shroud = tube(30, 28, 70, (15, 0, 0), (1, 0, 0))
    g = m.group(root+" / 核心机匣", [0, -45, 0])
    for sign, name in [(1, "上"), (-1, "下")]:
        m.part(f"压气机{name}半机匣", shroud ^ box([76, 70, 36], (15, 0, sign*18)), SILVER, g,
               [0, 0, sign*55])
    fasteners(m, g, [(x, y, 0) for x in [-12, 10, 33] for y in [-30, 30]], 1.1, 4,
              (0, 0, 1), "机匣合缝螺栓")

    root = m.group("燃烧室", [-25, 0, 78])
    g = m.group(root+" / 环形火焰筒", [-25, 0, 90])
    m.part("燃烧室外机匣", tube(27, 25, 25, (-26, 0, 0), (1, 0, 0)), SILVER, g, [0, 0, 30])
    m.part("外火焰筒", tube(23, 22, 23, (-26, 0, 0), (1, 0, 0)), GOLD, g, [0, 25, 0])
    m.part("内火焰筒", tube(11, 10, 23, (-26, 0, 0), (1, 0, 0)), COPPER, g, [0, -25, 0])
    m.part("火焰筒头部", tube(23, 10, 1.6, (-14, 0, 0), (1, 0, 0)), GOLD, g, [18, 0, 0])
    g = m.group(root+" / 喷油与点火", [-25, 0, 120])
    for i, a in enumerate(np.linspace(0, 2*np.pi, 10, endpoint=False)):
        v = [0, np.cos(a), np.sin(a)]
        p = [-14, 17*np.cos(a), 17*np.sin(a)]
        m.part(f"燃油喷嘴 {i+1:02d}", cyl(1.5, 8, p, (1, 0, 0)), COPPER, g,
               np.asarray(v)*18+[18, 0, 0])
    for i, y in enumerate([-26, 26]):
        m.part(f"点火器 {i+1}", cyl(1.5, 12, (-21, y, 0), (0, 1, 0)), WHITE, g, [0, y, 10])

    root = m.group("涡轮与排气", [-110, 0, 0])
    for i, (x, radius) in enumerate([(-46, 19), (-58, 23)]):
        g = m.group(root+f" / {'高' if i == 0 else '低'}压涡轮", [-95-i*28, 0, 0])
        m.part(f"涡轮轮盘 {i+1}", tube(8, 5.3 if i == 0 else 2.9, 4, (x, 0, 0), (1, 0, 0)), GOLD, g)
        blade_ring(m, g, x, radius, 16, f"涡轮 {i+1} 动叶", ORANGE)
        m.part(f"涡轮导向器承环 {i+1}", tube(radius+1.2, radius-.5, 3, (x+5, 0, 0), (1, 0, 0)), DARK, g, [9, 0, 0])
        blade_ring(m, g, x+5, radius*.94, 10, f"涡轮 {i+1} 导向叶片", GOLD, 9)
    g = m.group(root+" / 排气段", [-155, 0, 0])
    m.part("排气外喷管", orient(conical_shell(25, 19, 29, 1), [1, 0, 0]).translate([-91, 0, 0]), DARK, g)
    m.part("尾锥", orient(as_manifold(trimesh.creation.cone(radius=8, height=25)), [-1, 0, 0]).translate([-66, 0, 0]), SILVER, g, [-30, 0, 0])

    root = m.group("支承与附件", [0, -70, -25])
    g = m.group(root+" / 主轴支承", [0, -80, -25])
    for i, x in enumerate([51, 2, -64]):
        m.part(f"主轴承外圈 {i+1}", tube(9, 7, 4, (x, 0, 0), (1, 0, 0)), STEEL, g,
               [0, -12*i, -12*i])
        m.part(f"主轴承内圈 {i+1}", tube(6.5, 5.1, 4, (x, 0, 0), (1, 0, 0)), GOLD, g,
               [0, 12+8*i, -12*i])
    g = m.group(root+" / 附件机匣", [0, -95, -55])
    m.part("附件齿轮箱壳", shell_box([30, 14, 10], 1, (5, -24, -29)), SILVER, g)
    m.part("附件传动齿轮", orient(gear(5, 18, 3), [0, 1, 0]).translate([2, -24, -29]), GOLD, g, [0, -18, 0])
    m.part("滑油泵", cyl(4, 12, (20, -24, -29), [1, 0, 0]), DARK, g, [20, 0, 0])
    m.part("供油管", pipe([[20, -24, -29], [20, -30, -15], [-10, -30, -15]], 1.1), COPPER, g, [0, -20, 15])
    return m


def airliner():
    m = Model()
    root = m.group("机身结构", [0, 0, 30])
    stations = [(-107, .8, 1, 5), (-86, 8, 8, 2), (-60, 13, 13, 0),
                (58, 13, 13, 0), (83, 10, 11, 0), (103, .7, 1.5, -1)]
    skin = hollow_loft(stations, .7)
    g = m.group(root+" / 蒙皮与舱门", [0, 0, 55])
    for a, b, name in [(-108, -60, "尾段"), (-60, 5, "后客舱"), (5, 66, "前客舱"), (66, 104, "机头")]:
        segment = skin ^ box([b-a, 40, 42], ((a+b)/2, 0, 3))
        for sign, side in [(-1, "右"), (1, "左")]:
            m.part(f"{name}{side}侧蒙皮", segment ^ box([220, 20, 45], (0, sign*10, 4)), WHITE, g,
                   [0, sign*35, 15])
    for sign, side in [(-1, "右"), (1, "左")]:
        for x, label in [(65, "前登机门"), (-57, "后登机门")]:
            m.part(side+label, rounded_rect(6, 12, 1, .9).rotate([90, 0, 0]).translate([x, sign*12.7, 0]), SILVER, g,
                   [12, sign*55, 0])
    g = m.group(root+" / 隔框与纵向构件", [0, 0, 16])
    for i, x in enumerate(np.linspace(-59, 59, 11)):
        ring = tube(12.3, 11.4, 1, (x, 0, 0), [1, 0, 0])
        m.part(f"机身环形隔框 {i+1:02d}", ring, STEEL, g, [0, 0, 3*i])
    for i, angle in enumerate(np.linspace(0, 2*np.pi, 10, endpoint=False)):
        y, z = 11.9*np.cos(angle), 11.9*np.sin(angle)
        m.part(f"长桁 {i+1:02d}", beam([-62, y, z], [62, y, z], .45), SILVER, g,
               [0, 16*np.cos(angle), 16*np.sin(angle)])
    g = m.group(root+" / 地板与压力舱壁", [0, -25, 5])
    for i, x in enumerate([-48, -24, 0, 24, 48]):
        m.part(f"地板横梁 {i+1}", box([1.5, 23, 1.3], (x, 0, -3)), SILVER, g, [0, 0, -i*4])
    m.part("客舱地板", box([125, 22, .7], (0, 0, -2)), SILVER, g, [0, 0, 12])
    m.part("后压力舱壁", cyl(10.9, 1, (-70, 0, 0), [1, 0, 0]), TEAL, g, [-20, 0, 0])

    for sign, side in [(-1, "右"), (1, "左")]:
        root = m.group(f"{side}主翼", [0, sign*62, 0])
        outline = [[24, sign*10], [-8, sign*95], [-33, sign*90], [-13, sign*10]]
        wing = prism(outline, 3.4, -3)
        g = m.group(root+" / 翼盒", [0, sign*68, 0])
        for z, name in [(-1.6, "上"), (-4.4, "下")]:
            m.part(f"{side}翼{name}蒙皮", wing ^ box([250, 240, .6], (0, 0, z)), WHITE, g, [0, 0, 20 if z > -3 else -20])
        for fraction, name in [(.3, "前"), (.72, "后")]:
            line = []
            for y, front, back in [(12, 23, -13), (91, -8, -33)]:
                x = front*(1-fraction)+back*fraction
                line.append([x, sign*y, -3])
            m.part(f"{side}翼{name}翼梁", beam(line[0], line[1], .8), STEEL, g, [8 if fraction < .5 else -8, 0, 0])
        for i, y in enumerate(np.linspace(16, 85, 9)):
            rib = wing ^ box([240, 1, 2.1], (0, sign*y, -3))
            m.part(f"{side}翼肋 {i+1:02d}", rib, GOLD, g, [0, sign*i*3, 0])
        g = m.group(root+" / 增升与操纵面", [-20, sign*64, -10])
        for i, (y0, y1) in enumerate([(17, 39), (40, 62)]):
            flap = prism([[-15, sign*y0], [-24, sign*y1], [-34, sign*y1], [-24, sign*y0]], .9, -4)
            m.part(f"{side}后缘襟翼 {i+1}", flap, SILVER, g, [-20, sign*i*9, -10])
            m.part(f"{side}襟翼作动筒 {i+1}", beam([-16, sign*y0, -3], [-25, sign*y0, -6], .9), DARK, g, [-25, 0, -20])
        m.part(f"{side}副翼", prism([[-26, sign*65], [-34, sign*89], [-39, sign*87], [-32, sign*64]], .9, -3.5), BLUE, g,
               [-32, sign*10, 0])
        for i, y in enumerate([28, 47, 66]):
            m.part(f"{side}扰流板 {i+1}", box([8, 12, .6], (-12, sign*y, -.9)), SILVER, g,
                   [0, 0, 28+i*6])

        root = m.group(f"{side}动力装置", [25, sign*45, -60])
        g = m.group(root+" / 发动机与吊架", [25, sign*45, -65])
        m.part(f"{side}发动机吊架", box([19, 5, 12], (4, sign*36, -9)), SILVER, g, [0, 0, 22])
        shroud = tube(10.5, 9.4, 31, (7, sign*37, -21), [1, 0, 0])
        for vertical, label in [(1, "上"), (-1, "下")]:
            m.part(f"{side}发动机{label}整流罩", shroud ^ box([40, 26, 13], (7, sign*37, -21+vertical*6.5)), WHITE, g,
                   [0, 0, vertical*18])
        m.part(f"{side}发动机核心机", tube(5, 2, 29, (6, sign*37, -21), [1, 0, 0]), DARK, g)
        m.part(f"{side}风扇盘", cyl(2.6, 2, (22, sign*37, -21), [1, 0, 0]), SILVER, g, [18, 0, 0])
        for i, a in enumerate(np.linspace(0, 2*np.pi, 12, endpoint=False)):
            blade = turn(box([6.5, 1.1, .7], (5, 0, 0)), math.degrees(a))
            blade = orient(blade, [1, 0, 0]).translate([22, sign*37, -21])
            m.part(f"{side}发动机风扇叶片 {i+1:02d}", blade, STEEL, g, [20, 6*np.cos(a), 6*np.sin(a)])

    root = m.group("尾翼与起落架", [-32, 0, -25])
    g = m.group(root+" / 尾翼操纵", [-45, 0, 45])
    for sign, side in [(-1, "右"), (1, "左")]:
        m.part(side+"水平安定面", prism([[-78, sign*4], [-94, sign*36], [-105, sign*34], [-95, sign*4]], 1.6, 9), WHITE, g,
               [0, sign*32, 0])
        m.part(side+"升降舵", prism([[-96, sign*6], [-107, sign*34], [-113, sign*33], [-103, sign*6]], .9, 9), BLUE, g,
               [-22, sign*32, 0])
    fin = turn(prism([[-82, 8], [-99, 40], [-107, 40], [-101, 8]], 1.6), 90, [1, 0, 0])
    m.part("垂直安定面", fin, WHITE, g, [0, 0, 28])
    rudder = turn(prism([[-102, 9], [-108, 40], [-113, 39], [-109, 9]], 1.0), 90, [1, 0, 0])
    m.part("方向舵", rudder, BLUE, g, [-18, 0, 28])
    for i, (x, y, name) in enumerate([(63, 0, "前"), (-18, -12, "右主"), (-18, 12, "左主")]):
        g = m.group(root+f" / {name}起落架", [i*18, y*3, -75])
        m.part(name+"减震支柱", tube(1.8, 1.1, 9, (x, y, -18)), SILVER, g, [10, 0, 0])
        m.part(name+"活塞杆", cyl(1, 9, (x, y, -23)), STEEL, g, [0, 0, -15])
        m.part(name+"轮轴", cyl(1.1, 11, (x, y, -27), [0, 1, 0]), SILVER, g)
        m.part(name+"撑杆", beam([x-5, y, -10], [x, y, -24], .8), DARK, g, [-10, 0, 0])
        for sign, side in [(-1, "右"), (1, "左")]:
            m.part(name+side+"轮胎", tube(3.7, 2, 2.3, (x, y+sign*4.3, -27), [0, 1, 0]), RUBBER, g, [0, sign*10, 0])
            m.part(name+side+"轮毂", tube(1.9, .7, 2.4, (x, y+sign*4.3, -27), [0, 1, 0]), SILVER, g, [0, sign*16, 0])

    root = m.group("客舱与驾驶舱", [0, -50, 30])
    g = m.group(root+" / 座椅与仪表", [0, -60, 40])
    for row, x in enumerate([-44, -22, 0, 22, 44]):
        for side, y in [("右", -7), ("左", 7)]:
            seat = union(box([8, 7, 1.5], (x, y, 0)), box([1.5, 7, 9], (x-3.5, y, 4)))
            m.part(f"取样客舱座椅 {row+1}{side}", seat, BLUE, g, [0, y*2, row*6])
    m.part("驾驶舱仪表台", box([8, 16, 6], (78, 0, 0)), DARK, g, [25, 0, 0])
    m.part("驾驶舱风挡", ellipsoid([13, 18, 2], (82, 0, 8.8)), GLASS, g, [28, 0, 20])
    for sign, side in [(-1, "右"), (1, "左")]:
        for i, x in enumerate(np.arange(-55, 64, 10)):
            m.part(f"{side}取样舷窗 {i+1:02d}", ellipsoid([2.1, .6, 2.7], (x, sign*13.1, 4)), GLASS, g,
                   [0, sign*22, i*.5])
    return m


def luxury_liner():
    m = Model()
    root = m.group("船体与舱室", [0, 0, -30])
    outline = [[-102, -19], [-61, -29], [56, -28], [88, -16], [108, 0],
               [88, 16], [56, 28], [-61, 29], [-102, 19]]
    outer = prism(outline, 25, -2)
    hull = outer-prism(np.asarray(outline)*[.978, .94], 26, .5)
    g = m.group(root+" / 船壳与双层底", [0, 0, -45])
    for sign, side in [(-1, "右舷"), (1, "左舷")]:
        m.part(side+"船壳", hull ^ box([240, 65, 40], (0, sign*32.5, -2)), BLUE, g,
               [0, sign*45, 0])
    m.part("内底板", prism(np.asarray(outline)*[.97, .90], 1.1, -11), STEEL, g, [0, 0, -20])
    m.part("船底纵桁", box([186, 1.4, 3], (-2, 0, -10)), SILVER, g, [0, 0, -35])
    g = m.group(root+" / 肋骨与水密舱壁", [0, 0, -18])
    for i, x in enumerate(np.linspace(-80, 72, 13)):
        rib = outer ^ box([1, 60, 24], (x, 0, -1))
        rib = rib-box([3, 49, 23], (x, 0, 3))
        m.part(f"取样横向肋骨 {i+1:02d}", rib, STEEL, g, [0, 0, i*1.8])
    for i, x in enumerate([-70, -30, 10, 48, 79]):
        bulkhead = outer ^ box([1, 60, 23], (x, 0, -1))
        m.part(f"水密舱壁 {i+1}", bulkhead, SILVER, g, [i*5, 0, 16])

    root = m.group("上层建筑", [0, 0, 65])
    for i in range(5):
        z, length, width = 12+i*8, 166-i*17, 53-i*6
        g = m.group(root+f" / 第 {i+1} 层甲板", [0, 0, 35+i*24])
        m.part(f"第 {i+1} 层甲板板", rounded_rect(length, width, 6, 1, z).translate([-9-i*2, 0, 0]), WHITE, g)
        for sign, side in [(-1, "右舷"), (1, "左舷")]:
            wall = box([length-12, 1, 6], (-9-i*2, sign*(width/2-2), z+3.6))
            m.part(f"第 {i+1} 层{side}舱室外壁", wall, WHITE, g, [0, sign*25, 0])
            for j, x in enumerate(np.linspace(-length/2+9, length/2-9, 6)):
                m.part(f"第 {i+1} 层{side}取样窗 {j+1}", box([6.5, .5, 2.5], (x-9-i*2, sign*(width/2-1.4), z+4)), GLASS, g,
                       [0, sign*32, j])
        for j, x in enumerate([-length*.22, length*.22]):
            m.part(f"第 {i+1} 层代表性舱室隔墙 {j+1}", box([.8, width-6, 6], (x-9, 0, z+3.6)), SILVER, g,
                   [0, 0, 10+j*8])
    g = m.group(root+" / 航行与公共设施", [0, 0, 165])
    bridge = shell_box([23, 35, 8], .9, (55, 0, 47))
    m.part("驾驶台舱体", bridge, WHITE, g, [25, 0, 0])
    m.part("驾驶台风挡", box([.5, 31, 3], (67, 0, 48)), GLASS, g, [38, 0, 0])
    m.part("雷达桅杆", cyl(.9, 17, (49, 0, 62)), SILVER, g, [0, 0, 25])
    m.part("雷达天线", box([18, 2, 1.5], (49, 0, 71)), DARK, g, [0, 0, 37])
    pool = rounded_rect(28, 13, 4, 2, 47).translate([-34, 0, 0])-rounded_rect(25, 10, 3, 3, 48).translate([-34, 0, 0])
    m.part("泳池池体", pool, TEAL, g, [-25, 0, 0])
    for sign in [-1, 1]:
        m.part("右排气烟道" if sign < 0 else "左排气烟道", shell_box([13, 9, 18], .8, (2, sign*7, 53)), DARK, g,
               [0, sign*18, 25])

    root = m.group("轮机与电力", [-25, -60, -30])
    for i, (x, y) in enumerate([(-28, -10), (-28, 10), (8, -10), (8, 10)]):
        g = m.group(root+f" / 柴油发电机组 {i+1}", [-30-i*10, -55-i*13, -30])
        m.part(f"发电机组 {i+1} 底座", box([28, 13, 2], (x, y, -9)), STEEL, g, [0, 0, -10])
        m.part(f"柴油机 {i+1} 缸体", box([18, 10, 7], (x+3, y, -4)), DARK, g)
        m.part(f"同步发电机 {i+1}", cyl(4.5, 8, (x-11, y, -4), [1, 0, 0]), TEAL, g, [-18, 0, 0])
        m.part(f"机组 {i+1} 排气歧管", pipe([[x+7, y, -1], [x+7, y, 6], [2, y, 6]], 1.1), COPPER, g, [0, 0, 22])
    g = m.group(root+" / 配电与辅机", [30, -90, 0])
    m.part("主配电板", box([13, 6, 12], (33, 0, -3)), WHITE, g, [0, 0, 15])
    m.part("推进变频柜", box([10, 6, 11], (47, 0, -3)), BLUE, g, [18, 0, 15])
    m.part("海水冷却泵", union(cyl(3.5, 9, (62, -9, -6), [1, 0, 0]), cyl(3, 5, (57, -9, -6), [0, 1, 0])), TEAL, g,
           [32, 0, 0])

    root = m.group("推进与操舵", [-70, 0, -50])
    for sign, side in [(-1, "右舷"), (1, "左舷")]:
        g = m.group(root+f" / {side}轴系", [-75, sign*37, -40])
        y = sign*10
        m.part(side+"推进电机", cyl(5, 15, (-69, y, -6), [1, 0, 0]), TEAL, g, [25, 0, 0])
        m.part(side+"推进轴", cyl(1.2, 35, (-91, y, -6), [1, 0, 0]), SILVER, g)
        m.part(side+"尾轴管", tube(2.7, 1.6, 17, (-94, y, -6), [1, 0, 0]), DARK, g, [0, 0, 12])
        m.part(side+"螺旋桨毂", cyl(2.2, 4, (-110, y, -6), [1, 0, 0]), GOLD, g, [-20, 0, 0])
        for i, a in enumerate(np.linspace(0, 360, 4, endpoint=False)):
            blade = turn(ellipsoid([8, 3, 1], (4, 0, 0)), a)
            m.part(f"{side}螺旋桨叶片 {i+1}", orient(blade, [1, 0, 0]).translate([-110, y, -6]), GOLD, g,
                   [-24, 10*np.cos(np.deg2rad(a)), 10*np.sin(np.deg2rad(a))])
        m.part(side+"舵叶", box([7, 1.4, 10], (-103, y, -9)), SILVER, g, [-5, sign*20, -15])
    g = m.group(root+" / 艏侧推", [65, 0, -60])
    m.part("艏侧推隧道", tube(4.7, 4, 28, (74, 0, -6), [0, 1, 0]), DARK, g)
    m.part("侧推器电机", cyl(3.2, 6, (74, 0, -2)), TEAL, g, [0, 0, 20])
    m.part("侧推器叶轮", orient(gear(3.4, 8, .8), [0, 1, 0]).translate([74, 0, -6]), GOLD, g, [0, 25, 0])

    root = m.group("救生与系泊", [0, 75, 0])
    for sign, side in [(-1, "右舷"), (1, "左舷")]:
        g = m.group(root+f" / {side}救生装置", [0, sign*70, 18])
        for i, x in enumerate([-58, -29, 0, 29]):
            m.part(f"{side}救生艇 {i+1}", ellipsoid([20, 6.5, 5], (x, sign*29, 16)), ORANGE, g,
                   [0, sign*22, i*4])
            davit = union(beam([x, sign*25, 14], [x, sign*25, 22], .6), beam([x, sign*25, 22], [x, sign*30, 22], .6))
            m.part(f"{side}吊艇架 {i+1}", davit, WHITE, g, [0, -sign*12, 18])
        m.part(side+"锚", union(beam([92, sign*13, 2], [92, sign*13, -9], .8),
                               beam([86, sign*13, -6], [98, sign*13, -6], .9)), DARK, g, [25, sign*10, -25])
    return m


def model_y_surface():
    # Approximate cross sections constrained by Tesla's 2020–2024 dimensions,
    # at 1:25 display scale. The curved surface is shared by the actual panels.
    controls = np.asarray([
        [-95.02, 27, 38, 13], [-88, 35, 42, 10], [-70, 38.2, 50, 8],
        [-55.6, 38.42, 57, 7], [-34, 38.4, 63.4, 7], [-10, 37.5, 64.96, 7],
        [15, 36.3, 62, 7], [38, 36.6, 45, 8], [55, 37.5, 38, 9],
        [77, 34, 33, 12], [92, 28, 28, 16], [95.02, 25, 26, 19],
    ])
    x = np.linspace(controls[0, 0], controls[-1, 0], 100)
    values = PchipInterpolator(controls[:, 0], controls[:, 1:], axis=0)(x)
    outer = loft_x([(float(a), float(w), float((top-bottom)/2), float((top+bottom)/2))
                    for a, (w, top, bottom) in zip(x, values, strict=True)])
    inner = loft_x([(float(a), float(w-.9), float((top-bottom)/2-.9), float((top+bottom)/2))
                    for a, (w, top, bottom) in zip(x[1:-1], values[1:-1], strict=True)])
    return outer, outer-inner


def model_y():
    m = Model()
    root = m.group("车身与门盖", [0, 0, 50])
    g = m.group(root+" / 承载车身", [0, 0, 35])
    envelope, skin = model_y_surface()
    for x in [-55.58, 60.02]:
        for y in [-32.72, 32.72]:
            skin = skin-cyl(16.0, 24, (x, y, 15.2), [0, 1, 0])
    masks = {
        "hood": box([47, 58, 36], (66, 0, 45)),
        "hatch": box([19, 58, 44], (-86, 0, 40)),
        "roof": box([62, 55.4, 40], (-16, 0, 65)),
        "windshield": box([29, 56, 35], (29.5, 0, 57.5)),
        "rear-window": box([34, 53, 35], (-64, 0, 55.5)),
        "front-bumper": box([10, 90, 20], (93, 0, 18)),
        "rear-bumper": box([10, 90, 17], (-94, 0, 16.5)),
    }
    panels = {name: skin ^ mask for name, mask in masks.items()}
    door_panels, windows = {}, {}
    for sign in [-1, 1]:
        for name, low, high in [("前", -3, 42), ("后", -55, -5)]:
            door_panels[sign, name] = skin ^ box([high-low, 22, 22], ((low+high)/2, sign*38, 27))
            lo, hi = (-3, 35) if name == "前" else (-68, -5)
            windows[sign, name] = skin ^ box([hi-lo, 17, 27], ((lo+hi)/2, sign*37, 51.5))
    body = skin-union(*panels.values(), *door_panels.values(), *windows.values())
    body = union(body, *[box([172, 3, 5], (0, y, 13)) for y in [-22, 22]],
                 box([3, 68, 6], (0, 0, 18)),
                 *[box([3, 56, 12], (x, 0, 17)) for x in [-80, 80]],
                 box([188, 2, 4], (0, 0, 12)), beam([91, 0, 13], [91, 0, 28], .8))
    body = body ^ envelope
    body = union(body, box([184, 1.5, 2], (0, 0, 20)) ^ envelope)
    m.part("车身焊接结构参考件", body, WHITE, g)
    m.part("地板", box([156, 59, 1.8], (-2, 0, 14)), SILVER, g, [0, 0, -22])
    for sign, side in [(-1, "右"), (1, "左")]:
        m.part(side+"门槛梁", box([100, 3, 6], (-2, sign*32, 18)), STEEL, g, [0, sign*20, 0])
        m.part(side+"B柱", beam([-5, sign*31, 23], [-5, sign*27, 54], 1.2), SILVER, g, [0, sign*25, 0])
    g = m.group(root+" / 前后盖与保险杠", [0, 0, 65])
    m.part("前舱盖", panels["hood"], WHITE, g, [35, 0, 30])
    m.part("尾门", panels["hatch"]-panels["rear-window"], WHITE, g, [-40, 0, 20])
    for x, name, key in [(90, "前", "front-bumper"), (-92, "后", "rear-bumper")]:
        m.part(name+"保险杠蒙皮", panels[key], WHITE, g,
               [np.sign(x)*45, 0, 0])
        m.part(name+"防撞横梁", box([3, 59, 4], (x-np.sign(x)*4, 0, 24)) ^ envelope, STEEL, g,
               [np.sign(x)*30, 0, -12])
    m.part("前保险杠下进气格栅", box([1.2, 40, 2.6], (94, 0, 19)), DARK, g, [50, 0, -10])
    for sign, side in [(-1, "右"), (1, "左")]:
        g = m.group(root+f" / {side}车门与玻璃", [0, sign*52, 45])
        for x, name in [(19.5, "前"), (-30, "后")]:
            panel = door_panels[sign, name]
            m.part(side+name+"车门外板", panel, WHITE, g, [0, sign*20, 0])
            m.part(side+name+"车门内饰板", box([40, 1.4, 21], (x, sign*32.7, 32)), DARK, g, [0, -sign*12, 0])
            m.part(side+name+"门玻璃", windows[sign, name], DARK, g, [0, sign*25, 25])
            m.part(side+name+"车门把手", box([8, 1, 1.1], (x-12, sign*37.5, 35)), DARK, g, [0, sign*34, 0])
            m.part(side+name+"玻璃升降器", union(beam([x-12, sign*33, 25], [x+12, sign*33, 40], .7),
                                                beam([x+12, sign*33, 25], [x-12, sign*33, 40], .7)), SILVER, g,
                   [0, -sign*25, 0])
        m.part(side+"后视镜", ellipsoid([7, 5, 3.4], (30, sign*40, 40)), DARK, g, [20, sign*22, 25])
    g = m.group(root+" / 顶部玻璃", [0, 0, 95])
    m.part("全景玻璃车顶", panels["roof"], DARK, g, [0, 0, 28])
    m.part("前风挡玻璃", panels["windshield"], DARK, g, [30, 0, 12])
    m.part("后风挡玻璃", panels["rear-window"], DARK, g,
           [-32, 0, 12])

    root = m.group("座舱与电器", [0, -70, 35])
    g = m.group(root+" / 五座与仪表", [0, -70, 40])
    for x, rows in [(19, [(-17, "前右"), (17, "前左")]), (-25, [(-18, "后右"), (0, "后中"), (18, "后左")])]:
        for y, name in rows:
            m.part(name+"座椅坐垫", rounded_rect(21, 16, 3, 4, 23).translate([x, y, 0]), DARK, g, [0, y, 10])
            m.part(name+"座椅靠背", rounded_rect(6, 16, 2, 23, 36).translate([x-9, y, 0]), DARK, g,
                   [-12, y, 25])
            m.part(name+"头枕", rounded_rect(5, 10, 2, 6, 50).translate([x-9, y, 0]), DARK, g,
                   [-12, y, 40])
    m.part("仪表台", box([12, 52, 5], (36, 0, 33)), DARK, g, [25, 0, 0])
    m.part("中控屏幕模组", box([1.5, 20, 10], (26, 0, 37)), DARK, g, [35, 0, 20])
    m.part("方向盘", tube(6.5, 5.3, 1.5, (27, 16, 38), [1, 0, 0]), DARK, g, [20, 24, 18])
    g = m.group(root+" / 灯具与低压电源", [30, -70, 0])
    for sign, side in [(-1, "右"), (1, "左")]:
        lamp = turn(ellipsoid([12, 4.3, 1.1]), sign*30).translate([81, sign*24, 29.2])
        m.part(side+"前大灯模组", lamp, GLASS, g, [35, sign*20, 0])
        m.part(side+"尾灯模组", box([2, 19, 2.5], (-90, sign*19, 30)), RED, g, [-35, sign*20, 0])
    m.part("低压蓄电池", box([12, 16, 8], (67, 12, 28)), DARK, g, [25, 0, 15])

    root = m.group("高压电池与热管理", [0, 0, -65])
    g = m.group(root+" / 密封高压电池包", [0, 0, -70])
    m.part("高压电池包总成 · 未虚构内部电芯", rounded_rect(119, 52, 4, 5, 10.7), BLUE, g)
    m.part("电池包底部护板", rounded_rect(125, 56, 3, .8, 7.3), SILVER, g, [0, 0, -20])
    m.part("高压接线与维护接口", box([13, 10, 4], (-63, 0, 10)), ORANGE, g, [-20, 0, 0])
    g = m.group(root+" / 热管理", [60, 0, -50])
    m.part("前冷凝器散热器模组", box([2.5, 46, 14], (77, 0, 24)), DARK, g, [28, 0, 0])
    m.part("冷却风扇", cyl(9, 3, (72, 0, 24), [1, 0, 0]), DARK, g, [10, 0, 0])
    m.part("空调压缩机", cyl(4, 12, (64, -13, 22), [1, 0, 0]), SILVER, g, [0, -25, 0])
    m.part("冷却液循环泵", cyl(3.5, 7, (60, 13, 22), [0, 1, 0]), TEAL, g, [0, 25, 0])
    m.part("冷却液管路", pipe([[60, 13, 22], [60, 25, 15], [-38, 25, 15]], 1), DARK, g, [0, 30, 15])

    root = m.group("双驱动单元与底盘", [0, 0, -35])
    for x, name in [(-55.58, "后"), (60.02, "前")]:
        g = m.group(root+f" / {name}驱动与副车架", [x*.8, 0, -40])
        m.part(name+"副车架", shell_box([33, 52, 5], 1.5, (x, 0, 15)), STEEL, g, [0, 0, -15])
        m.part(name+"驱动单元壳体", tube(8, 6.8, 23, (x, 0, 21), [0, 1, 0]), SILVER, g)
        m.part(name+"驱动电机定子 · 构型近似", tube(6.5, 4.3, 17, (x, 0, 21), [0, 1, 0]), COPPER, g,
               [0, -22, 0], basis="engineering_inference")
        m.part(name+"驱动电机转子 · 构型近似", cyl(4, 18, (x, 0, 21), [0, 1, 0]), STEEL, g,
               [0, 22, 0], basis="engineering_inference")
        m.part(name+"逆变器", box([17, 22, 4], (x, 0, 31)), SILVER, g, [0, 0, 22])
        for sign, side in [(-1, "右"), (1, "左")]:
            m.part(name+side+"半轴", cyl(1.3, 25, (x, sign*23, 17), [0, 1, 0]), STEEL, g, [0, sign*26, 0])

    for x, end in [(-55.58, "后"), (60.02, "前")]:
        for sign, side in [(-1, "右"), (1, "左")]:
            root = m.group(end+side+"轮端与悬架", [x*.25, sign*75, -10])
            g = m.group(root+" / 车轮与制动", [x*.25, sign*88, -10])
            y = sign*32.72
            m.part(end+side+"轮胎", tube(15.2, 9.1, 9, (x, y, 15.2), [0, 1, 0]), RUBBER, g, [0, sign*26, 0])
            spokes = [beam([x, y, 15], [x+7.5*np.cos(a), y, 15+7.5*np.sin(a)], .8)
                      for a in np.linspace(0, 2*np.pi, 5, endpoint=False)]
            rim = union(tube(9, 7.5, 8.6, (x, y, 15), [0, 1, 0]), cyl(2.7, 9, (x, y, 15), [0, 1, 0]), *spokes)
            m.part(end+side+"轮毂", rim, SILVER, g, [0, sign*42, 0])
            m.part(end+side+"制动盘", tube(8, 2.5, 1.1, (x, y-sign*4.8, 15), [0, 1, 0]), STEEL, g, [0, sign*10, 0])
            m.part(end+side+"制动卡钳", box([4, 4, 9], (x+6, y-sign*4.8, 15)), DARK, g, [18, 0, 0])
            m.part(end+side+"轮毂轴承", tube(3.8, 1.5, 4, (x, y-sign*7, 15), [0, 1, 0]), SILVER, g, [0, -sign*12, 0])
            fasteners(m, g, [(x+4*np.cos(a), y+sign*4.8, 15+4*np.sin(a)) for a in np.linspace(0, 2*np.pi, 5, endpoint=False)],
                      .6, 3, [0, sign, 0], end+side+"车轮螺母")
            g = m.group(root+" / 悬架连杆", [x*.25, sign*56, -10])
            m.part(end+side+"转向节", union(box([4, 4, 11], (x, y-sign*10, 17)),
                                          cyl(2.6, 6, (x, y-sign*10, 15), [0, 1, 0])), SILVER, g)
            for i, z in enumerate([12, 27]):
                wishbone = union(beam([x-10, sign*17, z], [x, sign*29, z], .9),
                                 beam([x+10, sign*17, z], [x, sign*29, z], .9))
                m.part(end+side+("下" if i == 0 else "上")+"悬架连杆", wishbone, STEEL, g,
                       [0, -sign*(12+i*15), i*15])
            m.part(end+side+"减振器", tube(2.6, 1.5, 13, (x, sign*26, 22)), DARK, g, [0, 0, 27])
            m.part(end+side+"减振活塞杆", cyl(1.2, 11, (x, sign*26, 25)), SILVER, g, [0, 0, 45])
            # A helical spring is a continuous solid, not a disconnected stack of rings.
            spring = helix(4, 12, .55, 5).translate([x, sign*26, 18])
            m.part(end+side+"悬架螺旋弹簧", spring, STEEL, g, [0, -sign*15, 32])
    return m


def helix(radius, height, wire, turns, samples=100):
    angles = np.linspace(0, 2*np.pi*turns, samples)
    points = np.column_stack([radius*np.cos(angles), radius*np.sin(angles), height*angles/angles[-1]])
    # Consecutive slightly overlapping cylinders make one watertight spring.
    return union(*[beam(a, b, wire) for a, b in pairwise(points)])


def iphone():
    m = Model()
    root = m.group("机身与显示", [0, 0, 5])
    g = m.group(root+" / 外壳与密封", [0, 0, 0])
    frame = rounded_rect(72, 150, 10, 8)-rounded_rect(67, 145, 8, 10)
    m.part("金属中框", frame, SILVER, g, material="钛合金/铝结构 · 外形近似")
    m.part("内部承载板", rounded_rect(66, 144, 7, .5, -2.8), SILVER, g, [0, 0, -8])
    m.part("后盖", rounded_rect(69, 147, 8, .9, -4.5), WHITE, g, [0, 0, -55], material="后玻璃 · 模组")
    seal = rounded_rect(68, 146, 8, .3, 4.15)-rounded_rect(65, 143, 7, 1, 4.15)
    m.part("显示屏周边密封胶框", seal, RUBBER, g, [0, 0, 18], role="seal")
    g = m.group(root+" / 显示模组", [0, 0, 50])
    m.part("显示屏", rounded_rect(68, 146, 8, 1.1, 4.9), DARK, g,
           material="层压显示总成 · 不将界面图案作为零件")
    m.part("显示排线", box([7, 34, .35], (15, -24, 3.2)), COPPER, g, [28, 0, -12])
    m.part("显示连接器压板", box([12, 17, .5], (20, -17, 2.5)), STEEL, g, [36, 0, 0])
    fasteners(m, g, [(16, -12, 3), (24, -22, 3)], .55, 1.8, name="显示压板螺钉")

    root = m.group("电源与主板", [40, 0, 0])
    g = m.group(root+" / 电池与充电", [-50, -15, -15])
    m.part("电池", rounded_rect(43, 91, 4, 3.1).translate([-9, -12, 0]), DARK, g,
           material="密封锂电池 · 不拆电芯")
    m.part("电池连接排线", box([8, 14, .3], (15, -7, .8)), COPPER, g, [25, 0, 0])
    m.part("电池连接器压板", box([9, 10, .5], (20, -7, 2)), STEEL, g, [30, 0, 12])
    m.part("无线充电线圈模组", tube(23, 17, .6, (-4, -7, -3.6)), COPPER, g, [0, 0, -25])
    m.part("MagSafe 环形磁体模组", tube(26, 24, .7, (-4, -7, -3.6)), DARK, g, [0, 0, -38])
    g = m.group(root+" / 逻辑主板与屏蔽", [55, 0, 0])
    board = union(box([12, 98, 1.2], (24, 5, -.1)), box([41, 15, 1.2], (9, 58, -.1)))
    m.part("逻辑主板总成", board, TEAL, g, material="多层 PCB · 不虚构芯片内部")
    for i, (x, y, w, h) in enumerate([(24, 31, 11, 22), (24, 2, 11, 19), (10, 57, 24, 13)]):
        m.part(f"主板屏蔽罩 {i+1}", shell_box([w, h, 1.2], .22, (x, y, 1.1)), SILVER, g,
               [0, 0, 15+i*7])
    fasteners(m, g, [(29, 46, 2), (29, -28, 2), (10, 63, 2)], .5, 2.0, name="主板固定螺钉")

    root = m.group("相机与传感", [0, 45, 0])
    g = m.group(root+" / 后置相机总成", [-35, 55, -15])
    m.part("相机承载支架", rounded_rect(30, 35, 5, 1, -1).translate([-16, 49, 0]), STEEL, g)
    for label, x, y, radius in [("主摄相机模组", -25, 58, 6.5), ("超广角相机模组", -25, 41, 5.8),
                                ("长焦相机模组", -9, 49, 6.1)]:
        module = union(box([radius*2, radius*2, 3], (x, y, -.5)), cyl(radius, 4, (x, y, -3.5)))
        m.part(label, module, DARK, g, [(x+16)*2, (y-49)*2, -18])
        m.part(label+"外部保护玻璃", cyl(radius*.88, .5, (x, y, -5.8)), GLASS, g,
               [(x+16)*2, (y-49)*2, -38], material="保护镜片 · 不拆内部镜组")
    m.part("LiDAR 传感器模组", cyl(2.5, 3.8, (1, 37, -3)), DARK, g, [25, -15, -15])
    m.part("闪光灯模组", cyl(2.5, 3, (1, 62, -3)), WHITE, g, [25, 15, -15])
    fasteners(m, g, [(-32, 65, 2), (-1, 65, 2), (-31, 33, 2)], .55, 2.1, name="相机支架螺钉")
    g = m.group(root+" / TrueDepth 与听筒", [0, 55, 30])
    m.part("TrueDepth 相机总成", box([22, 5.5, 2], (0, 67, 1)), DARK, g)
    m.part("上扬声器", box([20, 4, 2.5], (0, 71, -1)), DARK, g, [0, 18, 0])
    m.part("前置感光排线", box([24, 2, .3], (0, 65, 3.2)), COPPER, g, [0, -12, 14])
    m.part("TrueDepth 固定压板", box([26, 6, .4], (0, 67, 2.5)), SILVER, g, [0, 0, 22])

    root = m.group("音频与接口", [0, -55, -5])
    g = m.group(root+" / 下部维修件", [0, -65, -5])
    m.part("Taptic Engine", rounded_rect(23, 10, 2, 3).translate([-17, -62, 0]), STEEL, g, [-25, 0, 0])
    m.part("下扬声器", rounded_rect(21, 11, 2, 3).translate([17, -62, 0]), DARK, g, [25, 0, 0])
    m.part("USB-C 连接器模组", tube(4.2, 3.1, 8, (0, -71, 0), [0, 1, 0]).scale([1.5, 1, .65]), SILVER, g,
           [0, -25, 0])
    m.part("USB-C 接口排线", box([34, 6, .35], (0, -66, -1.8)), COPPER, g, [0, -12, -15])
    m.part("主麦克风", box([4, 3, 2], (-8, -71, 0)), DARK, g, [-15, -22, 0])
    fasteners(m, g, [(-28, -64, 2), (-6, -64, 2), (7, -65, 2), (27, -65, 2)], .55, 2.3,
              name="下部模组固定螺钉")
    g = m.group(root+" / 按键与卡托", [48, -25, 5])
    for x, y, length, name in [(-36.6, 37, 7, "操作按钮"), (-36.6, 19, 13, "音量上键"),
                                (-36.6, 1, 13, "音量下键"), (36.6, 22, 18, "侧边按钮"),
                                (36.6, -26, 13, "相机控制按钮")]:
        m.part(name, box([1.3, length, 2], (x, y, 0)), SILVER, g, [np.sign(x)*20, 0, 0])
    m.part("SIM 卡托 · 区域版本示意", rounded_rect(3, 18, 1, 1).translate([-35, -31, 0]), SILVER, g,
           [-30, 0, 0], basis="configuration_dependent")
    fasteners(m, g, [(-18, -75, 0), (18, -75, 0)], .6, 2, [0, -1, 0], "底边固定螺钉")
    return m


def rotary_joint(m, group, center, axis, radius, length, prefix="", inferred=False):
    """A generic servo/strain-wave joint; never an asserted brand-specific design."""
    def at(solid):
        return orient(solid, axis).translate(center)

    components = [
        ("壳体", tube(radius, radius*.86, length), DARK, 0),
        ("定子绕组", tube(radius*.82, radius*.56, length*.42, (0, 0, -length*.17)), COPPER, -length*.8),
        ("永磁转子", tube(radius*.52, radius*.22, length*.43, (0, 0, -length*.17)), STEEL, -length*1.6),
        ("输入轴", cyl(radius*.19, length*1.1), SILVER, -length*2.4),
        ("刚轮", gear(radius*.80, 30, length*.14).translate([0, 0, length*.18])-cyl(radius*.61, length), GOLD, length*.9),
        ("柔轮", tube(radius*.60, radius*.48, length*.20, (0, 0, length*.19)), STEEL, length*1.6),
        ("波发生器", ellipsoid([radius*.93, radius*.78, length*.15], (0, 0, length*.19))-cyl(radius*.21, length), SILVER, length*2.3),
        ("输出轴承", tube(radius*.91, radius*.72, length*.13, (0, 0, length*.39)), STEEL, length*3.0),
        ("输出法兰", tube(radius*.87, radius*.26, length*.10, (0, 0, length*.54)), SILVER, length*3.7),
        ("编码器码盘", tube(radius*.45, radius*.22, length*.05, (0, 0, -length*.54)), TEAL, -length*3.0),
        ("制动器摩擦盘", tube(radius*.76, radius*.53, length*.07, (0, 0, -length*.41)), DARK, -length*3.7),
    ]
    for label, solid, color, step in components:
        m.part(prefix+label, at(solid), color, group, np.asarray(axis)*step,
               basis="engineering_inference" if inferred else "generic_component_reference",
               role="actuator_component")


def optimus():
    m = Model()
    root = m.group("头部与躯干", [0, 0, 30])
    g = m.group(root+" / 头部感知", [0, 0, 70])
    shell = ellipsoid([24, 23, 31], (0, 0, 168))-ellipsoid([21, 20, 28], (0, 0, 168))
    m.part("头部后壳", shell ^ box([30, 15, 36], (0, -7.5, 168)), WHITE, g, [0, -25, 10])
    m.part("头部前罩", shell ^ box([30, 15, 36], (0, 7.5, 168)), DARK, g, [0, 30, 10])
    for x, name in [(-4, "右"), (4, "左")]:
        m.part(name+"视觉相机模组 · 推定", union(box([4, 4, 3], (x, 7, 170)), cyl(1.2, 2, (x, 9, 170), [0, 1, 0])), DARK, g,
               [x*4, 20, 0], basis="engineering_inference")
    m.part("头部视觉处理板 · 推定", box([14, 1, 12], (0, -3, 168)), TEAL, g, [0, -20, 0], basis="engineering_inference")
    m.part("颈部轴承", tube(5.8, 3.4, 5, (0, 0, 151)), STEEL, g, [0, 0, -18])
    g = m.group(root+" / 躯干骨架与电源", [0, -40, 18])
    torso = ellipsoid([42, 29, 52], (0, 0, 128))-ellipsoid([38, 25, 48], (0, 0, 128))
    for sign, name in [(1, "胸部前"), (-1, "背部后")]:
        m.part(name+"外壳", torso ^ box([50, 20, 60], (0, sign*10, 128)), WHITE, g, [0, sign*42, 0])
    m.part("躯干承载框架 · 推定", shell_box([27, 19, 38], 2, (0, 0, 128)), STEEL, g,
           [0, -15, 0], basis="engineering_inference")
    m.part("躯干电池包 · 推定", rounded_rect(20, 10, 2, 25, 126).translate([0, -3, 0]), BLUE, g,
           [0, -40, 0], basis="engineering_inference")
    m.part("主控制计算板 · 推定", box([18, 1, 18], (0, 7, 135)), TEAL, g, [0, 25, 0], basis="engineering_inference")
    m.part("电源分配板 · 推定", box([16, 1, 9], (0, 7, 118)), TEAL, g, [0, 25, -15], basis="engineering_inference")
    m.part("骨盆承载壳体 · 推定", shell_box([29, 18, 14], 1.8, (0, 0, 98)), DARK, g,
           [0, 0, -22], basis="engineering_inference")
    m.part("腰部轴承", tube(9, 6.2, 6, (0, 0, 108)), STEEL, g, [0, 0, 18])
    fasteners(m, g, [(x, y, 147) for x in [-10, 10] for y in [-7, 7]], .8, 4, name="躯干壳体紧固件")

    for sign, side in [(-1, "右"), (1, "左")]:
        root = m.group(side+"上肢", [sign*58, 0, 10])
        for name, center, axis, radius, length, vector in [
            ("肩", [sign*24, 0, 142], [1, 0, 0], 7.5, 11, [sign*50, 0, 30]),
            ("肘", [sign*32, 0, 110], [0, 1, 0], 5.8, 10, [sign*65, 0, 0]),
            ("腕", [sign*36, 2, 84], [0, 0, 1], 4, 7, [sign*80, 0, -20]),
        ]:
            g = m.group(root+f" / {name}执行器 · 内部推定", vector)
            rotary_joint(m, g, center, axis, radius, length, side+name, inferred=True)
        g = m.group(root+" / 臂部骨架与护壳", [sign*55, 0, 0])
        for name, a, b, radius in [("上臂", [sign*26, 0, 136], [sign*31, 0, 116], 5.7),
                                   ("前臂", [sign*33, 1, 104], [sign*36, 2, 89], 5.2)]:
            m.part(side+name+"承载连杆 · 推定", beam(a, b, radius*.35), STEEL, g, [0, -20, 0], basis="engineering_inference")
            shell = beam(a, b, radius)-beam(a, b, radius-.9)
            m.part(side+name+"护壳", shell, WHITE, g, [0, 25, 0])
        m.part(side+"上肢线束 · 推定", pipe([[sign*24, -4, 140], [sign*30, -4, 112], [sign*35, -3, 87]], .7), DARK, g,
               [0, -30, 0], basis="engineering_inference")

        root = m.group(side+"灵巧手", [sign*100, 0, -22])
        g = m.group(root+" / 掌部与腱驱动推定", [sign*100, 0, -20])
        m.part(side+"手掌骨架 · 推定", shell_box([10, 6, 11], .8, (sign*36, 2, 77)), STEEL, g,
               [0, -15, 0], basis="engineering_inference")
        m.part(side+"掌背护盖", box([10.6, 1.1, 11.5], (sign*36, -.7, 77)), WHITE, g, [0, -25, 0])
        for digit, dx, length in [("食指", -3.5, 12), ("中指", -1.2, 14), ("无名指", 1.2, 13), ("小指", 3.5, 10)]:
            path = m.group(root+" / "+digit+" · 指节推定", [sign*100+dx*5, 18, -32])
            for j in range(3):
                z = 71-j*length/3
                m.part(f"{side}{digit}第 {j+1} 指骨", box([1.7, 2.5, length/3-.3], (sign*36+dx, 2, z)), SILVER, path,
                       [0, 0, -j*9], basis="engineering_inference")
                m.part(f"{side}{digit}关节销 {j+1}", cyl(.65, 2.5, (sign*36+dx, 2, z+length/6), [1, 0, 0]), DARK, path,
                       [sign*7, 0, -j*9], basis="engineering_inference")
            m.part(side+digit+"驱动腱 · 推定", beam([sign*36+dx, 3.3, 80], [sign*36+dx, 3.3, 60], .25), COPPER, path,
                   [0, 9, 0], basis="engineering_inference")
        path = m.group(root+" / 拇指 · 指节推定", [sign*123, 15, -15])
        for j in range(2):
            a = [sign*(42+j*2.5), 3, 78-j*5]
            b = [sign*(44+j*2.5), 5, 73-j*5]
            m.part(f"{side}拇指第 {j+1} 指骨", beam(a, b, 1.3), SILVER, path, [sign*j*9, 0, -j*6], basis="engineering_inference")
        m.part(side+"拇指关节销", cyl(.7, 3, [sign*42, 3, 79], [0, 1, 0]), DARK, path,
               [0, 12, 0], basis="engineering_inference")

        root = m.group(side+"下肢", [sign*42, 0, -50])
        for name, center, axis, radius, length, vector in [
            ("髋", [sign*10, 0, 93], [1, 0, 0], 7, 12, [sign*35, 0, -25]),
            ("膝", [sign*11, 0, 54], [1, 0, 0], 6.7, 12, [sign*48, 0, -62]),
            ("踝", [sign*12, 1, 16], [1, 0, 0], 4.8, 10, [sign*60, 0, -95]),
        ]:
            path = m.group(root+f" / {name}执行器 · 内部推定", vector)
            rotary_joint(m, path, center, axis, radius, length, side+name, inferred=True)
        g = m.group(root+" / 腿部结构与足底", [sign*38, 0, -50])
        for name, a, b, radius in [("大腿", [sign*10, 0, 86], [sign*11, 0, 61], 7.2),
                                   ("小腿", [sign*11, 0, 47], [sign*12, 0, 23], 6.1)]:
            m.part(side+name+"骨架 · 推定", beam(a, b, 2.1), STEEL, g, [0, -18, 0], basis="engineering_inference")
            m.part(side+name+"护壳", beam(a, b, radius)-beam(a, b, radius-.9), WHITE, g, [0, 23, 0])
        m.part(side+"脚掌承载板 · 推定", rounded_rect(16, 28, 4, 2, 8).translate([sign*12, 7, 0]), SILVER, g,
               [0, 20, -20], basis="engineering_inference")
        m.part(side+"脚底橡胶垫", rounded_rect(16.5, 28.5, 4, 1.3, 5.9).translate([sign*12, 7, 0]), RUBBER, g,
               [0, 20, -33])
        for i, y in enumerate([-1, 15]):
            m.part(f"{side}足部力传感器 {i+1} · 推定", box([9, 6, 1.6], (sign*12, y, 10)), TEAL, g,
                   [0, y, 22], basis="engineering_inference")
    return m


def falcon():
    m = Model()
    root = m.group("一级储箱与结构", [0, 0, -30])
    g = m.group(root+" / 推进剂储箱", [0, 0, -25])
    # Tank sections are hollow, so the engine/feed system remains inspectable.
    m.part("一级 RP-1 煤油箱筒段", tube(9, 8.5, 44, (0, 0, 36)), WHITE, g, [27, 0, -12])
    m.part("一级 LOX 液氧箱筒段", tube(9, 8.5, 75, (0, 0, 96)), WHITE, g, [-27, 0, 12])
    for z, name in [(14, "一级后"), (58.5, "一级共底"), (134, "一级前")]:
        dome = ellipsoid([17, 17, 5], (0, 0, z))-ellipsoid([15.8, 15.8, 4.1], (0, 0, z))
        m.part(name+"封头", dome, SILVER, g, [0, 25, z*.2])
    m.part("一级液氧贯穿输送管", tube(1.2, .85, 114, (0, 0, 71)), SILVER, g, [18, 0, 0])
    m.part("一级外部线缆走线罩", box([1.2, 2.5, 115], (9, 0, 73)), WHITE, g, [28, 0, 0])
    g = m.group(root+" / 推力结构与增压", [0, 0, -45])
    m.part("一级发动机推力承载环", tube(8.9, 6.8, 5, (0, 0, 12)), DARK, g)
    for i, a in enumerate(np.linspace(0, 2*np.pi, 8, endpoint=False)):
        m.part(f"发动机承力辐条 {i+1}", beam([0, 0, 12], [7*np.cos(a), 7*np.sin(a), 12], .6), STEEL, g,
               [12*np.cos(a), 12*np.sin(a), -8])
    for i, a in enumerate(np.linspace(0, 2*np.pi, 3, endpoint=False)):
        m.part(f"一级增压气瓶 {i+1} · 布置近似", ellipsoid([3, 3, 10], [5*np.cos(a), 5*np.sin(a), 119]), DARK, g,
               [18*np.cos(a), 18*np.sin(a), 20], basis="engineering_inference")

    root = m.group("一级九发动机", [0, 0, -75])
    centers = [(0, 0)]+[(6*np.cos(a), 6*np.sin(a)) for a in np.linspace(0, 2*np.pi, 8, endpoint=False)]
    for i, (x, y) in enumerate(centers):
        g = m.group(root+f" / Merlin {i+1:02d}", [x*6, y*6, -85])
        m.part(f"Merlin {i+1:02d} 喷管", conical_shell(2.4, .9, 7, .35).translate([x, y, -7]), DARK, g,
               [0, 0, -16], role="first_stage_engine_nozzle")
        m.part(f"Merlin {i+1:02d} 燃烧室", tube(1.15, .7, 3, (x, y, 1.5)), SILVER, g,
               [0, 0, 8], basis="engineering_inference")
        m.part(f"Merlin {i+1:02d} 喷注器头", cyl(1.35, .7, (x, y, 3.5)), COPPER, g,
               [0, 0, 18], basis="engineering_inference")
        m.part(f"Merlin {i+1:02d} 涡轮泵 · 形状近似", cyl(1, 3, (x+1.8, y, 4.7), [0, 1, 0]), STEEL, g,
               [9, 0, 13], basis="engineering_inference")
        m.part(f"Merlin {i+1:02d} 燃气发生器 · 近似", cyl(.6, 2.5, (x-1.5, y, 4.5)), GOLD, g,
               [-9, 0, 13], basis="engineering_inference")
        m.part(f"Merlin {i+1:02d} 供给管路 · 近似", pipe([[x+1.8, y, 4.7], [x+1.8, y, 2], [x, y, 2]], .45, .15), COPPER, g,
               [0, 10, 10], basis="engineering_inference")
        m.part(f"Merlin {i+1:02d} 万向安装环", tube(1.8, 1.4, .7, (x, y, 6.5)), DARK, g,
               [0, 0, 25], basis="engineering_inference")

    root = m.group("回收组件", [0, 0, 15])
    for i, a in enumerate(np.linspace(0, 2*np.pi, 4, endpoint=False)):
        direction = np.array([np.cos(a), np.sin(a), 0])
        g = m.group(root+f" / 着陆腿 {i+1}", (direction*43).tolist())
        m.part(f"着陆腿 {i+1} 主支柱", beam(direction*9+[0, 0, 44], direction*17+[0, 0, 5], .9), DARK, g,
               direction*12, role="landing_leg_primary")
        m.part(f"着陆腿 {i+1} 斜撑", beam(direction*9+[0, 0, 13], direction*17+[0, 0, 5], .65), STEEL, g,
               direction*20+[0, 0, -8])
        m.part(f"着陆腿 {i+1} 铰接销", cyl(1.1, 3, direction*9+[0, 0, 44], [-np.sin(a), np.cos(a), 0]), SILVER, g,
               direction*8+[0, 0, 18])
        m.part(f"着陆腿 {i+1} 脚垫", box([4, 4, .8], direction*17+[0, 0, 4]), DARK, g,
               direction*22+[0, 0, -18])
        g = m.group(root+f" / 栅格舵 {i+1}", (direction*47+[0, 0, 28]).tolist())
        frame = box([10, 10, 1.4])-box([8.8, 8.8, 3])
        bars = [box([9, .45, 1.0], (0, v, 0)) for v in [-3, 0, 3]]
        bars += [box([.45, 9, 1.0], (v, 0, 0)) for v in [-3, 0, 3]]
        grid = turn(union(frame, *bars), math.degrees(a)).translate((direction*14+[0, 0, 128]).tolist())
        m.part(f"栅格舵 {i+1} 整体栅格", grid, STEEL, g, direction*18, role="grid_fin")
        m.part(f"栅格舵 {i+1} 转轴", cyl(.9, 6, direction*10+[0, 0, 128], direction), SILVER, g,
               direction*-10)
        m.part(f"栅格舵 {i+1} 作动器 · 近似", cyl(1.5, 5, direction*7+[0, 0, 128]), DARK, g,
               [0, 0, -20], basis="engineering_inference")

    root = m.group("级间与二级", [0, 0, 75])
    g = m.group(root+" / 级间与分离", [0, 0, 55])
    m.part("复合材料级间筒", tube(9.2, 8.3, 15, (0, 0, 143)), DARK, g, [25, 0, 0])
    m.part("级间前连接环", tube(9.3, 7.7, 1.2, (0, 0, 151)), SILVER, g, [0, 0, 18])
    m.part("级间后连接环", tube(9.3, 7.7, 1.2, (0, 0, 135)), SILVER, g, [0, 0, -18])
    g = m.group(root+" / 二级储箱与发动机", [0, 0, 103])
    m.part("二级 RP-1 箱筒段", tube(8.8, 8.25, 16, (0, 0, 163)), WHITE, g, [24, 0, 0])
    m.part("二级 LOX 箱筒段", tube(8.8, 8.25, 24, (0, 0, 183.5)), WHITE, g, [-24, 0, 0])
    m.part("二级共底", cyl(8.3, .7, (0, 0, 171.5)), SILVER, g, [0, 22, 0])
    m.part("二级前封头", ellipsoid([16.6, 16.6, 4], (0, 0, 196))-ellipsoid([15.6, 15.6, 3.4], (0, 0, 196)), SILVER, g,
           [0, 0, 24])
    m.part("二级液氧输送管", tube(.95, .6, 35, (0, 0, 176)), SILVER, g, [14, 0, 10])
    m.part("MVac 真空喷管", conical_shell(7.4, 1.8, 13, .4).translate([0, 0, 140]), DARK, g,
           [35, 0, -20], role="second_stage_engine_nozzle")
    m.part("MVac 燃烧室 · 近似", tube(2, 1.4, 4, (0, 0, 155)), SILVER, g,
           [35, 0, 0], basis="engineering_inference")
    m.part("二级航电安装板", cyl(7.5, .8, (0, 0, 198)), TEAL, g, [0, 0, 35])
    for i, a in enumerate(np.linspace(0, 2*np.pi, 3, endpoint=False)):
        m.part(f"二级航电盒 {i+1} · 近似", box([3, 3, 2], (4*np.cos(a), 4*np.sin(a), 200)), DARK, g,
               [13*np.cos(a), 13*np.sin(a), 45], basis="engineering_inference")

    root = m.group("载荷与整流罩", [0, 0, 160])
    g = m.group(root+" / 整流罩双瓣", [0, 0, 165])
    matrix = trimesh.geometry.align_vectors([1, 0, 0], [0, 0, 1])
    fairing = hollow_loft([(200, 10, 10, 0), (207, 13, 13, 0), (231, 13, 13, 0),
                           (245, 7, 7, 0), (251, .5, .5, 0)], .65).transform(matrix[:3].tolist())
    for sign, side in [(-1, "右"), (1, "左")]:
        m.part(side+"半整流罩", fairing ^ box([25, 40, 70], (sign*12.5, 0, 224)), WHITE, g,
               [sign*45, 0, 20])
    g = m.group(root+" / 载荷接口", [0, 0, 195])
    m.part("载荷适配器", conical_shell(7.7, 5, 7, .6).translate([0, 0, 201]), SILVER, g)
    m.part("载荷分离环", tube(5.5, 4.6, 1.1, (0, 0, 208.5)), GOLD, g, [0, 0, 20])
    m.part("占位卫星载荷 · 非标准箭体部件", box([10, 10, 14], (0, 0, 217)), TEAL, g,
           [0, 0, 45], basis="illustrative_payload", role="payload_proxy")
    return m


def spiral(outer, inner, turns, width, height):
    angles = np.linspace(0, turns*2*np.pi, 180)
    radii = np.linspace(inner, outer, len(angles))
    a = np.column_stack([(radii+width/2)*np.cos(angles), (radii+width/2)*np.sin(angles)])
    b = np.column_stack([(radii-width/2)*np.cos(angles), (radii-width/2)*np.sin(angles)])
    return prism(np.concatenate([a, b[::-1]]), height)


def mechanical_watch():
    m = Model()
    root = m.group("表壳与密封", [0, 0, 0])
    g = m.group(root+" / 壳镜与底盖", [0, 0, 0])
    case = tube(29, 24.6, 10)
    lugs = [box([5, 10, 7], (x, y, 0)) for x in [-11.5, 11.5] for y in [-28, 28]]
    m.part("中层表壳与表耳", union(case, *lugs), GOLD, g)
    bezel = tube(30.5, 24.5, 3.5, (0, 0, 7))
    # Fluting is surface geometry on one physical bezel, never 60 fake parts.
    ridges = [cyl(.65, 3.5, (29.9*np.cos(a), 29.9*np.sin(a), 7), sections=12)
              for a in np.linspace(0, 2*np.pi, 60, endpoint=False)]
    m.part("三角坑纹表圈", union(bezel, *ridges), GOLD, g, [0, 0, 35])
    m.part("蓝宝石表镜", cyl(24.2, .8, (0, 0, 9.5)), GLASS, g, [0, 0, 60], appearance=GLASS_PROXY)
    m.part("表镜密封圈", tube(24.7, 24.0, .8, (0, 0, 8.9)), RUBBER, g, [0, 0, 48], role="seal")
    m.part("旋入式底盖", union(cyl(25, 1.3, (0, 0, -5.9)), tube(24, 22.8, 1.5, (0, 0, -4.7))), SILVER, g,
           [0, 0, -95])
    m.part("底盖密封圈", tube(24.7, 24.0, .7, (0, 0, -5)), RUBBER, g, [0, 0, -80], role="seal")
    g = m.group(root+" / 表冠与柄轴", [40, 0, 0])
    m.part("表冠", cyl(3.6, 5, (32, 0, 1), [1, 0, 0]), GOLD, g, [30, 0, 0])
    m.part("表冠管", tube(2, 1.1, 7, (27, 0, 1), [1, 0, 0]), SILVER, g, [15, 0, 0])
    m.part("上链柄轴", cyl(.8, 22, (18, 0, 1), [1, 0, 0]), STEEL, g)
    m.part("表冠密封圈", tube(2.1, 1.6, .6, (29, 0, 1), [1, 0, 0]), RUBBER, g, [22, 0, 0], role="seal")

    root = m.group("表盘与日历", [0, 0, 80])
    g = m.group(root+" / 显示与三针", [0, 0, 85])
    dial = cyl(24, .8, (0, 0, 5.5))-box([6, 4.5, 2], (17, 0, 5.5))
    m.part("绿色表盘", dial, [24, 78, 65], g)
    for i, a in enumerate(np.arange(0, 360, 30)):
        if i == 3:
            continue
        m.part(f"镶嵌时标 {i+1:02d}", turn(box([1.4, 4.4, .5], (0, 20.5, 6.2)), -a), GOLD, g,
               [8*np.sin(np.deg2rad(a)), 8*np.cos(np.deg2rad(a)), 18])
    for name, size, center, angle, z in [("时针", [1.8, 15, .4], [0, 6, 6.8], 55, 28),
                                         ("分针", [1.1, 20, .35], [0, 9, 7.3], -55, 40),
                                         ("秒针", [.45, 24, .3], [0, 8, 7.8], 15, 52)]:
        m.part(name, union(turn(box(size, center), angle), tube(1.1, .55, size[2], (0, 0, center[2]))), GOLD, g,
               [0, 0, z])
    g = m.group(root+" / 日期机构", [0, 0, 55])
    m.part("日历环 · 刻字不独立计件", tube(22.8, 17.5, .6, (0, 0, 4)), WHITE, g)
    m.part("日历驱动轮", gear(4.4, 24, .6).translate([12, -6, 3]), GOLD, g, [20, -8, 0])
    m.part("日历定位杆", union(beam([15, 8, 3], [20, 3, 3], .5), cyl(1, .8, [15, 8, 3])), STEEL, g,
           [25, 12, 5])
    m.part("日历定位弹簧", spiral(2.1, .6, 1.2, .18, .35).translate([15, 8, 3.6]), BLUE, g, [25, 12, 16])

    root = m.group("机械机芯", [0, 0, -40])
    g = m.group(root+" / 主夹板与桥板", [0, 0, -35])
    mainplate = cyl(23.4, 1.1, (0, 0, -1.6))-cyl(5.7, 3, (3, -12, -1.6))
    m.part("机芯主夹板 · 通用轮廓", mainplate, SILVER, g, [0, 0, -30])
    for name, x, y, sx, sy, z in [("轮系桥板", 3, 2, 25, 12, 2.0), ("发条盒桥板", -11, 9, 17, 10, 2.4),
                                  ("摆轮双支承桥板", 2, -12, 18, 3, 2.4), ("自动上链夹板", -8, -3, 14, 9, -2.8)]:
        bridge = rounded_rect(sx, sy, 2, .8, z).translate([x, y, 0])
        bridge = bridge-cyl(1, 3, (x, y, z))
        m.part(name, bridge, SILVER, g, [x*.8, y*.8, 22 if z > 0 else -38])
    fasteners(m, g, [(-18, 10, 3), (-5, 11, 3), (12, 7, 3), (-7, -12, 3), (11, -12, 3)], .55, 2,
              name="夹板固定螺钉")
    g = m.group(root+" / 发条盒", [-25, 15, -45])
    m.part("发条盒鼓", union(tube(8.2, 7.6, 2.7, (-10, 9, .1)), cyl(8.2, .5, (-10, 9, -1.2))), GOLD, g)
    m.part("发条盒盖", cyl(8.2, .4, (-10, 9, 1.7)), SILVER, g, [0, 0, 18])
    m.part("主发条", spiral(7.0, 1.5, 4, .35, .7).translate([-10, 9, .1]), STEEL, g, [0, 0, 30])
    m.part("发条轴", cyl(1.2, 4.5, (-10, 9, .1)), SILVER, g, [0, 0, -20])
    m.part("发条盒齿圈", (gear(8.9, 40, .7)-cyl(7.1, 2)).translate([-10, 9, -.7]), GOLD, g, [0, 0, -33])

    g = m.group(root+" / 走时轮系", [22, 0, -50])
    for i, (name, x, y, radius, teeth) in enumerate([
        ("中心轮", -1, 1, 5.6, 32), ("三轮", 7.5, 5, 4.3, 28),
        ("四轮", 10, -3.2, 3.8, 26), ("擒纵轮", 5.7, -9.1, 2.8, 15),
    ]):
        wheel = gear(radius, teeth, .45)-cyl(radius*.52, 1)
        spokes = [beam([0, 0, 0], [radius*.74*np.cos(a), radius*.74*np.sin(a), 0], .35)
                  for a in np.linspace(0, 2*np.pi, 5, endpoint=False)]
        wheel = union(wheel, *spokes, cyl(.8, .45))
        m.part(name+"轮片", wheel.translate([x, y, .0]), GOLD, g, [x*1.5, y*1.5, -i*10])
        m.part(name+"小钢轮与轴", union(cyl(.5, 4), gear(1.3, 10, .9).translate([0, 0, -.5])).translate([x, y, 0]), STEEL, g,
               [x*1.5, y*1.5, -35-i*10])
        for z, side in [(-1.8, "下"), (2, "上")]:
            m.part(name+side+"轴承宝石", tube(1.05, .35, .45, (x, y, z)), RED, g,
                   [x*1.5, y*1.5, 22 if z > 0 else -65], material="合成红宝石")
    g = m.group(root+" / 擒纵与振荡", [0, -35, -55])
    fork = union(beam([4.3, -11.8, 1], [3.2, -15.2, 1], .45),
                 beam([4.3, -11.8, 1], [7, -11, 1], .45), beam([4.3, -11.8, 1], [2, -10, 1], .45))
    m.part("擒纵叉 · 通用拓扑", fork, STEEL, g, [15, -10, 15])
    for x, y, name in [(7, -11, "进瓦"), (2, -10, "出瓦")]:
        m.part("擒纵叉"+name, box([.8, .65, .45], (x, y, 1)), RED, g, [x*3, -12, 28])
    balance = union(tube(5.3, 4.6, .6), box([9.4, .65, .5]), cyl(.8, .8))
    m.part("摆轮", balance.translate([0, -14, .2]), GOLD, g, [-15, -15, 0])
    m.part("摆轴", cyl(.4, 4.5, (0, -14, .2)), STEEL, g, [-15, -15, -20])
    m.part("游丝 · 曲线近似", spiral(4.2, .8, 3.5, .12, .18).translate([0, -14, 1.1]), BLUE, g,
           [-15, -15, 30])
    m.part("游丝内桩", tube(.9, .45, .5, (0, -14, 1.1)), SILVER, g, [-15, -15, 42])
    for z, side in [(-2, "下"), (2.5, "上")]:
        m.part(side+"避震宝石座", tube(1.8, 1.1, .55, (0, -14, z)), GOLD, g, [-15, -15, 58 if z > 0 else -40])
        m.part(side+"摆轴托钻", cyl(1, .35, (0, -14, z)), RED, g, [-15, -15, 68 if z > 0 else -50])
    g = m.group(root+" / 自动上链与拨针", [-25, 0, -85])
    rotor = tube(21, 14, 1, (0, 0, -3.5)) ^ box([46, 23, 3], (0, -11.5, -3.5))
    rotor = union(rotor, beam([0, 0, -3.5], [-17, -6, -3.5], .9), beam([0, 0, -3.5], [17, -6, -3.5], .9))
    m.part("自动上链摆陀", rotor, GOLD, g, [0, 0, -35])
    m.part("摆陀轴承", tube(2.7, 1.1, 1.2, (0, 0, -3.5)), STEEL, g, [0, 0, -50])
    for i, (x, y) in enumerate([(-10, -4), (-4, -7)]):
        m.part(f"自动上链换向轮 {i+1}", gear(3.1, 20, .6).translate([x, y, -2.4]), GOLD, g,
               [x, y, -65-i*12])
    m.part("上链离合轮", orient(gear(2.4, 14, 1.2), [1, 0, 0]).translate([14, 0, 1]), STEEL, g, [25, 0, 10])
    m.part("拨针离合杆", box([10, 1, .5], (10, -3, 2)), SILVER, g, [25, -10, 25])

    root = m.group("表链与表扣", [0, 0, -10])
    for sign, side in [(-1, "下"), (1, "上")]:
        g = m.group(root+f" / {side}段三排表链", [0, sign*55, -12])
        m.part(side+"端链", rounded_rect(21, 8, 1.5, 4).translate([0, sign*31, -1]), GOLD, g,
               [0, sign*12, 0])
        for i in range(6):
            width, y, z = 21-i*.6, sign*(39+i*7), -1-i*.4
            for x, fraction, name in [(-width*.36, .25, "右侧"), (0, .42, "中间"), (width*.36, .25, "左侧")]:
                m.part(f"{side}链节 {i+1:02d} {name}块", rounded_rect(width*fraction, 6.2, 1, 3, z).translate([x, y, 0]), GOLD, g,
                       [x*1.7, sign*i*4, 0])
            m.part(f"{side}链节连接销 {i+1:02d}", cyl(.5, width-1, (0, y, z), [1, 0, 0]), STEEL, g,
                   [30, sign*i*4, -10], role="fastener")
    g = m.group(root+" / 折叠表扣", [0, -75, -45])
    m.part("表扣外盖", shell_box([18, 18, 3], .8, (0, -88, -4)), GOLD, g)
    m.part("表扣内折叠片", box([14, 17, .8], (0, -86, -5)), SILVER, g, [0, 0, -18])
    m.part("表扣外折叠片", box([14, 17, .8], (0, -86, -6)), SILVER, g, [0, -12, -32])
    m.part("表扣保险扣", shell_box([20, 6, 3], .7, (0, -80, -4)), GOLD, g, [0, 15, 12])
    return m


def refinery():
    m = Model()
    root = m.group("基础与钢结构", [0, 0, -25])
    g = m.group(root+" / 基础与管廊", [0, 0, -25])
    m.part("工艺单元基础", box([190, 140, 4]), SILVER, g)
    for i, (x, y) in enumerate([(x, y) for x in [-5, 34, 73] for y in [-56, -12]]):
        m.part(f"管廊立柱 {i+1}", box([2, 2, 34], (x, y, 19)), BLUE, g, [0, y*.3, 10])
    for i, x in enumerate([-5, 34, 73]):
        m.part(f"管廊横梁 {i+1}", box([2, 46, 2], (x, -34, 37)), BLUE, g, [0, 0, 25])

    root = m.group("常压蒸馏与侧线塔", [0, 50, 20])
    for k, (name, x, y, radius, height, trays) in enumerate([
        ("常压蒸馏塔", -55, 31, 10, 86, 10), ("侧线汽提塔", -24, 34, 7.5, 68, 7),
        ("稳定塔", 3, 36, 6.5, 55, 6),
    ]):
        g = m.group(root+f" / {name}压力边界", [k*18-15, 60+k*22, 15])
        m.part(name+"焊接筒体", tube(radius, radius-.7, height, (x, y, height/2+5)), WHITE, g,
               [-30, 0, 0], material="钢制焊接壳体 · 不按任意平面切块")
        for z, side in [(5, "下"), (height+5, "上")]:
            cap = ellipsoid([radius*2, radius*2, radius], (x, y, z))-ellipsoid([radius*2-1.4, radius*2-1.4, radius-1.2], (x, y, z))
            cap = cap ^ box([radius*3, radius*3, radius], (x, y, z+(-1 if side == "下" else 1)*radius/2))
            m.part(name+side+"封头", cap, SILVER, g, [0, 0, -20 if side == "下" else 30])
        m.part(name+"裙座", tube(radius*.94, radius*.80, 5, (x, y, 2.5)), STEEL, g, [0, 0, -20])
        g = m.group(root+f" / {name}塔内件", [k*18+20, 60+k*22, 35])
        for i, z in enumerate(np.linspace(16, height-4, trays)):
            tray = cyl(radius-.9, .45, (x, y, z))-box([radius*.55, radius*2, 1], (x+radius*.72, y, z))
            # One perforated tray is one manufactured plate; holes are geometry.
            holes = [cyl(.45, 1, (x+dx, y+dy, z), sections=12) for dx in [-3, 0, 3] for dy in [-3, 0, 3]]
            tray = tray-union(*holes)
            m.part(f"{name}取样筛板塔盘 {i+1:02d}", tray, STEEL, g, [0, 0, i*9])
            m.part(f"{name}降液板 {i+1:02d}", box([.5, radius*1.3, 4], (x+radius*.6, y, z-2)), GOLD, g,
                   [15, 0, i*9])
        for i, z in enumerate(np.arange(22, height, 24)):
            m.part(f"{name}检修平台 {i+1}", tube(radius+3.5, radius+.4, .9, (x, y, z)), GOLD, g,
                   [-18, 0, i*12])
        m.part(name+"直梯", ladder([x+radius+2, y-2.5, 4], height-4, 5), GOLD, g, [-25, 0, 10])
        m.part(name+"进料接管", tube(1.9, 1.3, 10, (x, y-radius-4, height*.44), [0, 1, 0]), COPPER, g,
               [0, -25, 0])
        m.part(name+"顶出料接管", tube(1.5, 1, 9, (x, y, height+9)), COPPER, g, [0, 0, 45])

    root = m.group("换热与加热", [-50, -40, 10])
    for k, y in enumerate([-22, -47]):
        g = m.group(root+f" / 管壳式换热器 {k+1}", [-70-k*20, -40-k*25, 15])
        x, z = -49, 14
        m.part(f"换热器 {k+1} 壳程筒体", tube(7, 6.3, 42, (x, y, z), [1, 0, 0]), DARK, g, [0, -22, 0])
        for j, end in enumerate([-1, 1]):
            m.part(f"换热器 {k+1} 管箱端盖 {j+1}", cyl(7.3, 2, (x+end*23, y, z), [1, 0, 0]), SILVER, g,
                   [end*30, 0, 0])
            m.part(f"换热器 {k+1} 管板 {j+1}", cyl(6.2, .8, (x+end*20.2, y, z), [1, 0, 0]), STEEL, g,
                   [end*15, 0, 0])
        for j, (dy, dz) in enumerate([(a, b) for a in [-3, 0, 3] for b in [-3, 0, 3]]):
            m.part(f"换热器 {k+1} 取样传热管 {j+1:02d}", tube(.65, .4, 40, (x, y+dy, z+dz), [1, 0, 0]), COPPER, g,
                   [0, dy*4, dz*4])
        for j, dx in enumerate([-13, 0, 13]):
            baffle = cyl(6, .6, (x+dx, y, z), [1, 0, 0]) ^ box([2, 15, 10], (x+dx, y, z+(2 if j % 2 else -2)))
            m.part(f"换热器 {k+1} 折流板 {j+1}", baffle, STEEL, g, [dx*.5, 0, 25+j*8])
        for j, dx in enumerate([-14, 14]):
            m.part(f"换热器 {k+1} 鞍座 {j+1}", box([4, 17, 7], (x+dx, y, 6)), SILVER, g, [0, 0, -18])
    g = m.group(root+" / 原油加热炉", [55, -65, 10])
    m.part("加热炉外壳", shell_box([30, 29, 29], 1, (49, -33, 19)), WHITE, g, [30, 0, 0])
    m.part("辐射炉管盘管", pipe([[38, -40, 9], [60, -40, 9], [60, -40, 28], [38, -40, 28],
                              [38, -27, 28], [60, -27, 28], [60, -27, 9], [38, -27, 9]], 1.2), COPPER, g,
           [-20, 0, 20])
    m.part("烟囱", tube(4, 3.3, 40, (49, -33, 54)), DARK, g, [0, 0, 40])
    for i, x in enumerate([42, 56]):
        m.part(f"炉底燃烧器 {i+1}", cyl(2.4, 4, (x, -33, 4)), DARK, g, [0, 0, -20])

    root = m.group("储运与泵阀", [65, 0, 5])
    for i, x in enumerate([40, 76]):
        g = m.group(root+f" / 储罐 {i+1}", [65+i*30, 30, 10])
        m.part(f"储罐 {i+1} 筒体", tube(15, 14.3, 24, (x, 32, 16)), SILVER, g, [0, 25, 0])
        m.part(f"储罐 {i+1} 底板", cyl(15, .8, (x, 32, 3.5)), STEEL, g, [0, 0, -15])
        m.part(f"储罐 {i+1} 顶盖", conical_shell(15, 1, 5, .5).translate([x, 32, 28]), WHITE, g, [0, 0, 30])
        m.part(f"储罐 {i+1} 人孔盖", cyl(2.5, .7, (x+8, 32, 30)), GOLD, g, [15, 0, 40])
    for i, (x, y) in enumerate([(17, -48), (70, -56)]):
        g = m.group(root+f" / 离心泵 {i+1}", [70+i*25, -35-i*20, 5])
        m.part(f"泵 {i+1} 底座", box([23, 11, 1.5], (x, y, 4)), STEEL, g, [0, 0, -12])
        m.part(f"泵 {i+1} 电动机", cyl(3.7, 11, (x+5, y, 9), [1, 0, 0]), TEAL, g, [22, 0, 0])
        m.part(f"泵 {i+1} 联轴器", tube(2, .8, 4, (x-3, y, 9), [1, 0, 0]), SILVER, g, [10, 0, 15])
        m.part(f"泵 {i+1} 泵壳", tube(4.7, 3.8, 4, (x-8, y, 9), [1, 0, 0]), DARK, g, [-15, 0, 0])
        m.part(f"泵 {i+1} 叶轮", orient(gear(3.5, 8, 1.2), [1, 0, 0]).translate([x-8, y, 9]), GOLD, g, [-30, 0, 0])
        m.part(f"泵 {i+1} 端盖", cyl(4.8, .7, (x-10.5, y, 9), [1, 0, 0]), SILVER, g, [-45, 0, 0])
        m.part(f"泵 {i+1} 机械密封", tube(1.8, 1.0, 1.5, (x-5, y, 9), [1, 0, 0]), RUBBER, g, [-30, 0, 15])
    root = m.group("工艺管线", [0, -65, 35])
    g = m.group(root+" / 管段与法兰", [0, -75, 40])
    for i, y in enumerate([-47, -34, -21]):
        m.part(f"管廊工艺管段 {i+1}", tube(1.5, 1.1, 85, (34, y, 41), [1, 0, 0]), ORANGE, g,
               [0, -i*15, i*10])
        for x in [-8.5, 76.5]:
            m.part(f"管段 {i+1} {'入口' if x < 0 else '出口'}法兰", tube(2.5, 1.1, 1, (x, y, 41), [1, 0, 0]), SILVER, g,
                   [np.sign(x)*15, -i*15, i*10])
    g = m.group(root+" / 隔离与调节阀", [25, -95, 60])
    for i, (x, y) in enumerate([(10, -47), (40, -34), (62, -21)]):
        m.part(f"阀门 {i+1} 阀体", tube(3.1, 1.5, 6, (x, y, 41), [1, 0, 0]), DARK, g)
        m.part(f"阀门 {i+1} 阀芯", cyl(1.4, 5, (x, y, 41)), GOLD, g, [0, 0, 15])
        m.part(f"阀门 {i+1} 阀杆", cyl(.65, 7, (x, y, 46)), STEEL, g, [0, 0, 25])
        m.part(f"阀门 {i+1} 手轮", tube(3.2, 2.5, .8, (x, y, 50)), RED, g, [0, 0, 37])
    return m


def ladder(start, height, width):
    x, y, z = start
    pieces = [beam([x, y, z], [x, y, z+height], .45),
              beam([x, y+width, z], [x, y+width, z+height], .45)]
    pieces += [beam([x, y, z+dz], [x, y+width, z+dz], .35) for dz in np.arange(3, height, 5)]
    return union(*pieces)


def industrial_arm():
    m = Model()
    root = m.group("机架与连杆", [0, 0, 0])
    g = m.group(root+" / 基座与承载连杆", [0, 0, 0])
    base = union(cyl(27, 7, (0, 0, 3.5)), box([46, 46, 3], (0, 0, .5)))
    m.part("安装底座", base, DARK, g, [0, 0, -35])
    m.part("底座旋转承载壳", tube(20, 17, 16, (0, 0, 19)), ORANGE, g, [0, 0, 25])
    for i, (a, b, radius, name) in enumerate([
        ([0, 0, 43], [28, 0, 94], 10, "大臂"),
        ([30, 0, 96], [81, 0, 116], 8, "前臂"),
    ]):
        shell = beam(a, b, radius)-beam(a, b, radius-2)
        m.part(name+"承载壳体", shell, ORANGE, g, [-25 if i == 0 else 25, 0, 35+i*15])
        m.part(name+"内部线缆导管", beam(a, b, 1.5)-beam(a, b, .9), DARK, g, [0, -20, 30])
    fasteners(m, g, [(x, y, 4) for x in [-17, 17] for y in [-17, 17]], 1.8, 7, name="底座地脚螺栓")

    joints = [
        ("J1 基座回转", [0, 0, 19], [0, 0, 1], 16, 14, [-45, 0, -5]),
        ("J2 肩关节", [0, 0, 42], [0, 1, 0], 13, 22, [-40, -65, 20]),
        ("J3 肘关节", [28, 0, 94], [0, 1, 0], 11, 20, [0, -70, 50]),
        ("J4 前臂回转", [83, 0, 116], [1, 0, 0], 8.2, 15, [55, 0, 60]),
        ("J5 腕部俯仰", [100, 0, 114], [0, 1, 0], 7.2, 13, [70, -50, 35]),
        ("J6 工具回转", [102, 0, 101], [0, 0, 1], 6.3, 10, [85, 0, -5]),
    ]
    for name, center, axis, radius, length, vector in joints:
        root = m.group(name, vector)
        g = m.group(root+" / 伺服与减速器", vector)
        rotary_joint(m, g, center, axis, radius, length, name+" · ")
        g = m.group(root+" / 端盖与连接件", (np.asarray(vector)+np.asarray(axis)*length*2).tolist())
        endcap = tube(radius*.98, radius*.20, length*.06)
        m.part(name+"端盖", orient(endcap, axis).translate(np.asarray(center)-np.asarray(axis)*length*.57), ORANGE, g,
               np.asarray(axis)*(-length*4))
        # Positions are generic flange bolt patterns, not a manufacturer drawing.
        for i, angle in enumerate(np.linspace(0, 2*np.pi, 4, endpoint=False)):
            point = [radius*.78*np.cos(angle), radius*.78*np.sin(angle), length*.61]
            solid = orient(bolt(radius*.055, length*.18, point), axis).translate(center)
            m.part(f"{name} 法兰螺钉 {i+1}", solid, STEEL, g, np.asarray(axis)*(length*1.5), role="fastener")

    root = m.group("末端平行夹爪", [100, 0, -45])
    g = m.group(root+" / 夹爪驱动", [100, 0, -45])
    m.part("夹爪基座", shell_box([25, 17, 8], 1.3, (102, 0, 90)), ORANGE, g)
    m.part("夹爪上盖", box([25, 17, 1], (102, 0, 94.5)), DARK, g, [0, 0, 16])
    m.part("夹爪微型电机", cyl(3, 10, (102, -3, 89), [0, 1, 0]), DARK, g, [0, -22, 0])
    m.part("同步传动齿轮", gear(4.7, 18, 1.3).translate([102, 0, 88]), GOLD, g, [0, 0, -18])
    m.part("夹爪导轨", box([22, 2, 2], (102, 3, 88)), SILVER, g, [0, 15, 0])
    for sign, side in [(-1, "右"), (1, "左")]:
        g = m.group(root+f" / {side}滑块与夹指", [100+sign*25, 0, -55])
        m.part(side+"滑块", box([8, 6, 3], (102+sign*6, 3, 87)), SILVER, g, [sign*12, 0, 10])
        m.part(side+"齿条", box([12, 2, 1.5], (102+sign*4, sign*4.5, 88)), GOLD, g, [sign*18, sign*12, 0])
        finger = union(box([4, 8, 19], (102+sign*9, 0, 77)), box([8, 8, 3], (102+sign*7, 0, 68)))
        m.part(side+"夹指", finger, DARK, g, [sign*20, 0, -18])
        m.part(side+"夹指橡胶垫", box([1, 8, 9], (102+sign*4, 0, 72)), RUBBER, g, [sign*8, 0, -30])
    return m


BUILDERS = {
    "aero-engine": aero_engine, "airliner": airliner, "luxury-liner": luxury_liner,
    "model-y": model_y, "iphone": iphone, "optimus": optimus, "falcon": falcon,
    "mechanical-watch": mechanical_watch, "refinery": refinery, "industrial-arm": industrial_arm,
}

# Loaded after the shared primitives above; no alternate quantum SDK is used.
from .superconducting import superconducting_computer

BUILDERS["superconducting-computer"] = superconducting_computer
