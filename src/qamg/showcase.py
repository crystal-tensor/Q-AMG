"""Reproducible conceptual assemblies, explicitly separate from photo inference.

All geometry below is authored here. Names of commercial products identify an
appearance reference; no official CAD, measured dimensions or internal BOM is
claimed. QB measurements choose illustrative layout parameters, not the meshes.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from .assembly_plan import selected_disassembly
from .backends import make_backend
from .benchmark import dump
from .gallery import CATALOGUE
from .image_pipeline import export_parts
from .mesh_parts import as_manifold, merge_solids, validate_partition
from .problem import StructureProblem
from .quantum import QuantumStructurePrior
from .schema import ChoiceGroup, Option, PairRule, PlannerSettings, ShowcaseContext

WHITE = [226, 230, 229]
SILVER = [151, 167, 180]
DARK = [37, 47, 58]
BLUE = [60, 108, 143]
TEAL = [46, 135, 124]
GOLD = [201, 164, 84]
ORANGE = [214, 112, 53]
RED = [179, 65, 66]


def box(size, center=(0., 0., 0.)):
    return as_manifold(trimesh.creation.box(size)).translate(center)


def orient(solid, axis):
    matrix = trimesh.geometry.align_vectors([0, 0, 1], axis)
    return solid.transform(matrix[:3].tolist())


def turn(solid, degrees, axis=(0, 0, 1)):
    matrix = trimesh.transformations.rotation_matrix(math.radians(degrees), axis)
    return solid.transform(matrix[:3].tolist())


def cyl(radius, height, center=(0., 0., 0.), axis=(0, 0, 1), sections=64):
    solid = as_manifold(trimesh.creation.cylinder(radius=radius, height=height, sections=sections))
    return orient(solid, axis).translate(center)


def tube(radius, inner, height, center=(0., 0., 0.), axis=(0, 0, 1)):
    return cyl(radius, height, center, axis)-cyl(inner, height+.1, center, axis)


def ellipsoid(size, center=(0., 0., 0.)):
    mesh = trimesh.creation.icosphere(subdivisions=3)
    mesh.apply_scale(np.array(size)/2)
    return as_manifold(mesh).translate(center)


def beam(start, end, radius):
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    return cyl(radius, float(np.linalg.norm(end-start)), (start+end)/2, end-start)


def union(*solids):
    return merge_solids(list(solids))


def prism(points, height, z=0):
    points = np.asarray(points, dtype=float)
    n = len(points)
    vertices = np.concatenate([np.column_stack([points, np.full(n, z-height/2)]),
                               np.column_stack([points, np.full(n, z+height/2)]),
                               [[*points.mean(axis=0), z-height/2], [*points.mean(axis=0), z+height/2]]])
    faces = []
    for i in range(n):
        j = (i+1) % n
        faces += [[i, j, n+j], [i, n+j, n+i], [2*n, j, i], [2*n+1, n+i, n+j]]
    mesh = trimesh.Trimesh(vertices, faces, process=True)
    mesh.fix_normals()
    return as_manifold(mesh)


def rounded_rect(width, length, radius, height, z=0):
    points = []
    for x, y, angle in [(width/2-radius, length/2-radius, 0),
                         (-width/2+radius, length/2-radius, 90),
                         (-width/2+radius, -length/2+radius, 180),
                         (width/2-radius, -length/2+radius, 270)]:
        for a in np.linspace(angle, angle+90, 9):
            points.append([x+radius*np.cos(np.deg2rad(a)), y+radius*np.sin(np.deg2rad(a))])
    return prism(points, height, z)


def loft_x(stations):
    """Closed elliptical stations (x, lateral radius, vertical radius, z)."""
    count = 64
    angles = np.arange(count)*2*np.pi/count
    rings = [np.column_stack([np.full(count, x), ry*np.cos(angles), z+rz*np.sin(angles)])
             for x, ry, rz, z in stations]
    vertices = np.concatenate([*rings, [[stations[0][0], 0, stations[0][3]],
                                        [stations[-1][0], 0, stations[-1][3]]]])
    faces = []
    for row in range(len(stations)-1):
        for i in range(count):
            a, b = row*count+i, row*count+(i+1) % count
            c, d = b+count, a+count
            faces += [[a, b, c], [a, c, d]]
    for i in range(count):
        j = (i+1) % count
        faces += [[len(vertices)-2, j, i], [len(vertices)-1, (len(stations)-1)*count+i,
                                            (len(stations)-1)*count+j]]
    mesh = trimesh.Trimesh(vertices, faces, process=True)
    mesh.fix_normals()
    return as_manifold(mesh)


def fan(radius, blades=18):
    pieces = [cyl(radius*.2, 4)]
    blade = prism([[radius*.1, -1], [radius*.75, -radius*.13], [radius, -1],
                   [radius*.8, radius*.10], [radius*.1, 1]], 1.4)
    blade = turn(blade, 18, [1, 0, 0])
    for angle in np.linspace(0, 360, blades, endpoint=False):
        pieces.append(turn(blade, angle))
    return union(*pieces)


def gear(radius, teeth=20, height=2):
    angles = np.arange(teeth*4)*2*np.pi/(teeth*4)
    radii = np.tile([radius*.86, radius, radius, radius*.86], teeth)
    return prism(np.column_stack([radii*np.cos(angles), radii*np.sin(angles)]), height)


@dataclass
class DemoPart:
    label: str
    solid: object
    color: list[int]
    group: str
    explode: list[float]
    appearance: dict | None = None


class Model:
    def __init__(self):
        self.parts = []

    def add(self, label, solid, color, group, explode, appearance=None):
        self.parts.append(DemoPart(label, solid, color, group, list(explode), appearance))


def aero_engine():
    m = Model()
    m.add("进气短舱", tube(43, 39.5, 33, (59, 0, 0), (1, 0, 0)), WHITE, "外部机匣", [60, 0, 25])
    shroud = tube(34, 31.5, 90, (-7, 0, 0), (1, 0, 0))
    m.add("上半机匣", shroud ^ box([100, 80, 40], (-7, 0, 20)), SILVER, "外部机匣", [0, 0, 70])
    m.add("下半机匣", shroud ^ box([100, 80, 40], (-7, 0, -20)), SILVER, "外部机匣", [0, 0, -60])
    m.add("前风扇总成", orient(fan(37, 22), [1, 0, 0]).translate([72, 0, 0]), [106, 123, 137], "转子与核心机", [80, 0, 0])
    m.add("风扇锥", orient(as_manifold(trimesh.creation.cone(radius=7, height=15)), [1, 0, 0]).translate([74, 0, 0]), SILVER, "转子与核心机", [105, 0, 0])
    m.add("主轴", cyl(3.7, 140, (0, 0, 0), (1, 0, 0)), SILVER, "转子与核心机", [-90, 0, 0])
    for i, x in enumerate([34, 17, 0]):
        m.add(f"压气机级 {i+1}", orient(fan(22-i*2, 16), [1, 0, 0]).translate([x, 0, 0]), BLUE, "转子与核心机", [20-i*35, 45, 0])
    m.add("燃烧段外壳", tube(22, 18.5, 25, (-22, 0, 0), (1, 0, 0)), GOLD, "热端与排气", [-20, -60, 0])
    m.add("涡轮级", orient(fan(20, 20), [1, 0, 0]).translate([-43, 0, 0]), ORANGE, "热端与排气", [-80, 0, 0])
    m.add("尾喷管", tube(24, 21, 28, (-67, 0, 0), (1, 0, 0)), DARK, "热端与排气", [-115, 0, 0])
    return m


def airliner():
    m = Model()
    shell = loft_x([(-107, 1, 1, 5), (-86, 8, 8, 2), (-60, 13, 13, 0),
                    (58, 13, 13, 0), (83, 10, 11, 0), (103, 1, 2, -1)])
    m.add("机身", shell, WHITE, "机体", [0, 0, 22])
    for sign, label in [(-1, "右"), (1, "左")]:
        wing = prism([[24, sign*10], [-8, sign*95], [-33, sign*90], [-13, sign*10]], 3, -3)
        m.add(f"{label}主翼", wing, SILVER, "升力与操纵面", [0, sign*65, 0])
        tail = prism([[-77, sign*4], [-93, sign*37], [-108, sign*35], [-95, sign*4]], 2.3, 9)
        m.add(f"{label}水平尾翼", tail, WHITE, "升力与操纵面", [-30, sign*35, 20])
        engine = union(tube(10, 8, 31, (7, sign*37, -17), (1, 0, 0)),
                       orient(fan(7.5, 12), [1, 0, 0]).translate([23, sign*37, -17]))
        m.add(f"{label}发动机", engine, DARK, "动力与起落架", [30, sign*28, -45])
    fin = turn(prism([[-83, 8], [-100, 42], [-111, 43], [-105, 8]], 2), 90, [1, 0, 0])
    m.add("垂直尾翼", fin, BLUE, "升力与操纵面", [-50, 0, 45])
    windows = [ellipsoid([2.2, .8, 2.5], (x, sign*12.8, 3.5))
               for sign in [-1, 1] for x in np.arange(-58, 70, 6)]
    windows += [ellipsoid([12, 18, 5], (82, 0, 8))]
    m.add("舷窗与驾驶舱窗组件", union(*windows), BLUE, "机体", [0, 0, 48])
    for i, (x, y) in enumerate([(63, 0), (-18, -12), (-18, 12)]):
        gear_set = union(cyl(3.8, 5, (x, y, -24), [0, 1, 0]), beam([x, y, -13], [x, y, -22], 1.3))
        m.add(f"起落架组件 {i+1}", gear_set, DARK, "动力与起落架", [0, y*2, -42])
    return m


def luxury_liner():
    m = Model()
    outline = [[-98, -21], [-55, -29], [55, -27], [89, -13], [105, 0], [89, 13], [55, 27], [-55, 29], [-98, 21]]
    hull = prism(outline, 25, -5)
    m.add("主船体", hull, BLUE, "船体与甲板", [0, 0, -35])
    for i in range(5):
        length, width, z = 163-i*16, 52-i*6, 11+i*8
        level = rounded_rect(length, width, 7, 7, z).translate([-9-i*2, 0, 0])
        m.add(f"上层甲板 {i+1}", level, WHITE, "船体与甲板", [0, 0, 18*(i+1)])
        windows = []
        for sign in [-1, 1]:
            for x in np.arange(-length/2+5, length/2-5, 5):
                windows.append(box([3.2, .6, 2.6], (x-9-i*2, sign*(width/2+.25), z+.7)))
        m.add(f"甲板窗带 {i+1}", union(*windows), DARK, "舱室与设施", [0, 25+i*8, 18*(i+1)])
    m.add("驾驶楼", rounded_rect(20, 36, 3, 10, 42).translate([55, 0, 0]), WHITE, "舱室与设施", [30, 0, 60])
    m.add("顶层泳池", rounded_rect(29, 13, 4, 1.5, 47).translate([-28, 0, 0]), TEAL, "舱室与设施", [-30, 0, 95])
    funnels = union(box([14, 11, 16], (2, -7, 52)), box([14, 11, 16], (2, 7, 52)))
    m.add("双烟囱组件", funnels, DARK, "舱室与设施", [0, 0, 110])
    for sign, name in [(-1, "右舷"), (1, "左舷")]:
        boats = [ellipsoid([15, 5, 4], (x, sign*29, 12)) for x in [-58, -36, -14, 8, 30]]
        m.add(f"{name}救生艇组", union(*boats), ORANGE, "救生与推进", [0, sign*45, 0])
    m.add("推进器组", union(cyl(3, 16, (-103, -10, -8), [1, 0, 0]),
                            cyl(3, 16, (-103, 10, -8), [1, 0, 0])), GOLD, "救生与推进", [-50, 0, -15])
    return m


def model_y():
    m = Model()
    outer = loft_x([(-91, 25, 9, 25), (-75, 35, 14, 25), (-35, 37, 15, 25),
                    (46, 37, 15, 25), (83, 32, 10, 25), (91, 25, 6, 25)])
    body = outer-box([139, 58, 40], (-2, 0, 34))
    for x in [-54, 54]:
        for y in [-35, 35]:
            body = body-cyl(16.5, 22, (x, y, 15), [0, 1, 0])
    m.add("车身外壳", body, WHITE, "车身与座舱", [0, 0, 50])
    cabin_mesh = trimesh.convex.convex_hull(np.array([[x, sign*y, z] for x, y, z in
        [(-64, 31, 33), (-43, 27, 56), (-15, 27, 63), (28, 27, 60), (61, 32, 34)] for sign in [-1, 1]]))
    m.add("全景车顶与车窗近似件", as_manifold(cabin_mesh), DARK, "车身与座舱", [0, 0, 85])
    m.add("底盘", box([167, 60, 4], (0, 0, 9)), SILVER, "电池与驱动", [0, 0, -40])
    m.add("电池包示意", rounded_rect(116, 51, 4, 6, 14), BLUE, "电池与驱动", [0, 0, -70])
    for x, name in [(-54, "后"), (54, "前")]:
        m.add(f"{name}驱动单元", union(cyl(8, 28, (x, 0, 20), [0, 1, 0]),
                                       cyl(2, 65, (x, 0, 15), [0, 1, 0])), SILVER, "电池与驱动", [x*.7, 0, -20])
        for sign, side in [(-1, "右"), (1, "左")]:
            m.add(f"{side}{name}轮胎", tube(14.5, 8.5, 9, (x, sign*37, 15), [0, 1, 0]), DARK, "四轮组件", [0, sign*50, 0])
            rim = union(cyl(3, 9.4, (x, sign*37, 15), [0, 1, 0]),
                        tube(8.4, 6.9, 9.4, (x, sign*37, 15), [0, 1, 0]))
            spokes = []
            for angle in np.linspace(0, 2*np.pi, 5, endpoint=False):
                spokes.append(beam([x, sign*37, 15], [x+7*np.cos(angle), sign*37, 15+7*np.sin(angle)], .9))
            m.add(f"{side}{name}轮毂", union(rim, *spokes), SILVER, "四轮组件", [0, sign*70, 0])
    m.add("前灯组", union(ellipsoid([4, 20, 2.5], (86, -20, 30)),
                          ellipsoid([4, 20, 2.5], (86, 20, 30))), BLUE, "车身与座舱", [40, 0, 20])
    m.add("尾灯组", union(box([2, 21, 2], (-90, -19, 31)), box([2, 21, 2], (-90, 19, 31))), RED, "车身与座舱", [-40, 0, 20])
    return m


def iphone():
    m = Model()
    frame = rounded_rect(70, 145, 9, 7)-rounded_rect(65, 140, 7, 9)
    m.add("金属中框", frame, SILVER, "机身与显示", [0, 0, 0])
    screen = rounded_rect(66, 141, 7.5, 1.5, 4.3)
    m.add("显示屏", screen, DARK, "机身与显示", [0, 0, 35])
    m.add("屏幕图形示意层", union(rounded_rect(54, 53, 18, .15, 5.12).translate([0, -27, 0]),
                                     rounded_rect(48, 44, 16, .15, 5.13).translate([0, 24, 0])), BLUE, "机身与显示", [0, 0, 37])
    m.add("听筒与前置组件", rounded_rect(20, 5, 2, .25, 5.4).translate([0, 59, 0]), DARK, "机身与显示", [0, 0, 48])
    m.add("后盖", rounded_rect(67, 142, 8, 1.4, -4.4), WHITE, "机身与显示", [0, 0, -35])
    m.add("电池", rounded_rect(43, 91, 4, 3.3).translate([-8, -12, 0]), DARK, "内部结构示意", [-45, 0, -10])
    board = union(box([13, 95, 2], (23, 4, 0)), box([49, 19, 2], (0, 56, 0)))
    m.add("主板近似件", board, TEAL, "内部结构示意", [45, 0, -8])
    chips = union(box([10, 12, 2.3], (22, 28, 1.8)), box([9, 10, 2.3], (22, 5, 1.8)))
    m.add("芯片屏蔽罩", chips, SILVER, "内部结构示意", [65, 0, 10])
    camera = union(rounded_rect(27, 31, 6, 3, -6).translate([-18, 48, 0]),
                   cyl(6.5, 3, (-24, 55, -8)), cyl(6.5, 3, (-11, 43, -8)), cyl(6.5, 3, (-24, 39, -8)))
    m.add("三摄外观组件", camera, DARK, "相机与音频", [-45, 40, -22])
    m.add("扬声器与接口组件", union(box([43, 8, 3], (0, -61, 0)), box([12, 10, 2.7], (0, -68, 0))), SILVER, "相机与音频", [0, -45, -8])
    m.add("侧键组件", union(box([2, 15, 3], (-35.8, 15, 0)), box([2, 10, 3], (-35.8, 37, 0)),
                            box([2, 20, 3], (35.8, 20, 0))), SILVER, "机身与显示", [40, 0, 0])
    return m


def optimus():
    m = Model()
    m.add("头部外壳", ellipsoid([24, 23, 31], (0, 0, 168)), WHITE, "头部与躯干", [0, 0, 45])
    m.add("面部面罩", ellipsoid([19, 5, 24], (0, 10.5, 168)), DARK, "头部与躯干", [0, 35, 45])
    m.add("躯干外壳", loft_x([(-19, 9, 12, 130), (0, 13, 24, 130), (19, 9, 12, 130)]), WHITE, "头部与躯干", [0, 35, 18])
    m.add("躯干骨架与腰部", union(cyl(8, 25, (0, 0, 108)), box([24, 13, 33], (0, -2, 130)),
                                    cyl(5, 15, (0, 0, 153))), DARK, "头部与躯干", [0, -35, 0])
    m.add("骨盆", rounded_rect(29, 18, 5, 17, 98), DARK, "下肢与足部", [0, 0, -20])
    for sign, name in [(-1, "右"), (1, "左")]:
        m.add(f"{name}肩关节", cyl(8, 12, (sign*24, 0, 142), [1, 0, 0]), DARK, "上肢与手部", [sign*27, 0, 5])
        m.add(f"{name}上臂", union(beam([sign*26, 0, 137], [sign*31, 0, 114], 6),
                                  ellipsoid([13, 15, 18], (sign*29, 0, 126))), WHITE, "上肢与手部", [sign*40, 0, 0])
        m.add(f"{name}肘关节", cyl(6, 13, (sign*32, 0, 110), [0, 1, 0]), DARK, "上肢与手部", [sign*48, 0, 0])
        m.add(f"{name}前臂", beam([sign*32, 0, 105], [sign*35, 2, 85], 5.6), WHITE, "上肢与手部", [sign*55, 0, -10])
        fingers = [box([9, 6, 10], (sign*35, 2, 79))]
        fingers += [beam([sign*35+x, 2, 76], [sign*35+x, 4, 66], .85) for x in [-3, -1, 1, 3]]
        fingers += [beam([sign*40, 2, 81], [sign*42, 4, 73], 1)]
        m.add(f"{name}手部近似件", union(*fingers), DARK, "上肢与手部", [sign*66, 0, -18])
        m.add(f"{name}大腿", beam([sign*9, 0, 91], [sign*11, 0, 57], 7.8), WHITE, "下肢与足部", [sign*25, 0, -20])
        m.add(f"{name}膝关节", cyl(7, 16, (sign*11, 0, 54), [1, 0, 0]), DARK, "下肢与足部", [sign*35, 0, -32])
        m.add(f"{name}小腿", beam([sign*11, 0, 48], [sign*12, 0, 17], 6.5), WHITE, "下肢与足部", [sign*42, 0, -40])
        m.add(f"{name}足部", rounded_rect(16, 29, 5, 9, 8).translate([sign*12, 6, 0]), DARK, "下肢与足部", [sign*50, 15, -50])
    return m


def falcon():
    m = Model()
    m.add("一级箭体外观", cyl(9, 136, (0, 0, 70)), WHITE, "一级与发动机", [0, 0, -25])
    bells = []
    profile = [[2.6, 0], [1.3, 7], [.9, 7], [2.2, 0], [2.6, 0]]
    bell = as_manifold(trimesh.creation.revolve(profile, sections=48))
    bells.append(bell.translate([0, 0, -5]))
    bells += [bell.translate([6*np.cos(a), 6*np.sin(a), -5]) for a in np.linspace(0, 2*np.pi, 8, endpoint=False)]
    m.add("九发动机组示意", union(*bells), DARK, "一级与发动机", [0, 0, -65])
    for i, a in enumerate(np.linspace(0, 2*np.pi, 4, endpoint=False)):
        direction = np.array([np.cos(a), np.sin(a), 0])
        leg = union(beam(direction*9+[0, 0, 42], direction*15+[0, 0, 6], 1.1),
                    beam(direction*9+[0, 0, 11], direction*15+[0, 0, 6], .8))
        m.add(f"着陆支腿 {i+1}", leg, DARK, "回收组件", (direction*35).tolist())
        grid = []
        for offset in [-4, -2, 0, 2, 4]:
            grid.append(box([10, .6, .6], (0, offset, 0)))
            grid.append(box([.6, 10, .6], (offset, 0, 0)))
        fin = turn(union(*grid), np.rad2deg(a)).translate((direction*14+[0, 0, 127]).tolist())
        m.add(f"栅格舵示意 {i+1}", fin, SILVER, "回收组件", (direction*35+[0, 0, 15]).tolist())
    m.add("级间段", tube(9.1, 7.5, 13, (0, 0, 145)), DARK, "二级与载荷", [0, 0, 25])
    m.add("二级箭体外观", cyl(8.8, 44, (0, 0, 174)), WHITE, "二级与载荷", [0, 0, 55])
    m.add("二级发动机喷管示意", orient(bell, [0, 0, -1]).translate([0, 0, 152]), SILVER, "二级与载荷", [35, 0, 40])
    # loft_x has its axis on X; the proper inverse rotation maps X to +Z.
    matrix = trimesh.geometry.align_vectors([1, 0, 0], [0, 0, 1])
    fairing = loft_x([(197, 10, 10, 0), (203, 13, 13, 0), (228, 13, 13, 0),
                      (243, 8, 8, 0), (250, .4, .4, 0)]).transform(matrix[:3].tolist())
    m.add("整流罩左半体", fairing ^ box([30, 60, 70], (15, 0, 225)), WHITE, "二级与载荷", [35, 0, 85])
    m.add("整流罩右半体", fairing ^ box([30, 60, 70], (-15, 0, 225)), WHITE, "二级与载荷", [-35, 0, 85])
    return m


def mechanical_watch():
    m = Model()
    m.add("表壳", tube(29, 24.5, 10), GOLD, "表壳与表带", [0, 0, 0])
    bezel = tube(30.5, 24.5, 3.5, (0, 0, 7))
    flutes = [cyl(1, 3.8, (29.5*np.cos(a), 29.5*np.sin(a), 7)) for a in np.linspace(0, 2*np.pi, 60, endpoint=False)]
    m.add("纹理表圈近似件", union(bezel, *flutes), GOLD, "表壳与表带", [0, 0, 25])
    m.add("表镜", cyl(24.2, .8, (0, 0, 9.4)), BLUE, "显示与指针", [0, 0, 45],
          {"type": "transparent_display_proxy", "rgba": [159, 204, 215, 50]})
    dial = cyl(24, 1, (0, 0, 5.4))
    m.add("绿色表盘", dial, [24, 78, 65], "显示与指针", [0, 0, 16])
    indices = [turn(box([1.4, 5, .5], (0, 20, 6.2)), a) for a in np.arange(0, 360, 30)]
    hands = [turn(box([1.8, 16, .5], (0, 7, 6.7)), 55), turn(box([1.1, 21, .4], (0, 10, 7.3)), -55),
             cyl(2, 1, (0, 0, 7))]
    m.add("时标与指针示意", union(*indices, *hands), GOLD, "显示与指针", [0, 0, 32])
    m.add("机芯夹板示意", cyl(23.5, 2.5, (0, 0, -.5))-cyl(13, 4, (4, 0, -.5)), SILVER, "机芯示意", [0, 0, -25])
    for i, (x, y, radius) in enumerate([(-7, 2, 9), (8, 9, 7), (9, -8, 8), (-7, -13, 5)]):
        m.add(f"机芯齿轮示意 {i+1}", gear(radius, 22, 1.5).translate([x, y, 0]), GOLD, "机芯示意", [x*2, y*2, -40-i*6])
    m.add("底盖", cyl(25, 1.5, (0, 0, -6)), SILVER, "表壳与表带", [0, 0, -70])
    for sign, name in [(-1, "下"), (1, "上")]:
        links = []
        for i in range(7):
            links += [rounded_rect(21-i*.45, 7, 1.4, 3.2, -1-i*.5).translate([0, sign*(32+i*7.5), 0])]
        m.add(f"{name}表带组件", union(*links), GOLD, "表壳与表带", [0, sign*45, -10])
    m.add("表冠", cyl(3.7, 7, (32, 0, 1), [1, 0, 0]), GOLD, "表壳与表带", [35, 0, 0])
    return m


def refinery():
    m = Model()
    m.add("设施底座", box([180, 130, 4], (0, 0, 0)), SILVER, "基础与管廊", [0, 0, -25])
    for i, (x, y, radius, height) in enumerate([(-55, 30, 10, 90), (-22, 33, 8, 72), (8, 35, 7, 58)]):
        tower = union(cyl(radius, height, (x, y, height/2+3)), ellipsoid([radius*2, radius*2, radius], (x, y, height+3)))
        m.add(f"塔器 {i+1}", tower, WHITE, "塔器与平台", [0, 25+i*12, 30])
        platforms = [tube(radius+3.5, radius+.6, 1.2, (x, y, z)) for z in np.arange(20, height, 20)]
        platforms += [beam([x+radius+2, y-3, 4], [x+radius+2, y-3, height], .6),
                      beam([x+radius+2, y+3, 4], [x+radius+2, y+3, height], .6)]
        m.add(f"塔器平台与扶梯 {i+1}", union(*platforms), GOLD, "塔器与平台", [-30, 25, 15+i*8])
    for i, x in enumerate([38, 72]):
        tank = union(cyl(15, 25, (x, 30, 16)), ellipsoid([30, 30, 12], (x, 30, 29)))
        m.add(f"储罐 {i+1}", tank, SILVER, "储存与换热", [40+i*15, 0, 20])
    for i, y in enumerate([-20, -44]):
        exchanger = union(cyl(6.8, 45, (-50, y, 14), [1, 0, 0]),
                          box([4, 18, 9], (-65, y, 6)), box([4, 18, 9], (-35, y, 6)))
        m.add(f"换热器组件 {i+1}", exchanger, DARK, "储存与换热", [-40, -30-i*15, 15])
    supports = [box([2, 2, 32], (x, y, 18)) for x in [-8, 28, 64] for y in [-54, -12]]
    supports += [box([2, 45, 2], (x, -33, 34)) for x in [-8, 28, 64]]
    m.add("管廊钢架", union(*supports), BLUE, "基础与管廊", [25, -30, 0])
    pipes = [beam([-8, y, z], [75, y, z], 1.5) for y, z in [(-45, 38), (-33, 38), (-21, 38)]]
    pipes += [beam([-8, -45, 38], [-8, 30, 38], 1.5), beam([-8, 30, 38], [-22, 30, 38], 1.5)]
    m.add("工艺管线示意", union(*pipes), ORANGE, "基础与管廊", [20, -30, 40])
    m.add("加热炉示意", union(box([30, 29, 29], (49, -32, 18)), cyl(4, 44, (49, -32, 54))), WHITE, "储存与换热", [55, -35, 20])
    return m


def industrial_arm():
    m = Model()
    m.add("底座", union(cyl(27, 8, (0, 0, 4)), box([46, 46, 4], (0, 0, 0))), DARK, "底座与肩部", [0, 0, -30])
    m.add("旋转台", cyl(19, 20, (0, 0, 18)), ORANGE, "底座与肩部", [0, 0, 25])
    m.add("肩部关节", cyl(15, 31, (0, 0, 37), [0, 1, 0]), DARK, "底座与肩部", [0, -45, 20])
    m.add("大臂", union(beam([0, 0, 39], [28, 0, 94], 10), box([14, 18, 30], (9, 0, 65))), ORANGE, "连杆与肘部", [-30, 0, 30])
    m.add("肘部关节", cyl(12, 26, (28, 0, 94), [0, 1, 0]), DARK, "连杆与肘部", [0, -45, 35])
    m.add("前臂", beam([30, 0, 96], [84, 0, 117], 8), ORANGE, "连杆与肘部", [25, 0, 45])
    m.add("腕部旋转组件", union(cyl(9, 18, (89, 0, 117), [1, 0, 0]), cyl(6, 18, (100, 0, 113))), DARK, "腕部与末端", [55, 0, 25])
    m.add("末端法兰", cyl(10, 4, (100, 0, 101)), SILVER, "腕部与末端", [55, 0, 0])
    m.add("夹爪基体", box([24, 16, 8], (100, 0, 94)), ORANGE, "腕部与末端", [55, 0, -15])
    for sign, name in [(-1, "右"), (1, "左")]:
        jaw = union(box([4, 8, 20], (100+sign*9, 0, 82)), box([8, 8, 4], (100+sign*7, 0, 73)))
        m.add(f"{name}夹指", jaw, DARK, "腕部与末端", [55+sign*15, 0, -35])
    return m


BUILDERS = dict(zip([entry["slug"] for entry in CATALOGUE],
                    [aero_engine, airliner, luxury_liner, model_y, iphone, optimus,
                     falcon, mechanical_watch, refinery, industrial_arm], strict=True))


def generate_showcase(slug: str, output: Path) -> dict:
    started = time.perf_counter()
    entry = next(e for e in CATALOGUE if e["slug"] == slug)
    model = BUILDERS[slug]()
    parts = []
    for item in model.parts:
        rgb = np.asarray(item.color)/255
        parts.append(item.solid.set_properties(3, lambda position, properties, color=rgb: color))
    context = ShowcaseContext(example_id=slug, title=entry["title"], seed=42)
    groups = []
    for key, label, low, high in [("lateral", "横向分离", .8, 1.15), ("vertical", "竖向分离", .85, 1.2),
                                   ("longitudinal", "纵向间距", .9, 1.25)]:
        groups.append(ChoiceGroup(key=key, label=label, options=[
            Option(key="compact", label="紧凑展开", cost=1, attributes={"factor": low}),
            Option(key="spread", label="宽松展开", cost=0, attributes={"factor": high})]))
    problem = StructureProblem(context, groups, [PairRule(left=(0, 1), right=(1, 1), cost=3,
                                  reason="illustrative_layout_spread_penalty_not_measured_physics")])
    settings = PlannerSettings(backend="qb", encoding="compact_binary", iterations=4, depth=1, shots=512)
    plan = QuantumStructurePrior(make_backend("qb", settings.max_qubits)).plan(problem, settings, 42)
    candidate = plan.best
    if candidate is None:
        raise ValueError("No observed layout candidate; no classical substitution")
    factors = [g.options[o].attributes["factor"] for g, o in zip(groups, problem.selection(candidate.bits), strict=True)]
    vectors = [(np.asarray(p.explode)*[factors[2], factors[0], factors[1]]).tolist() for p in model.parts]
    ids = [f"part-{i:03d}" for i in range(len(parts))]
    anchor = int(np.argmax([p.volume() for p in parts]))
    sequence = selected_disassembly(parts, ids, vectors, anchor, samples=12)
    validation = validate_partition(merge_solids(parts), parts)
    validation["checks"].update({"authored_conceptual_example": True, "photo_reconstruction_performed": False,
                                  "official_product_cad": False, "manufacturing_joints_verified": False})
    validation["assembly_overlap_fraction"] = max(0., sum(p.volume() for p in parts)/validation["merged_volume_mm3"]-1)
    provenance = {"mode": "qb", "execution_source": plan.execution_source, "quantum_sdk": "QuantumBridge",
                  "quantum_role": "illustrative_layout_parameter_selection", "geometry_source": "authored_procedural_concept",
                  "geometry_is_inferred": False, "photo_reconstruction_performed": False,
                  "official_product_cad": False, "generation_rounds": 1, "fallback_used": False,
                  "hardware_verified": False, "quantum_speedup_verified": False,
                  "selected_measurement": {"bits": candidate.bits, "job_id": candidate.first_job_id,
                                           "observed_count": candidate.count, "kind": "direct_or_exactly_decoded"}}
    output.mkdir(parents=True, exist_ok=True)
    structure = export_parts(parts, output, provenance, sequence, validation, [p.label for p in model.parts],
                              explosion_vectors=vectors, appearance=[p.appearance or {} for p in model.parts])
    names = list(dict.fromkeys(p.group for p in model.parts))
    structure["groups"] = [{"id": f"demo-group-{i}", "parent": "root", "label": name} for i, name in enumerate(names)]
    for record, item in zip(structure["parts"], model.parts, strict=True):
        record["parent"] = f"demo-group-{names.index(item.group)}"
    structure["engineering_scope"] = "人工参数化结构示意；尺寸、内部和部件关系未经原产品验证"
    dump(output/"structure.json", structure)
    dump(output/"trace.json", [{"round": 0, "problem": problem.as_dict(), "plan": plan.as_dict()}])
    dump(output/"problem.json", problem.as_dict())
    if plan.circuit_qasm3:
        (output/"layout.qasm").write_text(plan.circuit_qasm3)
    description = {**entry, "kind": "procedural_demo", "geometry_source": "Q-AMG authored procedural geometry",
                   "reference_scope": "品牌外形参考入口，不是模型下载来源、尺寸依据或官方 CAD" if entry["reference"] else "通用结构概念",
                   "limitations": ["结构与内部为人工示意", "比例不是原产品实测尺寸", "分件不是官方 BOM", "碰撞采样不认证连续拆解路径"]}
    dump(output/"demo.json", description)
    summary = {"status": "generated", "kind": "procedural_demo", "spec": context.model_dump(),
               "settings": {"planner": settings.model_dump()}, "title": entry["title"],
               "description": entry["description"], "part_count": len(parts), "selected_bits": candidate.bits,
               "decisions": problem.describe(candidate.bits), "resources": plan.resources,
               "provenance": provenance, "validation": validation, "demo": description,
               "timings": {"end_to_end_s": time.perf_counter()-started},
               "disassembly": {"complete_for_generated_mesh": sequence["complete_for_generated_mesh"],
                               "moves": len(sequence["moves"]), "continuous_collision_proof": False}}
    dump(output/"run.json", summary)
    return summary
