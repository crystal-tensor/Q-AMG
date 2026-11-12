"""CPU depth-buffer thumbnail rendering from the actual exported GLB geometry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from qamg.gallery import public_catalogue


def render(directory: Path, slug: str):
    scene = trimesh.load(directory/"assembly.glb", force="scene")
    structure = json.loads((directory/"structure.json").read_text())
    access = structure.get("access_views")
    preset = next((v for v in access["views"] if v["id"] == access["thumbnail"]), None) if access else None
    part_records = {part["id"]: part for part in structure["parts"]}
    elevation = np.deg2rad(12 if access else 65 if slug in {"iphone", "mechanical-watch"} else 27)
    azimuth = np.deg2rad(75 if access else 55)
    eye = np.array([np.cos(azimuth)*np.cos(elevation), np.sin(azimuth)*np.cos(elevation), np.sin(elevation)])
    right = np.cross([0, 0, 1], eye)
    right /= np.linalg.norm(right)
    basis = np.array([right, np.cross(eye, right), eye]).T
    triangles, colors = [], []
    for name in scene.graph.nodes_geometry:
        record = part_records.get(name, {})
        if preset and record.get("view_scope", "interior") not in preset["scopes"]:
            continue
        transform, geom = scene.graph[name]
        mesh = scene.geometry[geom].copy()
        mesh.apply_transform(transform)
        if preset and preset["reveal"]:
            mesh.apply_translation(np.asarray(record.get("reveal_translation_mm", [0, 0, 0]))*.001)
        if mesh.visual.kind == "texture":
            color = np.asarray(mesh.visual.material.baseColorFactor, dtype=float)/255
            rgba = np.tile(color, (len(mesh.faces), 1))
        else:
            rgba = np.asarray(mesh.visual.face_colors, dtype=float)/255
        if record.get("appearance", {}).get("type") == "pbr_surface":
            rgb = rgba[:, :3]
            rgba[:, :3] = np.where(rgb <= .0031308, rgb*12.92, 1.055*rgb**(1/2.4)-.055)
        light = np.clip(mesh.face_normals @ np.array([.35, .2, .92]), 0, 1)
        rgba[:, :3] *= .67+.33*light[:, None]
        triangles.append(mesh.triangles @ basis)
        colors.append(rgba)
    triangles, colors = np.concatenate(triangles), np.concatenate(colors)
    width, height = 480, 300
    low, high = triangles[:, :, :2].reshape(-1, 2).min(axis=0), triangles[:, :, :2].reshape(-1, 2).max(axis=0)
    center, size = (low+high)/2, high-low
    scale = min(width/size[0], height/size[1])*.88
    pixels = triangles.copy()
    pixels[:, :, 0] = (triangles[:, :, 0]-center[0])*scale+width/2
    pixels[:, :, 1] = -(triangles[:, :, 1]-center[1])*scale+height/2
    canvas = np.full((height, width, 3), [238., 243., 235.])
    depth = np.full((height, width), -np.inf)
    opaque = colors[:, 3] >= .99
    order = [*np.flatnonzero(opaque), *np.flatnonzero(~opaque)[np.argsort(triangles[~opaque, :, 2].mean(axis=1))]]
    for index in order:
        tri, color = pixels[index], colors[index]
        x0, x1 = max(0, int(np.floor(tri[:, 0].min()))), min(width-1, int(np.ceil(tri[:, 0].max())))
        y0, y1 = max(0, int(np.floor(tri[:, 1].min()))), min(height-1, int(np.ceil(tri[:, 1].max())))
        denominator = (tri[1, 1]-tri[2, 1])*(tri[0, 0]-tri[2, 0])+(tri[2, 0]-tri[1, 0])*(tri[0, 1]-tri[2, 1])
        if abs(denominator) < 1e-10:
            continue
        yy, xx = np.mgrid[y0:y1+1, x0:x1+1]+.5
        a = ((tri[1, 1]-tri[2, 1])*(xx-tri[2, 0])+(tri[2, 0]-tri[1, 0])*(yy-tri[2, 1]))/denominator
        b = ((tri[2, 1]-tri[0, 1])*(xx-tri[2, 0])+(tri[0, 0]-tri[2, 0])*(yy-tri[2, 1]))/denominator
        c = 1-a-b
        z = a*tri[0, 2]+b*tri[1, 2]+c*tri[2, 2]
        window = depth[y0:y1+1, x0:x1+1]
        mask = (a >= -1e-6) & (b >= -1e-6) & (c >= -1e-6) & (z > window)
        rgb = canvas[y0:y1+1, x0:x1+1]
        rgb[mask] = color[3]*color[:3]*255+(1-color[3])*rgb[mask]
        if opaque[index]:
            window[mask] = z[mask]
    Image.fromarray(np.uint8(np.clip(canvas, 0, 255))).save(directory/"thumbnail.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/examples"))
    args = parser.parse_args()
    for entry in public_catalogue():
        directory = args.root/entry["slug"]
        if (directory/"assembly.glb").exists():
            render(directory, entry["slug"])
            print(entry["slug"], flush=True)


if __name__ == "__main__":
    main()
