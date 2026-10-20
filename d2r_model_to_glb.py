#!/usr/bin/env python3
"""
d2r_model_to_glb.py - convert a Diablo II: Resurrected .model file to glTF binary (.glb)
so it can be opened in Blender (File > Import > glTF 2.0).

D2R .model files are Granny 3D (GR2) containers, format v7, 64-bit little-endian,
with BitKnit2-compressed sections.  Decompression/reflection is done by the bundled
`carbon_granny` package (MIT, see LICENSE-carbon_granny-MIT.txt).

Usage:
    python d2r_model_to_glb.py legs_med_lod0.model [out.glb] [--meters]

Vertex encoding (verified against the bounding boxes stored in the file):
    Position  : 4 x int16   -> xyz = value / 32767 * Mesh.ExtendedData.VertexScale  (w is padding)
    Normal    : 4 x int8    -> xyz = value / 127
    Tangent   : 4 x int8    -> xyz = value / 127, w = handedness (+/-127)
    BoneWeights: 4 x uint8  -> value / 255 (sums to 255)
    BoneIndices: 4 x uint8  -> index into Mesh.BoneBindings[] (then looked up by BoneName)
    UV0       : 2 x float16
Units are Maya "feet-ish": ArtToolInfo.UnitsPerMeter = 3.28084. Pass --meters to rescale.
Axes: Y-up, right-handed (same as glTF), so no axis conversion is needed.
"""
import json
import struct
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from carbon_granny.reader import read_raw


def pad4(b, fill=b"\x00"):
    return b + fill * ((4 - len(b) % 4) % 4)


def convert(src, dst, meters=False):
    fi = read_raw(open(src, "rb").read()).file_info
    upm = fi["ArtToolInfo"]["UnitsPerMeter"]
    scale = (1.0 / upm) if meters else 1.0

    bones = fi["Skeletons"][0]["Bones"]
    name_to_bone = {b["Name"]: i for i, b in enumerate(bones)}

    binary = bytearray()
    buffer_views, accessors = [], []

    def add_view(data, target=None):
        while len(binary) % 4:
            binary.append(0)
        off = len(binary)
        binary.extend(data)
        bv = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target:
            bv["target"] = target
        buffer_views.append(bv)
        return len(buffer_views) - 1

    def add_accessor(data, comp, ctype, count, target=None, minmax=None, normalized=False):
        a = {"bufferView": add_view(data, target), "componentType": comp,
             "count": count, "type": ctype}
        if normalized:
            a["normalized"] = True
        if minmax:
            a["min"], a["max"] = minmax
        accessors.append(a)
        return len(accessors) - 1

    # ---- skeleton nodes -------------------------------------------------
    nodes, children = [], {i: [] for i in range(len(bones))}
    for i, b in enumerate(bones):
        t = b["Transform"]
        ss = t["scaleShear"]
        node = {"name": b["Name"],
                "translation": [c * scale for c in t["position"]],
                "rotation": list(t["orientation"]),  # Granny: x y z w  (same as glTF)
                "scale": [ss[0], ss[4], ss[8]]}
        nodes.append(node)
        if b["ParentIndex"] >= 0:
            children[b["ParentIndex"]].append(i)
    for i, ch in children.items():
        if ch:
            nodes[i]["children"] = ch
    root_bones = [i for i, b in enumerate(bones) if b["ParentIndex"] < 0]

    # inverse bind matrices (Granny stores 4x4 with translation in elements 12..14 == glTF layout)
    ibm = np.array([b["InverseWorldTransform"] for b in bones], dtype=np.float32)
    if meters:
        ibm[:, 12:15] *= scale
    ibm_acc = add_accessor(ibm.tobytes(), 5126, "MAT4", len(bones))
    skin = {"name": fi["Skeletons"][0]["Name"], "joints": list(range(len(bones))),
            "inverseBindMatrices": ibm_acc, "skeleton": root_bones[0]}

    # ---- materials ------------------------------------------------------
    materials, mat_index = [], {}

    def tex_paths(m):
        out = {}
        for mp in m.get("Maps", []):
            tex = (mp.get("Map") or {}).get("Texture")
            if tex:
                out[mp["Usage"]] = tex["FromFileName"]
        return out

    # ---- meshes ---------------------------------------------------------
    gl_meshes, mesh_nodes = [], []
    for m in fi["Meshes"]:
        vs = m["PrimaryVertexData"]["Vertices"]
        n = len(vs)
        vscale = m["ExtendedData"]["VertexScale"]
        P = np.array([v["Position"][:3] for v in vs], np.float64) / 32767.0 * vscale * scale
        N = np.array([v["Normal"][:3] for v in vs], np.float64) / 127.0
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        T = np.array([v["Tangent"] for v in vs], np.float64)
        TAN = np.concatenate([T[:, :3] / 127.0, -np.sign(T[:, 3:4])], axis=1)  # handedness flips with the V flip
        UV = np.array([v["TextureCoordinates0"] for v in vs], np.float64)
        UV[:, 1] = 1.0 - UV[:, 1]  # Maya V-up -> glTF V-down (Blender flips it back on import)
        W = np.array([v["BoneWeights"] for v in vs], np.float64)
        W = (W / W.sum(1, keepdims=True)).astype(np.float32)
        local_to_skel = np.array([name_to_bone[bb["BoneName"]] for bb in m["BoneBindings"]], np.uint16)
        J = local_to_skel[np.array([v["BoneIndices"] for v in vs])]
        idx = np.array([list(x.values())[0] & 0xFFFF for x in m["PrimaryTopology"]["Indices16"]], np.uint16)

        attrs = {
            "POSITION": add_accessor(P.astype(np.float32).tobytes(), 5126, "VEC3", n, 34962,
                                     ([float(x) for x in P.min(0)], [float(x) for x in P.max(0)])),
            "NORMAL": add_accessor(N.astype(np.float32).tobytes(), 5126, "VEC3", n, 34962),
            "TANGENT": add_accessor(TAN.astype(np.float32).tobytes(), 5126, "VEC4", n, 34962),
            "TEXCOORD_0": add_accessor(UV.astype(np.float32).tobytes(), 5126, "VEC2", n, 34962),
            "JOINTS_0": add_accessor(J.astype(np.uint16).tobytes(), 5123, "VEC4", n, 34962),
            "WEIGHTS_0": add_accessor(W.tobytes(), 5126, "VEC4", n, 34962),
        }
        # one glTF primitive per D2R material group (TriFirst/TriCount slice of the index buffer)
        groups = m["PrimaryTopology"]["Groups"] or [
            {"MaterialIndex": 0, "TriFirst": 0, "TriCount": len(idx) // 3}]
        prims = []
        for gi, g in enumerate(groups):
            sl = idx[g["TriFirst"] * 3:(g["TriFirst"] + g["TriCount"]) * 3]
            ind = add_accessor(sl.tobytes(), 5123, "SCALAR", len(sl), 34963)
            mat = m["MaterialBindings"][g["MaterialIndex"]]["Material"]
            if mat["Name"] not in mat_index:
                mat_index[mat["Name"]] = len(materials)
                ex = mat.get("ExtendedData") or {}
                materials.append({
                    "name": mat["Name"],
                    "doubleSided": bool(ex.get("IsDoubleSided", 0)),
                    "extras": {"d2r_textures": tex_paths(mat), "d2r_shader": ex.get("technique")},
                })
            prims.append({"attributes": attrs, "indices": ind, "material": mat_index[mat["Name"]],
                          "mode": 4, "extras": {"d2r_group": gi}})
        gl_meshes.append({"name": m["Name"], "primitives": prims})
        nodes.append({"name": m["Name"], "mesh": len(gl_meshes) - 1, "skin": 0})
        mesh_nodes.append(len(nodes) - 1)

    gltf = {
        "asset": {"version": "2.0", "generator": "d2r_model_to_glb.py",
                  "extras": {"source": os.path.basename(src), "unitsPerMeter": upm, "meters": meters}},
        "scene": 0, "scenes": [{"nodes": root_bones + mesh_nodes}],
        "nodes": nodes, "skins": [skin], "meshes": gl_meshes, "materials": materials,
        "accessors": accessors, "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(binary)}],
    }
    js = pad4(json.dumps(gltf, separators=(",", ":")).encode(), b" ")
    bn = pad4(bytes(binary))
    with open(dst, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(js) + 8 + len(bn)))
        f.write(struct.pack("<I4s", len(js), b"JSON") + js)
        f.write(struct.pack("<I4s", len(bn), b"BIN\x00") + bn)
    tv = sum(len(m["PrimaryVertexData"]["Vertices"]) for m in fi["Meshes"])
    print(f"wrote {dst}: {len(fi['Meshes'])} meshes, {tv} verts, {len(bones)} bones, "
          f"{len(materials)} materials, {os.path.getsize(dst)//1024} KiB")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)
    src = args[0]
    dst = args[1] if len(args) > 1 else os.path.splitext(src)[0] + ".glb"
    convert(src, dst, meters="--meters" in sys.argv)
