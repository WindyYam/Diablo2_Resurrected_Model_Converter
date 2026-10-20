#!/usr/bin/env python3
"""
d2r_glb_to_model.py - write an edited glTF/GLB back into a D2R `.model`.

    python d2r_glb_to_model.py EDITED.glb ORIGINAL.model OUT.model [--meters] [--only a,b] [--keep-node-transform]

The ORIGINAL .model is used as a template: skeleton, materials, textures, bone bindings, type
information etc. are kept byte-for-byte; only the geometry of the meshes found in the GLB is
replaced (vertices, triangles, material groups, bounds, per-bone OBBs, VertexScale).

Workflow
    1. python d2r_model_to_glb.py legs_med_lod0.model legs.glb         (add --meters if you like)
    2. Blender: File > Import > glTF 2.0, edit meshes (keep the armature & vertex-group names),
       File > Export > glTF 2.0 (.glb)   [Include: Selected/Visible; Mesh: Apply Modifiers;
       Skinning ON; "Tangents" optional - they are recomputed if absent]
    3. python d2r_glb_to_model.py exported.glb legs_med_lod0.model new_legs_med_lod0.model
       (pass --meters if step 1 used --meters)

Matching rules
    * a GLB mesh node is matched to an original mesh by node name or mesh name (Blender's ".001"
      suffixes are ignored); template meshes missing from the GLB get an invisible placeholder
    * a mesh may have as many glTF primitives as the original has material groups (1 each here)
    * skin joints are matched to skeleton bones by node NAME; influences on bones that are not
      bound to the mesh are moved to the nearest bound ancestor bone, max 4 influences / vertex
Conventions converted for you: V flipped (Maya V-up <-> glTF V-down), tangent handedness,
units (--meters), skinned-mesh node transform ignored (glTF spec).
"""
import itertools
import json
import os
import re
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d2r_model_io import D2RModel, MeshData, compute_tangents    # noqa: E402
from carbon_granny.reader import read_raw                         # noqa: E402

CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


# ------------------------------------------------------------------------- GLB reading
def load_glb(path):
    d = open(path, "rb").read()
    if d[:4] != b"glTF":
        raise ValueError("not a .glb file (export from Blender as 'glTF Binary (.glb)')")
    off, js, binary = 12, None, b""
    while off < len(d):
        ln, ty = struct.unpack_from("<I4s", d, off)
        chunk = d[off + 8: off + 8 + ln]
        if ty == b"JSON":
            js = json.loads(chunk)
        elif ty.startswith(b"BIN"):
            binary = chunk
        off += 8 + ln
    return js, binary


def accessor(js, binary, idx):
    a = js["accessors"][idx]
    bv = js["bufferViews"][a["bufferView"]]
    dt, nc = CT[a["componentType"]], NC[a["type"]]
    isz = np.dtype(dt).itemsize
    stride = bv.get("byteStride") or isz * nc
    arr = np.ndarray((a["count"], nc), dt, binary, bv.get("byteOffset", 0) + a.get("byteOffset", 0),
                     (stride, isz)).copy()
    if a.get("normalized"):
        arr = arr.astype(np.float64) / (np.iinfo(dt).max if np.issubdtype(dt, np.integer) else 1.0)
        arr = np.maximum(arr, -1.0)
    return arr


def node_matrix(n):
    if "matrix" in n:
        return np.array(n["matrix"], float).reshape(4, 4).T
    t, r, s = n.get("translation", [0, 0, 0]), n.get("rotation", [0, 0, 0, 1]), n.get("scale", [1, 1, 1])
    x, y, z, w = r
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M = np.eye(4)
    M[:3, :3] = R * np.array(s)[None, :]
    M[:3, 3] = t
    return M


def world_matrices(js):
    parent = {}
    for i, n in enumerate(js["nodes"]):
        for c in n.get("children", []):
            parent[c] = i
    cache = {}

    def W(i):
        if i not in cache:
            m = node_matrix(js["nodes"][i])
            cache[i] = W(parent[i]) @ m if i in parent else m
        return cache[i]
    return W


def norm_name(s):
    return re.sub(r"\.\d{3}$", "", (s or "").strip()).lower()


def smooth_normals(P, tris):
    f = np.cross(P[tris[:, 1]] - P[tris[:, 0]], P[tris[:, 2]] - P[tris[:, 0]])
    N = np.zeros_like(P)
    for k in range(3):
        np.add.at(N, tris[:, k], f)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


def placeholder_meshdata(mdl, mi, used_indices):
    m = mdl.meshes[mi]
    groups = m["PrimaryTopology"]["Groups"]
    if not groups:
        raise ValueError(f"'{m['Name']}' has no material groups for an empty placeholder")
    if not m["BoneBindings"]:
        raise ValueError(f"'{m['Name']}' has no bone bindings for an empty placeholder")

    pattern = None
    for vertex_count in range(3, 33):
        for tri in itertools.permutations(range(vertex_count), 3):
            candidate = struct.pack("<3H", *tri)
            if candidate not in used_indices and mdl.raw0.find(candidate) < 0:
                pattern = candidate
                used_indices.add(candidate)
                indices = tri
                break
        if pattern is not None:
            break
    if pattern is None:
        raise ValueError(f"could not find unique placeholder indices for '{m['Name']}'")

    vertex_count = max(indices) + 1
    center = (np.array(m["ExtendedData"]["bbMin"], dtype=np.float64) +
              np.array(m["ExtendedData"]["bbMax"], dtype=np.float64)) / 2
    P = np.tile(center, (vertex_count, 1))
    N = np.tile([0.0, 1.0, 0.0], (vertex_count, 1))
    T = np.tile([1.0, 0.0, 0.0, 1.0], (vertex_count, 1))
    UV = np.zeros((vertex_count, 2))
    bone = mdl.n2b[m["BoneBindings"][0]["BoneName"]]
    J = np.zeros((vertex_count, 4), dtype=np.int64)
    J[:, 0] = bone
    W = np.zeros((vertex_count, 4))
    W[:, 0] = 1.0
    tri = np.array([indices], dtype=np.int64)
    return MeshData(m["Name"], P, N, T, UV, J, W,
                    [tri] + [np.zeros((0, 3), dtype=np.int64) for _ in groups[1:]])


# --------------------------------------------------------------------- GLB -> MeshData
def glb_to_meshdata(js, binary, node_idx, mdl, meters, keep_node_transform, log, group_mats=None):
    node = js["nodes"][node_idx]
    gm = js["meshes"][node["mesh"]]
    skin = js["skins"][node["skin"]] if "skin" in node else None
    joint_bone = None
    if skin:
        joint_bone = np.array([mdl.n2b.get(js["nodes"][j].get("name", ""), -1) for j in skin["joints"]])
        miss = sorted({js["nodes"][j].get("name", "") for j, b in zip(skin["joints"], joint_bone) if b < 0})
        if miss:
            log(f"    note: {len(miss)} skin joint(s) are not in the D2R skeleton and are ignored: {miss[:6]}")
    else:
        log("    note: mesh has no skin; all vertices will be bound to binding 0")

    Ps, Ns, Ts, UVs, Js, Ws, groups, base = [], [], [], [], [], [], [], 0
    prim_mats = []
    have_tan = True
    for prim in gm["primitives"]:
        if prim.get("mode", 4) != 4:
            raise ValueError("only triangle-list primitives are supported")
        at = prim["attributes"]
        P = accessor(js, binary, at["POSITION"]).astype(np.float64)
        n = len(P)
        if "indices" in prim:
            idx = accessor(js, binary, prim["indices"]).ravel().astype(np.int64)
        else:
            idx = np.arange(n)
        tris = idx.reshape(-1, 3)
        N = accessor(js, binary, at["NORMAL"]).astype(np.float64) if "NORMAL" in at else smooth_normals(P, tris)
        UV = accessor(js, binary, at["TEXCOORD_0"]).astype(np.float64) if "TEXCOORD_0" in at else np.zeros((n, 2))
        if "TEXCOORD_0" not in at:
            log("    warning: no UVs in GLB; zero UVs written")
        if "TANGENT" in at:
            T = accessor(js, binary, at["TANGENT"]).astype(np.float64)
        else:
            have_tan, T = False, None
        # skinning (sets 0 and 1; keep the 4 strongest influences)
        jl, wl, s = [], [], 0
        while f"JOINTS_{s}" in at and f"WEIGHTS_{s}" in at:
            jl.append(accessor(js, binary, at[f"JOINTS_{s}"]).astype(np.int64))
            wl.append(accessor(js, binary, at[f"WEIGHTS_{s}"]).astype(np.float64))
            s += 1
        if jl and joint_bone is not None:
            Jall, Wall = np.concatenate(jl, 1), np.concatenate(wl, 1)
            Ball = joint_bone[Jall]
            Wall = np.where(Ball >= 0, Wall, 0.0)
            order = np.argsort(-Wall, axis=1, kind="stable")[:, :4]
            Wk = np.take_along_axis(Wall, order, 1)
            Bk = np.take_along_axis(Ball, order, 1)
            Bk = np.where(Wk > 0, Bk, -1)
        else:
            Bk, Wk = np.full((n, 4), -1), np.zeros((n, 4))
            if skin is None:
                pass
        # transforms
        if skin is None or keep_node_transform:
            M = world_matrices(js)(node_idx)
            P = (np.c_[P, np.ones(n)] @ M.T)[:, :3]
            Nm = np.linalg.inv(M[:3, :3]).T
            N = N @ Nm.T
            if T is not None:
                T = np.c_[T[:, :3] @ M[:3, :3].T, T[:, 3]]
        if meters:
            P = P * mdl.upm
        N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        UVf = np.c_[UV[:, 0], 1.0 - UV[:, 1]]                 # glTF V-down -> Maya V-up
        if T is not None:
            T = np.c_[T[:, :3] / np.maximum(np.linalg.norm(T[:, :3], axis=1, keepdims=True), 1e-12),
                      -np.sign(T[:, 3:4] + (T[:, 3:4] == 0))]  # handedness flips with V
        Ps.append(P); Ns.append(N); Ts.append(T); UVs.append(UVf); Js.append(Bk); Ws.append(Wk)
        groups.append(tris + base)
        mi_ = prim.get("material")
        prim_mats.append(norm_name(js["materials"][mi_].get("name")) if mi_ is not None and "materials" in js else None)
        base += n
    P, N, UV = np.concatenate(Ps), np.concatenate(Ns), np.concatenate(UVs)
    J, W = np.concatenate(Js), np.concatenate(Ws)
    # order the primitives like the template's material groups (by material name when unambiguous)
    if group_mats and len(groups) == len(group_mats) and len(groups) > 1 \
            and len(set(prim_mats)) == len(prim_mats) and set(prim_mats) == set(group_mats) \
            and len(set(group_mats)) == len(group_mats):
        groups = [groups[prim_mats.index(gm)] for gm in group_mats]
    tri_all = np.concatenate(groups)
    T = np.concatenate(Ts) if have_tan else compute_tangents(P, N, UV, tri_all)
    return MeshData(gm.get("name", node.get("name")), P, N, T, UV, J, W, groups)


# ------------------------------------------------------------------------------- main
def convert(glb, template, out, meters=False, only=None, keep_node_transform=False, log=print, force=False):
    mdl = D2RModel(template)
    js, binary = load_glb(glb)
    log(f"template: {os.path.basename(template)}  ({len(mdl.meshes)} meshes, {len(mdl.bones)} bones)")
    cand = [i for i, n in enumerate(js["nodes"]) if "mesh" in n]
    edits, seen = {}, {}
    for ni in cand:
        n = js["nodes"][ni]
        mi = None
        for nm in (n.get("name"), js["meshes"][n["mesh"]].get("name")):
            if nm:
                mi = next((i for i, m in enumerate(mdl.meshes) if norm_name(m["Name"]) == norm_name(nm)), None)
                if mi is not None:
                    break
        if mi is None:
            log(f"  skipping GLB mesh '{n.get('name')}': no mesh of that name in the template")
            continue
        if only and mdl.meshes[mi]["Name"] not in only:
            continue
        if mi in seen:
            raise ValueError(f"GLB has two meshes matching '{mdl.meshes[mi]['Name']}' "
                             f"('{seen[mi]}', '{n.get('name')}'); join or rename them in Blender")
        seen[mi] = n.get("name")
        log(f"  reading '{n.get('name')}' -> {mdl.meshes[mi]['Name']}")
        topo = mdl.meshes[mi]["PrimaryTopology"]["Groups"]
        gmats = [norm_name(mdl.meshes[mi]["MaterialBindings"][g["MaterialIndex"]]["Material"]["Name"]) for g in topo]
        e = glb_to_meshdata(js, binary, ni, mdl, meters, keep_node_transform, log, gmats)
        ngrp = len(topo)
        if len(e.groups) != ngrp:
            total = sum(len(g) for g in e.groups)
            orig_total = sum(g["TriCount"] for g in topo)
            if len(e.groups) == 1 and total == orig_total:
                tris = e.groups[0]
                e.groups = [tris[g["TriFirst"]:g["TriFirst"] + g["TriCount"]] for g in topo]
                log(f"    warning: GLB mesh has 1 primitive but '{mdl.meshes[mi]['Name']}' has {ngrp} material groups; "
                    f"triangle count is unchanged, so the original group split was reused by triangle order")
            else:
                raise ValueError(
                    f"'{mdl.meshes[mi]['Name']}' has {ngrp} material groups in the template "
                    f"({', '.join(gmats)}), but the GLB mesh has {len(e.groups)} primitive(s) "
                    f"({total} triangles vs {orig_total} originally). Each D2R material group must stay a separate "
                    f"material slot in Blender. Re-export the GLB with the current d2r_model_to_glb.py "
                    f"(it now writes one primitive per group), redo the edit, and keep the material slots.")
        edits[mi] = e
    if not edits:
        raise ValueError("no mesh in the GLB matched a mesh of the template model")
    used_indices = set()
    placeholders = set()
    for mi, m in enumerate(mdl.meshes):
        if mi in edits or (only and m["Name"] not in only):
            continue
        log(f"  missing from GLB: {m['Name']}; creating an invisible placeholder")
        edits[mi] = placeholder_meshdata(mdl, mi, used_indices)
        placeholders.add(mi)
    untouched = [m["Name"] for mi, m in enumerate(mdl.meshes)
                 if mi not in edits and only and m["Name"] not in only]
    if untouched:
        log(f"  left unchanged by --only: {untouched}")
    # ---- unit sanity check: a missing/extra --meters (or Blender unit scale) changes size by ~3.28x
    for mi, e in edits.items():
        if mi in placeholders:
            continue
        ext = mdl.meshes[mi]["ExtendedData"]
        orig = float(np.linalg.norm(np.array(ext["bbMax"]) - np.array(ext["bbMin"])))
        new = float(np.linalg.norm(e.P.max(0) - e.P.min(0)))
        r = new / max(orig, 1e-9)
        if (r > 2.5 or r < 0.4) and not force:
            hint = ""
            if abs(r - mdl.upm) / mdl.upm < 0.15:
                hint = " This looks like the GLB was exported WITHOUT --meters but you passed --meters (or the same mistake in Blender's unit scale)."
            elif abs(r - 1 / mdl.upm) * mdl.upm < 0.15:
                hint = " This looks like the GLB was exported WITH --meters but you did not pass --meters."
            raise ValueError(f"'{mdl.meshes[mi]['Name']}' would be {r:.2f}x the size of the original mesh "
                             f"(diagonal {new:.3f} vs {orig:.3f}).{hint} Use --force if this is intentional.")
    g2 = mdl.apply(edits, log=log)
    size = g2.save(out)
    # ---- verify what we wrote by re-reading it with the independent reader
    chk = read_raw(open(out, "rb").read()).file_info
    for mi, e in edits.items():
        m = chk["Meshes"][mi]
        nv, nt = len(m["PrimaryVertexData"]["Vertices"]), len(m["PrimaryTopology"]["Indices16"]) // 3
        grp = sum(g["TriCount"] for g in m["PrimaryTopology"]["Groups"])
        idx = np.array([list(x.values())[0] & 0xFFFF for x in m["PrimaryTopology"]["Indices16"]])
        assert nv == len(e.P) and nt == sum(len(g) for g in e.groups) and grp == nt and idx.max(initial=0) < nv, \
            f"verification failed for {m['Name']}"
        bb = np.array(m["ExtendedData"]["bbMax"]) - np.array(m["ExtendedData"]["bbMin"])
        log(f"  verified {m['Name']}: {nv} verts, {nt} tris, bbox size {np.round(bb, 3).tolist()}, "
            f"VertexScale {m['ExtendedData']['VertexScale']:.5f}")
    log(f"wrote {out} ({size // 1024} KiB, sections stored uncompressed)")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Write an edited GLB back into a D2R .model (template = original .model)")
    ap.add_argument("glb")
    ap.add_argument("template", help="the ORIGINAL .model used as template")
    ap.add_argument("out", help="output .model")
    ap.add_argument("--meters", action="store_true", help="GLB was exported with --meters")
    ap.add_argument("--only", help="comma-separated mesh names to replace (others stay untouched)")
    ap.add_argument("--force", action="store_true", help="skip the size sanity check")
    ap.add_argument("--keep-node-transform", action="store_true", help="also apply node transforms of skinned meshes")
    a = ap.parse_args()
    convert(a.glb, a.template, a.out, meters=a.meters, only=set(a.only.split(",")) if a.only else None,
            keep_node_transform=a.keep_node_transform, force=a.force)
