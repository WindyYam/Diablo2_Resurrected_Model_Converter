#!/usr/bin/env python3
"""test_writer.py MODEL  - exercises the writer on a real .model (no game needed).

 1. GLB round trip: model -> glb -> model; geometry must match to quantisation precision
 2. shrink   : drop 35 % of a mesh's triangles (and the vertices only they use)
 3. grow     : append a shifted copy of a mesh's geometry (vertex/index arrays get bigger)
 4. rescale  : scale a mesh up so it exceeds VertexScale (all meshes get re-quantised)
After each edit the output is re-read with an independent reader and we check that
  * everything that is NOT the edited geometry is identical to the original object graph,
  * untouched meshes are bit-identical (or within 1 quantisation step after a rescale),
  * every pointer / count / group / mixed-marshalling entry is consistent, CRC is valid.
"""
import copy
import os
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d2r_model_io import D2RModel, Gr2, MeshData, decode_vertices, VTX     # noqa: E402
from d2r_model_to_glb import convert as to_glb                              # noqa: E402
from d2r_glb_to_model import convert as from_glb, load_glb                  # noqa: E402
from carbon_granny.reader import read_raw                                   # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def strip(fi):
    """object graph without the parts a geometry edit is allowed to change."""
    fi = copy.deepcopy(fi)
    def topo(t):
        t["Indices16"] = None; t["Groups"] = None
    for t in fi["TriTopologies"]:
        topo(t)
    for m in fi["Meshes"]:
        m["PrimaryVertexData"]["Vertices"] = None
        topo(m["PrimaryTopology"])
        for k in ("bbMin", "bbMax", "VertexScale"):
            m["ExtendedData"][k] = None
        for b in m["BoneBindings"]:
            b["OBBMin"] = b["OBBMax"] = None
    return repr(fi)


def only_mesh_diff(fi_a, fi_b, mesh_keep_same):
    """fi_a vs fi_b equal outside geometry; meshes in mesh_keep_same fully equal as well."""
    return strip(fi_a) == strip(fi_b)


def structure_ok(path):
    g = Gr2.load(path)
    raw = open(path, "rb").read()
    ok = struct.unpack_from("<I", raw, 0x28)[0] == zlib.crc32(raw[g.table_off:]) & 0xFFFFFFFF
    ok &= struct.unpack_from("<I", raw, 0x24)[0] == len(raw)
    for s in g.sections:
        for r in s.relocs:
            ok &= r[0] + 8 <= len(s.data) and r[2] <= len(g.sections[r[1]].data)
        for m in s.mixed:
            ok &= m[1] <= len(s.data)
    for sec in g.sections:
        ok &= sec.f16 <= len(sec.data) and sec.f8 <= len(sec.data)
    return bool(ok)


def md_from_model(mdl, mi):
    m = mdl.meshes[mi]
    a = mdl.vertex_array(mi)
    d = decode_vertices(a, mdl.mesh_scale(mi))
    bind = np.array([mdl.n2b[b["BoneName"]] for b in m["BoneBindings"]])
    tris = np.array([list(x.values())[0] & 0xFFFF for x in m["PrimaryTopology"]["Indices16"]]).reshape(-1, 3)
    N = d["N"] / np.linalg.norm(d["N"], axis=1, keepdims=True)
    return MeshData(m["Name"], d["P"], N, d["T"], d["UV"], bind[d["BI"]], d["BW"], [tris])


def mesh_geom(fi, mi, scale=None):
    m = fi["Meshes"][mi]
    vs = m["PrimaryVertexData"]["Vertices"]
    sc = m["ExtendedData"]["VertexScale"]
    P = np.array([v["Position"][:3] for v in vs]) / 32767.0 * sc
    idx = np.array([list(x.values())[0] & 0xFFFF for x in m["PrimaryTopology"]["Indices16"]])
    return P, idx.reshape(-1, 3), m


def run(model_path, work="/tmp"):
    mdl = D2RModel(model_path)
    fi0 = mdl.fi
    n = len(mdl.meshes)

    # ------------------------------------------------------------------ 1
    print("\n[1] GLB round trip")
    glb, out = os.path.join(work, "t_rt.glb"), os.path.join(work, "t_rt.model")
    to_glb(model_path, glb)
    from_glb(glb, model_path, out, log=lambda *_: None)
    fi1 = read_raw(open(out, "rb").read()).file_info
    check(strip(fi0) == strip(fi1), "everything except geometry identical to the original")
    check(structure_ok(out), "CRC, size, pointers, markers consistent")
    for mi in range(n):
        P0, T0, m0 = mesh_geom(fi0, mi)
        P1, T1, m1 = mesh_geom(fi1, mi)
        lsb = m0["ExtendedData"]["VertexScale"] / 32767
        check(np.abs(P0 - P1).max() <= lsb * 1.01 and np.array_equal(T0, T1),
              f"{m0['Name']}: positions within 1 LSB ({np.abs(P0 - P1).max():.2e}), triangles identical")
        w = np.array([v["BoneWeights"] for v in m1["PrimaryVertexData"]["Vertices"]])
        check((w.sum(1) == 255).all(), f"{m0['Name']}: skin weights sum to 255")
        uv0 = np.array([v["TextureCoordinates0"] for v in m0["PrimaryVertexData"]["Vertices"]])
        uv1 = np.array([v["TextureCoordinates0"] for v in m1["PrimaryVertexData"]["Vertices"]])
        check(np.abs(uv0 - uv1).max() < 1e-3, f"{m0['Name']}: UVs preserved ({np.abs(uv0 - uv1).max():.1e})")
        t0 = np.array([v["Tangent"] for v in m0["PrimaryVertexData"]["Vertices"]], float)
        t1 = np.array([v["Tangent"] for v in m1["PrimaryVertexData"]["Vertices"]], float)
        check(np.mean(np.sign(t0[:, 3]) == np.sign(t1[:, 3])) > 0.999 and np.abs(t0[:, :3] - t1[:, :3]).max() <= 1,
              f"{m0['Name']}: tangents + handedness preserved")

    # ------------------------------------------------------------------ 2 shrink
    print("\n[2] shrink mesh 1 by 35% triangles")
    e = md_from_model(mdl, 1)
    keep = np.arange(len(e.groups[0])) % 20 >= 7
    t = e.groups[0][keep]
    used = np.unique(t)
    remap = -np.ones(len(e.P), int); remap[used] = np.arange(len(used))
    e2 = MeshData(e.name, e.P[used], e.N[used], e.T[used], e.UV[used], e.J[used], e.W[used], [remap[t]])
    g = mdl.apply({1: e2}, log=lambda *_: None)
    out = os.path.join(work, "t_shrink.model"); g.save(out)
    fi = read_raw(open(out, "rb").read()).file_info
    P, T, _ = mesh_geom(fi, 1)
    check(len(P) == len(used) and len(T) == len(t) and T.max() < len(P), f"counts now {len(P)} verts / {len(T)} tris")
    check(fi["Meshes"][1]["PrimaryTopology"]["Groups"][0]["TriCount"] == len(t), "material group TriCount updated")
    check(strip(fi0) == strip(fi), "rest of the object graph identical")
    for mi in (0, 2):
        check(np.array_equal(mdl.vertex_array(mi), D2RModel(out).vertex_array(mi)), f"untouched mesh {mi} bit-identical")
    check(structure_ok(out), "structure/CRC ok")
    check(np.abs(P - e.P[used]).max() <= mdl.mesh_scale(1) / 32767 * 1.01, "vertex positions correct")

    # ------------------------------------------------------------------ 3 grow
    print("\n[3] grow mesh 0 (append a shifted copy)")
    e = md_from_model(mdl, 0)
    nv = len(e.P)
    e3 = MeshData(e.name, np.concatenate([e.P, e.P + [0.3, 0, 0]]), np.concatenate([e.N, e.N]),
                  np.concatenate([e.T, e.T]), np.concatenate([e.UV, e.UV]), np.concatenate([e.J, e.J]),
                  np.concatenate([e.W, e.W]), [np.concatenate([e.groups[0], e.groups[0] + nv])])
    g = mdl.apply({0: e3}, log=lambda *_: None)
    out = os.path.join(work, "t_grow.model"); g.save(out)
    fi = read_raw(open(out, "rb").read()).file_info
    P, T, m = mesh_geom(fi, 0)
    check(len(P) == 2 * nv and len(T) == 2 * len(e.groups[0]), f"counts now {len(P)} verts / {len(T)} tris")
    check(strip(fi0) == strip(fi), "rest of the object graph identical (incl. Materials, Skeleton, other meshes)")
    for mi in (1, 2):
        check(np.array_equal(mdl.vertex_array(mi), D2RModel(out).vertex_array(mi)), f"untouched mesh {mi} bit-identical")
    check(abs(m["ExtendedData"]["bbMax"][0] - (np.max(e.P[:, 0]) + 0.3)) < 1e-4, "bounding box follows the new geometry")
    check(structure_ok(out), "structure/CRC ok")
    mm = D2RModel(out)
    check([x[0] for x in mm.g2.sections[0].mixed] == [2 * nv, 6065, 1855], "mixed-marshalling counts updated")

    # ------------------------------------------------------------------ 4 rescale
    print("\n[4] scale mesh 2 x1.4 (exceeds VertexScale)")
    e = md_from_model(mdl, 2)
    e4 = MeshData(e.name, e.P * 1.4, e.N, e.T, e.UV, e.J, e.W, e.groups)
    g = mdl.apply({2: e4}, log=lambda *_: None)
    out = os.path.join(work, "t_scale.model"); g.save(out)
    fi = read_raw(open(out, "rb").read()).file_info
    ns = fi["Meshes"][0]["ExtendedData"]["VertexScale"]
    check(ns > mdl.mesh_scale(0) and all(m["ExtendedData"]["VertexScale"] == ns for m in fi["Meshes"]),
          f"VertexScale grown to {ns:.4f} in every mesh")
    for mi in (0, 1):
        P0, _, _ = mesh_geom(fi0, mi)
        P1, _, _ = mesh_geom(fi, mi)
        check(np.abs(P0 - P1).max() <= ns / 32767 * 1.01, f"untouched mesh {mi} still decodes to the same shape "
                                                             f"({np.abs(P0 - P1).max():.2e})")
    P1, _, _ = mesh_geom(fi, 2)
    check(np.abs(P1 - e4.P).max() <= ns / 32767 * 1.01, "scaled mesh decodes correctly")
    check(structure_ok(out), "structure/CRC ok")

    # ------------------------------------------------------------------ 5 odd index count
    print("\n[5] odd triangle count (index array needs alignment padding)")
    e = md_from_model(mdl, 1)
    e5 = MeshData(e.name, e.P, e.N, e.T, e.UV, e.J, e.W, [e.groups[0][:-1]])      # 8675 tris
    g = mdl.apply({1: e5}, log=lambda *_: None)
    out = os.path.join(work, "t_odd.model"); g.save(out)
    fi = read_raw(open(out, "rb").read()).file_info
    P, T, _ = mesh_geom(fi, 1)
    check(len(T) == 8675 and np.array_equal(T, e.groups[0][:-1]), "8675 triangles stored and read back exactly")
    check(strip(fi0) == strip(fi) and structure_ok(out), "rest identical, structure ok")
    check(np.array_equal(mdl.vertex_array(2), D2RModel(out).vertex_array(2)), "mesh after the odd array unaffected")

    # ------------------------------------------------------------------ 6 meters
    print("\n[6] --meters round trip")
    glbm, outm = os.path.join(work, "t_m.glb"), os.path.join(work, "t_m.model")
    to_glb(model_path, glbm, meters=True)
    from_glb(glbm, model_path, outm, meters=True, log=lambda *_: None)
    fim = read_raw(open(outm, "rb").read()).file_info
    for mi in range(n):
        P0, _, _ = mesh_geom(fi0, mi); P1, _, _ = mesh_geom(fim, mi)
        check(np.abs(P0 - P1).max() < 2e-4, f"mesh {mi}: positions survive metre conversion ({np.abs(P0 - P1).max():.1e})")

    # ------------------------------------------------------------------ 7 no tangents in GLB (Blender default)
    print("\n[7] GLB without TANGENT attribute (tangents recomputed)")
    import json
    from d2r_glb_to_model import load_glb
    js, binary = load_glb(glb)
    for gm in js["meshes"]:
        for pr in gm["primitives"]:
            pr["attributes"].pop("TANGENT")
    jb = json.dumps(js, separators=(",", ":")).encode(); jb += b" " * ((-len(jb)) % 4)
    glbn, outn = os.path.join(work, "t_nt.glb"), os.path.join(work, "t_nt.model")
    with open(glbn, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(jb) + 8 + len(binary)))
        f.write(struct.pack("<I4s", len(jb), b"JSON") + jb + struct.pack("<I4s", len(binary), b"BIN\0") + binary)
    from_glb(glbn, model_path, outn, log=lambda *_: None)
    fin = read_raw(open(outn, "rb").read()).file_info
    for mi in range(n):
        t0 = np.array([v["Tangent"] for v in fi0["Meshes"][mi]["PrimaryVertexData"]["Vertices"]], float)
        t1 = np.array([v["Tangent"] for v in fin["Meshes"][mi]["PrimaryVertexData"]["Vertices"]], float)
        cos = ((t0[:, :3] * t1[:, :3]).sum(1) / (np.linalg.norm(t0[:, :3], axis=1) * np.linalg.norm(t1[:, :3], axis=1) + 1e-9))
        check(np.mean(np.sign(t0[:, 3]) == np.sign(t1[:, 3])) > 0.99 and np.mean(cos > 0.9) > 0.98,
              f"mesh {mi}: recomputed tangents agree with the original ({np.mean(cos > 0.9):.3f} dir, "
              f"{np.mean(np.sign(t0[:, 3]) == np.sign(t1[:, 3])):.3f} sign)")

    # ------------------------------------------------------------------ 8 multi-material-group meshes
    print("\n[8] meshes with several material groups")
    tpl = os.path.join(work, "t_2grp.model")
    L = mdl.loc[0]
    g2 = copy.deepcopy(mdl.g2); s0 = g2.sections[0]
    nt = L["nidx"] // 3
    mat = struct.unpack_from("<I", s0.data, L["grp"])[0]
    struct.pack_into("<I", s0.data, L["grp_cnt"], 2)
    g2.replace_range(0, L["grp"], 12, struct.pack("<III", mat, 0, 1000) + struct.pack("<III", mat, 1000, nt - 1000))
    g2.save(tpl)
    ftpl = read_raw(open(tpl, "rb").read()).file_info
    check([g["TriCount"] for g in ftpl["Meshes"][0]["PrimaryTopology"]["Groups"]] == [1000, nt - 1000],
          "synthetic 2-group template reads back")
    glb2 = os.path.join(work, "t_2grp.glb")
    to_glb(tpl, glb2)
    js2, _ = load_glb(glb2)
    check(len(js2["meshes"][0]["primitives"]) == 2, "exporter writes one primitive per material group")
    out2 = os.path.join(work, "t_2grp_out.model")
    from_glb(glb2, tpl, out2, log=lambda *_: None)
    fo = read_raw(open(out2, "rb").read()).file_info
    gs = [(g["TriFirst"], g["TriCount"]) for g in fo["Meshes"][0]["PrimaryTopology"]["Groups"]]
    check(gs == [(0, 1000), (1000, nt - 1000)], f"group split preserved through the round trip {gs}")
    check(strip(read_raw(open(tpl, 'rb').read()).file_info) == strip(fo) and structure_ok(out2), "rest identical, structure ok")
    # old single-primitive GLB + unchanged triangle count -> original split reused (with a warning)
    msgs = []
    from_glb(glb, tpl, os.path.join(work, "t_2grp_fb.model"), log=msgs.append)
    fb = read_raw(open(os.path.join(work, "t_2grp_fb.model"), "rb").read()).file_info
    check(any("original group split was reused" in x for x in msgs) and
          [g["TriCount"] for g in fb["Meshes"][0]["PrimaryTopology"]["Groups"]] == [1000, nt - 1000],
          "merged single primitive + unchanged tri count -> original split reused")
    # ... but a changed triangle count cannot be split -> clear error
    shr = os.path.join(work, "t_grow.glb"); to_glb(os.path.join(work, "t_grow.model"), shr)   # mesh 0 has more tris now
    try:
        from_glb(shr, tpl, os.path.join(work, "never.model"), log=lambda *_: None)
        check(False, "merged primitive with changed tri count must raise")
    except ValueError as ex:
        check("material groups" in str(ex), "merged primitive + changed tri count -> clear error")

    # ------------------------------------------------------------------ 9 unit mismatch guard
    print("\n[9] unit-mismatch guard")
    for kw, name in ((dict(meters=True), "GLB without --meters read with --meters"),):
        try:
            from_glb(glb, model_path, os.path.join(work, "never.model"), log=lambda *_: None, **kw)
            check(False, name + " must raise")
        except ValueError as ex:
            check("WITHOUT --meters" in str(ex), name + " -> stopped with a hint")
    try:
        from_glb(glbm, model_path, os.path.join(work, "never.model"), log=lambda *_: None)
        check(False, "meters GLB read without --meters must raise")
    except ValueError as ex:
        check("did not pass --meters" in str(ex), "GLB exported with --meters read without it -> stopped with a hint")

    print("\n" + ("ALL TESTS PASSED" if not FAILS else f"{len(FAILS)} FAILED:\n  " + "\n  ".join(FAILS)))
    return not FAILS


if __name__ == "__main__":
    sys.exit(0 if run(sys.argv[1] if len(sys.argv) > 1 else "legs_med_lod0.model") else 1)
