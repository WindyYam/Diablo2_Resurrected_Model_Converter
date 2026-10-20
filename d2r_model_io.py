#!/usr/bin/env python3
"""
d2r_model_io.py - read, patch and WRITE Diablo II: Resurrected `.model` files.

A `.model` is a Granny 3D (GR2) v7 64-bit-LE container.  This module

  * parses the container (sections, relocation tables, mixed-marshalling tables),
    decompressing BitKnit2 sections with the bundled `carbon_granny` decoder;
  * can replace byte ranges inside a section while keeping every pointer, array count,
    marshalling entry, region marker and the file CRC consistent (`Gr2.replace_range`);
  * writes the container back out with every section stored UNCOMPRESSED
    (compression = 0, which Granny supports).  There is no BitKnit2 *encoder*, so output
    files are larger (~3x) than the originals;
  * knows the D2R character-mesh layout well enough to replace the geometry of a mesh
    (`D2RModel.apply`): vertices, triangles, material groups, bounding box, VertexScale and
    per-bone OBBs are all regenerated.

Layout facts this relies on (all verified on legs_med_lod0.model, see FORMAT_NOTES.md):
  * array reference = u32 count + 64-bit pointer (12 bytes); the pointer is a relocation entry
    at `src`, the count sits at `src - 4`;
  * TriTopology = Groups(12) Indices(12) Indices16(12) ...  => Groups ptr is at `Indices16 ptr - 24`;
  * a mixed-marshalling entry is (count, offsetInSection, typeSection, typeOffset); for meshes it
    describes the vertex array, so its count must follow the vertex count;
  * section header first16bit/first8bit mark where the 16-bit / 8-bit data regions start;
  * header CRC = CRC32 of the file from the start of the section table (0x68) to the end;
  * vertex = Position 4xi16 | BoneWeights 4xu8 | BoneIndices 4xu8 | Normal 4xi8 | Tangent 4xi8 | UV0 2xf16
    (28 bytes) with position = i16/32767 * VertexScale.
"""
import copy
import os
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from carbon_granny.codecs import decompress_bitknit2          # noqa: E402
from carbon_granny.reader import decompress_section, read_raw  # noqa: E402

GR2_MAGIC_LE64 = bytes.fromhex("e59b495e6f631f141e13eba990beedc4")
SEC_HDR = 44

VTX = np.dtype([("pos", "<i2", (4,)), ("bw", "u1", (4,)), ("bi", "u1", (4,)),
                ("n", "i1", (4,)), ("t", "i1", (4,)), ("uv", "<f2", (2,))])
assert VTX.itemsize == 28
EXPECTED_LAYOUT = [(17, "Position", 4), (14, "BoneWeights", 4), (12, "BoneIndices", 4),
                   (13, "Normal", 4), (13, "Tangent", 4), (21, "TextureCoordinates0", 2)]


# =====================================================================================
#  Container
# =====================================================================================
class Section:
    __slots__ = ("comp", "align", "data", "relocs", "mixed", "f16", "f8")


class Gr2:
    """In-memory GR2 file: every section decompressed, relocations/mixed tables as lists."""

    @classmethod
    def load(cls, src):
        d = open(src, "rb").read() if isinstance(src, (str, os.PathLike)) else bytes(src)
        if d[:16] != GR2_MAGIC_LE64:
            raise ValueError("not a 64-bit little-endian Granny GR2 file (D2R .model expected)")
        u = lambda o: struct.unpack_from("<I", d, o)[0]
        self = cls()
        if u(0x20) != 7:
            raise ValueError(f"unsupported GR2 version {u(0x20)} (only v7)")
        self.header_size = u(0x10)
        self.table_off = 0x20 + u(0x2C)
        n = u(0x30)
        self.prefix = bytearray(d[: self.table_off])
        self.root_type = [u(0x34), u(0x38)]
        self.root_node = [u(0x3C), u(0x40)]
        self.sections = []
        for i in range(n):
            (comp, off, csize, usize, align, f16, f8, roff, nrel, moff, nmix) = \
                struct.unpack_from("<11I", d, self.table_off + SEC_HDR * i)
            s = Section()
            s.comp, s.align, s.f16, s.f8 = comp, align, f16, f8
            if usize:
                data = decompress_section(comp, d[off:off + csize], usize,
                                          {"first16": f16, "first8": f8})
                if len(data) != usize:
                    raise ValueError(f"section {i}: bad decompressed size")
                s.data = bytearray(data)
            else:
                s.data = bytearray()

            def table(count, size, o):
                if not count:
                    return b""
                if comp == 4:
                    cs = struct.unpack_from("<I", d, o)[0]
                    return decompress_bitknit2(d[o + 4:o + 4 + cs], count * size)
                return d[o:o + count * size]

            rb, mb = table(nrel, 12, roff), table(nmix, 16, moff)
            s.relocs = [list(struct.unpack_from("<III", rb, 12 * k)) for k in range(nrel)]
            s.mixed = [list(struct.unpack_from("<IIII", mb, 16 * k)) for k in range(nmix)]
            self.sections.append(s)
        return self

    # ------------------------------------------------------------------ writing
    def to_bytes(self):
        n = len(self.sections)
        out = bytearray(self.prefix)
        table_pos = len(out)
        out += bytes(SEC_HDR * n)
        if len(out) != self.header_size:
            raise ValueError("header size mismatch")
        pad = lambda: out.extend(b"\0" * ((-len(out)) % 4))
        heads = []
        for s in self.sections:
            pad()
            doff = len(out)
            out += s.data
            pad()
            roff = len(out)
            for r in s.relocs:
                out += struct.pack("<III", *r)
            moff = len(out)
            for m in s.mixed:
                out += struct.pack("<IIII", *m)
            heads.append((0, doff, len(s.data), len(s.data), s.align, s.f16, s.f8,
                          roff, len(s.relocs), moff, len(s.mixed)))
        for i, h in enumerate(heads):
            struct.pack_into("<11I", out, table_pos + SEC_HDR * i, *h)
        struct.pack_into("<II", out, 0x34, *self.root_type)
        struct.pack_into("<II", out, 0x3C, *self.root_node)
        struct.pack_into("<I", out, 0x24, len(out))
        struct.pack_into("<I", out, 0x28, zlib.crc32(bytes(out[self.table_off:])) & 0xFFFFFFFF)
        return bytes(out)

    def save(self, path):
        b = self.to_bytes()
        with open(path, "wb") as f:
            f.write(b)
        return len(b)

    # ------------------------------------------------------------ range replace
    def replace_range(self, si, start, old_len, new):
        """Replace section[si].data[start:start+old_len] by `new`, fixing everything that moves."""
        sec = self.sections[si]
        end = start + old_len
        delta = len(new) - old_len
        if delta % 4:
            raise ValueError("size change must be a multiple of 4 to keep alignment")
        for sj, s in enumerate(self.sections):
            for r in s.relocs:
                if sj == si and start <= r[0] < end:
                    raise ValueError(f"pointer fixup at {r[0]:#x} lies inside the replaced range")
                if r[1] == si and start < r[2] < end:
                    raise ValueError(f"pointer into the middle of the replaced range ({r[2]:#x})")
            for m in s.mixed:
                if sj == si and start < m[1] < end:
                    raise ValueError("marshalling entry inside the replaced range")
        sec.data[start:end] = new
        sh = lambda off: off + delta if off >= end else off
        for sj, s in enumerate(self.sections):
            for r in s.relocs:
                if sj == si:
                    r[0] = sh(r[0])
                if r[1] == si:
                    r[2] = sh(r[2])
            for m in s.mixed:
                if sj == si:
                    m[1] = sh(m[1])
                if m[2] == si:
                    m[3] = sh(m[3])
        sec.f16, sec.f8 = sh(sec.f16), sh(sec.f8)
        for ref in (self.root_type, self.root_node):
            if ref[0] == si:
                ref[1] = sh(ref[1])


# =====================================================================================
#  Vertex helpers
# =====================================================================================
def decode_vertices(arr, scale):
    """structured VTX array -> dict of float arrays (positions in file units)."""
    return {
        "P": arr["pos"][:, :3].astype(np.float64) / 32767.0 * scale,
        "N": arr["n"][:, :3].astype(np.float64) / 127.0,
        "T": np.concatenate([arr["t"][:, :3] / 127.0, np.sign(arr["t"][:, 3:4].astype(float))], 1),
        "UV": arr["uv"].astype(np.float64),
        "BI": arr["bi"].astype(np.int64),
        "BW": arr["bw"].astype(np.float64),
    }


def encode_vertices(P, N, T, UV, Jlocal, W, scale, pad_w):
    """Quantise to the D2R vertex format.  Jlocal = per-mesh bone-binding indices (n,4)."""
    n = len(P)
    out = np.zeros(n, VTX)
    out["pos"][:, :3] = np.clip(np.rint(P / scale * 32767.0), -32767, 32767)
    out["pos"][:, 3] = pad_w
    order = np.argsort(-W, axis=1, kind="stable")
    Ws = np.clip(np.take_along_axis(W, order, 1), 0, None)
    Js = np.take_along_axis(Jlocal, order, 1)
    Wn = Ws / Ws.sum(1, keepdims=True) * 255.0
    fl = np.floor(Wn)
    rem = (255 - fl.sum(1)).astype(int)
    rank = np.argsort(np.argsort(-(Wn - fl), axis=1, kind="stable"), axis=1)
    fl += rank < rem[:, None]
    out["bw"] = fl.astype(np.uint8)
    out["bi"] = np.where(fl > 0, Js, Js[:, [0]]).astype(np.uint8)
    out["n"][:, :3] = np.clip(np.rint(N * 127.0), -127, 127)
    out["t"][:, :3] = np.clip(np.rint(T[:, :3] * 127.0), -127, 127)
    out["t"][:, 3] = np.where(T[:, 3] >= 0, 127, -127)
    out["uv"] = UV.astype(np.float16)
    return out


def compute_tangents(P, N, UV, tris):
    """Per-vertex tangent + handedness in the *Granny file* convention (UV as stored, w = +/-1).
    Verified against the original data: direction cos 0.994, handedness 99.7 % identical."""
    t = tris
    p0, p1, p2 = P[t[:, 0]], P[t[:, 1]], P[t[:, 2]]
    u0, u1, u2 = UV[t[:, 0]], UV[t[:, 1]], UV[t[:, 2]]
    e1, e2, d1, d2 = p1 - p0, p2 - p0, u1 - u0, u2 - u0
    r = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(r) < 1e-20, 1e-20, r)
    tan = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / r[:, None]
    bit = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) / r[:, None]
    T, B = np.zeros_like(P), np.zeros_like(P)
    for k in range(3):
        np.add.at(T, t[:, k], tan)
        np.add.at(B, t[:, k], bit)
    T = T - N * (N * T).sum(1, keepdims=True)
    nrm = np.linalg.norm(T, axis=1, keepdims=True)
    fallback = np.cross(N, np.array([0.0, 1.0, 0.0]))
    T = np.where(nrm > 1e-12, T / np.maximum(nrm, 1e-12), fallback)
    w = np.sign((np.cross(N, T) * B).sum(1))
    w[w == 0] = 1.0
    return np.concatenate([T, w[:, None]], 1)


class MeshData:
    """Replacement geometry for one mesh, in D2R *file* conventions (see d2r_glb_to_model)."""

    def __init__(self, name, P, N, T, UV, J, W, groups):
        self.name, self.P, self.N, self.T, self.UV = name, P, N, T, UV
        self.J, self.W, self.groups = J, W, groups      # J = skeleton bone indices (n,4), -1 invalid


# =====================================================================================
#  D2R model
# =====================================================================================
class D2RModel:
    def __init__(self, path):
        self.path = path
        self.raw = open(path, "rb").read()
        self.g2 = Gr2.load(self.raw)
        self.fi = read_raw(self.raw).file_info
        sk = self.fi["Skeletons"][0]["Bones"]
        self.bones = sk
        self.n2b = {b["Name"]: i for i, b in enumerate(sk)}
        self.IWT = np.array([b["InverseWorldTransform"] for b in sk], np.float64).reshape(-1, 4, 4)
        self.upm = self.fi["ArtToolInfo"]["UnitsPerMeter"]
        self.meshes = self.fi["Meshes"]
        self.warnings = []
        self._locate()

    # ---------------------------------------------------------------- locating
    def _locate(self):
        s0 = self.g2.sections[0]
        raw0 = bytes(s0.data)
        self.raw0 = raw0
        rel_src = {r[0]: r for r in s0.relocs}
        rel_dst = {}
        for r in s0.relocs:
            rel_dst.setdefault((r[1], r[2]), []).append(r[0])
        u32 = lambda o: struct.unpack_from("<I", raw0, o)[0]
        used_mixed, used_idx = set(), set()
        self.loc = []
        for mi, m in enumerate(self.meshes):
            vs = m["PrimaryVertexData"]["Vertices"]
            lay = [(t["type"], t["name"], t["array_width"]) for t in vs.granny_type]
            if lay != EXPECTED_LAYOUT:
                raise ValueError(f"mesh {m['Name']}: unsupported vertex layout {lay}")
            nv = len(vs)
            L = {"mesh": mi, "nv": nv}
            k = next((k for k, e in enumerate(s0.mixed) if e[0] == nv and k not in used_mixed), None)
            if k is None:
                raise ValueError(f"mesh {m['Name']}: vertex marshalling entry not found")
            used_mixed.add(k)
            L["mixed"], L["vtx"] = k, s0.mixed[k][1]
            src = [s for s in rel_dst.get((0, L["vtx"]), []) if u32(s - 4) == nv]
            if len(src) != 1:
                raise ValueError(f"mesh {m['Name']}: cannot find the vertex array reference")
            L["vtx_cnt"] = src[0] - 4
            arr = np.frombuffer(raw0, VTX, nv, L["vtx"])
            if not np.array_equal(arr["pos"][0, :3], np.array(vs[0]["Position"][:3])):
                raise ValueError(f"mesh {m['Name']}: vertex data does not match the located blob")
            # indices
            topo = m["PrimaryTopology"]
            ind = np.array([list(x.values())[0] & 0xFFFF for x in topo["Indices16"]], "<u2")
            if len(topo["Indices"]):
                raise ValueError(f"mesh {m['Name']}: 32-bit index buffers are not supported")
            for extra in ("VertexToVertexMap", "VertexToTriangleMap", "SideToNeighborMap",
                          "BonesForTriangle", "TriangleToBoneIndices", "PolygonIndexStarts",
                          "PolygonIndices", "TriAnnotationSets"):
                if len(topo[extra]):
                    raise ValueError(f"mesh {m['Name']}: topology has non-empty '{extra}' (unsupported)")
            pat = ind.tobytes()
            pos = raw0.find(pat)
            if pos < 0 or raw0.find(pat, pos + 2) >= 0:
                raise ValueError(f"mesh {m['Name']}: index array not found uniquely")
            if pos in used_idx:
                raise ValueError("two meshes share one index buffer (unsupported)")
            used_idx.add(pos)
            L["idx"], L["nidx"] = pos, len(ind)
            s16 = [s for s in rel_dst.get((0, pos), []) if u32(s - 4) == len(ind)]
            if len(s16) != 1:
                raise ValueError(f"mesh {m['Name']}: index array reference not found")
            L["idx_cnt"] = s16[0] - 4
            g = rel_src.get(s16[0] - 24)
            ngrp = len(topo["Groups"])
            if g is None or u32(s16[0] - 28) != ngrp:
                raise ValueError(f"mesh {m['Name']}: material-group array not found")
            L["grp"], L["grp_cnt"], L["ngrp"] = g[2], s16[0] - 28, ngrp
            for gi, gr in enumerate(topo["Groups"]):
                if struct.unpack_from("<III", raw0, g[2] + 12 * gi) != \
                        (gr["MaterialIndex"], gr["TriFirst"], gr["TriCount"]):
                    raise ValueError(f"mesh {m['Name']}: material group {gi} mismatch")
            # bounds + scale
            ext = m["ExtendedData"]
            bbp = struct.pack("<6f", *ext["bbMin"], *ext["bbMax"])
            bb = raw0.find(bbp)
            L["bb"] = bb if bb >= 0 and raw0.find(bbp, bb + 1) < 0 else None
            L["vscale"] = None
            if L["bb"] is not None:
                sp = struct.pack("<f", ext["VertexScale"])
                after = raw0.find(sp, bb + 24, bb + 24 + 96)
                before = raw0.rfind(sp, max(0, bb - 64), bb)
                L["vscale"] = after if after >= 0 else (before if before >= 0 else None)
            # per-bone OBBs
            obbs = []
            for bbind in m["BoneBindings"]:
                p = struct.pack("<6f", *bbind["OBBMin"], *bbind["OBBMax"])
                hits, o = [], raw0.find(p)
                while o >= 0:
                    r = rel_src.get(o - 8)
                    if r and self._cstr(r[1], r[2]) == bbind["BoneName"]:
                        hits.append(o)
                    o = raw0.find(p, o + 1)
                obbs.append(hits[0] if len(hits) == 1 else None)
            L["obb"] = obbs
            self.loc.append(L)

    def _cstr(self, si, off):
        d = self.g2.sections[si].data
        return bytes(d[off:d.index(b"\0", off)]).decode("utf-8", "replace")

    # ---------------------------------------------------------------- accessors
    def vertex_array(self, mi):
        L = self.loc[mi]
        return np.frombuffer(self.raw0, VTX, L["nv"], L["vtx"])

    def mesh_scale(self, mi):
        return float(self.meshes[mi]["ExtendedData"]["VertexScale"])

    def bone_local_map(self, mi):
        """skeleton bone index -> mesh-local binding index (nearest bound ancestor if unbound)."""
        local = {self.n2b[b["BoneName"]]: j for j, b in enumerate(self.meshes[mi]["BoneBindings"])}
        out = np.full(len(self.bones), -1, np.int64)
        for b in range(len(self.bones)):
            c = b
            while c >= 0 and c not in local:
                c = self.bones[c]["ParentIndex"]
            out[b] = local[c] if c >= 0 else -1
        direct = np.array([b in local for b in range(len(self.bones))])
        return out, direct

    def mesh_index(self, name):
        key = name.strip().lower()
        for i, m in enumerate(self.meshes):
            if m["Name"].lower() == key:
                return i
        return None

    # ----------------------------------------------------------------- editing
    def apply(self, edits, log=print):
        """edits: {mesh_index: MeshData}.  Returns a new Gr2 with the geometry replaced."""
        g2 = copy.deepcopy(self.g2)
        s0 = g2.sections[0]
        patches, repl = [], []
        pk = lambda fmt, off, *v: patches.append((off, struct.pack(fmt, *v)))

        # ---- global quantisation scale
        old_scale = max(self.mesh_scale(i) for i in range(len(self.meshes)))
        need = max((np.abs(e.P).max() for e in edits.values()), default=0.0)
        new_scale = old_scale if need <= old_scale else float(need) * (1 + 1e-6)
        rescale = new_scale != old_scale
        if rescale:
            log(f"  VertexScale {old_scale:.6f} -> {new_scale:.6f} "
                f"(re-quantising untouched meshes too)")

        for mi, m in enumerate(self.meshes):
            L = self.loc[mi]
            if mi not in edits and not rescale:
                continue
            old_arr = self.vertex_array(mi)
            pad_w = int(old_arr["pos"][0, 3])
            if mi in edits:
                e = edits[mi]
                lmap, direct = self.bone_local_map(mi)
                J = np.where(e.J >= 0, lmap[np.clip(e.J, 0, None)], -1)
                W = np.where(e.J >= 0, e.W, 0.0)
                moved = int(((e.J >= 0) & (e.W > 0) & ~direct[np.clip(e.J, 0, None)]).sum())
                if moved:
                    log(f"  {m['Name']}: {moved} influences on bones not bound to this mesh "
                        f"were moved to their nearest bound ancestor")
                J = np.where(J < 0, 0, J)
                dead = W.sum(1) <= 1e-9
                if dead.any():
                    log(f"  {m['Name']}: {int(dead.sum())} vertices had no usable skin weights "
                        f"-> bound 100% to binding 0")
                    W[dead] = [1, 0, 0, 0]
                    J[dead] = 0
                tris = np.concatenate(e.groups) if e.groups else np.zeros((0, 3), np.int64)
                if len(e.P) > 65535:
                    raise ValueError(f"{m['Name']}: {len(e.P)} vertices (max 65535 for 16-bit indices)")
                if len(e.groups) != L["ngrp"]:
                    raise ValueError(f"{m['Name']}: expected {L['ngrp']} material group(s) "
                                     f"(glTF primitives), got {len(e.groups)}")
                T = e.T if e.T is not None else compute_tangents(e.P, e.N, e.UV, tris)
                arr = encode_vertices(e.P, e.N, T, e.UV, J, W, new_scale, pad_w)
                idx = tris.astype("<u2").ravel()
                nv, nt = len(arr), len(tris)
                # --- in-place patches
                pk("<I", L["vtx_cnt"], nv)
                s0.mixed[L["mixed"]][0] = nv
                pk("<I", L["idx_cnt"], len(idx))
                first = 0
                for gi, gt in enumerate(e.groups):
                    mat = struct.unpack_from("<I", self.raw0, L["grp"] + 12 * gi)[0]
                    pk("<III", L["grp"] + 12 * gi, mat, first, len(gt))
                    first += len(gt)
                if L["bb"] is not None:
                    pk("<6f", L["bb"], *e.P.min(0), *e.P.max(0))
                else:
                    log(f"  {m['Name']}: bounding-box field not located - left unchanged")
                self._patch_obbs(mi, arr, e.P, new_scale, patches)
                ib = idx.tobytes()
                ib += b"\0" * ((-(len(ib) - 2 * L["nidx"])) % 4)
                repl.append((L["idx"], 2 * L["nidx"], ib))
                log(f"  {m['Name']}: verts {L['nv']} -> {nv}, tris {L['nidx'] // 3} -> {nt}")
            else:  # untouched but re-quantised because the scale grew
                d = decode_vertices(old_arr, self.mesh_scale(mi))
                arr = old_arr.copy()
                arr["pos"][:, :3] = np.clip(np.rint(d["P"] / new_scale * 32767.0), -32767, 32767)
            repl.append((L["vtx"], 28 * L["nv"], arr.tobytes()))
            if rescale:
                if L["vscale"] is None:
                    raise ValueError(f"{m['Name']}: VertexScale field not located; cannot grow it")
                pk("<f", L["vscale"], new_scale)

        for off, b in patches:                       # 1) in-place edits on the original layout
            s0.data[off:off + len(b)] = b
        for start, old_len, new in sorted(repl, key=lambda r: -r[0]):   # 2) resizes, high -> low
            g2.replace_range(0, start, old_len, new)
        return g2

    def _patch_obbs(self, mi, arr, P, scale, patches):
        """Regenerate per-bone oriented bounding boxes (bone-local min/max of weighted vertices)."""
        L, m = self.loc[mi], self.meshes[mi]
        for j, bb in enumerate(m["BoneBindings"]):
            off = L["obb"][j]
            if off is None:
                continue
            sel = ((arr["bi"] == j) & (arr["bw"] > 0)).any(1)
            if not sel.any():
                continue
            M = self.IWT[self.n2b[bb["BoneName"]]]
            Q = (np.c_[P[sel], np.ones(sel.sum())] @ M)[:, :3]
            patches.append((off, struct.pack("<6f", *Q.min(0), *Q.max(0))))


# =====================================================================================
#  CLI: repack / selftest
# =====================================================================================
def _same(a, b):
    return repr(a) == repr(b)


def selftest(path):
    print(f"== self-test on {path}")
    mdl = D2RModel(path)
    g2 = mdl.g2
    print(f"sections: {[len(s.data) for s in g2.sections]}  relocs(sec0)={len(g2.sections[0].relocs)}")
    out = g2.to_bytes()
    print(f"1. no-op repack: {len(mdl.raw)} -> {len(out)} bytes (all sections uncompressed)")
    fi2 = read_raw(out).file_info
    ok = _same(mdl.fi, fi2)
    print(f"   object graph identical after repack: {ok}")
    out2 = Gr2.load(out).to_bytes()
    print(f"   save(load(save(x))) is byte-identical: {out == out2}")
    crc = struct.unpack_from('<I', out, 0x28)[0]
    print(f"   header CRC {crc:08x} matches data: {crc == zlib.crc32(out[g2.table_off:]) & 0xffffffff}")
    # original file CRC convention
    ocrc = struct.unpack_from('<I', mdl.raw, 0x28)[0]
    print(f"   (original file CRC rule reproduces stored value: "
          f"{ocrc == zlib.crc32(mdl.raw[0x68:]) & 0xffffffff})")
    print("2. vertex quantiser round-trip (decode -> encode):")
    for mi, m in enumerate(mdl.meshes):
        a = mdl.vertex_array(mi)
        d = decode_vertices(a, mdl.mesh_scale(mi))
        lm = np.array([[b for b in row] for row in d["BI"]])
        b = encode_vertices(d["P"], d["N"], d["T"], d["UV"], lm, d["BW"], mdl.mesh_scale(mi), int(a["pos"][0, 3]))
        same = {f: float(np.mean(np.all(a[f].reshape(len(a), -1) == b[f].reshape(len(b), -1), 1)))
                for f in ("pos", "bw", "bi", "n", "t", "uv")}
        print(f"   {m['Name']}: identical-row fraction per field {same}")
    print("3. per-bone OBB regeneration vs stored values:")
    for mi, m in enumerate(mdl.meshes):
        a = mdl.vertex_array(mi)
        P = decode_vertices(a, mdl.mesh_scale(mi))["P"]
        worst, n_ok, n_all = 0.0, 0, 0
        for j, bb in enumerate(m["BoneBindings"]):
            sel = ((a["bi"] == j) & (a["bw"] > 0)).any(1)
            if not sel.any():
                continue
            Q = (np.c_[P[sel], np.ones(sel.sum())] @ mdl.IWT[mdl.n2b[bb["BoneName"]]])[:, :3]
            err = max(np.abs(Q.min(0) - np.array(bb["OBBMin"])).max(), np.abs(Q.max(0) - np.array(bb["OBBMax"])).max())
            worst, n_all = max(worst, err), n_all + 1
            n_ok += err < 2e-3
        print(f"   {m['Name']}: {n_ok}/{n_all} bones within 2e-3, worst error {worst:.4f}")
    print("4. located fields:", [{k: (v if not isinstance(v, list) else f"{sum(x is not None for x in v)}/{len(v)}")
                                  for k, v in L.items() if k in ('bb', 'vscale', 'obb', 'grp')} for L in mdl.loc])
    return ok


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "repack":
        mdl = D2RModel(sys.argv[2])
        n = mdl.g2.save(sys.argv[3])
        print(f"wrote {sys.argv[3]} ({n} bytes)")
    elif len(sys.argv) >= 3 and sys.argv[1] == "selftest":
        sys.exit(0 if selftest(sys.argv[2]) else 1)
    else:
        sys.exit("usage: d2r_model_io.py repack IN.model OUT.model | selftest IN.model")
