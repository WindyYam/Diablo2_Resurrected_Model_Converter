#!/usr/bin/env python3
"""
Diablo II: Resurrected .texture  <->  DDS converter.

Reverse-engineered layout (little endian), verified on amazon_body_alb.texture:

  0x00  4s   magic            "<DE(" (3C 44 45 28)
  0x04  u32  format word      bytes: [0]=0x3E [1]=0x00 [2]=0x02 [3]=mip count
                              (low 3 bytes = format/type id, top byte = mip count)
  0x08  u32  width
  0x0C  u32  height
  0x10  u32  1                (array size / depth - always 1 in the sample)
  0x14  u32  0
  0x18  u32  0
  0x1C  u32  mip count        (11 for 1024x1024)
  0x20  u32  4                (self-relative offset to the mip table: 0x20 + 4 = 0x24)
  0x24  mip table, `mip count` entries of 8 bytes:
            u32 size           byte size of this mip level
            u32 rel_offset     offset to the data, RELATIVE TO THE POSITION OF THIS FIELD
  ....  pixel data of every mip, largest first, tightly packed
        (identical to the payload of a DDS file)

For 1024x1024 / 11 mips the table ends at 0x7C and the data starts at 0x7C.
The pixel data in the sample is BC3 (DXT5).

Usage:
  d2r_texture.py info     in.texture
  d2r_texture.py to-dds   in.texture out.dds   [--format bc3|bc7|...]  [--dx10]
  d2r_texture.py from-dds in.dds out.texture   [--template orig.texture] [--fmt-word 0x0B02003E]
"""
import argparse
import struct
import sys

MAGIC = b"<DE("

# --------------------------------------------------------------------------
# Format tables
# --------------------------------------------------------------------------
# DXGI formats we know how to describe: name -> (dxgi id, bytes per 4x4 block, legacy fourcc)
DXGI = {
    "bc1": (71, 8, b"DXT1"),
    "bc1_srgb": (72, 8, None),
    "bc2": (74, 16, b"DXT3"),
    "bc2_srgb": (75, 16, None),
    "bc3": (77, 16, b"DXT5"),
    "bc3_srgb": (78, 16, None),
    "bc4": (80, 8, b"BC4U"),
    "bc4_snorm": (81, 8, b"BC4S"),
    "bc5": (83, 16, b"BC5U"),
    "bc5_snorm": (84, 16, b"BC5S"),
    "bc6h": (95, 16, None),
    "bc7": (98, 16, None),
    "bc7_srgb": (99, 16, None),
    "rgba8": (28, 0, None),      # uncompressed 32bpp
    "rgba8_srgb": (29, 0, None),
    "bgra8": (87, 0, None),
    "bgra8_srgb": (91, 0, None),
}
DXGI_BY_ID = {v[0]: k for k, v in DXGI.items()}
FOURCC_TO_NAME = {v[2]: k for k, v in DXGI.items() if v[2]}

# Format word (low 3 bytes of the dword at 0x04) -> format name.
# Only what has been verified from real files goes in here; extend as you
# discover more (run `info` on other .texture files and look at the word).
KNOWN_FORMAT_WORDS = {
    0x02003E: "bc3",
}
NAME_TO_WORD = {v: k for k, v in KNOWN_FORMAT_WORDS.items()}


def mip_size(name, w, h):
    _, bpb, _ = DXGI[name]
    if bpb == 0:  # uncompressed 32bpp
        return w * h * 4
    return max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * bpb


def guess_format_from_sizes(w, h, size0):
    """When the format word is unknown, guess the family from the byte size."""
    if size0 == w * h * 4:
        return "rgba8"
    blocks = max(1, (w + 3) // 4) * max(1, (h + 3) // 4)
    if size0 == blocks * 8:
        return "bc1"
    if size0 == blocks * 16:
        return "bc3"  # could also be bc5/bc7 - override with --format
    return None


# --------------------------------------------------------------------------
# .texture reader / writer
# --------------------------------------------------------------------------
class Texture:
    def __init__(self, width, height, mips, fmt_word, extra=(1, 0, 0)):
        self.width, self.height = width, height
        self.mips = mips               # list of bytes, largest first
        self.fmt_word = fmt_word       # full dword at 0x04
        self.extra = extra             # dwords at 0x10, 0x14, 0x18


def read_texture(path):
    d = open(path, "rb").read()
    if d[:4] != MAGIC:
        raise ValueError("not a D2R .texture file (bad magic %r)" % d[:4])
    fmt_word, w, h, e0, e1, e2, nmips, tbl_rel = struct.unpack_from("<8I", d, 4)
    tbl = 0x20 + tbl_rel
    mips = []
    for i in range(nmips):
        pos = tbl + 8 * i
        size, rel = struct.unpack_from("<II", d, pos)
        off = pos + 4 + rel
        if off + size > len(d):
            raise ValueError("mip %d points outside the file" % i)
        mips.append(d[off:off + size])
    return Texture(w, h, mips, fmt_word, (e0, e1, e2))


def write_texture(path, tex):
    n = len(tex.mips)
    tbl = 0x24
    data_start = tbl + 8 * n
    out = bytearray()
    word = (tex.fmt_word & 0x00FFFFFF) | (n << 24)
    out += MAGIC
    out += struct.pack("<8I", word, tex.width, tex.height,
                       *tex.extra, n, tbl - 0x20)
    assert len(out) == 0x24
    off = data_start
    for i, m in enumerate(tex.mips):
        pos = tbl + 8 * i
        out += struct.pack("<II", len(m), off - (pos + 4))
        off += len(m)
    for m in tex.mips:
        out += m
    open(path, "wb").write(out)


# --------------------------------------------------------------------------
# DDS reader / writer
# --------------------------------------------------------------------------
DDSD_CAPS, DDSD_HEIGHT, DDSD_WIDTH = 0x1, 0x2, 0x4
DDSD_PIXELFORMAT, DDSD_MIPMAPCOUNT, DDSD_LINEARSIZE = 0x1000, 0x20000, 0x80000
DDPF_ALPHAPIXELS, DDPF_FOURCC, DDPF_RGB = 0x1, 0x4, 0x40


def write_dds(path, tex, name, force_dx10=False):
    dxgi_id, bpb, fourcc = DXGI[name]
    n = len(tex.mips)
    use_dx10 = force_dx10 or fourcc is None
    flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_LINEARSIZE
    caps = 0x1000
    if n > 1:
        flags |= DDSD_MIPMAPCOUNT
        caps |= 0x8 | 0x400000
    if bpb == 0:  # uncompressed: pitch instead of linear size
        flags = (flags & ~DDSD_LINEARSIZE) | 0x8
        pitch = tex.width * 4
    else:
        pitch = len(tex.mips[0])
    if use_dx10 or bpb == 0:
        pf = struct.pack("<II4sIIIII", 32, DDPF_FOURCC, b"DX10", 0, 0, 0, 0, 0)
    else:
        pf = struct.pack("<II4sIIIII", 32, DDPF_FOURCC, fourcc, 0, 0, 0, 0, 0)
    hdr = struct.pack("<4sIIIIIII11I", b"DDS ", 124, flags, tex.height, tex.width,
                      pitch, 0, n, *([0] * 11))
    hdr += pf + struct.pack("<IIIII", caps, 0, 0, 0, 0)
    if use_dx10 or bpb == 0:
        hdr += struct.pack("<IIIII", dxgi_id, 3, 0, 1, 0)  # 3 = TEXTURE2D
    with open(path, "wb") as f:
        f.write(hdr)
        for m in tex.mips:
            f.write(m)


def read_dds(path):
    d = open(path, "rb").read()
    if d[:4] != b"DDS ":
        raise ValueError("not a DDS file")
    (size, flags, h, w, pitch, depth, nmips) = struct.unpack_from("<7I", d, 4)
    pf_flags, fourcc = struct.unpack_from("<I4s", d, 80)
    pos = 128
    name = None
    if pf_flags & DDPF_FOURCC and fourcc == b"DX10":
        dxgi_id, rdim, misc, arr, _ = struct.unpack_from("<5I", d, 128)
        pos = 148
        name = DXGI_BY_ID.get(dxgi_id)
        if name is None:
            raise ValueError("unsupported DXGI format id %d" % dxgi_id)
        if arr != 1 or (misc & 4):
            raise ValueError("texture arrays / cubemaps are not supported")
    elif pf_flags & DDPF_FOURCC:
        name = FOURCC_TO_NAME.get(fourcc)
        if name is None:
            raise ValueError("unsupported FourCC %r" % fourcc)
    elif pf_flags & DDPF_RGB:
        bits, rm, gm, bm, am = struct.unpack_from("<5I", d, 88)
        if bits != 32:
            raise ValueError("only 32-bit uncompressed DDS is supported")
        name = "bgra8" if rm == 0x00FF0000 else "rgba8"
    else:
        raise ValueError("unrecognised DDS pixel format")
    if not (flags & DDSD_MIPMAPCOUNT) or nmips == 0:
        nmips = 1
    mips = []
    mw, mh = w, h
    for _ in range(nmips):
        sz = mip_size(name, mw, mh)
        if pos + sz > len(d):
            raise ValueError("DDS truncated")
        mips.append(d[pos:pos + sz])
        pos += sz
        mw, mh = max(1, mw // 2), max(1, mh // 2)
    return Texture(w, h, mips, 0), name


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def resolve_format(tex, override=None):
    if override:
        return override.lower()
    name = KNOWN_FORMAT_WORDS.get(tex.fmt_word & 0x00FFFFFF)
    if name:
        return name
    name = guess_format_from_sizes(tex.width, tex.height, len(tex.mips[0]))
    print("warning: unknown format word 0x%08X, guessing '%s' from data size. "
          "Use --format to override." % (tex.fmt_word, name), file=sys.stderr)
    if name is None:
        raise SystemExit("cannot determine pixel format; pass --format")
    return name


def cmd_info(a):
    t = read_texture(a.input)
    name = KNOWN_FORMAT_WORDS.get(t.fmt_word & 0xFFFFFF, "unknown")
    print("size        : %dx%d" % (t.width, t.height))
    print("format word : 0x%08X  (type id 0x%06X, mips byte %d) -> %s"
          % (t.fmt_word, t.fmt_word & 0xFFFFFF, t.fmt_word >> 24, name))
    print("extra dwords:", t.extra)
    print("mips        :", len(t.mips))
    for i, m in enumerate(t.mips):
        print("   mip %2d  %7d bytes" % (i, len(m)))


def cmd_to_dds(a):
    t = read_texture(a.input)
    name = resolve_format(t, a.format)
    exp = mip_size(name, t.width, t.height)
    if exp != len(t.mips[0]):
        raise SystemExit("format '%s' expects %d bytes for mip 0 but file has %d"
                         % (name, exp, len(t.mips[0])))
    write_dds(a.output, t, name, a.dx10)
    print("wrote %s (%s, %dx%d, %d mips)" % (a.output, name, t.width, t.height, len(t.mips)))


def cmd_from_dds(a):
    t, name = read_dds(a.input)
    if a.fmt_word is not None:
        word = int(a.fmt_word, 0)
    elif a.template:
        tpl = read_texture(a.template)
        word = tpl.fmt_word
        t.extra = tpl.extra
        tname = KNOWN_FORMAT_WORDS.get(word & 0xFFFFFF)
        if tname and tname != name:
            print("warning: template is %s but DDS is %s" % (tname, name), file=sys.stderr)
    else:
        base = NAME_TO_WORD.get(name)
        if base is None:
            raise SystemExit(
                "The .texture format word for '%s' is not known yet.\n"
                "Pass --template <an original .texture of the same format> or "
                "--fmt-word 0x...." % name)
        word = base
    t.fmt_word = word
    write_texture(a.output, t)
    print("wrote %s (%s, %dx%d, %d mips, word 0x%08X)"
          % (a.output, name, t.width, t.height, len(t.mips), (word & 0xFFFFFF) | (len(t.mips) << 24)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("info"); s.add_argument("input"); s.set_defaults(f=cmd_info)
    s = sp.add_parser("to-dds"); s.add_argument("input"); s.add_argument("output")
    s.add_argument("--format", choices=sorted(DXGI)); s.add_argument("--dx10", action="store_true")
    s.set_defaults(f=cmd_to_dds)
    s = sp.add_parser("from-dds"); s.add_argument("input"); s.add_argument("output")
    s.add_argument("--template"); s.add_argument("--fmt-word")
    s.set_defaults(f=cmd_from_dds)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
