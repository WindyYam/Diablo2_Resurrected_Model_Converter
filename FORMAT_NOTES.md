# D2R `.model` format notes (from legs_med_lod0.model)

## Container: Granny 3D (GR2), v7, 64-bit little-endian
| Offset | Meaning |
|---|---|
| 0x00 | 16-byte magic `E5 9B 49 5E 6F 63 1F 14 1E 13 EB A9 90 BE ED C4` (GR2, 64-bit LE) |
| 0x10 | u32 headerSize (0x1C8), u32 headerFormat (0 = uncompressed header) |
| 0x20 | u32 version (7), u32 fileSize, u32 crc, u32 sectionsOffset (0x48), u32 numSections (8) |
| 0x34 | rootType {u32 section, u32 offset} -> section 6; rootNode -> section 0 offset 0; u32 tag 0x80000039 |
| 0x68 | 8 x 44-byte section headers |

Section header (11 x u32): compression, offsetInFile, compressedSize, uncompressedSize, alignment,
first16bit, first8bit, relocationsOffset, numRelocations, mixedMarshallingOffset, numMixedMarshalling.

* compression 4 = **BitKnit2** (stream starts with bytes `B1 75`). 0 = none.
* Relocation table of a compressed section = u32 compressedSize + BitKnit2 stream of numRelocations*12 bytes
  {u32 offsetInSection, u32 targetSection, u32 targetOffset}. Mixed-marshalling table is the same idea (16 B/entry).
* In this file only sections 0 (data, 215,680 B -> 411,344 B) and 6 (type info, 636 B -> 20,944 B) are non-empty.

## Content (reflected via Granny type graph)
Root `FileInfo`: ArtToolInfo (Maya 2018, UnitsPerMeter 3.28084, Y-up), Textures, Materials, Skeletons,
VertexDatas, TriTopologies, Meshes, Models, TrackGroups, Animations, ExtendedData.

This file: 1 skeleton (`pelvis_bind_jnt`, 125 bones, ParentIndex + local Transform + InverseWorldTransform),
3 meshes (`legs_lit_bodyShape` 2118v/3498t, `legs_lit_armorShape` 6065v/8676t, `light_skirtShape` 1855v/2742t),
8 texture paths (`data/hd/character/player/amazon/textures/*.texture`), 11 material nodes (Maya shader graph dump).
No animations (they live in separate `.animations` files).

## Vertex layout (per mesh `PrimaryVertexData.Vertices`)
| Member | Granny type | Decode |
|---|---|---|
| Position | BinormalInt16 x4 | xyz = i16 / 32767 * Mesh.ExtendedData.VertexScale ; w = constant 5336 (padding) |
| BoneWeights | NormalUInt8 x4 | u8 / 255 (sums to 255) |
| BoneIndices | UInt8 x4 | index into Mesh.BoneBindings[] -> BoneName -> skeleton bone |
| Normal | BinormalInt8 x4 | xyz = i8 / 127 |
| Tangent | BinormalInt8 x4 | xyz = i8 / 127, w = +/-127 (handedness) |
| TextureCoordinates0 | Real16 x2 | half float; Maya V-up |
Indices: `PrimaryTopology.Indices16` (uint16 triangle list). Groups give MaterialIndex/TriFirst/TriCount.
Verified: decoded positions reproduce Mesh.ExtendedData.bbMin/bbMax to 4 decimals.

## Writer-relevant facts (all verified by test_writer.py)
* **CRC** = CRC32 of the file from the start of the section table (0x68) to the end. Reproduces the stored value on the original.
* **Array reference** = u32 count + 64-bit pointer (12 bytes, no padding); the pointer is a relocation entry, count at `src-4`.
* **TriTopology** = Groups(12) | Indices(12) | Indices16(12) | ...; Groups entry = {MaterialIndex, TriFirst, TriCount} (3 x u32).
  The writer locates `Indices16` through its relocation and matching group-array contents, then uses topology-array order
  to disambiguate duplicate data. Bounding-box fields are patched by reflected field offsets, not ambiguous value searches.
* **Mixed-marshalling entry** (16 B) = {count, offsetInSection, typeSection, typeOffset}; for meshes it is the vertex array
  (count + start offset), so its count must follow the vertex count.
* **first16bit / first8bit** section markers = start of the 16-bit / 8-bit data regions. The three Indices16 arrays are packed
  back to back and end exactly at first8bit; they move when an index array changes size.
* Granny pointers are 4-byte aligned in this file (8-byte pointers at 4-aligned offsets) -> all resizes keep multiples of 4.
* **VertexScale** is identical in every mesh and equals the largest |coordinate| of the whole model (skirt top y = 4.337393 = int16 32767).
  Per-mesh bbMin/bbMax live in the same ExtendedData block, in float32.
* **Per-bone OBBs** (BoneBindings[].OBBMin/OBBMax) = min/max of the bone-local (InverseWorldTransform) positions of all vertices
  with weight > 0 on that bone. Regenerated values match the stored ones to 1e-4 on every bone of the sample.
* **Tangent handedness** in the file matches a tangent computed with UV exactly as stored (Maya V-up); glTF needs the opposite sign
  after the V flip - both tools handle this.
* Unused bone-index slots (weight 0) hold arbitrary leftovers in the original; the writer repeats the first index (no effect).
* An uncompressed rewrite (compression = 0 for every section, relocation / marshalling tables raw) reads back to an identical object
  graph and is byte-stable under repeated load/save.

## Not yet verified
* **Whether D2R itself loads a file whose sections are stored uncompressed** (Granny supports it; no game was available here).
  `legs_med_lod0_UNCOMPRESSED_noop.model` is a pure container repack for exactly that test.
* Blender-exporter specifics (joint ordering, skinned-node transforms) - handled per the glTF spec, not tested with Blender itself.
* The `.texture` files (separate format) and `.skeleton` / `.animations` siblings.
