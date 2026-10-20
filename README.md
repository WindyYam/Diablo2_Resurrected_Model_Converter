# D2R Model Tool

Python utilities to convert Diablo II: Resurrected models to GLB for Blender editing, write edited geometry back to a model, and convert textures to or from DDS.

## GUI

Launch the graphical interface:

```powershell
py d2r_model_tool_gui.py
```

## Model Usage

Convert a D2R model to GLB:

```powershell
py d2r_model_to_glb.py input.model [output.glb]
```

Open the GLB in Blender, edit it, and export as **glTF Binary (.glb)**. Then write the edited geometry back using the original model as a template:

```powershell
py d2r_glb_to_model.py edited.glb original.model output.model
```

Keep mesh and bone names recognizable in Blender.

## Texture Usage

Show texture dimensions, format, and mip information:

```powershell
py d2r_texture.py info input.texture
```

Convert a D2R texture to DDS:

```powershell
py d2r_texture.py to-dds input.texture output.dds
```

Convert DDS back to a D2R texture. Use an original texture as a template to preserve its format settings:

```powershell
py d2r_texture.py from-dds input.dds output.texture --template original.texture
```

For more details on the model format and current limitations, see [FORMAT_NOTES.md](FORMAT_NOTES.md).