# SPDX-License-Identifier: Apache-2.0
"""Headless Blender converter: Kimodo BVH -> FBX / glTF-binary.

Invoked by the Kimodo MCP server as:

    blender.exe -b --factory-startup -P blender_export.py -- \
        --input motion.bvh --output motion.fbx --target unreal

Design notes
------------
* Kimodo BVH is **Y-up, metres**. Blender's BVH importer converts Y-up -> Z-up
  with its default axis settings, which is what Unreal wants.
* Unreal treats 1 FBX unit as 1 cm. The reliable Blender recipe is: set the
  scene unit scale to 0.01 and export with "Apply Scalings = FBX All"; that
  writes centimetres without touching object transforms.
* Blender 4.5+/5.x ships a rewritten FBX exporter as ``bpy.ops.wm.fbx_export``
  alongside the classic ``bpy.ops.export_scene.fbx``. Both are probed.
"""

from __future__ import annotations

import json
import os
import sys

import bpy

RESULT_PREFIX = "__KIMODO_BLENDER_RESULT__"


def argv_after_dashes():
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return []


def parse_args():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input .bvh path")
    ap.add_argument("--output", required=True, help="output .fbx or .glb path")
    ap.add_argument(
        "--target",
        default="unreal",
        choices=("unreal", "unity", "raw"),
        help="unit/axis convention preset",
    )
    ap.add_argument("--scale", type=float, default=None, help="extra uniform scale applied on import")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--rename-root", default=None, help="rename the root bone (e.g. 'root' for UE)")
    return ap.parse_args(argv_after_dashes())


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_bvh(path: str, fps: float, scale: float):
    kwargs = dict(
        filepath=path,
        target="ARMATURE",
        global_scale=scale,
        frame_start=1,
        use_fps_scale=False,
        update_scene_fps=True,
        update_scene_duration=True,
        use_cyclic=False,
        rotate_mode="NATIVE",
        axis_forward="-Z",
        axis_up="Y",
    )
    try:
        bpy.ops.import_anim.bvh(**kwargs)
    except TypeError:
        # Older/newer builds may not expose every keyword; retry with the essentials.
        bpy.ops.import_anim.bvh(
            filepath=path,
            target="ARMATURE",
            global_scale=scale,
            update_scene_fps=True,
            update_scene_duration=True,
        )
    arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    if not arms:
        raise RuntimeError("BVH import produced no armature")
    arm = arms[0]
    clamp_scene_to_action(arm)
    return arm


def clamp_scene_to_action(arm) -> None:
    """Trim the scene range to the imported action.

    Blender's BVH importer does not always shrink scene.frame_end, which would
    otherwise bake a long static tail into the exported clip.
    """
    action = None
    if arm.animation_data and arm.animation_data.action:
        action = arm.animation_data.action

    start = end = None
    if action is not None:
        try:
            fr = action.frame_range
            start, end = float(fr[0]), float(fr[1])
        except Exception:
            pass
        if start is None or end is None or end <= start:
            keys = [k.co[0] for fc in action.fcurves for k in fc.keyframe_points]
            if keys:
                start, end = float(min(keys)), float(max(keys))

    if start is not None and end is not None and end > start:
        scene = bpy.context.scene
        scene.frame_start = int(round(start))
        scene.frame_end = int(round(end))


def export_fbx(path: str, target: str) -> str:
    scene = bpy.context.scene
    if target == "unreal":
        scene.unit_settings.system = "METRIC"
        scene.unit_settings.scale_length = 0.01
        apply_scale = "FBX_SCALE_ALL"
    elif target == "unity":
        scene.unit_settings.system = "METRIC"
        scene.unit_settings.scale_length = 1.0
        apply_scale = "FBX_SCALE_NONE"
    else:
        apply_scale = "FBX_SCALE_NONE"

    common = dict(
        filepath=path,
        use_selection=False,
        object_types={"ARMATURE"},
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,
        axis_forward="-Z",
        axis_up="Y",
        apply_scale_options=apply_scale,
        global_scale=1.0,
        path_mode="COPY",
    )

    if hasattr(bpy.ops.export_scene, "fbx"):
        bpy.ops.export_scene.fbx(**common)
        return "export_scene.fbx"

    if hasattr(bpy.ops.wm, "fbx_export"):
        # New (4.5+) exporter: much smaller keyword surface.
        bpy.ops.wm.fbx_export(filepath=path)
        return "wm.fbx_export"

    raise RuntimeError("no FBX exporter available in this Blender build")


def export_gltf(path: str) -> str:
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format="GLB" if path.lower().endswith(".glb") else "GLTF_SEPARATE",
        export_animations=True,
        export_frame_range=True,
        export_force_sampling=True,
        export_yup=True,
    )
    return "export_scene.gltf"


def main() -> int:
    args = parse_args()
    scale = args.scale if args.scale is not None else 1.0

    clear_scene()
    arm = import_bvh(args.input, args.fps, scale)

    if args.rename_root:
        try:
            bones = arm.data.bones
            root = next((b for b in bones if b.parent is None), None)
            if root is not None:
                old = root.name
                arm.data.bones[old].name = args.rename_root
        except Exception:
            pass

    scene = bpy.context.scene
    frame_count = scene.frame_end - scene.frame_start + 1
    dims = tuple(round(float(v), 4) for v in arm.dimensions)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    ext = os.path.splitext(args.output)[1].lower()
    if ext in (".glb", ".gltf"):
        exporter = export_gltf(args.output)
    else:
        exporter = export_fbx(args.output, args.target)

    result = {
        "ok": os.path.exists(args.output),
        "input": os.path.abspath(args.input),
        "output": os.path.abspath(args.output),
        "exporter": exporter,
        "target": args.target,
        "armature": arm.name,
        "bones": len(arm.data.bones),
        "frames": int(frame_count),
        "scene_fps": float(scene.render.fps / max(scene.render.fps_base, 1e-9)),
        "armature_dimensions_blender_units": dims,
        "bytes": os.path.getsize(args.output) if os.path.exists(args.output) else 0,
    }
    print(RESULT_PREFIX + " " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(RESULT_PREFIX + " " + json.dumps({"ok": False, "error": str(exc)}), flush=True)
        raise SystemExit(1)
