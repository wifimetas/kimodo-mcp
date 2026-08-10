# SPDX-License-Identifier: Apache-2.0
"""Headless Kimodo generation runner used by the Kimodo MCP server.

Runs inside the Kimodo virtualenv (it imports torch + kimodo). Reads a JSON job
file, generates motion, writes NPZ / BVH / demo-example outputs, and prints a
single machine-readable result line:

    __KIMODO_RESULT__ {"ok": true, ...}

Why this exists instead of just shelling out to ``kimodo_gen``:
  * The Aero-Ex low-VRAM fork only wires multi-tier offloading into the demo app
    (``python -m kimodo.demo --offload``). The plain CLI does not use it, so on a
    6 GB card the 5.4 GB text encoder and the diffusion model fight over VRAM.
    This runner reproduces the demo's sequence: pre-encode prompts through the
    disk-backed CachedTextEncoder, evict the encoder, move the diffusion model
    back to the GPU, then denoise.
  * It degrades gracefully: on a stock upstream install (no memory_manager /
    embedding_cache) it behaves exactly like ``kimodo_gen``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback

# Must be set before torch allocates anything.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

RESULT_PREFIX = "__KIMODO_RESULT__"


def log(msg: str) -> None:
    print(f"[runner] {msg}", flush=True)


def emit(payload: dict) -> None:
    print(RESULT_PREFIX + " " + json.dumps(payload), flush=True)


# --------------------------------------------------------------------------
# helpers mirrored from kimodo/scripts/generate.py so output layout matches
# --------------------------------------------------------------------------
def _single_file_path(path: str, ext: str) -> str:
    if not path.endswith(ext):
        path = path.rstrip(os.sep) + ext
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def _output_dir_and_path(path: str, default_base: str, ext: str):
    folder = os.path.splitext(path)[0] if os.path.splitext(path)[1] else path
    os.makedirs(folder, exist_ok=True)
    base_name = os.path.basename(folder.rstrip(os.sep))
    return folder, os.path.join(folder, default_base + ext), base_name


def build_cfg_kwargs(job: dict) -> dict:
    cfg_type = job.get("cfg_type")
    cfg_weight = job.get("cfg_weight")

    if cfg_type == "nocfg":
        return {"cfg_type": "nocfg"}

    if cfg_weight is None and cfg_type is None:
        return {}

    if cfg_weight is None:
        raise ValueError(f"cfg_type={cfg_type!r} requires cfg_weight")

    if isinstance(cfg_weight, (int, float)):
        cfg_weight = [float(cfg_weight)]
    cfg_weight = [float(w) for w in cfg_weight]

    if cfg_type is None:
        cfg_type = "regular" if len(cfg_weight) == 1 else "separated"

    if cfg_type == "regular":
        if len(cfg_weight) != 1:
            raise ValueError("cfg_type 'regular' needs exactly one cfg_weight value")
        return {"cfg_type": "regular", "cfg_weight": cfg_weight[0]}

    if cfg_type == "separated":
        if len(cfg_weight) != 2:
            raise ValueError("cfg_type 'separated' needs exactly two cfg_weight values [text, constraint]")
        return {"cfg_type": "separated", "cfg_weight": cfg_weight}

    raise ValueError(f"unknown cfg_type {cfg_type!r}")


def split_prompts(job: dict, fps: float):
    """Return (texts, num_frames, durations)."""
    prompts = job.get("prompts")
    if isinstance(prompts, str):
        prompts = [p.strip() for p in prompts.split(".") if p.strip()]
    if not prompts:
        raise ValueError("no prompts given")
    texts = []
    for p in prompts:
        p = p.strip()
        if not p.endswith("."):
            p = p + "."
        texts.append(p)

    durations = job.get("durations") or [5.0]
    if isinstance(durations, (int, float)):
        durations = [float(durations)]
    durations = [float(d) for d in durations]
    if len(durations) == 1 and len(texts) > 1:
        durations = durations * len(texts)
    if len(durations) != len(texts):
        raise ValueError(
            f"got {len(texts)} prompt(s) but {len(durations)} duration(s); pass one duration or one per prompt"
        )

    num_frames = [int(round(d * fps)) for d in durations]
    if any(n < 2 for n in num_frames):
        raise ValueError("each duration must produce at least 2 frames")
    return texts, num_frames, durations


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, help="path to job JSON file")
    args = ap.parse_args()

    with open(args.job, "r", encoding="utf-8") as f:
        job = json.load(f)

    t0 = time.time()

    import torch  # noqa: E402

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log(f"device={device}")
    if device == "cpu":
        log("WARNING: no CUDA device visible; generation will be very slow")

    offload = bool(job.get("offload", True)) and device.startswith("cuda")

    memory_manager = None
    if offload:
        try:
            from kimodo.demo.memory_manager import manager as memory_manager  # type: ignore

            memory_manager.offload_enabled = True
            log("multi-tier offloading enabled (low-VRAM mode)")
        except Exception as exc:  # stock upstream install
            memory_manager = None
            log(f"offloading unavailable on this install ({exc}); continuing without it")

    from kimodo import load_model
    from kimodo.constraints import load_constraints_lst
    from kimodo.exports.motion_io import save_kimodo_npz
    from kimodo.tools import save_json, seed_everything

    model_name = job.get("model") or "kimodo-soma-rp"
    log(f"loading model {model_name} ...")
    model, resolved_model = load_model(
        model_name,
        device=device,
        default_family="Kimodo",
        return_resolved_name=True,
    )
    log(f"model loaded: {resolved_model} (fps={model.fps})")

    raw_encoder = getattr(model, "text_encoder", None)
    if memory_manager is not None:
        memory_manager.register_model(resolved_model, model)
        if raw_encoder is not None:
            memory_manager.register_encoder(raw_encoder)

    # Disk-backed embedding cache: lets us encode once, then evict the encoder.
    cached_encoder = None
    if raw_encoder is not None:
        try:
            from kimodo.demo.embedding_cache import CachedTextEncoder  # type: ignore

            cached_encoder = CachedTextEncoder(raw_encoder, model_name=resolved_model)
            model.text_encoder = cached_encoder
            log("text-embedding disk cache active")
        except Exception as exc:
            log(f"embedding cache unavailable ({exc}); encoding inline")

    texts, num_frames, durations = split_prompts(job, model.fps)
    for t, n in zip(texts, num_frames):
        log(f"prompt: {t!r} -> {n} frames")

    constraints_path = job.get("constraints_path") or None
    constraint_lst = []
    if constraints_path:
        constraint_lst = load_constraints_lst(constraints_path, model.skeleton)
        log(f"loaded {len(constraint_lst)} constraint set(s) from {constraints_path}")

    cfg_kwargs = build_cfg_kwargs(job)
    if cfg_kwargs:
        log(f"cfg: {cfg_kwargs}")

    seed = job.get("seed")
    if seed is not None:
        seed_everything(int(seed))
        log(f"seed={seed}")

    # ---- phase 1: encode prompts, then get the encoder out of VRAM ----
    if cached_encoder is not None and memory_manager is not None:
        prompt_set = set(texts)
        weights = cfg_kwargs.get("cfg_weight")
        if weights is None or (isinstance(weights, list) and any(w > 0 for w in weights)) or (
            isinstance(weights, float) and weights > 0
        ):
            # unconditional embeddings are needed for classifier-free guidance
            prompt_set.add("")
            prompt_set.add(" ")
        log("phase 1/2: encoding prompts ...")
        cached_encoder.reload()
        cached_encoder.prewarm(sorted(prompt_set))
        memory_manager.purge_encoder_completely()
        memory_manager.touch_and_move(resolved_model, device)
        memory_manager.report_residency()

    # ---- phase 2: denoise ----
    log("phase 2/2: generating ...")
    gen_t0 = time.time()
    post_processing = bool(job.get("postprocess", True))
    if "g1" in resolved_model:
        post_processing = False

    output = model(
        texts,
        num_frames,
        constraint_lst=constraint_lst,
        num_denoising_steps=int(job.get("diffusion_steps", 100)),
        num_samples=int(job.get("num_samples", 1)),
        multi_prompt=True,
        num_transition_frames=int(job.get("num_transition_frames", 5)),
        post_processing=post_processing,
        return_numpy=True,
        **cfg_kwargs,
    )
    gen_secs = time.time() - gen_t0
    log(f"generation finished in {gen_secs:.1f}s")

    if memory_manager is not None:
        try:
            memory_manager.purge_encoder_completely()
        except Exception:
            pass

    # ---- save ----
    n_samples = int(output["posed_joints"].shape[0])
    output_base = job["output_stem"]
    files = {"npz": [], "bvh": [], "csv": [], "amass": [], "example_dir": []}

    def slice_sample(i):
        return {
            k: (v[i] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == n_samples else v)
            for k, v in output.items()
        }

    if n_samples == 1:
        npz_path = _single_file_path(output_base, ".npz")
        save_kimodo_npz(npz_path, slice_sample(0))
        files["npz"].append(npz_path)
    else:
        out_dir, _, base_name = _output_dir_and_path(output_base, "motion", ".npz")
        for i in range(n_samples):
            p = os.path.join(out_dir, f"{base_name}_{i:02d}.npz")
            save_kimodo_npz(p, slice_sample(i))
            files["npz"].append(p)
    log(f"wrote {len(files['npz'])} npz file(s)")

    # AMASS (SMPLX models only)
    if resolved_model == "kimodo-smplx-rp":
        try:
            from kimodo.exports.smplx import AMASSConverter

            converter = AMASSConverter(skeleton=model.skeleton, fps=model.fps)
            if n_samples == 1:
                p = _single_file_path(output_base + "_amass", ".npz")
                converter.convert_save_npz(output, p)
                files["amass"].append(p)
            else:
                out_dir, _, _ = _output_dir_and_path(output_base, "amass", ".npz")
                p = os.path.join(out_dir, "amass.npz")
                converter.convert_save_npz(output, p)
                files["amass"].append(p)
        except Exception as exc:
            log(f"AMASS export failed: {exc}")

    # MuJoCo qpos CSV (G1 models only)
    if resolved_model == "kimodo-g1-rp":
        try:
            from kimodo.exports.mujoco import MujocoQposConverter

            converter = MujocoQposConverter(model.skeleton)
            qpos = converter.dict_to_qpos(output, device)
            if n_samples == 1:
                p = _single_file_path(output_base, ".csv")
                converter.save_csv(qpos, p)
            else:
                out_dir, _, base_name = _output_dir_and_path(output_base, "qpos", ".csv")
                p = os.path.join(out_dir, base_name + ".csv")
                converter.save_csv(qpos, p)
            files["csv"].append(p)
        except Exception as exc:
            log(f"MuJoCo CSV export failed: {exc}")

    # BVH (SOMA skeletons only)
    if job.get("bvh"):
        skeleton = model.skeleton
        if "somaskel" not in getattr(skeleton, "name", ""):
            log("BVH export skipped: only SOMA skeletons are supported")
        else:
            from kimodo.exports.bvh import save_motion_bvh
            from kimodo.skeleton import SOMASkeleton30, global_rots_to_local_rots

            if isinstance(skeleton, SOMASkeleton30):
                skeleton = skeleton.somaskel77.to(device)

            standard_tpose = bool(job.get("bvh_standard_tpose", False))

            def write_bvh(i, path):
                joints_pos = torch.from_numpy(output["posed_joints"][i]).to(device)
                joints_rot = torch.from_numpy(output["global_rot_mats"][i]).to(device)
                local_rot_mats = global_rots_to_local_rots(joints_rot, skeleton)
                root_positions = joints_pos[:, skeleton.root_idx, :]
                save_motion_bvh(
                    path,
                    local_rot_mats,
                    root_positions,
                    skeleton=skeleton,
                    fps=model.fps,
                    standard_tpose=standard_tpose,
                )

            if n_samples == 1:
                p = _single_file_path(output_base, ".bvh")
                write_bvh(0, p)
                files["bvh"].append(p)
            else:
                out_dir, _, base_name = _output_dir_and_path(output_base, "motion", ".bvh")
                for i in range(n_samples):
                    p = os.path.join(out_dir, f"{base_name}_{i:02d}.bvh")
                    write_bvh(i, p)
                    files["bvh"].append(p)
            log(f"wrote {len(files['bvh'])} bvh file(s)")

    # demo-compatible example dir (so the motion can be re-opened in kimodo_demo)
    if job.get("save_example_dir"):
        output_stem = os.path.splitext(output_base)[0].rstrip(os.sep)
        base_name = os.path.basename(output_stem)
        if n_samples == 1:
            example_dirs = [output_stem + "_example"]
            parent_dir = None
        else:
            parent_dir = output_stem + "_examples"
            os.makedirs(parent_dir, exist_ok=True)
            example_dirs = [os.path.join(parent_dir, f"{base_name}_example_{i:02d}") for i in range(n_samples)]

        if len(texts) == 1:
            meta_info = {"text": texts[0], "duration": durations[0]}
        else:
            meta_info = {"texts": texts, "durations": durations}
        meta_info["num_samples"] = int(job.get("num_samples", 1))
        if seed is not None:
            meta_info["seed"] = int(seed)
        meta_info["diffusion_steps"] = int(job.get("diffusion_steps", 100))
        if cfg_kwargs:
            ct = cfg_kwargs.get("cfg_type", "nocfg")
            cw = cfg_kwargs.get("cfg_weight")
            if ct == "nocfg":
                meta_info["cfg"] = {"enabled": False}
            elif ct == "separated":
                meta_info["cfg"] = {"enabled": True, "text_weight": cw[0], "constraint_weight": cw[1]}
            elif ct == "regular":
                meta_info["cfg"] = {"enabled": True, "text_weight": float(cw), "constraint_weight": float(cw)}

        for i, example_dir in enumerate(example_dirs):
            os.makedirs(example_dir, exist_ok=True)
            save_kimodo_npz(os.path.join(example_dir, "motion.npz"), slice_sample(i))
            if constraints_path:
                shutil.copy2(constraints_path, os.path.join(example_dir, "constraints.json"))
            save_json(os.path.join(example_dir, "meta.json"), meta_info)
            files["example_dir"].append(example_dir)
        log(f"wrote {len(example_dirs)} demo example dir(s)")

    result = {
        "ok": True,
        "model": resolved_model,
        "fps": float(model.fps),
        "prompts": texts,
        "durations": durations,
        "num_frames": num_frames,
        "total_frames": int(sum(num_frames)),
        "num_samples": n_samples,
        "seed": seed,
        "diffusion_steps": int(job.get("diffusion_steps", 100)),
        "cfg": cfg_kwargs or None,
        "constraints": constraints_path,
        "postprocess": post_processing,
        "offload": memory_manager is not None,
        "files": files,
        "generation_seconds": round(gen_secs, 2),
        "total_seconds": round(time.time() - t0, 2),
    }
    emit(result)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        hint = None
        text = f"{exc}\n{tb}".lower()
        if "out of memory" in text or "cuda oom" in text:
            hint = (
                "CUDA ran out of memory. Try: fewer samples (num_samples=1), shorter duration, "
                "fewer diffusion_steps, or set text_encoder_device='cpu' in the tool call."
            )
        elif "text encoder server" in text:
            hint = "Start the text encoder server (kimodo_textencoder) or check the local NF4 model path."
        emit({"ok": False, "error": str(exc), "hint": hint, "traceback": tb[-4000:]})
        raise SystemExit(1)
