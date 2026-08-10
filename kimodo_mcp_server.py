#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Kimodo MCP server -- gives Claude Code hands-on control of NVIDIA Kimodo.

Zero third-party dependencies on purpose: this speaks MCP's stdio JSON-RPC
directly so it can run on the stock system Python *or* inside the Kimodo
virtualenv without ever pip-installing anything into a fragile ML environment.
All heavy lifting (torch, kimodo, Blender) happens in subprocesses.

Transport : stdio, newline-delimited JSON-RPC 2.0
Logging   : stderr only (stdout is reserved for the protocol)
"""

from __future__ import annotations

import glob
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional

SERVER_NAME = "kimodo"
SERVER_VERSION = "1.0.0"
SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")
HERE = os.path.dirname(os.path.abspath(__file__))

RESULT_PREFIX = "__KIMODO_RESULT__"
BLENDER_RESULT_PREFIX = "__KIMODO_BLENDER_RESULT__"


def log(msg: str) -> None:
    print(f"[kimodo-mcp] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def _first_existing(*paths: str) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _find_blender() -> Optional[str]:
    env = os.environ.get("KIMODO_BLENDER")
    if env and os.path.exists(env):
        return env
    which = shutil.which("blender")
    if which:
        return which
    patterns = [
        r"C:\Program Files\Blender Foundation\*\blender.exe",
        r"C:\Program Files (x86)\Blender Foundation\*\blender.exe",
        "/usr/bin/blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",
    ]
    found: List[str] = []
    for pat in patterns:
        found.extend(glob.glob(pat))
    found.sort(reverse=True)  # newest version first
    return found[0] if found else None


class Config:
    def __init__(self) -> None:
        cfg: Dict[str, Any] = {}
        cfg_path = os.path.join(HERE, "kimodo_mcp_config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception as exc:
                log(f"could not read config file: {exc}")
        self.raw = cfg

        self.kimodo_home = os.environ.get("KIMODO_HOME") or cfg.get("kimodo_home") or ""
        self.python = (
            os.environ.get("KIMODO_PYTHON")
            or cfg.get("python")
            or _first_existing(
                os.path.join(self.kimodo_home, "venv", "Scripts", "python.exe"),
                os.path.join(self.kimodo_home, "venv", "bin", "python"),
            )
            or sys.executable
        )
        self.repo = (
            os.environ.get("KIMODO_REPO")
            or cfg.get("repo")
            or _first_existing(os.path.join(self.kimodo_home, "kimodo"))
            or self.kimodo_home
        )
        self.output_dir = os.path.abspath(
            os.environ.get("KIMODO_OUTPUT_DIR")
            or cfg.get("output_dir")
            or os.path.join(self.kimodo_home or HERE, "kimodo_out")
        )
        self.blender = os.environ.get("KIMODO_BLENDER") or cfg.get("blender") or _find_blender()
        self.default_model = os.environ.get("KIMODO_MODEL") or cfg.get("default_model") or "kimodo-soma-rp"
        self.offload = str(os.environ.get("KIMODO_OFFLOAD") or cfg.get("offload", "1")).lower() not in (
            "0",
            "false",
            "no",
        )
        self.text_encoder_device = os.environ.get("TEXT_ENCODER_DEVICE") or cfg.get("text_encoder_device") or ""
        self.export_target = cfg.get("export_target", "unreal")

        self.jobs_dir = os.path.join(self.output_dir, "_jobs")
        os.makedirs(self.jobs_dir, exist_ok=True)


CFG = Config()


# ---------------------------------------------------------------------------
# model registry (mirrors kimodo/model/registry.py so we can answer offline)
# ---------------------------------------------------------------------------
MODELS = [
    {
        "name": "kimodo-soma-rp",
        "repo_id": "nvidia/Kimodo-SOMA-RP-v1.1",
        "skeleton": "SOMA (77-joint human)",
        "data": "Bones Rigplay 1 (700h)",
        "exports": ["npz", "bvh"],
        "notes": "Default. Best overall quality for human characters. Use this for game animation.",
    },
    {
        "name": "kimodo-soma-rp-v1",
        "repo_id": "nvidia/Kimodo-SOMA-RP-v1",
        "skeleton": "SOMA (77-joint human)",
        "data": "Bones Rigplay 1 (700h)",
        "exports": ["npz", "bvh"],
        "notes": "Previous release; keep for reproducing older results.",
    },
    {
        "name": "kimodo-soma-seed",
        "repo_id": "nvidia/Kimodo-SOMA-SEED-v1.1",
        "skeleton": "SOMA (77-joint human)",
        "data": "BONES-SEED (288h, public)",
        "exports": ["npz", "bvh"],
        "notes": "Weaker than RP; mainly for benchmark comparability.",
    },
    {
        "name": "kimodo-smplx-rp",
        "repo_id": "nvidia/Kimodo-SMPLX-RP-v1",
        "skeleton": "SMPL-X",
        "data": "Bones Rigplay 1 (700h)",
        "exports": ["npz", "amass npz"],
        "notes": "Use when a downstream tool wants AMASS/SMPL-X (e.g. GMR retargeting). R&D licence.",
    },
    {
        "name": "kimodo-g1-rp",
        "repo_id": "nvidia/Kimodo-G1-RP-v1",
        "skeleton": "Unitree G1 humanoid robot",
        "data": "Bones Rigplay 1 (700h)",
        "exports": ["npz", "mujoco qpos csv"],
        "notes": "Robot skeleton. Post-processing is force-disabled for G1.",
    },
    {
        "name": "kimodo-g1-seed",
        "repo_id": "nvidia/Kimodo-G1-SEED-v1",
        "skeleton": "Unitree G1 humanoid robot",
        "data": "BONES-SEED (288h)",
        "exports": ["npz", "mujoco qpos csv"],
        "notes": "Robot skeleton, public-data variant.",
    },
]


# ---------------------------------------------------------------------------
# built-in game-animation presets
# ---------------------------------------------------------------------------
def _p(name, prompt, duration, category, **kw):
    d = {
        "name": name,
        "prompt": prompt,
        "duration": duration,
        "category": category,
        "diffusion_steps": kw.get("diffusion_steps", 120),
        "cfg_type": kw.get("cfg_type", "separated"),
        "cfg_weight": kw.get("cfg_weight", [2.0, 2.0]),
        "notes": kw.get("notes", ""),
    }
    return d


PRESETS = [
    _p("idle", "A person stands still in a relaxed combat-ready stance, breathing, with small weight shifts.", 4.0,
       "locomotion", notes="Generate 4-6s and cut a loop from the middle."),
    _p("idle_alert", "A person stands alert, scanning left and right, weapon held low, shifting weight.", 5.0, "locomotion"),
    _p("walk_forward", "A person walks forward steadily at a constant pace.", 4.0, "locomotion",
       notes="Pair with a root2d straight-line path constraint for a perfectly straight cycle."),
    _p("run_forward", "A person runs forward quickly at a steady sprint.", 3.0, "locomotion"),
    _p("strafe_left", "A person side-steps to the left while facing forward, weapon raised.", 3.0, "locomotion"),
    _p("strafe_right", "A person side-steps to the right while facing forward, weapon raised.", 3.0, "locomotion"),
    _p("turn_180", "A person pivots and turns around one hundred and eighty degrees, then steadies.", 2.5, "locomotion"),
    _p("crouch_walk", "A person crouches low and moves forward carefully, staying low to the ground.", 4.0, "locomotion"),
    _p("jump", "A person crouches, jumps forward, and lands absorbing the impact with bent knees.", 3.0, "traversal"),
    _p("dodge_roll", "A person dives forward into a shoulder roll and comes back up to a standing stance.", 2.5, "traversal"),
    _p("vault", "A person runs forward, plants a hand and vaults over a waist-high obstacle, then keeps moving.", 3.5, "traversal"),
    _p("climb", "A person reaches up, pulls themselves up over a ledge and stands up.", 4.0, "traversal"),
    _p("melee_combo", "A person throws a heavy right punch, then a left hook, then a spinning kick.", 4.0, "combat"),
    _p("heavy_slam", "A person raises both arms overhead and slams down hard into the ground, then recovers.", 3.0, "combat"),
    _p("aim_rifle", "A person raises a rifle to the shoulder, aims forward steadily, then lowers it.", 4.0, "combat"),
    _p("hit_reaction", "A person is struck in the chest, staggers backward two steps, then recovers their stance.", 2.5, "combat"),
    _p("death_fall", "A person is hit, collapses forward and falls to the ground, then lies still.", 3.5, "combat"),
    _p("cast_ability", "A person plants their feet, raises one arm and channels energy forward, then releases it.", 3.5, "combat"),
    _p("pick_up", "A person walks forward, crouches down, picks something up from the ground, and stands up.", 5.0, "interaction"),
    _p("push_button", "A person steps forward, reaches out with the right hand and presses a panel at chest height.", 3.5, "interaction"),
    _p("look_around", "A person stops, looks left, looks right, then looks up as if scanning a large room.", 4.0, "interaction"),
    _p("stagger_exhausted", "An exhausted person walks forward slowly, limping and hunched over.", 5.0, "flavour"),
    _p("victory", "A person raises both fists in the air in a victory celebration, then lowers them.", 3.0, "flavour"),
]


# ---------------------------------------------------------------------------
# job manager
# ---------------------------------------------------------------------------
class Job:
    def __init__(self, job_id: str, kind: str, label: str, cmd: List[str], cwd: str, env: Dict[str, str]):
        self.id = job_id
        self.kind = kind
        self.label = label
        self.cmd = cmd
        self.cwd = cwd
        self.env = env
        self.status = "queued"
        self.created = time.time()
        self.started: Optional[float] = None
        self.finished: Optional[float] = None
        self.returncode: Optional[int] = None
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.log_path = os.path.join(CFG.jobs_dir, f"{job_id}.log")
        self.proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None

    def to_dict(self, include_tail: int = 0) -> dict:
        d = {
            "job_id": self.id,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.created)),
            "elapsed_seconds": round((self.finished or time.time()) - (self.started or self.created), 1),
            "returncode": self.returncode,
            "log_file": self.log_path,
        }
        if self.result is not None:
            d["result"] = self.result
        if self.error:
            d["error"] = self.error
        if include_tail:
            d["log_tail"] = self.tail(include_tail)
        return d

    def tail(self, lines: int = 40) -> List[str]:
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as f:
                return [ln.rstrip("\n") for ln in f.readlines()[-lines:]]
        except Exception:
            return []

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        self.status = "running"
        self.started = time.time()
        try:
            with open(self.log_path, "w", encoding="utf-8", errors="replace") as logf:
                logf.write(f"$ {' '.join(shlex.quote(c) for c in self.cmd)}\n\n")
                logf.flush()
                self.proc = subprocess.Popen(
                    self.cmd,
                    cwd=self.cwd,
                    env=self.env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                assert self.proc.stdout is not None
                for line in self.proc.stdout:
                    logf.write(line)
                    logf.flush()
                    stripped = line.strip()
                    for prefix in (RESULT_PREFIX, BLENDER_RESULT_PREFIX):
                        if stripped.startswith(prefix):
                            try:
                                self.result = json.loads(stripped[len(prefix):].strip())
                            except Exception as exc:
                                self.error = f"could not parse result line: {exc}"
                self.returncode = self.proc.wait()
        except Exception as exc:  # noqa: BLE001
            self.error = f"{exc}\n{traceback.format_exc()}"
            self.returncode = -1

        self.finished = time.time()
        if self.status == "cancelled":
            pass
        elif self.returncode == 0 and (self.result is None or self.result.get("ok", True)):
            self.status = "succeeded"
        else:
            self.status = "failed"
            if not self.error:
                if self.result and self.result.get("error"):
                    self.error = self.result["error"]
                else:
                    self.error = "process exited with code %s; see log_tail" % self.returncode

    def cancel(self) -> bool:
        if self.proc and self.proc.poll() is None:
            self.status = "cancelled"
            try:
                self.proc.terminate()
            except Exception:
                return False
            return True
        return False


JOBS: Dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def launch_job(kind: str, label: str, cmd: List[str], cwd: Optional[str] = None,
               extra_env: Optional[dict] = None) -> Job:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if CFG.text_encoder_device:
        env["TEXT_ENCODER_DEVICE"] = CFG.text_encoder_device
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items() if v is not None})
    job = Job(uuid.uuid4().hex[:10], kind, label, cmd, cwd or CFG.repo or HERE, env)
    with JOBS_LOCK:
        JOBS[job.id] = job
    job.start()
    return job


def wait_for(job: Job, seconds: float) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline and job.status in ("queued", "running"):
        time.sleep(0.4)


def run_sync(cmd: List[str], timeout: int = 180, cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        cwd=cwd,
        env=env,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def safe_name(name: str) -> str:
    keep = "-_. "
    cleaned = "".join(c for c in name if c.isalnum() or c in keep).strip().replace(" ", "_")
    return cleaned or "motion"


def resolve_output_stem(name: str, subfolder: Optional[str] = None) -> str:
    parts = [CFG.output_dir]
    if subfolder:
        parts.append(safe_name(subfolder))
    stem = os.path.join(*parts, safe_name(name))
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    return stem


def preset_by_name(name: str) -> Optional[dict]:
    for p in PRESETS:
        if p["name"] == name:
            return p
    return None


def build_generate_job_payload(a: dict) -> dict:
    prompts = a.get("prompts")
    durations = a.get("durations")

    preset_name = a.get("preset")
    if preset_name:
        preset = preset_by_name(preset_name)
        if not preset:
            raise ValueError(
                f"unknown preset {preset_name!r}; call kimodo_list_presets for the list"
            )
        prompts = prompts or [preset["prompt"]]
        durations = durations or [preset["duration"]]
        a.setdefault("diffusion_steps", preset["diffusion_steps"])
        a.setdefault("cfg_type", preset["cfg_type"])
        a.setdefault("cfg_weight", preset["cfg_weight"])

    if isinstance(prompts, str):
        prompts = [prompts]
    if not prompts:
        raise ValueError("either 'prompts' or 'preset' is required")
    if isinstance(durations, (int, float)):
        durations = [float(durations)]

    name = a.get("name") or (preset_name or prompts[0][:40])
    stem = resolve_output_stem(name, a.get("subfolder"))

    return {
        "model": a.get("model") or CFG.default_model,
        "prompts": prompts,
        "durations": durations or [5.0],
        "num_samples": int(a.get("num_samples", 1)),
        "diffusion_steps": int(a.get("diffusion_steps", 100)),
        "num_transition_frames": int(a.get("num_transition_frames", 5)),
        "seed": a.get("seed"),
        "cfg_type": a.get("cfg_type"),
        "cfg_weight": a.get("cfg_weight"),
        "constraints_path": a.get("constraints_path"),
        "postprocess": bool(a.get("postprocess", True)),
        "bvh": bool(a.get("bvh", True)),
        "bvh_standard_tpose": bool(a.get("bvh_standard_tpose", False)),
        "save_example_dir": bool(a.get("save_example_dir", True)),
        "offload": bool(a.get("offload", CFG.offload)),
        "output_stem": stem,
    }


def spawn_generation(payload: dict, label: str) -> Job:
    job_file = os.path.join(CFG.jobs_dir, f"job_{uuid.uuid4().hex[:8]}.json")
    with open(job_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    cmd = [CFG.python, os.path.join(HERE, "kimodo_runner.py"), "--job", job_file]
    extra = {}
    if payload.get("text_encoder_device"):
        extra["TEXT_ENCODER_DEVICE"] = payload["text_encoder_device"]
    return launch_job("generate", label, cmd, cwd=CFG.repo, extra_env=extra)


# ---------------------------------------------------------------------------
# tool implementations
# ---------------------------------------------------------------------------
def tool_status(a: dict) -> dict:
    info: Dict[str, Any] = {
        "server_version": SERVER_VERSION,
        "platform": platform.platform(),
        "kimodo_home": CFG.kimodo_home,
        "kimodo_repo": CFG.repo,
        "python": CFG.python,
        "python_exists": os.path.exists(CFG.python),
        "output_dir": CFG.output_dir,
        "blender": CFG.blender,
        "default_model": CFG.default_model,
        "low_vram_offload": CFG.offload,
        "text_encoder_device": CFG.text_encoder_device or "(default: gpu)",
    }

    try:
        proc = run_sync([CFG.python, os.path.join(HERE, "kimodo_probe.py")], timeout=300)
        line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        info["environment"] = json.loads(line) if line.startswith("{") else {"raw": proc.stdout[-2000:]}
        if proc.returncode != 0:
            info["environment_stderr"] = proc.stderr[-2000:]
    except Exception as exc:  # noqa: BLE001
        info["environment_error"] = str(exc)

    hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    if os.path.isdir(hf_cache):
        info["downloaded_models"] = sorted(
            d.replace("models--", "").replace("--", "/")
            for d in os.listdir(hf_cache)
            if d.startswith("models--") and "imodo" in d.lower()
        )

    with JOBS_LOCK:
        info["jobs"] = {
            "running": sum(1 for j in JOBS.values() if j.status == "running"),
            "total_this_session": len(JOBS),
        }
    return info


def tool_list_models(a: dict) -> dict:
    return {"models": MODELS, "default": CFG.default_model,
            "tip": "SOMA models are the ones that export BVH -> FBX for Unreal."}


def tool_list_presets(a: dict) -> dict:
    cat = a.get("category")
    items = [p for p in PRESETS if not cat or p["category"] == cat]
    return {
        "presets": items,
        "categories": sorted({p["category"] for p in PRESETS}),
        "usage": "kimodo_generate with preset='walk_forward', or kimodo_batch_generate with presets=[...]",
    }


def tool_generate(a: dict) -> dict:
    payload = build_generate_job_payload(dict(a))
    if a.get("text_encoder_device"):
        payload["text_encoder_device"] = a["text_encoder_device"]
    label = a.get("name") or a.get("preset") or payload["prompts"][0][:50]
    job = spawn_generation(payload, label)

    wait = float(a.get("wait_seconds", 0) or 0)
    if wait > 0:
        wait_for(job, wait)

    out = job.to_dict(include_tail=25 if job.status != "succeeded" else 6)
    out["settings"] = {k: payload[k] for k in
                       ("model", "prompts", "durations", "num_samples", "diffusion_steps", "seed",
                        "cfg_type", "cfg_weight", "constraints_path", "output_stem")}
    if job.status in ("queued", "running"):
        out["next_step"] = (
            f"Generation is running in the background. Poll kimodo_job_status with job_id='{job.id}' "
            "(a 5s clip at 100 diffusion steps typically takes 1-4 min on a 6GB card, longer on the first "
            "run while the model downloads)."
        )
    elif job.status == "succeeded":
        out["next_step"] = ("Run kimodo_export_fbx on the .bvh file to get an Unreal-ready FBX, and "
                            "kimodo_inspect_motion on the .npz for quality metrics.")
    return out


def tool_batch_generate(a: dict) -> dict:
    items = a.get("items")
    presets = a.get("presets")
    if presets and not items:
        items = [{"preset": p} for p in presets]
    if not items:
        raise ValueError("pass 'items' (list of generate specs) or 'presets' (list of preset names)")

    subfolder = a.get("subfolder") or time.strftime("set_%Y%m%d_%H%M")
    shared = {k: a[k] for k in ("model", "num_samples", "diffusion_steps", "seed", "cfg_type",
                                "cfg_weight", "postprocess", "bvh", "offload") if k in a}

    jobs = []
    for item in items:
        spec = dict(shared)
        spec.update(item)
        spec.setdefault("subfolder", subfolder)
        payload = build_generate_job_payload(spec)
        label = spec.get("name") or spec.get("preset") or payload["prompts"][0][:40]
        job = spawn_generation(payload, label)
        jobs.append({"job_id": job.id, "label": label, "output_stem": payload["output_stem"]})
        time.sleep(0.15)

    return {
        "queued": len(jobs),
        "subfolder": os.path.join(CFG.output_dir, safe_name(subfolder)),
        "jobs": jobs,
        "note": (
            "Jobs run sequentially on the GPU only if you let them; they were all started now, which on a 6GB "
            "card can thrash VRAM. For best results poll with kimodo_list_jobs and keep concurrency at 1 by "
            "passing items one at a time, or accept the queueing and check back later."
        ),
    }


def tool_job_status(a: dict) -> dict:
    job_id = a["job_id"]
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise ValueError(f"no such job {job_id!r}")
    wait = float(a.get("wait_seconds", 0) or 0)
    if wait > 0:
        wait_for(job, wait)
    return job.to_dict(include_tail=int(a.get("log_lines", 15)))


def tool_job_logs(a: dict) -> dict:
    job_id = a["job_id"]
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise ValueError(f"no such job {job_id!r}")
    return {"job_id": job_id, "status": job.status, "log": job.tail(int(a.get("lines", 120)))}


def tool_list_jobs(a: dict) -> dict:
    with JOBS_LOCK:
        jobs = sorted(JOBS.values(), key=lambda j: j.created, reverse=True)
    status = a.get("status")
    return {"jobs": [j.to_dict() for j in jobs if not status or j.status == status]}


def tool_cancel_job(a: dict) -> dict:
    with JOBS_LOCK:
        job = JOBS.get(a["job_id"])
    if not job:
        raise ValueError("no such job")
    return {"job_id": job.id, "cancelled": job.cancel(), "status": job.status}


def tool_make_constraints(a: dict) -> dict:
    """Author a Kimodo constraints JSON without opening the demo UI."""
    constraints: List[dict] = []
    fps = float(a.get("fps", 30.0))

    path_spec = a.get("root_path")
    if path_spec:
        pts = path_spec.get("points") or []
        if len(pts) < 2:
            raise ValueError("root_path.points needs at least 2 [x, z] points (metres, first should be [0, 0])")
        frames = path_spec.get("frame_indices")
        if not frames:
            duration = float(path_spec.get("duration", a.get("duration", 5.0)))
            total = max(int(round(duration * fps)) - 1, 1)
            frames = [int(round(i * total / (len(pts) - 1))) for i in range(len(pts))]
        entry = {
            "type": "root2d",
            "frame_indices": [int(f) for f in frames],
            "smooth_root_2d": [[float(p[0]), float(p[1])] for p in pts],
        }
        headings = path_spec.get("headings_deg")
        if headings:
            import math

            entry["global_root_heading"] = [
                [math.cos(math.radians(h)), math.sin(math.radians(h))] for h in headings
            ]
        constraints.append(entry)

    for ee in a.get("end_effectors") or []:
        etype = ee.get("type", "end-effector")
        if etype not in ("left-hand", "right-hand", "left-foot", "right-foot", "end-effector"):
            raise ValueError(f"bad end-effector type {etype!r}")
        entry = {
            "type": etype,
            "frame_indices": [int(f) for f in ee["frame_indices"]],
            "root_positions": [[float(v) for v in rp] for rp in ee["root_positions"]],
            "local_joints_rot": ee["local_joints_rot"],
        }
        if etype == "end-effector":
            entry["joint_names"] = ee["joint_names"]
        if ee.get("smooth_root_2d"):
            entry["smooth_root_2d"] = ee["smooth_root_2d"]
        constraints.append(entry)

    for raw in a.get("raw") or []:
        constraints.append(raw)

    if not constraints:
        raise ValueError("nothing to write -- pass root_path, end_effectors, or raw")

    name = safe_name(a.get("name") or "constraints")
    out_dir = os.path.join(CFG.output_dir, "_constraints")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(constraints, f, indent=2)

    return {
        "constraints_path": path,
        "count": len(constraints),
        "constraints": constraints if len(json.dumps(constraints)) < 4000 else "(large -- see file)",
        "reminder": (
            "Kimodo is Y-up, metres, and canonicalised: the root starts at XZ (0,0) at frame 0, so author "
            "positions relative to that origin. Pass constraints_path to kimodo_generate."
        ),
    }


def tool_convert(a: dict) -> dict:
    src, dst = a["input"], a["output"]
    exe = _first_existing(
        os.path.join(os.path.dirname(CFG.python), "kimodo_convert.exe"),
        os.path.join(os.path.dirname(CFG.python), "kimodo_convert"),
    )
    cmd = [exe] if exe else [CFG.python, "-m", "kimodo.scripts.motion_convert"]
    cmd += [src, dst]
    for flag, key in (("--from", "from_format"), ("--to", "to_format"), ("--source-fps", "source_fps")):
        if a.get(key) is not None:
            cmd += [flag, str(a[key])]
    if a.get("no_z_up"):
        cmd.append("--no-z-up")
    if a.get("bvh_standard_tpose"):
        cmd.append("--bvh_standard_tpose")

    proc = run_sync(cmd, timeout=600, cwd=CFG.repo)
    return {
        "ok": proc.returncode == 0 and os.path.exists(dst),
        "command": " ".join(cmd),
        "output": dst if os.path.exists(dst) else None,
        "stdout": proc.stdout[-3000:],
        "stderr": proc.stderr[-3000:],
    }


def tool_export_fbx(a: dict) -> dict:
    if not CFG.blender:
        raise ValueError(
            "Blender was not found. Set KIMODO_BLENDER (or 'blender' in kimodo_mcp_config.json) to blender.exe."
        )
    src = a["input"]
    if not os.path.exists(src):
        raise ValueError(f"input BVH not found: {src}")

    fmt = (a.get("format") or "fbx").lower()
    dst = a.get("output")
    if not dst:
        dst = os.path.splitext(src)[0] + ("." + fmt)
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)

    cmd = [
        CFG.blender, "-b", "--factory-startup",
        "-P", os.path.join(HERE, "blender_export.py"), "--",
        "--input", src, "--output", dst,
        "--target", a.get("target") or CFG.export_target,
        "--fps", str(a.get("fps", 30.0)),
    ]
    if a.get("scale") is not None:
        cmd += ["--scale", str(a["scale"])]
    if a.get("rename_root"):
        cmd += ["--rename-root", a["rename_root"]]

    proc = run_sync(cmd, timeout=int(a.get("timeout", 600)))
    result = None
    for line in (proc.stdout or "").splitlines():
        if line.strip().startswith(BLENDER_RESULT_PREFIX):
            try:
                result = json.loads(line.strip()[len(BLENDER_RESULT_PREFIX):].strip())
            except Exception:
                pass

    ok = bool(result and result.get("ok")) or os.path.exists(dst)
    out = {
        "ok": ok,
        "output": dst if os.path.exists(dst) else None,
        "blender": CFG.blender,
        "details": result,
    }
    if not ok:
        out["stdout"] = (proc.stdout or "")[-4000:]
        out["stderr"] = (proc.stderr or "")[-4000:]
    else:
        out["import_hint"] = (
            "In Unreal: Import as Skeletal Mesh -> Import Animations on, Convert Scene on, "
            "Force Front XAxis off, Import Uniform Scale 1.0. Then retarget onto your character's skeleton "
            "with an IK Retargeter."
        )
    return out


def tool_inspect_motion(a: dict) -> dict:
    paths = a.get("paths") or ([a["path"]] if a.get("path") else [])
    if not paths:
        raise ValueError("pass 'path' or 'paths'")
    cmd = [CFG.python, os.path.join(HERE, "kimodo_inspect.py")] + list(paths)
    if a.get("fps"):
        cmd += ["--fps", str(a["fps"])]
    proc = run_sync(cmd, timeout=180)
    if proc.returncode != 0:
        return {"ok": False, "stdout": proc.stdout[-3000:], "stderr": proc.stderr[-3000:]}
    try:
        return {"ok": True, "report": json.loads(proc.stdout)}
    except Exception:
        return {"ok": False, "raw": proc.stdout[-4000:]}


def tool_list_outputs(a: dict) -> dict:
    root = a.get("folder") or CFG.output_dir
    exts = tuple(e.lower() for e in (a.get("extensions") or [".npz", ".bvh", ".fbx", ".glb", ".csv"]))
    entries = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "_jobs"]
        for fn in filenames:
            if fn.lower().endswith(exts):
                full = os.path.join(dirpath, fn)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                entries.append({
                    "path": full,
                    "size_kb": round(st.st_size / 1024, 1),
                    "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)),
                })
    entries.sort(key=lambda e: e["modified"], reverse=True)
    limit = int(a.get("limit", 60))
    return {"folder": root, "count": len(entries), "files": entries[:limit]}


def tool_list_examples(a: dict) -> dict:
    base = os.path.join(CFG.repo, "kimodo", "assets", "demo", "examples")
    if not os.path.isdir(base):
        return {"examples": [], "note": f"examples folder not found at {base}"}
    out = []
    for model_dir in sorted(os.listdir(base)):
        mp = os.path.join(base, model_dir)
        if not os.path.isdir(mp):
            continue
        for ex in sorted(os.listdir(mp)):
            ep = os.path.join(mp, ex)
            if not os.path.isdir(ep):
                continue
            meta = {}
            meta_path = os.path.join(ep, "meta.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    pass
            cpath = os.path.join(ep, "constraints.json")
            out.append({
                "model": model_dir,
                "example": ex,
                "dir": ep,
                "constraints": cpath if os.path.exists(cpath) else None,
                "meta": meta,
            })
    return {"examples": out, "tip": "Pass an example's constraints.json to kimodo_generate as constraints_path."}


DEMO_JOB_KEY = {"id": None}


def tool_start_demo(a: dict) -> dict:
    with JOBS_LOCK:
        existing = JOBS.get(DEMO_JOB_KEY["id"] or "")
    if existing and existing.status == "running":
        return {"already_running": True, "job_id": existing.id, "log_tail": existing.tail(10)}

    cmd = [CFG.python, "-m", "kimodo.demo"]
    if a.get("offload", CFG.offload):
        cmd.append("--offload")
    if a.get("model"):
        cmd += ["--model", a["model"]]
    job = launch_job("demo", "kimodo interactive demo", cmd, cwd=CFG.repo)
    DEMO_JOB_KEY["id"] = job.id
    wait_for(job, float(a.get("wait_seconds", 25)))
    return {
        "job_id": job.id,
        "status": job.status,
        "log_tail": job.tail(30),
        "note": ("The viser demo prints its local URL in the log (usually http://127.0.0.1:8080). Open it in a "
                 "browser for timeline + constraint authoring, then Save Constraints and feed the JSON back to "
                 "kimodo_generate."),
    }


def tool_stop_demo(a: dict) -> dict:
    jid = a.get("job_id") or DEMO_JOB_KEY["id"]
    if not jid:
        return {"stopped": False, "reason": "no demo started from this server"}
    with JOBS_LOCK:
        job = JOBS.get(jid)
    if not job:
        return {"stopped": False, "reason": "job not found"}
    return {"stopped": job.cancel(), "status": job.status}


# ---------------------------------------------------------------------------
# tool registry
# ---------------------------------------------------------------------------
def T(name, description, schema, fn, read_only=True, destructive=False, idempotent=True, open_world=False):
    return {
        "name": name,
        "description": description,
        "inputSchema": schema,
        "annotations": {
            "title": name,
            "readOnlyHint": read_only,
            "destructiveHint": destructive,
            "idempotentHint": idempotent,
            "openWorldHint": open_world,
        },
        "_fn": fn,
    }


def OBJ(props, required=None):
    return {
        "type": "object",
        "properties": props,
        "required": required or [],
        "additionalProperties": False,
    }


GEN_PROPS = {
    "prompts": {
        "type": "array", "items": {"type": "string"},
        "description": "One or more motion descriptions in plain English, e.g. ['A person walks forward.']. "
                       "Multiple prompts are generated as one continuous clip, blended in order.",
    },
    "durations": {
        "type": "array", "items": {"type": "number"},
        "description": "Seconds per prompt. Pass one value to use it for all prompts. Default 5.",
    },
    "preset": {"type": "string",
               "description": "Name of a built-in game-animation preset (see kimodo_list_presets). Fills prompt/duration/cfg."},
    "name": {"type": "string", "description": "Output file stem, e.g. 'hero_walk_fwd'."},
    "subfolder": {"type": "string", "description": "Subfolder under the output dir to group a set of animations."},
    "model": {"type": "string", "description": "Model short name (see kimodo_list_models). Default kimodo-soma-rp."},
    "num_samples": {"type": "integer", "minimum": 1, "maximum": 8,
                    "description": "Variations to generate. Each costs VRAM; keep at 1-2 on a 6GB card."},
    "diffusion_steps": {"type": "integer", "minimum": 10, "maximum": 300,
                        "description": "Denoising steps. 50-100 fast, 100-200 higher quality. Default 100."},
    "num_transition_frames": {"type": "integer",
                              "description": "Frames used to blend between multi-prompt segments. Default 5."},
    "seed": {"type": "integer",
             "description": "Seed for reproducible output. Re-roll this to get a different take of the same prompt."},
    "cfg_type": {"type": "string", "enum": ["nocfg", "regular", "separated"],
                 "description": "Guidance mode. 'separated' takes [text_weight, constraint_weight]."},
    "cfg_weight": {"type": "array", "items": {"type": "number"},
                   "description": "One value for 'regular', two ([text, constraint]) for 'separated'. Raise the "
                                  "constraint weight when the motion ignores your constraints."},
    "constraints_path": {"type": "string",
                         "description": "Path to a constraints JSON (from kimodo_make_constraints or the demo's "
                                        "Save Constraints)."},
    "postprocess": {"type": "boolean", "description": "Foot-skate cleanup + constraint optimisation. Default true."},
    "bvh": {"type": "boolean",
            "description": "Also export BVH (SOMA models only). Default true -- needed for the FBX step."},
    "bvh_standard_tpose": {"type": "boolean",
                           "description": "Export BVH against the standard T-pose rest pose instead of the "
                                          "BONES-SEED rest pose."},
    "save_example_dir": {"type": "boolean", "description": "Also write a demo-loadable example folder. Default true."},
    "offload": {"type": "boolean", "description": "Multi-tier VRAM offloading for low-VRAM GPUs. Default on."},
    "text_encoder_device": {"type": "string", "enum": ["cpu", "cuda"],
                            "description": "Force the 5.4GB text encoder onto the CPU if VRAM is tight."},
    "wait_seconds": {"type": "number",
                     "description": "Block up to this many seconds waiting for the job before returning. "
                                    "Default 0 (return immediately with a job_id)."},
}

TOOLS: List[dict] = [
    T("kimodo_status",
      "Health check for the whole Kimodo pipeline: install paths, Python/torch/CUDA, GPU name and free VRAM, "
      "whether low-VRAM offloading is supported, which model checkpoints are already downloaded, Blender path, "
      "and running jobs. Call this first if anything misbehaves.",
      OBJ({}), tool_status),

    T("kimodo_list_models",
      "List the Kimodo model variants with their skeleton, training data, supported export formats and when to "
      "use each one.",
      OBJ({}), tool_list_models),

    T("kimodo_list_presets",
      "List the built-in game-animation presets (idle, walk, run, dodge, melee, hit reaction, death, ...) with "
      "their tuned prompts and settings. Use a preset name in kimodo_generate or kimodo_batch_generate.",
      OBJ({"category": {"type": "string",
                        "description": "Filter: locomotion, traversal, combat, interaction, flavour."}}),
      tool_list_presets),

    T("kimodo_generate",
      "Generate a motion clip from text (and optional kinematic constraints) and write NPZ + BVH into the output "
      "folder. Runs in the background and returns a job_id immediately unless you pass wait_seconds. This is the "
      "main tool: everything else supports it.",
      OBJ(GEN_PROPS), tool_generate, read_only=False, idempotent=False),

    T("kimodo_batch_generate",
      "Queue several clips at once -- e.g. a whole animation set for a character. Pass 'presets' (list of preset "
      "names) or 'items' (list of kimodo_generate argument objects). Shared settings can be given at the top level.",
      OBJ({
          "presets": {"type": "array", "items": {"type": "string"}, "description": "Preset names to generate."},
          "items": {"type": "array", "items": {"type": "object"},
                    "description": "Full generate specs, one per clip."},
          "subfolder": {"type": "string", "description": "Group the set under this subfolder."},
          "model": GEN_PROPS["model"], "num_samples": GEN_PROPS["num_samples"],
          "diffusion_steps": GEN_PROPS["diffusion_steps"], "seed": GEN_PROPS["seed"],
          "cfg_type": GEN_PROPS["cfg_type"], "cfg_weight": GEN_PROPS["cfg_weight"],
          "postprocess": GEN_PROPS["postprocess"], "bvh": GEN_PROPS["bvh"], "offload": GEN_PROPS["offload"],
      }), tool_batch_generate, read_only=False, idempotent=False),

    T("kimodo_job_status",
      "Check a generation job: status, elapsed time, the result payload (output file paths, timings) and the tail "
      "of its log. Pass wait_seconds to block until it finishes.",
      OBJ({"job_id": {"type": "string"},
           "wait_seconds": {"type": "number", "description": "Block up to this long waiting for completion."},
           "log_lines": {"type": "integer", "description": "How many log lines to include. Default 15."}},
          ["job_id"]),
      tool_job_status),

    T("kimodo_job_logs",
      "Read the raw log tail of a job -- use this when a generation fails and you need the traceback.",
      OBJ({"job_id": {"type": "string"}, "lines": {"type": "integer"}}, ["job_id"]), tool_job_logs),

    T("kimodo_list_jobs",
      "List every job started in this session (generations, batches, the demo server) with status, elapsed time "
      "and log paths. Use it to see what is still occupying the GPU before starting more work.",
      OBJ({"status": {"type": "string",
                      "enum": ["queued", "running", "succeeded", "failed", "cancelled"]}}),
      tool_list_jobs),

    T("kimodo_cancel_job",
      "Terminate a running job (a generation or the interactive demo server) and release its VRAM. Partial output "
      "files may be left behind.",
      OBJ({"job_id": {"type": "string"}}, ["job_id"]), tool_cancel_job,
      read_only=False, destructive=True, idempotent=False),

    T("kimodo_make_constraints",
      "Author a Kimodo constraints JSON file without opening the demo UI. Currently supports 2D root paths and "
      "waypoints (walk this shape on the ground) plus optional per-waypoint headings, and passthrough of raw "
      "constraint objects for end-effector or full-body keyframes. Coordinates are metres, Y-up, XZ ground plane, "
      "with the root starting at (0,0).",
      OBJ({
          "name": {"type": "string", "description": "File name stem for the constraints JSON."},
          "fps": {"type": "number", "description": "Model fps used to convert durations to frames. Default 30."},
          "duration": {"type": "number",
                       "description": "Clip duration in seconds, used to spread points over frames."},
          "root_path": {
              "type": "object",
              "description": "2D ground path for the character's root.",
              "properties": {
                  "points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}},
                             "description": "List of [x, z] in metres. Start at [0, 0]."},
                  "frame_indices": {"type": "array", "items": {"type": "integer"},
                                    "description": "Optional explicit frames for each point; otherwise spread evenly."},
                  "duration": {"type": "number"},
                  "headings_deg": {"type": "array", "items": {"type": "number"},
                                   "description": "Optional facing angle in degrees per point (0 = facing +Z)."},
              },
          },
          "end_effectors": {"type": "array", "items": {"type": "object"},
                            "description": "Raw end-effector constraint objects (type, frame_indices, "
                                           "root_positions, local_joints_rot, joint_names)."},
          "raw": {"type": "array", "items": {"type": "object"},
                  "description": "Constraint objects passed through verbatim."},
      }), tool_make_constraints, read_only=False, idempotent=False),

    T("kimodo_convert",
      "Convert between motion formats with kimodo_convert: Kimodo NPZ, AMASS NPZ, SOMA BVH, G1 MuJoCo CSV.",
      OBJ({
          "input": {"type": "string"}, "output": {"type": "string"},
          "from_format": {"type": "string", "enum": ["amass", "kimodo", "soma-bvh", "g1-csv"]},
          "to_format": {"type": "string", "enum": ["kimodo", "amass", "soma-bvh", "g1-csv"]},
          "source_fps": {"type": "number"},
          "no_z_up": {"type": "boolean"},
          "bvh_standard_tpose": {"type": "boolean"},
      }, ["input", "output"]), tool_convert, read_only=False, idempotent=False),

    T("kimodo_export_fbx",
      "Convert a generated BVH into a game-engine file using headless Blender. Default target is Unreal "
      "(centimetres, Z-up, baked animation, no leaf bones). Set format='glb' for a glTF instead.",
      OBJ({
          "input": {"type": "string", "description": "Path to the .bvh produced by kimodo_generate."},
          "output": {"type": "string",
                     "description": "Output path. Defaults to the input path with the new extension."},
          "format": {"type": "string", "enum": ["fbx", "glb"], "description": "Default fbx."},
          "target": {"type": "string", "enum": ["unreal", "unity", "raw"],
                     "description": "Unit/axis preset. Default unreal."},
          "scale": {"type": "number",
                    "description": "Extra uniform scale applied on import, if the rig comes in the wrong size."},
          "fps": {"type": "number"},
          "rename_root": {"type": "string",
                          "description": "Rename the root bone, e.g. 'root' for Unreal root motion."},
          "timeout": {"type": "integer"},
      }, ["input"]), tool_export_fbx, read_only=False, idempotent=False),

    T("kimodo_inspect_motion",
      "Quality-check generated motion NPZ files: frame count, duration, root travel and average speed, ground "
      "penetration, foot-skate during contact, jitter, and how cleanly the clip would loop. Use this to decide "
      "whether to keep a take or re-roll the seed.",
      OBJ({"path": {"type": "string"}, "paths": {"type": "array", "items": {"type": "string"}},
           "fps": {"type": "number"}}),
      tool_inspect_motion),

    T("kimodo_list_outputs",
      "List generated animation files in the output folder, newest first.",
      OBJ({"folder": {"type": "string"}, "extensions": {"type": "array", "items": {"type": "string"}},
           "limit": {"type": "integer"}}),
      tool_list_outputs),

    T("kimodo_list_examples",
      "List the constraint/motion examples shipped with Kimodo. Their constraints.json files are good starting "
      "points for path-following, keyframed poses and end-effector control.",
      OBJ({}), tool_list_examples),

    T("kimodo_start_demo",
      "Launch the interactive Kimodo authoring demo (timeline, constraint tracks, 3D preview) in the background "
      "with low-VRAM offloading. Use it when a motion needs hand-authored keyframes; save the constraints there "
      "and feed the JSON back into kimodo_generate.",
      OBJ({"model": {"type": "string"}, "offload": {"type": "boolean"}, "wait_seconds": {"type": "number"}}),
      tool_start_demo, read_only=False, idempotent=False),

    T("kimodo_stop_demo",
      "Stop the interactive demo started by this server and free its VRAM.",
      OBJ({"job_id": {"type": "string"}}), tool_stop_demo,
      read_only=False, destructive=True, idempotent=False),
]

TOOL_BY_NAME: Dict[str, dict] = {t["name"]: t for t in TOOLS}


def public_tools() -> List[dict]:
    return [{k: v for k, v in t.items() if not k.startswith("_")} for t in TOOLS]


# ---------------------------------------------------------------------------
# JSON-RPC plumbing
# ---------------------------------------------------------------------------
def jsonrpc_result(req_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def jsonrpc_error(req_id: Any, code: int, message: str, data: Any = None) -> dict:
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


def handle_initialize(params: dict) -> dict:
    requested = params.get("protocolVersion")
    version = requested if requested in SUPPORTED_PROTOCOLS else SUPPORTED_PROTOCOLS[0]
    return {
        "protocolVersion": version,
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        "instructions": (
            "Kimodo is NVIDIA's text-to-motion diffusion model. Typical flow for game animation:\n"
            "1. kimodo_status to confirm the GPU and install are healthy.\n"
            "2. kimodo_generate (or kimodo_batch_generate with presets) -- returns a job_id; poll "
            "kimodo_job_status.\n"
            "3. kimodo_inspect_motion on the resulting .npz to check foot skate, jitter and loopability.\n"
            "4. kimodo_export_fbx on the .bvh for an Unreal-ready FBX.\n"
            "Prompts work best as one clear sentence per action, present tense, describing a person "
            "('A person sprints forward and slides.'). Use kimodo_make_constraints when the motion has to "
            "follow an exact path. Generation is GPU-bound: keep one job running at a time on a 6GB card."
        ),
    }


def call_tool(name: str, arguments: dict) -> dict:
    tool = TOOL_BY_NAME.get(name)
    if not tool:
        return {
            "content": [{"type": "text", "text": f"Unknown tool {name!r}. Available: {', '.join(TOOL_BY_NAME)}"}],
            "isError": True,
        }
    try:
        payload = tool["_fn"](arguments or {})
        text = json.dumps(payload, indent=2, default=str)
        return {"content": [{"type": "text", "text": text}], "isError": False}
    except Exception as exc:  # noqa: BLE001
        log(f"tool {name} failed: {exc}\n{traceback.format_exc()}")
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "error": str(exc),
                    "tool": name,
                    "arguments": arguments,
                    "traceback": traceback.format_exc()[-2000:],
                }, indent=2),
            }],
            "isError": True,
        }


def handle_message(msg: dict) -> Optional[dict]:
    method = msg.get("method")
    req_id = msg.get("id")
    params = msg.get("params") or {}

    if method is None:
        return None  # a response to something we never send

    if method == "initialize":
        return jsonrpc_result(req_id, handle_initialize(params))
    if method in ("notifications/initialized", "initialized", "notifications/cancelled"):
        return None
    if method == "ping":
        return jsonrpc_result(req_id, {})
    if method == "tools/list":
        return jsonrpc_result(req_id, {"tools": public_tools()})
    if method == "tools/call":
        return jsonrpc_result(req_id, call_tool(params.get("name", ""), params.get("arguments") or {}))
    if method in ("resources/list", "prompts/list"):
        key = "resources" if method.startswith("resources") else "prompts"
        return jsonrpc_result(req_id, {key: []})

    if req_id is None:
        return None
    return jsonrpc_error(req_id, -32601, f"Method not found: {method}")


def serve() -> int:
    log(f"starting; python={CFG.python} repo={CFG.repo} out={CFG.output_dir} blender={CFG.blender}")
    stdout = sys.stdout
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError as exc:
            stdout.write(json.dumps(jsonrpc_error(None, -32700, f"Parse error: {exc}")) + "\n")
            stdout.flush()
            continue

        messages = msg if isinstance(msg, list) else [msg]
        for m in messages:
            try:
                response = handle_message(m)
            except Exception as exc:  # noqa: BLE001
                log(traceback.format_exc())
                response = jsonrpc_error(m.get("id"), -32603, f"Internal error: {exc}")
            if response is not None:
                stdout.write(json.dumps(response, default=str) + "\n")
                stdout.flush()
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        print(json.dumps({"tools": [t["name"] for t in TOOLS], "config": {
            "python": CFG.python, "repo": CFG.repo, "output_dir": CFG.output_dir,
            "blender": CFG.blender}}, indent=2))
        raise SystemExit(0)
    try:
        raise SystemExit(serve())
    except KeyboardInterrupt:
        raise SystemExit(0)
