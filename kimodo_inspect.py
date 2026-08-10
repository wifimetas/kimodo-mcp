# SPDX-License-Identifier: Apache-2.0
"""Quality-control inspection for Kimodo NPZ motion files.

Pure numpy, no torch, no kimodo import -- fast enough to run on every generation.
Prints one JSON object to stdout.

Metrics are chosen to answer "is this animation actually usable in a game?":
  * foot skate      -- horizontal speed of the planted foot during ground contact
  * ground penetration / float
  * jitter          -- mean per-joint jerk, the usual tell for diffusion artefacts
  * root travel + average locomotion speed
  * loopability     -- pose distance between first and last frame
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

FPS_DEFAULT = 30.0


def summarize(path: str, fps: float = FPS_DEFAULT) -> dict:
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())

    out: dict = {"file": os.path.abspath(path), "keys": keys, "fps": fps}

    if "posed_joints" not in keys:
        out["error"] = "not a Kimodo NPZ (no 'posed_joints' array)"
        return out

    joints = np.asarray(data["posed_joints"], dtype=np.float64)  # [T, J, 3]
    if joints.ndim == 4:  # [B, T, J, 3] -- take first sample
        joints = joints[0]
    T, J, _ = joints.shape
    out.update({"frames": int(T), "joints": int(J), "duration_seconds": round(T / fps, 3)})

    # --- root travel ---
    if "root_positions" in keys:
        root = np.asarray(data["root_positions"], dtype=np.float64)
        if root.ndim == 3:
            root = root[0]
    else:
        root = joints[:, 0, :]
    disp = root[-1] - root[0]
    planar_path = np.linalg.norm(np.diff(root[:, [0, 2]], axis=0), axis=-1).sum()
    out["root"] = {
        "start_xz": [round(float(root[0, 0]), 4), round(float(root[0, 2]), 4)],
        "end_xz": [round(float(root[-1, 0]), 4), round(float(root[-1, 2]), 4)],
        "net_displacement_m": round(float(np.linalg.norm(disp[[0, 2]])), 4),
        "path_length_m": round(float(planar_path), 4),
        "avg_speed_mps": round(float(planar_path / max(T - 1, 1) * fps), 4),
        "height_min_m": round(float(root[:, 1].min()), 4),
        "height_max_m": round(float(root[:, 1].max()), 4),
    }

    # --- ground contact / penetration ---
    ground_y = float(joints[..., 1].min())
    lowest_per_frame = joints[..., 1].min(axis=1)
    out["ground"] = {
        "lowest_point_m": round(ground_y, 4),
        "frames_below_ground": int((lowest_per_frame < -0.02).sum()),
        "frames_floating_above_5cm": int((lowest_per_frame > 0.05).sum()),
    }

    # --- foot skate ---
    # Measured mapping-free: on every frame flagged as having ground contact, take the
    # *slowest* candidate foot joint (that is the planted one) and look at its horizontal
    # world speed. A planted foot should be near zero. This avoids having to know which
    # foot_contacts column corresponds to which joint index.
    if T >= 2:
        mean_h = joints[..., 1].mean(axis=0)
        foot_idx = list(np.argsort(mean_h)[:4])
        vel = np.linalg.norm(np.diff(joints[:, foot_idx][:, :, [0, 2]], axis=0), axis=-1) * fps  # [T-1, k]
        planted = vel.min(axis=1)  # slowest foot per frame

        if "foot_contacts" in keys:
            contacts = np.asarray(data["foot_contacts"], dtype=np.float64)
            if contacts.ndim == 3:
                contacts = contacts[0]
            any_contact = contacts[: T - 1].max(axis=1) > 0.5
        else:
            any_contact = np.ones(T - 1, dtype=bool)

        n = int(any_contact.sum())
        if n > 0:
            skate = planted[any_contact]
            mean_skate = float(skate.mean())
            out["foot_skate"] = {
                "contact_frames": n,
                "planted_foot_mean_speed_mps": round(mean_skate, 4),
                "planted_foot_p90_speed_mps": round(float(np.percentile(skate, 90)), 4),
                "planted_foot_max_speed_mps": round(float(skate.max()), 4),
                "verdict": (
                    "good -- planted foot stays put" if mean_skate < 0.15
                    else "some sliding" if mean_skate < 0.4
                    else "heavy foot skate -- re-roll the seed or keep postprocess on"
                ),
            }
        else:
            out["foot_skate"] = {"contact_frames": 0, "verdict": "no ground contact detected (airborne motion?)"}

    # --- jitter (mean per-joint jerk magnitude) ---
    if T >= 4:
        acc = np.diff(joints, n=2, axis=0) * (fps ** 2)
        jerk = np.linalg.norm(np.diff(acc, axis=0) * fps, axis=-1)
        mean_jerk = float(jerk.mean())
        out["smoothness"] = {
            "mean_jerk_m_per_s3": round(mean_jerk, 2),
            "verdict": "smooth" if mean_jerk < 800 else "jittery -- raise diffusion_steps or re-roll the seed",
        }

    # --- loopability ---
    if T >= 2:
        centered = joints - joints[:, :1, :]
        pose_delta = float(np.linalg.norm(centered[0] - centered[-1], axis=-1).mean())
        out["loopability"] = {
            "first_last_pose_distance_m": round(pose_delta, 4),
            "verdict": (
                "loops cleanly" if pose_delta < 0.08
                else "not loop-ready -- trim in your DCC or generate a longer clip and cut a cycle"
            ),
        }

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--fps", type=float, default=FPS_DEFAULT)
    args = ap.parse_args()

    results = []
    for p in args.paths:
        try:
            results.append(summarize(p, args.fps))
        except Exception as exc:  # noqa: BLE001
            results.append({"file": p, "error": str(exc)})
    json.dump(results if len(results) > 1 else results[0], sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
