# Kimodo MCP

An [MCP](https://modelcontextprotocol.io) server that gives AI coding agents (Claude Code, or any MCP client) hands-on control of a local [NVIDIA Kimodo](https://github.com/nv-tlabs/kimodo) install — text-to-motion generation, constraint authoring, quality checks, and headless-Blender FBX export for Unreal/Unity — as first-class tools.

Ask your agent for *"a full locomotion set for my character"* and it can generate the clips, QC them for foot skate and jitter, re-roll the bad takes, and hand you Unreal-ready FBX files.

```
you:    generate a dodge roll for my player character and get it into Unreal
agent:  kimodo_generate  preset="dodge_roll" name="player_dodge"
        kimodo_job_status job_id=... wait_seconds=120
        kimodo_inspect_motion path=".../player_dodge.npz"    → foot skate 0.05 m/s, smooth
        kimodo_export_fbx input=".../player_dodge.bvh" target="unreal"
```

## Features

- **17 tools** covering the full pipeline: generate → inspect → convert → export
- **Zero dependencies** — the server speaks MCP's stdio JSON-RPC directly in stdlib Python. Nothing is ever pip-installed into your (fragile) ML environment; all heavy lifting (torch, kimodo, Blender) runs in subprocesses
- **Background jobs** — generation returns a `job_id` immediately; poll status, read logs, cancel
- **Low-VRAM aware** — on the [Aero-Ex low-VRAM fork](https://github.com/nv-tlabs/kimodo) it reproduces the demo's multi-tier offloading (encode prompts → evict the 5.4 GB text encoder to RAM → denoise on GPU), which makes 6 GB cards viable. On a stock install it silently falls back to normal behaviour
- **23 tuned game-animation presets** — idle, walk, run, strafe, dodge roll, vault, melee combo, hit reaction, death, and more
- **Constraint authoring without the demo UI** — describe a 2D root path with waypoints and headings, get a valid constraints JSON
- **Motion QC** — foot skate during contact, ground penetration, jitter (jerk), root speed, loopability
- **Game-engine export** — headless Blender turns BVH into Unreal-ready FBX (centimetres, Z-up, baked, no leaf bones) or GLB, with a Unity preset too

## Requirements

| What | Why |
|---|---|
| Python 3.9+ | runs the server itself (stdlib only) |
| A working [Kimodo](https://github.com/nv-tlabs/kimodo) install in its own venv | generation (`torch` + CUDA GPU recommended; the low-VRAM fork enables 6 GB cards) |
| [Blender](https://www.blender.org/) 3.x–5.x (optional) | only needed for `kimodo_export_fbx`; auto-detected from PATH or standard install locations |

The server runs fine without a GPU or Blender — the relevant tools just report what's missing.

**Don't have Kimodo installed yet?** This walkthrough covers the whole install (including the low-VRAM setup this server takes advantage of): [AI Animation Running Now Locally for FREE (6gb vram)](https://www.youtube.com/watch?v=nNmeuvew8LY) by PixelArtistry.

## Install

```bash
git clone https://github.com/wifimetas/kimodo-mcp
cd kimodo-mcp
cp kimodo_mcp_config.example.json kimodo_mcp_config.json
# edit kimodo_mcp_config.json to point at your Kimodo install
```

### Claude Code

```bash
claude mcp add kimodo -s user -- python /path/to/kimodo-mcp/kimodo_mcp_server.py
```

Tip: use your Kimodo venv's python as the command and you can skip most of the config file — the server derives the repo and install paths from it.

### Any MCP client (`.mcp.json` / equivalent)

```json
{
  "mcpServers": {
    "kimodo": {
      "command": "C:\\path\\to\\kimodo\\venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\kimodo-mcp\\kimodo_mcp_server.py"]
    }
  }
}
```

### Verify

```bash
python test_protocol.py            # protocol + schema checks, runs anywhere
python test_protocol.py --strict   # also require GPU, kimodo venv and Blender
```

## Configuration

Settings come from `kimodo_mcp_config.json` next to the server, or environment variables (env wins). Everything is optional if the server is launched with the Kimodo venv's python from inside a standard layout.

| Setting | Env var | Default |
|---|---|---|
| Kimodo install root | `KIMODO_HOME` | — |
| Python with kimodo installed | `KIMODO_PYTHON` | `<home>/venv/.../python` |
| Repo checkout (cwd for subprocesses) | `KIMODO_REPO` | `<home>/kimodo` |
| Output folder | `KIMODO_OUTPUT_DIR` | `<home>/kimodo_out` |
| Blender executable | `KIMODO_BLENDER` | auto-detected |
| Default model | `KIMODO_MODEL` | `kimodo-soma-rp` |
| Low-VRAM offload | `KIMODO_OFFLOAD` | `1` |
| Text encoder device | `TEXT_ENCODER_DEVICE` | GPU |

## Tools

| Tool | What it does |
|---|---|
| `kimodo_status` | Install paths, torch/CUDA, GPU + free VRAM, offload support, downloaded checkpoints, Blender, running jobs |
| `kimodo_list_models` | Model variants, skeletons, training data, export formats |
| `kimodo_list_presets` | Built-in game-animation presets (idle, walk, run, dodge, melee, hit, death, …) |
| `kimodo_generate` | The main one. Text to motion, writes NPZ + BVH. Backgrounded, returns a `job_id` |
| `kimodo_batch_generate` | Whole animation sets in one call |
| `kimodo_job_status` / `kimodo_job_logs` / `kimodo_list_jobs` / `kimodo_cancel_job` | Job control |
| `kimodo_make_constraints` | Author a constraints JSON (2D root paths/waypoints + headings, raw end-effector/full-body passthrough) without opening the demo |
| `kimodo_convert` | Format conversion: Kimodo NPZ / AMASS NPZ / SOMA BVH / G1 MuJoCo CSV |
| `kimodo_export_fbx` | Headless Blender: BVH to Unreal-ready FBX (cm, Z-up, baked) or GLB |
| `kimodo_inspect_motion` | QC: foot skate during contact, ground penetration, jitter, root speed, loopability |
| `kimodo_list_outputs` | Browse generated files, newest first |
| `kimodo_list_examples` | Kimodo's bundled constraint examples, ready to reuse |
| `kimodo_start_demo` / `kimodo_stop_demo` | Launch/stop the interactive timeline UI with `--offload` |

## Usage

Typical flow (the server's MCP instructions teach the agent this, so mostly you just describe what you want):

```
kimodo_status                                   # confirm GPU + install
kimodo_generate  preset="dodge_roll" name="player_dodge"
kimodo_job_status job_id=... wait_seconds=120
kimodo_inspect_motion path=".../player_dodge.npz"
kimodo_export_fbx input=".../player_dodge.bvh" target="unreal"
```

Path-constrained motion — make the character run an exact route:

```
kimodo_make_constraints name="flank_left" duration=6
    root_path={"points": [[0,0],[0,3],[-2,5]], "headings_deg":[0,0,-40]}
kimodo_generate prompts=["A person runs forward then cuts to the left."]
    durations=[6] constraints_path=".../flank_left.json"
    cfg_type="separated" cfg_weight=[2.0,3.0]
```

Whole animation set for a character:

```
kimodo_batch_generate presets=["idle","walk_forward","run_forward","dodge_roll","hit_reaction","death_fall"]
    subfolder="grunt_soldier"
```

### Prompting tips

One clear, present-tense sentence per action, describing "a person": *"A person sprints forward and slides."* Multiple prompts become one continuous clip, blended in order. Re-roll `seed` for a different take of the same prompt.

### Unreal import

Import the FBX as a **Skeletal Mesh** with *Import Animations* on, *Convert Scene* on, *Force Front XAxis* off, *Import Uniform Scale* 1.0, then retarget onto your skeleton with an IK Retargeter. The SOMA rig is 77 joints; the exporter converts Kimodo's Y-up metres to Unreal's Z-up centimetres, so characters come in at real-world scale. If a clip imports at the wrong size, re-export with `kimodo_export_fbx scale=<n>` rather than fixing it in-engine.

## Low-VRAM mode (how 6 GB cards work)

The low-VRAM fork only wires its multi-tier offloading (Disk/RAM/VRAM) into the interactive demo — the plain CLI doesn't use it, so the 5.4 GB NF4 text encoder and the diffusion model fight over VRAM. `kimodo_runner.py` reproduces what the demo actually does:

1. load the model, register it and the encoder with the fork's `MemoryManager`
2. encode all prompts through the disk-backed `CachedTextEncoder`
3. **evict the encoder to system RAM**, move the diffusion model back to the GPU
4. denoise using the cached embeddings

Text embeddings are cached to disk per unique prompt string, so encoding is a one-time cost: re-generating a preset is essentially just the diffusion pass (~10 s for a 2.5 s clip at 40 steps on an RTX 3050).

## Notes and limits

- **One generation at a time on small cards.** `kimodo_batch_generate` starts everything at once; on 6 GB, feed items one at a time unless you're fine with VRAM thrashing.
- **First run per model downloads the checkpoint** from Hugging Face — a job can sit "running" for a while with nothing in the log.
- **Kimodo doesn't make loops.** Generate longer than you need and cut a cycle; `kimodo_inspect_motion` reports first/last pose distance to help.
- **BVH/FBX export is SOMA-only.** SMPL-X models export AMASS NPZ, G1 exports MuJoCo CSV.

## License

This server is [Apache-2.0](LICENSE). Kimodo itself is Apache-2.0, but its **model checkpoints** are under the NVIDIA Open Model licence (the SMPL-X variant is R&D-only) — check those terms before shipping generated animation in a commercial product.
