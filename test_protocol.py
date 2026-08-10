#!/usr/bin/env python3
"""Smoke test: drive the Kimodo MCP server over stdio like a real MCP client."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(HERE, "kimodo_mcp_server.py")

proc = subprocess.Popen(
    [sys.executable, SERVER],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True, encoding="utf-8", bufsize=1,
)

failures = []
warnings = []
# Environment checks (GPU, kimodo venv, Blender) only fail with --strict;
# protocol checks always fail hard.
STRICT = "--strict" in sys.argv


def send(obj, expect_response=True):
    proc.stdin.write(json.dumps(obj) + "\n")
    proc.stdin.flush()
    if not expect_response:
        return None
    line = proc.stdout.readline()
    if not line:
        raise SystemExit("server closed stdout unexpectedly")
    return json.loads(line)


def check(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{'  ' + detail if detail and not cond else ''}")
    if not cond:
        failures.append(label)


def check_env(label, cond, detail=""):
    if STRICT:
        check(label, cond, detail)
        return
    print(f"{'PASS' if cond else 'WARN'}  {label}{'  ' + detail if detail and not cond else ''}")
    if not cond:
        warnings.append(label)


r = send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
          "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                     "clientInfo": {"name": "test", "version": "0"}}})
check("initialize returns result", "result" in r, json.dumps(r)[:300])
check("protocolVersion echoed", r["result"]["protocolVersion"] == "2025-06-18")
check("serverInfo present", r["result"]["serverInfo"]["name"] == "kimodo")
check("instructions present", len(r["result"].get("instructions", "")) > 100)

send({"jsonrpc": "2.0", "method": "notifications/initialized"}, expect_response=False)

r = send({"jsonrpc": "2.0", "id": 2, "method": "ping"})
check("ping", r.get("result") == {})

r = send({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
tools = r["result"]["tools"]
check("tools/list returns 17 tools", len(tools) == 17, str(len(tools)))
check("no private keys leaked", all(not k.startswith("_") for t in tools for k in t))
for t in tools:
    ok = ("name" in t and "description" in t and "inputSchema" in t
          and t["inputSchema"].get("type") == "object"
          and isinstance(t["inputSchema"].get("properties"), dict))
    check(f"schema valid: {t['name']}", ok, json.dumps(t)[:200])

r = send({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
          "params": {"name": "kimodo_list_models", "arguments": {}}})
check("list_models works", len(json.loads(r["result"]["content"][0]["text"])["models"]) == 6)

r = send({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
          "params": {"name": "kimodo_list_presets", "arguments": {"category": "combat"}}})
payload = json.loads(r["result"]["content"][0]["text"])
check("presets filter by category", all(p["category"] == "combat" for p in payload["presets"]))

r = send({"jsonrpc": "2.0", "id": 6, "method": "tools/call",
          "params": {"name": "kimodo_make_constraints", "arguments": {
              "name": "selftest_path", "duration": 4.0,
              "root_path": {"points": [[0, 0], [0, 2.0], [1.5, 3.5]], "headings_deg": [0, 0, 45]}}}})
payload = json.loads(r["result"]["content"][0]["text"])
check("constraints written", os.path.exists(payload["constraints_path"]))
with open(payload["constraints_path"]) as f:
    c = json.load(f)
check("constraint type root2d", c[0]["type"] == "root2d")
check("frames spread over duration", c[0]["frame_indices"] == [0, 60, 119], str(c[0]["frame_indices"]))
check("heading is cos/sin pairs", len(c[0]["global_root_heading"][0]) == 2)

r = send({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
          "params": {"name": "kimodo_make_constraints", "arguments": {"name": "bad"}}})
check("tool error is isError, not crash", r["result"]["isError"] is True)

r = send({"jsonrpc": "2.0", "id": 8, "method": "tools/call",
          "params": {"name": "nope", "arguments": {}}})
check("unknown tool -> isError", r["result"]["isError"] is True)

r = send({"jsonrpc": "2.0", "id": 9, "method": "does/not/exist"})
check("unknown method -> -32601", r.get("error", {}).get("code") == -32601)

r = send({"jsonrpc": "2.0", "id": 10, "method": "tools/call",
          "params": {"name": "kimodo_generate", "arguments": {"preset": "does_not_exist"}}})
check("bad preset rejected", r["result"]["isError"] is True)

r = send({"jsonrpc": "2.0", "id": 11, "method": "tools/call",
          "params": {"name": "kimodo_list_examples", "arguments": {}}})
payload = json.loads(r["result"]["content"][0]["text"])
check_env("bundled examples discovered", len(payload.get("examples", [])) > 0, json.dumps(payload)[:300])

r = send({"jsonrpc": "2.0", "id": 12, "method": "tools/call",
          "params": {"name": "kimodo_status", "arguments": {}}})
payload = json.loads(r["result"]["content"][0]["text"])
env = payload.get("environment", {})
check_env("status: torch present", "torch" in env, json.dumps(payload)[:600])
check_env("status: cuda available", env.get("cuda_available") is True, json.dumps(env)[:400])
check_env("status: kimodo importable", "kimodo_package" in env, json.dumps(env)[:400])
check_env("status: offload supported", env.get("offload_supported") is True)
check_env("status: embedding cache", env.get("embedding_cache") is True)
check_env("status: blender found", bool(payload.get("blender")))
print("\nGPU:", env.get("gpu"), "| VRAM total GB:", env.get("vram_total_gb"),
      "| models:", env.get("available_models"))
print("downloaded:", payload.get("downloaded_models"))

proc.stdin.close()
proc.wait(timeout=15)
print("\n--- server stderr ---")
print((proc.stderr.read() or "").strip()[-1200:])
print(f"\n{len(failures)} failure(s)" + (": " + ", ".join(failures) if failures else ""))
sys.exit(1 if failures else 0)
