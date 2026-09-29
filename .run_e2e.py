"""TEMPORARY: live E2E driver (same flow as the previously verified script).

Boots the REAL webhook receiver process (python -m core.github_webhook) on 127.0.0.1:8765,
sends the two signed demo deliveries, verifies RETAIN/RECALL against the real bank, runs
Ask PulseMind, then deletes the demo units and restores the CSV — leaving the bank with
exactly its original 83 units.
"""

from __future__ import annotations

import json
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")

# A random per-run secret, injected via process env before settings load. Never printed.
TEST_SECRET = secrets.token_hex(24)

import dataclasses  # noqa: E402

import os  # noqa: E402

os.environ["GITHUB_WEBHOOK_SECRET"] = TEST_SECRET
os.environ["GITHUB_WEBHOOK_PORT"] = "8765"
# This 8GB machine swaps heavily while the local embedding/reranker models load, which
# pushed first-time init past the server's default 300s budget on every attempt.
os.environ["HINDSIGHT_API_MODEL_INIT_TIMEOUT"] = "900"

from core.config import get_settings  # noqa: E402


def dataclasses_settings_replace(base, **changes):
    return dataclasses.replace(base, **changes)


settings = get_settings()
t0 = time.time()
results: dict[str, bool] = {}

# ---------------------------------------------------------------- receiver up
print("[1] starting the REAL receiver process (python -m core.github_webhook)")
receiver = subprocess.Popen(
    [PY, "-m", "core.github_webhook"],
    cwd=str(ROOT),
    env={**os.environ},
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding="utf-8",
    errors="replace",
)
base = "http://127.0.0.1:8765"
receiver_ready = False
for _ in range(300):  # up to 10 minutes: cold start loads models under heavy swap
    if receiver.poll() is not None:
        print("    receiver process exited early with code", receiver.returncode)
        break
    try:
        with urllib.request.urlopen(base + "/health", timeout=3) as response:
            if response.status == 200:
                receiver_ready = True
                break
    except Exception:  # noqa: BLE001
        time.sleep(2.0)
results["receiver up on 127.0.0.1:8765"] = receiver_ready
print(f"    receiver ready: {receiver_ready} ({time.time() - t0:.0f}s)")
if not receiver_ready:
    print(receiver.stdout.read() if receiver.stdout else "(no output)")
    raise SystemExit(2)

# ---------------------------------------------------------------- bank before
# The receiver process already started the heavyweight Hindsight server (embedding
# model + pg0). This machine is low on free RAM (os error 1455 when two servers load
# concurrently), so this driver process attaches to the SAME server as a plain remote
# client instead of starting a second embedded server. The server picks a random port,
# so discover it from the OS: the receiver PID's listening TCP ports.
print("[2] discover the receiver's Hindsight port and attach as a remote client")

def _listening_ports(pid: int) -> set[int]:
    out = subprocess.run(
        ["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, timeout=30
    ).stdout
    ports = set()
    target = str(pid)
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[3] == "LISTENING" and parts[4] == target:
            if ":" in parts[1]:
                ports.add(int(parts[1].rsplit(":", 1)[1]))
    return ports

server_port = None
for _ in range(60):
    if receiver.poll() is not None:
        break
    ports = _listening_ports(receiver.pid)
    ports.discard(8765)  # the webhook port; the Hindsight server is the other listener
    if ports:
        server_port = sorted(ports)[0]
        break
    time.sleep(2.0)
if not server_port:
    print("    could not find the receiver's Hindsight port; netstat lines for the PID:")
    out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True).stdout
    print("\n".join(line for line in out.splitlines() if line.endswith(str(receiver.pid)))[:2000])
    receiver.terminate()
    raise SystemExit(2)
print(f"    Hindsight server endpoint: http://127.0.0.1:{server_port}")

from core.hindsight_client import HindsightMemory as _HM  # noqa: E402

remote_settings = dataclasses_settings_replace(
    settings,
    hindsight_mode="remote",
    hindsight_base_url=f"http://127.0.0.1:{server_port}",
)
memory = _HM(remote_settings).start(timeout=600.0)
ids_before = {r.id for r in memory.list_memories(limit=1000)[0]}
print(f"    bank units before: {len(ids_before)}")
results["bank preserved (83 units)"] = len(ids_before) == 83

# ---------------------------------------------------------------- signed sends
import hashlib  # noqa: E402
import hmac  # noqa: E402

from scripts.send_test_webhook import build_payload, send  # noqa: E402


def recall_contains(query: str, marker: str) -> tuple[bool, int]:
    hits = memory.recall(query, budget="mid", max_tokens=2048)
    joined = " ".join(hit.text for hit in hits)
    found = any(marker in hit.tags for hit in hits) or marker.lower() in joined.lower()
    return found, len(hits)


print("[3] send signed PR #42 (Checkout V3) to the real endpoint")
event, payload = build_payload(
    kind="pr",
    number=42,
    title="Checkout V3: fewer steps to pay",
    description="Reduces checkout to two steps; retries failed UPI payments.",
    when="2026-03-20T09:00:00Z",
)
status, body = send(base + "/webhooks/github", event, payload, TEST_SECRET, timeout=300)
results["PR delivery accepted"] = status == 200 and body.get("ok") is True
print(f"    HTTP {status}: {json.dumps(body)[:200]}")
doc_pr = body.get("document_id", "")

print("[4] verify the PR change really landed in Hindsight")
new_ids = {r.id for r in memory.list_memories(limit=1000)[0]} - ids_before
results["PR retained (new units)"] = len(new_ids) > 0
found, hits = recall_contains("Checkout V3 pull request merged checkout", "source:github")
results["PR recallable with github evidence"] = bool(hits) and found
print(f"    new units: {len(new_ids)}; recall hits: {hits}; github evidence: {found}")

print("[5] send signed release v3.0.0")
ids_after_pr = {r.id for r in memory.list_memories(limit=1000)[0]}
event, payload = build_payload(
    kind="release",
    number=7,
    title="Checkout V3: fewer steps to pay",
    version="v3.0.0",
    when="2026-03-21T12:00:00Z",
)
status, body = send(base + "/webhooks/github", event, payload, TEST_SECRET, timeout=300)
results["release delivery accepted"] = status == 200 and body.get("ok") is True
print(f"    HTTP {status}: {json.dumps(body)[:200]}")
doc_rel = body.get("document_id", "")

print("[6] verify the release landed in Hindsight")
new_ids = {r.id for r in memory.list_memories(limit=1000)[0]} - ids_after_pr
results["release retained (new units)"] = len(new_ids) > 0
found, hits = recall_contains("Checkout V3 release version v3.0.0", "v3.0.0")
results["release recallable"] = bool(hits) and found
print(f"    new units: {len(new_ids)}; recall hits: {hits}; version evidence: {found}")

print("[7] security: invalid signature and push event against the REAL receiver")
ids_before_security = {r.id for r in memory.list_memories(limit=1000)[0]}
event, payload = build_payload(kind="pr", number=43, title="should not land")
status, body = send(base + "/webhooks/github", event, payload, "wrong-secret", timeout=60)
results["invalid signature rejected (403)"] = status == 403
print(f"    invalid signature: HTTP {status}")
event, payload = build_payload(kind="pr", number=44, title="push-like event")
status, _ = send(base + "/webhooks/github", "push", {"zen": "x", "repository": {"full_name": "pulsemind/demo-shop"}}, TEST_SECRET, timeout=60)
results["push ignored (200, nothing retained)"] = status == 200
results["no units added by security tests"] = (
    {r.id for r in memory.list_memories(limit=1000)[0]} == ids_before_security
)
print(f"    push: HTTP {status}")

print("[8] Ask PulseMind end-to-end (1 Groq reasoning call)")
from core.agent import PulseMindAgent  # noqa: E402

agent = PulseMindAgent(settings)
agent.memory = memory
answer = agent.ask("Did customer feedback improve after the Checkout V3 release?")
text = (answer.text or "").lower()
markers = [m for m in ("improve", "fell", "dropped", "14.6", "39.1", "checkout v3", "release") if m in text]
results["Ask executed without error"] = answer.error is None and bool(answer.memories_used)
results["Ask grounded in history"] = len(markers) >= 2
print(f"    recalled={len(answer.memories_used)} markers={markers} error={answer.error}")
print(f"    A: {' '.join((answer.text or '').split())[:300]}")

print("[9] Product Changes page renders the GitHub change (AppTest)")
import streamlit.testing.v1 as st_testing  # noqa: E402

source = (ROOT / "ui" / "product_changes.py").read_text(encoding="utf-8")
stub = (
    "\n\n# ---- test stub ----\n"
    f"import sys as _sys\n_sys.path.insert(0, r'{ROOT}')\n"
    "from core.agent import PulseMindAgent as PulseMindAgent\n"
    "from core.config import get_settings as get_settings\n"
    "from tests.test_agent import FakeLLM as FakeLLM\n"
    "agent = PulseMindAgent(get_settings(), memory=memory, llm=FakeLLM())\n"
    "agent.reload_data()\n"
    "render(agent)\n"
)
runnable = ROOT / ".render_pc_live.py"
runnable.write_text(source + stub, encoding="utf-8")
app = st_testing.AppTest.from_file(str(runnable.resolve())).run(timeout=120)
page_text = " ".join(str(element.value) for element in app.main)
results["Product Changes renders live"] = app.exception is None
results["PR visible on page"] = "Checkout V3: fewer steps to pay" in page_text and "#42" in page_text
results["release visible on page"] = "v3.0.0" in page_text
results["github provenance on page"] = "Open on GitHub" in page_text
runnable.unlink(missing_ok=True)
print(f"    page exception: {app.exception}; PR visible: {'#42' in page_text}; "
      f"version visible: {'v3.0.0' in page_text}; link present: {'Open on GitHub' in page_text}")

print("[10] cleanup: delete the two demo documents + CSV rows, restore 83 units")
for doc_id in (doc_pr, doc_rel):
    if not doc_id:
        continue
    request = urllib.request.Request(
        f"{memory.endpoint}/v1/default/banks/{settings.bank_id}/documents/{doc_id}", method="DELETE"
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            out = json.loads(response.read().decode() or "{}")
            print(f"    DELETE {doc_id}: units removed={out.get('memory_units_deleted')}")
    except urllib.error.HTTPError as exc:
        print(f"    DELETE {doc_id}: HTTP {exc.code}")

time.sleep(3.0)
leftover = [
    r for r in memory.list_memories(limit=1000)[0]
    if "source:github" in r.tags and "fewer steps to pay" in r.text.lower()
]
results["memory cleanup complete"] = not leftover
import pandas as pd  # noqa: E402

csv_path = settings.product_changes_csv
if csv_path.exists():
    frame = pd.read_csv(csv_path)
    if "github_url" in frame.columns:
        keep = frame[~frame["github_url"].astype(str).str.contains("pulsemind/demo-shop", na=False)]
        removed = len(frame) - len(keep)
        if removed:
            keep.to_csv(csv_path, index=False)
        results["csv cleanup complete"] = True
        print(f"    CSV rows removed: {removed}")

final_ids = {r.id for r in memory.list_memories(limit=1000)[0]}
results["bank restored to 83 units"] = len(final_ids) == 83
print(f"    bank units after cleanup: {len(final_ids)}")

receiver.terminate()
try:
    receiver.wait(timeout=30)
except subprocess.TimeoutExpired:
    receiver.kill()
memory.close()

print("\n" + "=" * 74)
print("VERDICT")
for name, ok in results.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
print(f"  elapsed: {time.time() - t0:.0f}s")
raise SystemExit(0 if all(results.values()) else 2)
