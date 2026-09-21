"""Captures the RAW newline-delimited JSON-RPC wire traffic against the
contract-repository server -- bypassing the mcp SDK's ClientSession object
abstraction and talking directly to the subprocess's stdin/stdout, so what
lands in week9/wire.json is the literal protocol bytes, not a Python
object dump.

MCP over stdio is JSON-RPC 2.0, one message per line. The exchange
captured here is exactly: initialize -> notifications/initialized (no
response expected) -> tools/list -> tools/call.

Run:
    python -m week9.wire_capture
"""

import json
import os
import subprocess
import sys


def send(proc, message: dict):
    line = json.dumps(message) + "\n"
    proc.stdin.write(line)
    proc.stdin.flush()
    return message


def recv(proc) -> dict:
    line = proc.stdout.readline()
    return json.loads(line)


def main():
    proc = subprocess.Popen(
        [sys.executable, "-m", "week9.servers.contract_repo_server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )

    exchange = []

    try:
        # 1. initialize
        req = send(proc, {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "week9-wire-capture", "version": "0.1"},
            },
        })
        exchange.append({"direction": "client->server", "raw": req})
        resp = recv(proc)
        exchange.append({"direction": "server->client", "raw": resp})

        # 2. notifications/initialized (no id, no response expected)
        note = send(proc, {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        })
        exchange.append({"direction": "client->server", "raw": note})

        # 3. tools/list
        req2 = send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        exchange.append({"direction": "client->server", "raw": req2})
        resp2 = recv(proc)
        exchange.append({"direction": "server->client", "raw": resp2})

        # 4. tools/call
        req3 = send(proc, {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "lookup_contract",
                "arguments": {"contract_id": "MSA-2026-014"},
            },
        })
        exchange.append({"direction": "client->server", "raw": req3})
        resp3 = recv(proc)
        exchange.append({"direction": "server->client", "raw": resp3})

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    out_path = os.path.join(os.path.dirname(__file__), "wire.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(exchange, f, indent=2)

    print(f"[OK] Captured {len(exchange)} raw JSON-RPC messages -> {out_path}")
    for entry in exchange:
        print(f"  {entry['direction']}: {entry['raw'].get('method', entry['raw'].get('id'))}")


if __name__ == "__main__":
    main()
