# ORIGIN: copied from breakout-scanner shadow/ledger.py at commit 111b71d on 2026-10-11 (sha256_file inlined from shadow/paths.py).
"""Append-only, hash-chained log.

Every logged file (a night's picks, a universe snapshot, an outcome file) is written once
('x' mode, then made read-only) and registered in ledger.jsonl. Each JSON file embeds the SHA-256
of the previous ledger entry's file ("prev_sha256"), so a file's own hash covers the previous one:
changing or removing any earlier file breaks every later link. verify() recomputes the chain.
"""
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path

import hashlib

GENESIS = "0" * 64


def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class Ledger:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "ledger.jsonl"

    def entries(self):
        if not self.path.exists():
            return []
        return [json.loads(ln) for ln in self.path.read_text().splitlines() if ln.strip()]

    def last_sha(self):
        e = self.entries()
        return e[-1]["sha256"] if e else GENESIS

    def entry(self, kind, date):
        return next((e for e in self.entries() if e["kind"] == kind and e["date"] == date), None)

    def has(self, kind, date):
        return self.entry(kind, date) is not None

    def _register(self, kind, date, rel, sha, prev):
        e = {"seq": len(self.entries()) + 1, "kind": kind, "date": date, "file": rel, "sha256": sha,
             "prev_sha256": prev, "registered_utc": now_utc()}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(e, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return e

    def write_json(self, kind, date, rel, obj):
        """Write obj (+ prev_sha256 and written_utc) to root/rel exactly once and register it."""
        if self.has(kind, date):
            raise FileExistsError(f"{kind} for {date} is already logged (append-only)")
        prev = self.last_sha()
        obj = dict(obj, kind=kind, prev_sha256=prev, written_utc=now_utc())
        return self._write(kind, date, rel, json.dumps(obj, indent=1, sort_keys=True, default=str).encode(), prev)

    def write_bytes(self, kind, date, rel, data):
        if self.has(kind, date):
            raise FileExistsError(f"{kind} for {date} is already logged (append-only)")
        return self._write(kind, date, rel, data, self.last_sha())

    def _write(self, kind, date, rel, data, prev):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(p, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
        return self._register(kind, date, rel, sha256_file(p), prev)

    def verify(self):
        """List of problems; empty = every file present, hashes match, links unbroken."""
        probs, prev = [], GENESIS
        for e in self.entries():
            p = self.root / e["file"]
            if e["prev_sha256"] != prev:
                probs.append(f"seq {e['seq']}: prev link {e['prev_sha256'][:12]} != {prev[:12]}")
            if not p.exists():
                probs.append(f"seq {e['seq']}: missing {e['file']}")
            else:
                if sha256_file(p) != e["sha256"]:
                    probs.append(f"seq {e['seq']}: hash mismatch {e['file']}")
                if p.suffix == ".json" and json.loads(p.read_text()).get("prev_sha256") != e["prev_sha256"]:
                    probs.append(f"seq {e['seq']}: embedded prev_sha256 differs in {e['file']}")
            prev = e["sha256"]
        return probs


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
