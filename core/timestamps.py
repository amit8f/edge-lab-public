"""Outside timestamps for pre-registrations (PROTOCOL rule 13).

For every research/prereg/*.md whose current sha256 has no proof yet, a copy named <ID>-<sha12>.md is
stamped with the OpenTimestamps client: only the hash is sent to the public calendars, which anchor it
in the Bitcoin blockchain. The proof is saved as <proofs>/<ID>-<sha12>.md.ots (on GitHub: the `state`
branch, under timestamps/). Then every proof is upgraded, so pending ones gain their Bitcoin
attestation once it exists (a few hours later). MANIFEST.csv lists file, sha256 and the request time.

To check one by hand: drop the prereg file and its .ots proof on https://opentimestamps.org, or run
`ots verify -f research/prereg/M1.md M1-<sha12>.md.ots` with the proof from the state branch.

Usage: python -m core.timestamps [--root .] [--proofs state/timestamps]
"""
import argparse
import csv
import hashlib
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def pending(root, tdir):
    """[(prereg path, sha256, proof name)] for pre-registration versions without a proof in tdir."""
    out = []
    for p in sorted((root / "research" / "prereg").glob("*.md")):
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        name = f"{p.stem}-{sha[:12]}.md.ots"
        if not (tdir / name).exists():
            out.append((p, sha, name))
    return out


def run(root, proofs_dir=None, ots="ots", call=subprocess.run):
    tdir = Path(proofs_dir) if proofs_dir else root / "research" / "timestamps"
    tdir.mkdir(parents=True, exist_ok=True)
    manifest = tdir / "MANIFEST.csv"
    new = []
    with tempfile.TemporaryDirectory() as tmp:
        for p, sha, name in pending(root, tdir):
            copy = Path(tmp) / name[:-4]
            shutil.copyfile(p, copy)
            call([ots, "stamp", str(copy)], check=True)
            shutil.move(str(copy) + ".ots", tdir / name)
            new.append({"file": p.relative_to(root).as_posix(), "sha256": sha, "proof": name,
                        "requested_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    if new:
        exists = manifest.exists()
        with open(manifest, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file", "sha256", "proof", "requested_utc"])
            if not exists:
                w.writeheader()
            w.writerows(new)
    proofs = sorted(str(x) for x in tdir.glob("*.ots"))
    if proofs:
        call([ots, "upgrade", *proofs], check=False)     # pending proofs stay pending until Bitcoin confirms
    print(f"timestamps: {len(new)} new, {len(proofs)} proofs")
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--proofs", help="directory for the .ots proofs (default research/timestamps)")
    a = ap.parse_args()
    run(Path(a.root), a.proofs)


if __name__ == "__main__":
    main()
