from pathlib import Path

from core import timestamps


def test_stamps_only_new_versions(tmp_path):
    pre = tmp_path / "research" / "prereg"
    pre.mkdir(parents=True)
    (pre / "M1.md").write_text("rules v1")
    calls = []

    def fake(cmd, check):
        calls.append(cmd[:2])
        if cmd[1] == "stamp":
            Path(cmd[2] + ".ots").write_bytes(b"proof")

    new = timestamps.run(tmp_path, call=fake)
    assert len(new) == 1 and new[0]["proof"].startswith("M1-") and ["ots", "upgrade"] in calls
    assert timestamps.run(tmp_path, call=fake) == []
    (pre / "M1.md").write_text("rules v2")
    assert len(timestamps.run(tmp_path, call=fake)) == 1
    rows = (tmp_path / "research" / "timestamps" / "MANIFEST.csv").read_text().splitlines()
    assert rows[0] == "file,sha256,proof,requested_utc" and len(rows) == 3
