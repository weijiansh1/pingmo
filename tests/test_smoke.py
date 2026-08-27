from pathlib import Path

from src.smoke import run_local_smoke


def test_local_smoke_writes_sac_and_moe_checkpoints(tmp_path: Path) -> None:
    report = run_local_smoke(tmp_path, updates=2, seed=11)
    assert report["sac"]["finite"]
    assert report["moe"]["finite"]
    assert (tmp_path / "sac_smoke.pt").exists()
    assert (tmp_path / "moe_smoke.pt").exists()
