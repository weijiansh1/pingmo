"""Parameter counts for three-loop teachers and a unified student, from the roll networks.

The roll student and teacher are rebuilt first to confirm the counting
(3,856 and 4,255,537).  Multi-loop observations are an assumption: each of the
three control loops (Fs->q, Fas->p, Frp->beta) keeps the roll loop's signals,
and four shared signals (alpha, n_z, phi, theta) are added.  Training time
extrapolates the GPU 4 benchmark and has not been measured for these loops.
"""
import json
from pathlib import Path
import sys

from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pinmo6  # noqa: E402,F401  (puts the training source on sys.path)
from src.envs.batched_p_channel import CURRENT_DIM, HISTORY_CHANNELS, OLDER_ACTIONS, THETA_DIM, HISTORY_STEPS  # noqa: E402
from src.teacher.ppo.networks import HistoryConv, PPOTeacherActor  # noqa: E402
from src.teacher.residual import ResidualMLPTrunk  # noqa: E402

LOOPS = 3
SHARED = 4
BENCHMARK = pinmo6.PINGMO / "teacher_gpu4_benchmark_2026-10-25/results.json"


def count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def student(channels: int, current: int, older: int, conv: int, width: int, blocks: int, outputs: int) -> int:
    return count(nn.ModuleList([
        HistoryConv(channels=channels, width=conv),
        ResidualMLPTrunk(2 * conv + current + older, width, blocks),
        nn.Linear(width, outputs, bias=False),
    ]))


def main() -> None:
    roll = dict(channels=HISTORY_CHANNELS, current=CURRENT_DIM, older=OLDER_ACTIONS, outputs=1)
    three = dict(channels=LOOPS * HISTORY_CHANNELS + SHARED, current=LOOPS * CURRENT_DIM + SHARED,
                 older=LOOPS * OLDER_ACTIONS, outputs=LOOPS)
    roll_tiny = student(**roll, conv=8, width=32, blocks=1)
    roll_teacher = count(PPOTeacherActor())
    assert roll_tiny == 3856 and roll_teacher == 4255537, (roll_tiny, roll_teacher)
    sizes = {
        "tiny_like (conv 8, 32 x 1)": student(**three, conv=8, width=32, blocks=1),
        "wider_tiny (conv 16, 64 x 2)": student(**three, conv=16, width=64, blocks=2),
        "small_like (conv 24, 256 x 4)": student(**three, conv=24, width=256, blocks=4),
        "teacher_scale (conv 24, 512 x 8)": student(**three, conv=24, width=512, blocks=8),
    }
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    seconds = benchmark["batch"]["amortized_seconds_per_teacher"]
    budget = {}
    for aircraft in (3000, 5000):
        teachers = aircraft * LOOPS
        gpu_hours = teachers * seconds / 3600
        budget[f"{aircraft}_aircraft"] = {"teachers": teachers, "gpu_hours": round(gpu_hours, 1),
                                          "hours_on_5_gpus": round(gpu_hours / 5, 1),
                                          "teacher_bank_actor_parameters": teachers * roll_teacher}
    result = {
        "check": {"roll_tiny_student": roll_tiny, "roll_teacher_actor": roll_teacher},
        "observation": {"loops": LOOPS, "shared_signals": SHARED, "history_steps": HISTORY_STEPS,
                        "dim": three["current"] + HISTORY_STEPS * three["channels"] + three["older"],
                        "roll_dim": CURRENT_DIM + HISTORY_STEPS * HISTORY_CHANNELS + OLDER_ACTIONS,
                        "privileged_theta_dim_roll": THETA_DIM},
        "three_loop_student_parameters": sizes,
        "per_loop_teacher_actor_parameters": roll_teacher,
        "teacher_seconds_per_gpu_from_benchmark": round(seconds, 3),
        "teacher_budget_extrapolated": budget,
    }
    (HERE / "model_size.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
