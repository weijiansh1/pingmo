"""Six-channel short-period 品模: GJB LOES transfer functions plus rigid-body kinematics.

The roll channel is the frozen single-s P-channel. It is imported from the
current training source (``ppo_fixes_2026-10-26``) so that ``p`` is the same
plant the existing 5,000-object library and teachers use.
"""
from pathlib import Path
import sys

PINGMO = Path(__file__).resolve().parents[2]
ROLL_SOURCE = PINGMO / "ppo_fixes_2026-10-26"
if str(ROLL_SOURCE) not in sys.path:
    sys.path.insert(0, str(ROLL_SOURCE))

G = 9.80665
