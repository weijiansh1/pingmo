"""Validated parameter contract for the GJB-inspired P-channel family."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PChannelParameters:
    l_fa: float
    lambda_s: float
    t_r: float
    zeta_d: float
    omega_d: float
    r_omega: float
    r_zeta: float
    tau_p: float

    def __post_init__(self) -> None:
        if self.l_fa <= 0 or self.t_r <= 0 or self.omega_d <= 0:
            raise ValueError("l_fa, t_r, and omega_d must be positive")
        if not 0 < self.zeta_d < 1:
            raise ValueError("zeta_d must be in (0, 1)")
        if self.lambda_s == 0 or self.r_omega <= 0 or self.r_zeta <= 0 or self.tau_p < 0:
            raise ValueError("invalid spiral, ratio, or delay parameter")

    @property
    def omega_phi(self) -> float:
        return self.r_omega * self.omega_d

    @property
    def zeta_phi(self) -> float:
        return self.r_zeta * self.zeta_d
