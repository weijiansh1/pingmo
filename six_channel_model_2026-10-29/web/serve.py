"""Six-channel library viewer: static page, library summary and on-demand simulation.

Binds 127.0.0.1 only; reached through the Jupyter proxy at /proxy/<port>/, so the
page uses relative URLs.  Read-only: it never writes to the libraries.
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402

from pinmo6.coupled import CoupledAircraft, InertiaMoments, simulate_coupled  # noqa: E402
from pinmo6.criteria import ROLL_PERFORMANCE  # noqa: E402
from pinmo6.sampler import SOURCE_LIBRARY  # noqa: E402
from pinmo6.sampler_v4 import binding_group, load_v4  # noqa: E402

SERVICE = "pingmo-six-channel"
REVISION = "six-channel-v4-viewer-2026-10-31"
LIBRARY = PROJECT / "library/six_channel_5000_v4/plants.jsonl"
STATIC_DIR = HERE / "static"
TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".woff2": "font/woff2",
         ".woff": "font/woff", ".ttf": "font/ttf"}
PAGES = {"/": "model.html", "/library": "library.html"}
DOCS = {"/docs/design": PROJECT / "交叉通道与其余限制_设计说明.md", "/docs/readme": PROJECT / "README.md"}
SAMPLE_EVERY = 20                                   # 1 kHz plant, 50 Hz to the browser

ENTRIES = load_v4(LIBRARY)
BY_ID = {record["plant_id"]: (record, parts) for record, parts in ENTRIES}


def _round(value, digits: int = 5):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return float(f"{value:.{digits}g}")


def summary() -> dict:
    fields = ["id", "split", "level", "group", "t_r", "lambda_s", "zeta_d", "omega_d", "r_omega", "zeta_phi",
              "sensitivity", "tau_p", "eta", "phi_over_beta", "dihedral", "psi_beta", "dbeta_over_k",
              "speed_range", "v0", "omega_sp", "zeta_sp", "cap", "tau_theta", "zeta_p", "y_b"]
    rows = []
    for record, parts in ENTRIES:
        roll, loes, lat = parts["roll"], parts["loes"], parts["lateral"]
        crit = record["criteria"]
        phugoid = record["longitudinal"]["phugoid"]
        rows.append([record["plant_id"], record["split"], record["joint_level"] or 0, binding_group(record),
                     *[_round(x) for x in (roll.t_r, roll.lambda_s, roll.zeta_d, roll.omega_d, roll.r_omega,
                                           roll.zeta_phi, record["roll"]["sensitivity_1s_deg_per_n"], roll.tau_p,
                                           record["eta"], crit["dutch_roll_shape"]["phi_over_beta"],
                                           record["dihedral_ratio"], crit["psi_beta_deg"],
                                           crit["sideslip_excursion"].get("delta_beta_over_k_deg"))],
                     record["envelope"]["speed_range"],
                     *[_round(x) for x in (loes.v0, loes.omega_sp, loes.zeta_sp, loes.cap, loes.tau_theta,
                                           None if phugoid is None else phugoid["zeta_p"], lat.y_b)]])
    original = {"t_r": [], "lambda_s": [], "zeta_d": [], "omega_d": [], "r_omega": [], "zeta_phi": [],
                "sensitivity": []}
    for line in SOURCE_LIBRARY.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        p = row["parameters"]
        for key in ("t_r", "lambda_s", "zeta_d", "omega_d", "r_omega", "zeta_phi"):
            original[key].append(_round(p[key], 4))
        original["sensitivity"].append(_round(row["sensitivity_measured_deg_per_n"], 4))
    real = []
    with (PROJECT / "literature/cr2144_lateral.csv").open(encoding="utf-8") as stream:
        for r in csv.DictReader(stream):
            f = lambda k: float(r[k])  # noqa: E731
            real.append({"aircraft": r["aircraft"], "fc": r["fc"],
                         "t_r": _round(1 / f("inv_TR_comp")), "lambda_s": _round(-f("inv_Ts_comp")),
                         "zeta_d": _round(f("zeta_d_comp")), "omega_d": _round(f("omega_d_comp")),
                         "r_omega": _round(f("omega_phi_comp") / f("omega_d_comp")),
                         "zeta_phi": _round(f("zeta_phi_comp")),
                         "eta": _round(f("Np_da_stab") / f("Lp_da_stab")),
                         "phi_over_beta": _round(f("phi_beta_mag_d_comp")),
                         "dihedral": _round(-f("Lp_beta_stab") / f("Np_beta_stab")), "y_b": _round(f("Y_v_1ps"))})
    manifest = json.loads((LIBRARY.parent / "manifest.json").read_text(encoding="utf-8"))
    return {"fields": fields, "rows": rows, "original": original, "real": real,
            "manifest": {k: manifest[k] for k in ("candidate_count", "valid_candidates", "pool_sizes", "seed")}}


SUMMARY = json.dumps(summary(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def detail(plant_id: str) -> dict:
    record, parts = BY_ID[plant_id]
    keep = ("plant_id", "split", "sampling_bucket", "joint_level", "ood_reasons", "component_levels", "roll",
            "lateral", "eta", "tau_rp", "dihedral_ratio", "sideslip_per_force_deg_per_n", "envelope",
            "longitudinal_loes", "inertia", "span_m", "criteria")
    out = {k: record[k] for k in keep}
    out["longitudinal"] = {k: record["longitudinal"][k] for k in ("phugoid", "inv_t_theta1", "x_u", "m_u")}
    out["binding_group"] = binding_group(record)
    out["loes_derived"] = {"cap": parts["loes"].cap, "t_theta2": parts["loes"].t_theta2,
                           "k_theta": parts["loes"].k_theta}
    return out


def _series(states, column, scale=1.0, offset=0.0):
    return [_round((v - offset) * scale, 4) for v in states[::SAMPLE_EVERY, column]]


@lru_cache(maxsize=256)
def steps(plant_id: str) -> dict:
    record, parts = BY_ID[plant_id]
    roll, loes = parts["roll"], parts["loes"]
    seconds = 8.0
    n = int(seconds * 1000)
    deg = math.degrees(1.0)
    # small steps show the channel dynamics; large motions are in maneuvers()
    f_as = 2.0 / record["roll"]["sensitivity_1s_deg_per_n"]           # 2 deg of bank in 1 s
    f_rp = 0.5 / record["sideslip_per_force_deg_per_n"]                 # 0.5 deg of second-order sideslip
    f_s = 0.5 * loes.fs_per_g                                           # 0.5 g steady
    aircraft = parts["aircraft"]()
    stick = simulate_coupled(aircraft, np.tile([0.0, f_as, 0.0], (n, 1)))
    pedal = simulate_coupled(aircraft, np.tile([0.0, 0.0, f_rp], (n, 1)))
    pitch = simulate_coupled(aircraft, np.tile([f_s, 0.0, 0.0], (n, 1)))
    return {
        "t": [_round(t, 4) for t in stick["time"][::SAMPLE_EVERY]],
        "inputs": {"stick_n": f_as, "pedal_n": f_rp, "pitch_n": f_s},
        "stick": {"p": _series(stick["states"], 5, deg), "beta": _series(stick["states"], 4, deg),
                  "r": _series(stick["states"], 6, deg), "phi": _series(stick["states"], 7, deg)},
        "pedal": {"beta": _series(pedal["states"], 4, deg), "r": _series(pedal["states"], 6, deg),
                  "p": _series(pedal["states"], 5, deg), "phi": _series(pedal["states"], 7, deg)},
        "pitch": {"nz": [_round(v, 4) for v in pitch["nz_g"][::SAMPLE_EVERY]],
                  "q": _series(pitch["states"], 2, deg),
                  "alpha": _series(pitch["states"], 1, deg, loes.alpha_trim),
                  "dv": _series(pitch["states"], 0, 1.0, loes.v0)},
    }


LINEAR_LIMIT = math.radians(15.0)


def _guarded_run(aircraft, loes, command, steps):
    """Step until done or until alpha change or sideslip passes 15 deg, where the linear
    aerodynamics stop meaning anything; returns states, n_z and the time it left that range."""
    aircraft.reset()
    states, nz, state = np.empty((steps, 12)), np.empty(steps), None
    for k in range(steps):
        state = aircraft.step(*command(k, state))
        states[k], nz[k] = state, aircraft.nz_g
        if not (abs(state[1] - loes.alpha_trim) <= LINEAR_LIMIT and abs(state[4]) <= LINEAR_LIMIT):
            return states[:k + 1], nz[:k + 1], k * 0.001
    return states, nz, None


@lru_cache(maxsize=128)
def maneuvers(plant_id: str) -> dict:
    record, parts = BY_ID[plant_id]
    loes = parts["loes"]
    deg = math.degrees(1.0)
    out = {}
    # a 1.5 s stick pulse sized on the linear lateral model so the bank settles near 60 deg once the
    # roll mode has subsided (1.5 s + 3 T_R), then hold or pull 1 g (a level 60 deg turn needs n = 2)
    a, b = parts["lateral"].matrices()
    settle = min(1.5 + 3.0 * parts["roll"].t_r, 10.0)
    t_lin = np.arange(0.0, settle + 1e-9, 0.001)
    pulse = ((t_lin >= parts["roll"].tau_p) & (t_lin < 1.5 + parts["roll"].tau_p)).astype(float)
    _, phi_pulse, _ = signal.lsim((a, b[:, [0]], np.array([[0.0, 0.0, 0.0, 1.0]]), np.zeros((1, 1))), pulse, t_lin)
    bank_stick = math.radians(60.0) / float(phi_pulse[-1])
    n = 15000
    turn = np.zeros((n, 3))
    turn[:1500, 1] = bank_stick
    pull = turn.copy()
    pull[1500:, 0] = loes.fs_per_g
    for name, inputs in (("no_pull", turn), ("pull_1g", pull)):
        aircraft = parts["aircraft"]()
        s, nz, left_range = _guarded_run(aircraft, loes, lambda k, state: inputs[k], n)
        out[name] = {"altitude": _series(s, 11, -1.0), "phi": _series(s, 7, deg), "psi": _series(s, 8, deg),
                     "nz": [_round(v, 4) for v in nz[::SAMPLE_EVERY]], "left_linear_range_s": left_range}
    out["t_turn"] = [_round(i * 0.001, 4) for i in range(0, n, SAMPLE_EVERY)]
    # MIL-STD-1797A 4.8.1: a yaw-control-free roll through 360 deg.  The linear model has no stick
    # limit, so the stick is sized to the level-1 roll performance of the object's speed range
    # (table XXX: phi_req within t_req), held until the bank has turned 360 deg, then released.
    bank_req, t_req = ROLL_PERFORMANCE[record["envelope"]["speed_range"]]
    a, b = parts["lateral"].matrices()
    t_lin = np.arange(0.0, t_req + 1e-9, 0.001)
    forcing = (t_lin >= parts["roll"].tau_p).astype(float)
    _, phi_per_n, _ = signal.lsim((a, b[:, [0]], np.array([[0.0, 0.0, 0.0, 1.0]]), np.zeros((1, 1))), forcing, t_lin)
    roll_stick = math.radians(bank_req) / float(phi_per_n[-1])
    iy, iz = parts["inertia"].iy, parts["inertia"].iz
    out["phillips"] = {"p_pitch_deg_s": math.degrees(loes.omega_sp * math.sqrt(iy / (iz - 1.0))) if iz > 1.0 else None,
                       "p_yaw_deg_s": math.degrees(record["roll"]["omega_d"] * math.sqrt(iz / (iy - 1.0)))
                       if iy > 1.0 else None}
    n = 6000
    for name, inertia in (("with_coupling", parts["inertia"]), ("without_coupling", InertiaMoments(1.0, 1.0, 0.0))):
        aircraft = CoupledAircraft(loes, parts["roll"], record["tau_rp"], parts["lateral"], parts["longitudinal"],
                                   inertia)
        release = {"t": None}

        def command(k, state, release=release):
            if release["t"] is None and state is not None and state[7] >= 2 * math.pi:
                release["t"] = k * 0.001
            return (0.0, 0.0 if release["t"] is not None else roll_stick, 0.0)

        states, _, left_range = _guarded_run(aircraft, loes, command, n)
        released = release["t"]
        out[name] = {"alpha": _series(states, 1, deg, loes.alpha_trim), "beta": _series(states, 4, deg),
                     "phi": _series(states, 7, deg), "released_s": released, "left_linear_range_s": left_range,
                     "max_abs_d_alpha_deg": float(np.max(np.abs(states[:, 1] - loes.alpha_trim)) * deg),
                     "max_abs_beta_deg": float(np.max(np.abs(states[:, 4])) * deg),
                     "max_roll_rate_deg_s": float(np.max(np.abs(states[:, 5])) * deg)}
    out["t_roll"] = [_round(i * 0.001, 4) for i in range(0, n, SAMPLE_EVERY)]
    out["inputs"] = {"bank_stick_n": bank_stick, "roll_stick_n": roll_stick,
                     "pull_n": loes.fs_per_g, "roll_requirement": [bank_req, t_req]}
    return out


def _static_file(relative: str) -> Path | None:
    """A file inside static/ with a known type; never anything outside it."""
    candidate = (STATIC_DIR / relative).resolve()
    if STATIC_DIR.resolve() not in candidate.parents or not candidate.is_file() or candidate.suffix not in TYPES:
        return None
    return candidate


ENGINE_LAG_S = 1.0
FREQUENCIES = np.logspace(-2, 2, 161)


def _poly(values) -> list[float]:
    return [float(v) for v in np.atleast_1d(values)]


def _roots(values) -> list[list[float]]:
    values = np.trim_zeros(np.atleast_1d(values), "f")
    if len(values) < 2:
        return []
    return [[float(r.real), float(r.imag)] for r in sorted(np.roots(values), key=lambda z: (z.real, z.imag))]


def _trim(num) -> np.ndarray:
    """Drop leading coefficients at round-off level (ss2tf leaves ~1e-16 * max, i.e. fake zeros at 1e12)."""
    num = np.atleast_1d(np.asarray(num, dtype=float))
    scale = np.max(np.abs(num)) if len(num) else 0.0
    while len(num) > 1 and abs(num[0]) <= 1e-9 * scale:
        num = num[1:]
    return num


def _channel(num, den, delay: float, source: str) -> dict:
    num = _trim(num)
    den = np.asarray(den, dtype=float)
    if not len(num):
        num = np.array([0.0])
    jw = 1j * FREQUENCIES
    response = np.polyval(num, jw) / np.polyval(den, jw) * np.exp(-jw * delay)
    with np.errstate(divide="ignore"):
        magnitude = 20.0 * np.log10(np.abs(response))
    return {"num": _poly(num), "den": _poly(den), "zeros": _roots(num), "poles": _roots(den), "delay": delay,
            "source": source, "bode": {"db": [_round(v, 5) for v in magnitude],
                                       "deg": [_round(v, 5) for v in np.degrees(np.unwrap(np.angle(response)))]}}


def _state_tf(a, b, column: int, row) -> tuple[np.ndarray, np.ndarray]:
    num, den = signal.ss2tf(a, b[:, [column]], np.atleast_2d(row), np.zeros((1, 1)))
    return num[0], den


def _modes(a, names, kind: str, roll=None) -> list[dict]:
    values, vectors = np.linalg.eig(a)
    real = [value.real for value in values if abs(value.imag) <= 1e-9]
    # the real pole nearest -1/T_R is the roll mode; a slow roll can sit close to a stable spiral
    roll_pole = min(real, key=lambda x: abs(x + 1.0 / roll.t_r)) if kind == "lat" and roll is not None and real else None
    out = []
    for k, value in enumerate(values):
        if value.imag < -1e-9:
            continue
        v = vectors[:, k]
        pivot = int(np.argmax(np.abs(v)))
        v = v / v[pivot]
        if abs(value.imag) > 1e-9:
            wn = abs(value)
            label = ("荷兰滚" if kind == "lat" else ("短周期" if wn > 0.3 else "长周期"))
            entry = {"mode": label, "omega": float(wn), "zeta": float(-value.real / wn)}
        else:
            if kind == "lat":
                label = "滚转" if value.real == roll_pole else "螺旋"
            else:
                label = "长周期实根" if abs(value.real) < 0.3 else "短周期实根"
            entry = {"mode": label, "time_constant": float(-1.0 / value.real) if value.real else None}
        entry.update({"eigenvalue": [float(value.real), float(value.imag)],
                      "vector": {n: [float(abs(c)), float(np.degrees(np.angle(c)))] for n, c in zip(names, v)}})
        out.append(entry)
    return out


@lru_cache(maxsize=256)
def model(plant_id: str) -> dict:
    """Numbers for the formula page: matrices, every channel, modes and the LOES checks."""
    from src.aircraft.p_channel import p_channel_polynomials
    record, parts = BY_ID[plant_id]
    roll, loes, lat, lon = parts["roll"], parts["loes"], parts["lateral"], parts["longitudinal"]
    a_lat, b_lat = lat.matrices()
    a_lon, b_lon = lon.matrices()
    g, v0 = 9.80665, loes.v0
    nz_row = np.array([-(v0 / g) * lon.z_u, -(v0 / g) * lon.z_a, 0.0, 0.0])
    sp = [1.0, 2 * loes.zeta_sp * loes.omega_sp, loes.omega_sp**2]
    channels = {}
    channels["q_Fs_loes"] = _channel(loes.k_theta * np.array([1.0, 1.0 / loes.t_theta2]), sp, loes.tau_theta, "loes")
    channels["alpha_Fs_loes"] = _channel([loes.k_theta], sp, loes.tau_theta, "loes")
    channels["nz_Fs_loes"] = _channel([loes.k_nz], sp, loes.tau_theta, "loes")
    for key, row in (("q_Fs", [0, 0, 1, 0]), ("theta_Fs", [0, 0, 0, 1]), ("alpha_Fs", [0, 1, 0, 0]),
                     ("u_Fs", [1, 0, 0, 0]), ("nz_Fs", nz_row)):
        channels[key] = _channel(*_state_tf(a_lon, b_lon, 0, row), loes.tau_theta, "state")
    lag = [ENGINE_LAG_S, 1.0]
    for key, row in (("u_dT", [1, 0, 0, 0]), ("theta_dT", [0, 0, 0, 1])):
        num, den = _state_tf(a_lon, b_lon, 1, row)
        channels[key] = _channel(num, np.polymul(den, lag), 0.0, "state")
    channels["p_Fas_loes"] = _channel(*p_channel_polynomials(roll), roll.tau_p, "loes")
    for key, column, row, delay in (("p_Fas", 0, [0, 1, 0, 0], roll.tau_p), ("beta_Fas", 0, [1, 0, 0, 0], roll.tau_p),
                                    ("r_Fas", 0, [0, 0, 1, 0], roll.tau_p), ("phi_Fas", 0, [0, 0, 0, 1], roll.tau_p),
                                    ("beta_Frp", 1, [1, 0, 0, 0], record["tau_rp"]), ("r_Frp", 1, [0, 0, 1, 0], record["tau_rp"]),
                                    ("p_Frp", 1, [0, 1, 0, 0], record["tau_rp"])):
        channels[key] = _channel(*_state_tf(a_lat, b_lat, column, row), delay, "state")
    # exactness of the derivative models against the LOES they were built from
    p_state, p_loes = channels["p_Fas"], channels["p_Fas_loes"]
    lat_error = max(np.max(np.abs(np.array(p_state["den"]) - p_loes["den"]) / np.maximum(1, np.abs(p_loes["den"]))),
                    np.max(np.abs(np.array(p_state["num"][-4:]) - p_loes["num"][-4:])) / np.max(np.abs(p_loes["num"])))
    quotient, remainder = np.polydiv(channels["q_Fs"]["den"], sp)
    phugoid = np.polymul(sp, quotient)
    lon_error = float(np.max(np.abs(remainder)) / np.max(np.abs(channels["q_Fs"]["den"])))
    return {
        "id": plant_id, "level": record["joint_level"], "group": binding_group(record), "ood_reasons": record["ood_reasons"],
        "loes": {**{k: getattr(loes, k) for k in ("omega_sp", "zeta_sp", "n_alpha", "v0", "fs_per_g", "tau_theta",
                                                   "alpha_trim")},
                 "t_theta2": loes.t_theta2, "cap": loes.cap, "k_theta": loes.k_theta, "k_nz": loes.k_nz},
        "roll": {**{k: getattr(roll, k) for k in ("l_fa", "lambda_s", "t_r", "zeta_d", "omega_d", "r_omega", "r_zeta",
                                                   "tau_p")},
                 "omega_phi": roll.omega_phi, "zeta_phi": roll.zeta_phi,
                 "sensitivity": record["roll"]["sensitivity_1s_deg_per_n"]},
        "lateral": {"A": a_lat.tolist(), "B": b_lat.tolist(), **{k: getattr(lat, k) for k in lat.__dataclass_fields__},
                    "eta": record["eta"], "tau_rp": record["tau_rp"]},
        "longitudinal": {"A": a_lon.tolist(), "B": b_lon.tolist(),
                         **{k: getattr(lon, k) for k in ("g", "x_u", "x_a", "z_u", "z_a", "m_u", "m_a", "m_q", "m_s",
                                                          "inv_t_theta1")},
                         "long_period": list(lon.long_period), "phugoid": lon.phugoid, "engine_lag_s": ENGINE_LAG_S},
        "inertia": record["inertia"], "span_m": record["span_m"], "envelope": record["envelope"],
        "criteria": record["criteria"], "component_levels": record["component_levels"],
        "modes": {"lateral": _modes(a_lat, ["β", "p", "r", "φ"], "lat", roll),
                  "longitudinal": _modes(a_lon, ["u", "α", "q", "θ"], "lon")},
        "channels": channels, "frequencies": [float(w) for w in FREQUENCIES],
        "checks": {"lateral_vs_roll_loes_max_rel": float(lat_error), "longitudinal_short_period_remainder": lon_error,
                   "phugoid_factor": _poly(quotient), "phugoid_times_short_period": _poly(phugoid)},
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "PingmoSixChannel/1"

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value, status: int = 200) -> None:
        self._send(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=float).encode("utf-8"),
                   "application/json", status)

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        path = url.path
        query = parse_qs(url.query)
        plant_id = query.get("id", [""])[0]
        try:
            if path == "/healthz":
                return self._json({"service": SERVICE, "pid": os.getpid(), "revision": REVISION,
                                   "objects": len(ENTRIES)})
            static = _static_file(PAGES.get(path, path.lstrip("/")))
            if static is not None:
                return self._send(static.read_bytes(), TYPES[static.suffix])
            if path in DOCS:
                return self._send(DOCS[path].read_bytes(), "text/plain")         # shown in the browser, not downloaded
            if path == "/api/summary":
                return self._send(SUMMARY, "application/json")
            if path in ("/api/object", "/api/steps", "/api/maneuvers", "/api/model"):
                if plant_id not in BY_ID:
                    return self._json({"error": "unknown id"}, 404)
                handler = {"/api/object": detail, "/api/steps": steps, "/api/maneuvers": maneuvers,
                           "/api/model": model}[path]
                return self._json(handler(plant_id))
            return self._json({"error": "not found"}, 404)
        except BrokenPipeError:
            return None

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8330)
    args = parser.parse_args()
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
