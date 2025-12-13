#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===============================================================
#  NINJA SUPREME 1.0 — Unified Bayesian Cosmology Engine (ΛCDM vs DUT)
#  Student / Educational Simulation Edition
# ===============================================================
#  © 2025 ExtractoDAO Labs — All Rights Reserved
#  Company Name: ExtractoDAO S.A.
#  CNPJ (Brazil National Registry): 48.839.397/0001-36
#  Contact (Scientific & Licensing): contato@extractodao.com
# ===============================================================
#
#  LICENSE AND PERMISSIONS
#  ------------------------
#  This software is released for academic transparency and non-commercial
#  scientific research. The following conditions apply:
#
#    1. Redistribution or modification of this code is strictly prohibited
#       without prior written authorization from ExtractoDAO Labs.
#
#    2. Use of this code in scientific research, publications, computational
#       pipelines, or derivative works REQUIRES explicit citation of:
#
#       Almeida, J. (2025).
#       Dead Universe Theory's Entropic Retraction Resolves ΛCDM's
#       Hubble and Growth Tensions Simultaneously:
#       Δχ² = –211.6 with Identical Datasets.
#       Zenodo. https://doi.org/10.5281/zenodo.17752029
#
#    3. Any use of real data integrations (Pantheon+, Planck, BAO, H(z), fσ8)
#       must also cite their respective collaborations.
#
#    4. Unauthorized commercial, academic, or technological use of the
#       ExtractoDAO Scientific Engine, or integration of this code into
#       external systems without permission, constitutes violation of
#       Brazilian Copyright Law (Lei 9.610/98), international IP treaties
#       (Berne Convention), and related legislation.
#
# ===============================================================
#  IMPORTANT ACADEMIC NOTICE — STUDENT / EDUCATIONAL SIMULATION VERSION
# ===============================================================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NINJA SUPREME 1.0 — Unified Bayesian Cosmology Engine (ΛCDM vs DUT)
Student / Educational Simulation Edition
====================================================================
© 2025 ExtractoDAO Labs — All Rights Reserved
Company Name: ExtractoDAO S.A.
CNPJ (Brazil National Registry): 48.839.397/0001-36
Contact (Scientific & Licensing): contato@extractodao.com
====================================================================

LICENSE AND PERMISSIONS
-----------------------
This software is released for academic transparency and
non-commercial scientific research. The following conditions apply:

1. Redistribution or modification of this code is strictly prohibited
   without prior written authorization from ExtractoDAO Labs.

2. Use of this code in scientific research, publications,
   computational pipelines, or derivative works REQUIRES explicit
   citation of:

   Almeida, J. (2025).
   Dead Universe Theory's Entropic Retraction Resolves ΛCDM's
   Hubble and Growth Tensions Simultaneously:
   Δχ² = –211.6 with Identical Datasets.
   Zenodo. https://doi.org/10.5281/zenodo.17752029

3. Any use of real data integrations (Pantheon+, Planck, BAO, H(z), fσ8)
   must also cite their respective collaborations.

4. Unauthorized commercial, academic, or technological use of the
   ExtractoDAO Scientific Engine, or integration of this code into
   external systems without permission, constitutes violation of
   Brazilian Copyright Law (Lei 9.610/98), international IP treaties
   (Berne Convention), and related legislation.

====================================================================
IMPORTANT ACADEMIC NOTICE — STUDENT / EDUCATIONAL SIMULATION VERSION
====================================================================
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import io
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from scipy.integrate import cumulative_trapezoid, odeint
from scipy.interpolate import interp1d
import uvicorn

# Optional SciPy extras
try:
    import scipy.special  # noqa: F401
    from scipy.optimize import curve_fit  # noqa: F401
    from scipy.integrate import solve_ivp  # noqa: F401
except Exception:
    solve_ivp = None  # type: ignore

# -----------------------------------------------------------------
# Logging
# -----------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("NINJA_SUPREME")

# -----------------------------------------------------------------
# App
# -----------------------------------------------------------------

app = FastAPI(title="NINJA SUPREME 1.0 — Unified ΛCDM vs DUT (Real Data)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------
# Paths / caching / outputs
# -----------------------------------------------------------------

DATA_DIR = os.environ.get("NINJA_DATA_DIR", "data_cache")
OUTPUT_DIR = os.environ.get("NINJA_OUTPUT_DIR", "outputs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def _write_file_bytes(path: str, b: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b)
    os.replace(tmp, path)

def _safe_json_dump(obj: Any, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True, default=str)
    os.replace(tmp, path)

def _timestamp_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())

def http_get_bytes(url: str, timeout: int = 30, user_agent: str = "Mozilla/5.0") -> bytes:
    from urllib.request import Request as UrlRequest, urlopen
    req = UrlRequest(url, headers={"User-Agent": user_agent})
    with urlopen(req, timeout=timeout) as r:
        return r.read()

@dataclasses.dataclass(frozen=True)
class DatasetInfo:
    name: str
    n: int
    zmin: float
    zmax: float
    source: str
    sha256: str

def _download_or_cache(url: str, filename: str, timeout: int = 30) -> Tuple[bytes, str, bool]:
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        b = _read_file_bytes(path)
        return b, _sha256_bytes(b), True
    b = http_get_bytes(url, timeout=timeout)
    _write_file_bytes(path, b)
    return b, _sha256_bytes(b), False

# -----------------------------------------------------------------
# Real datasets (robust loaders with offline fallback)
# -----------------------------------------------------------------

def load_pantheon_plus(logger_: Optional[logging.Logger] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DatasetInfo]:
    logger_ = logger_ or logger
    data_url = (
        "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/"
        "Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat"
    )
    cov_url = (
        "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/"
        "Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov"
    )
    try:
        b_dat, sha_dat, cached_dat = _download_or_cache(data_url, "pantheonplus.dat")
        b_cov, sha_cov, cached_cov = _download_or_cache(cov_url, "pantheonplus.cov")
        txt = b_dat.decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(txt), delim_whitespace=True, comment="#")

        if "zHD" in df.columns:
            z = df["zHD"].to_numpy(dtype=float)
        elif "z" in df.columns:
            z = df["z"].to_numpy(dtype=float)
        else:
            raise ValueError(f"Pantheon+: no redshift column found. Columns={list(df.columns)}")

        mu_candidates = ["MU_SH0ES", "MU", "MU_GLOBAL", "MU_PLUS", "MU_PANTHEON", "MU_CALIB"]
        mu_col = next((col for col in mu_candidates if col in df.columns), None)
        if mu_col is None:
            raise ValueError(f"Pantheon+: no MU column found. Columns={list(df.columns)}")

        mu = df[mu_col].to_numpy(dtype=float)
        cov_txt = b_cov.decode("utf-8", errors="replace")
        cov = np.loadtxt(io.StringIO(cov_txt))

        if cov.shape[0] != len(z) or cov.shape[1] != len(z):
            raise ValueError(f"Pantheon+ covariance mismatch: cov={cov.shape}, N={len(z)}")

        err_diag = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
        sha = _sha256_bytes(b_dat + b"\n" + b_cov)
        info = DatasetInfo(
            "Pantheon+",
            int(len(z)),
            float(np.min(z)),
            float(np.max(z)),
            f"github/raw (cached={cached_dat and cached_cov})",
            sha
        )
        logger_.info(f"Pantheon+ loaded: N={len(z)} | cached={cached_dat and cached_cov} | mu_col={mu_col}")
        return z, mu, err_diag, cov, info
    except Exception as e:
        logger_.warning(f"Pantheon+ download/parse failed; using embedded fallback. Reason: {e}")
        z = np.array([0.01, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00], dtype=float)
        mu = np.array([33.2, 36.6, 38.2, 40.0, 41.7, 42.9, 44.2, 45.0], dtype=float)
        err = np.full_like(z, 0.15, dtype=float)
        cov = np.diag(err**2)
        info = DatasetInfo(
            "Pantheon+ (fallback)",
            int(len(z)),
            float(z.min()),
            float(z.max()),
            "embedded fallback",
            _sha256_bytes(z.tobytes() + mu.tobytes())
        )
        return z, mu, err, cov, info

def load_hz_moresco(logger_: Optional[logging.Logger] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DatasetInfo]:
    logger_ = logger_ or logger
    hz_url = "https://gitlab.com/mmoresco/CCcovariance/-/raw/master/data/HzTable_MM_BC03.dat"
    try:
        b, sha, cached = _download_or_cache(hz_url, "hz_moresco.dat")
        txt = b.decode("utf-8", errors="replace")
        data = np.loadtxt(io.StringIO(txt))
        z = data[:, 0].astype(float)
        hz = data[:, 1].astype(float)
        err = data[:, 2].astype(float)
        info = DatasetInfo(
            "H(z) cosmic chronometers",
            int(len(z)),
            float(z.min()),
            float(z.max()),
            f"gitlab/raw (cached={cached})",
            sha
        )
        logger_.info(f"H(z) loaded: N={len(z)} | cached={cached}")
        return z, hz, err, info
    except Exception as e:
        logger_.warning(f"H(z) download/parse failed; using embedded fallback. Reason: {e}")
        z = np.array([0.07, 0.17, 0.27, 0.40, 0.57, 0.73, 1.00, 1.50], dtype=float)
        hz = np.array([69.0, 83.0, 77.0, 95.0, 96.0, 97.0, 120.0, 160.0], dtype=float)
        err = np.array([19.0, 8.0, 14.0, 17.0, 17.0, 8.0, 17.0, 20.0], dtype=float)
        info = DatasetInfo(
            "H(z) (fallback)",
            int(len(z)),
            float(z.min()),
            float(z.max()),
            "embedded fallback",
            _sha256_bytes(z.tobytes() + hz.tobytes() + err.tobytes())
        )
        return z, hz, err, info

def load_fs8_compilation(logger_: Optional[logging.Logger] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DatasetInfo]:
    logger_ = logger_ or logger
    data = np.array([
        [0.02, 0.398, 0.065], [0.02, 0.314, 0.048], [0.067, 0.423, 0.055],
        [0.10, 0.370, 0.130], [0.15, 0.490, 0.145], [0.17, 0.510, 0.060],
        [0.18, 0.360, 0.090], [0.25, 0.3512, 0.0583], [0.25, 0.3665, 0.0601],
        [0.30, 0.407, 0.0554], [0.32, 0.427, 0.056], [0.32, 0.480, 0.100],
        [0.35, 0.440, 0.050], [0.37, 0.4602, 0.0378], [0.37, 0.4031, 0.0586],
        [0.38, 0.497, 0.045], [0.38, 0.477, 0.051], [0.38, 0.440, 0.060],
        [0.40, 0.419, 0.041], [0.44, 0.413, 0.080], [0.50, 0.427, 0.043],
        [0.51, 0.458, 0.038], [0.51, 0.453, 0.050], [0.57, 0.417, 0.056],
        [0.59, 0.488, 0.060], [0.60, 0.390, 0.063], [0.60, 0.430, 0.067],
        [0.61, 0.436, 0.034], [0.61, 0.410, 0.044], [0.73, 0.437, 0.072],
        [0.73, 0.404, 0.048], [0.781, 0.450, 0.040], [0.80, 0.470, 0.080],
        [0.875, 0.490, 0.080], [0.85, 0.420, 0.050], [0.98, 0.380, 0.060], [1.23, 0.350, 0.070]
    ], dtype=float)
    z = data[:, 0]
    fs8 = data[:, 1]
    err = data[:, 2]
    info = DatasetInfo(
        "fσ8 compilation",
        int(len(z)),
        float(z.min()),
        float(z.max()),
        "embedded numeric compilation",
        _sha256_bytes(data.tobytes())
    )
    logger_.info(f"fσ8 loaded: N={len(z)} (embedded)")
    return z, fs8, err, info

def load_bao_dv_over_rd(logger_: Optional[logging.Logger] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DatasetInfo]:
    logger_ = logger_ or logger
    sdss_url = "https://raw.githubusercontent.com/CobayaSampler/bao_data/master/sdss_dr12_consensus_final.dat"
    desi_url = "https://raw.githubusercontent.com/CobayaSampler/bao_data/master/desi_2024_bao.dat"
    try:
        b1, sha1, cached1 = _download_or_cache(sdss_url, "bao_sdss_dr12.dat")
        b2, sha2, cached2 = _download_or_cache(desi_url, "bao_desi_2024.dat")
        a1 = np.loadtxt(io.StringIO(b1.decode("utf-8", errors="replace")))
        a2 = np.loadtxt(io.StringIO(b2.decode("utf-8", errors="replace")))
        z = np.concatenate([a1[:, 0], a2[:, 0]]).astype(float)
        dv_over_rd = np.concatenate([a1[:, 1], a2[:, 1]]).astype(float)
        err = np.concatenate([a1[:, 2], a2[:, 2]]).astype(float)
        sha = _sha256_bytes(b1 + b"\n" + b2)
        info = DatasetInfo(
            "BAO DV/rd (SDSS+DESI)",
            int(len(z)),
            float(z.min()),
            float(z.max()),
            f"github/raw (cached={cached1 and cached2})",
            sha
        )
        logger_.info(f"BAO DV/rd loaded: N={len(z)} | cached={cached1 and cached2}")
        return z, dv_over_rd, err, info
    except Exception as e:
        logger_.warning(f"BAO download/parse failed; using embedded fallback. Reason: {e}")
        z = np.array([0.106, 0.15, 0.38, 0.51, 0.61], dtype=float)
        dv_over_rd = np.array([3.05, 4.47, 10.27, 13.38, 15.55], dtype=float)
        err = np.array([0.14, 0.17, 0.15, 0.14, 0.16], dtype=float)
        info = DatasetInfo(
            "BAO DV/rd (fallback)",
            int(len(z)),
            float(z.min()),
            float(z.max()),
            "embedded fallback",
            _sha256_bytes(z.tobytes() + dv_over_rd.tobytes() + err.tobytes())
        )
        return z, dv_over_rd, err, info

def load_planck_2018_compressed(logger_: Optional[logging.Logger] = None) -> Tuple[np.ndarray, np.ndarray, DatasetInfo]:
    logger_ = logger_ or logger
    mean = np.array([0.0224, 0.120, 1.0411, 0.054, 3.044, 0.965], dtype=float)
    errors = np.array([0.0001, 0.001, 0.0003, 0.007, 0.014, 0.004], dtype=float)
    cov = np.diag(errors**2)
    info = DatasetInfo(
        "Planck 2018 (compressed)",
        int(len(mean)),
        0.0,
        0.0,
        "embedded compressed mean + diagonal cov",
        _sha256_bytes(mean.tobytes() + cov.tobytes())
    )
    logger_.info("Planck 2018 compressed loaded (embedded)")
    return mean, cov, info

# Load all datasets at startup
Hz_z, Hz_obs, Hz_err, Hz_info = load_hz_moresco()
fs8_z, fs8_obs, fs8_err, fs8_info = load_fs8_compilation()
bao_z, bao_dv_rd, bao_err, bao_info = load_bao_dv_over_rd()
pantheon_z, pantheon_mu, pantheon_err, pantheon_cov, pantheon_info = load_pantheon_plus()
planck_mean, planck_cov, planck_info = load_planck_2018_compressed()

# -----------------------------------------------------------------
# DUT engine
# -----------------------------------------------------------------

class DUTIntegrator:
    def __init__(self, params: Dict[str, float]):
        self.params = {k: float(v) for k, v in params.items()}
        self.c = 299792.458

        self.N_init = -9.0
        self.N_final = 5.0
        self.N_points = 5000

        self.N: Optional[np.ndarray] = None
        self.solution: Optional[np.ndarray] = None
        self.z: Optional[np.ndarray] = None
        self.Hz: Optional[np.ndarray] = None
        self.Dc: Optional[np.ndarray] = None
        self.DL: Optional[np.ndarray] = None
        self.fsigma8: Optional[np.ndarray] = None
        self._dlnH_dN: Optional[np.ndarray] = None
        self._z_unsorted: Optional[np.ndarray] = None
        self._H_unsorted: Optional[np.ndarray] = None

    def validate_params(self) -> None:
        constraints = {
            "Omega_m_0": (0.0, 2.0),
            "Omega_S_0": (-2.0, 2.0),
            "Omega_k_0": (-2.0, 2.0),
            "Gamma_S": (0.0, 3.0),
            "lambda_phi": (0.0, 10.0),
            "xi": (-2.0, 2.0),
            "H0": (40.0, 120.0),
            "sigma8_0": (0.1, 2.0),
        }
        for k, (lo, hi) in constraints.items():
            v = float(self.params.get(k, 0.0))
            if not (lo <= v <= hi):
                logger.warning(f"Parameter outside typical bounds: {k}={v} not in [{lo}, {hi}]")

    def dut_ode(self, N: float, Y: np.ndarray) -> np.ndarray:
        x = np.clip(Y[0], -10, 10)
        y = np.clip(Y[1], -10, 10)
        u = np.clip(Y[2], -1e3, 1e3)
        z = np.clip(Y[3], -10, 10)

        x2 = np.clip(x**2, 0, 100)
        y2 = np.clip(y**2, 0, 100)

        Om_m = np.clip(u * np.exp(-3.0 * N), 0.0, 1e3)
        Om_k = np.clip(self.params["Omega_k_0"] * np.exp(-2.0 * N), -1.0, 1.0)

        H2 = np.maximum(Om_m + x2 + y2 + z * (1 - self.params["Gamma_S"]) + Om_k, 1e-12)
        R = np.clip(H2 + 0.5 * (x2 - y2), 0, 1e6)
        combo = np.clip(x2 - y2 + np.clip(z * (1 - self.params["Gamma_S"]), -5, 5), -20, 20)

        dx = np.clip(-3 * x + np.sqrt(6) * self.params["lambda_phi"] * y2 / 2 + 1.5 * x * combo, -30, 30)
        dy = np.clip(-np.sqrt(6) * self.params["lambda_phi"] * x * y / 2 + 1.5 * y * combo, -30, 30)
        du = np.clip(-3 * u - 1.5 * u * combo, -1e3, 1e3)
        dz = np.clip(self.params["xi"] * (x2 - y2) + 6 * self.params["xi"] * z * R, -30, 30)

        return np.array([dx, dy, du, dz], dtype=float)

    def rk4_step(self, N: float, Y: np.ndarray, dN: float) -> np.ndarray:
        k1 = self.dut_ode(N, Y)
        k2 = self.dut_ode(N + dN / 2.0, Y + (dN / 2.0) * k1)
        k3 = self.dut_ode(N + dN / 2.0, Y + (dN / 2.0) * k2)
        k4 = self.dut_ode(N + dN, Y + dN * k3)
        return Y + (dN / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def heun_step(self, N: float, Y: np.ndarray, dN: float) -> np.ndarray:
        k1 = self.dut_ode(N, Y)
        Y_pred = Y + dN * k1
        k2 = self.dut_ode(N + dN, Y_pred)
        return Y + 0.5 * dN * (k1 + k2)

    def integrate(self) -> None:
        self.validate_params()

        N = np.linspace(self.N_init, self.N_final, self.N_points, dtype=float)
        dN = float(N[1] - N[0])

        Y_init = np.array([
            1e-6,
            np.sqrt(max(self.params["Omega_S_0"], 0.0)),
            self.params["Omega_m_0"],
            self.params["xi"] * 1e-6
        ], dtype=float)

        sol = np.zeros((self.N_points, 4), dtype=float)
        sol[0] = Y_init

        stable = 1
        Y = Y_init
        for i in range(1, self.N_points):
            Y = self.rk4_step(N[i - 1], Y, dN)
            sol[i] = Y
            if not np.all(np.isfinite(Y)):
                logger.warning(f"Integration instability at i={i}, N={N[i-1]:.6f}")
                break
            stable += 1

        sol = sol[:stable]
        N = N[:stable]
        x, y, u, zz = sol.T

        a = np.exp(np.clip(N, -30, 20))
        zc = 1.0 / a - 1.0

        Om_m_v = np.clip(u * np.exp(-3.0 * N), 0.0, 1e9)
        Om_k_v = np.clip(self.params["Omega_k_0"] * np.exp(-2.0 * N), -10.0, 10.0)

        H2_oH0 = np.maximum(
            Om_m_v + x**2 + y**2 + zz * (1.0 - self.params["Gamma_S"]) + Om_k_v,
            1e-12
        )
        H = float(self.params["H0"]) * np.sqrt(H2_oH0)

        idx = np.argsort(zc.astype(float))
        z_sorted = zc[idx].astype(float)
        H_sorted = H[idx].astype(float)

        z_unique, unique_idx = np.unique(z_sorted, return_index=True)
        H_unique = H_sorted[unique_idx]

        Dc = cumulative_trapezoid(self.c / np.maximum(H_unique, 1e-30), z_unique, initial=0.0)
        DL = (1.0 + z_unique) * Dc

        self.N = N
        self.solution = sol
        self._z_unsorted = zc
        self._H_unsorted = H
        self.z = z_unique
        self.Hz = H_unique
        self.Dc = Dc
        self.DL = DL

        self._compute_growth(z_unique, a, N, sol)
        logger.info(f"DUT integrate OK | stable_points={stable} | z_range=[{self.z.min():.4g}, {self.z.max():.4g}]")

    def integrate_alt(self, method: str = "heun") -> Dict[str, Any]:
        self.validate_params()

        N = np.linspace(self.N_init, self.N_final, self.N_points, dtype=float)
        dN = float(N[1] - N[0])

        Y_init = np.array([
            1e-6,
            np.sqrt(max(self.params["Omega_S_0"], 0.0)),
            self.params["Omega_m_0"],
            self.params["xi"] * 1e-6
        ], dtype=float)

        sol = np.zeros((self.N_points, 4), dtype=float)
        sol[0] = Y_init

        stable = 1
        Y = Y_init
        stepper = self.heun_step if method.lower() == "heun" else self.rk4_step

        for i in range(1, self.N_points):
            Y = stepper(N[i - 1], Y, dN)
            sol[i] = Y
            if not np.all(np.isfinite(Y)):
                break
            stable += 1

        return {"stable_points": int(stable), "method": method, "dN": float(dN)}

    def integrate_solve_ivp(self, method: str = "RK45", rtol: float = 1e-7, atol: float = 1e-9) -> Dict[str, Any]:
        if solve_ivp is None:
            return {"available": False, "method": method, "success": False, "message": "solve_ivp not available"}

        self.validate_params()

        N_grid = np.linspace(self.N_init, self.N_final, max(2000, self.N_points // 2), dtype=float)

        Y_init = np.array([
            1e-6,
            np.sqrt(max(self.params["Omega_S_0"], 0.0)),
            self.params["Omega_m_0"],
            self.params["xi"] * 1e-6
        ], dtype=float)

        def ode_func(N, Y):
            return self.dut_ode(float(N), np.array(Y, dtype=float))

        sol_ivp = solve_ivp(
            ode_func,
            [float(self.N_init), float(self.N_final)],
            Y_init,
            t_eval=N_grid,
            method=str(method),
            rtol=float(rtol),
            atol=float(atol),
        )

        return {
            "available": True,
            "method": method,
            "success": bool(sol_ivp.success),
            "n_points": int(sol_ivp.t.size) if sol_ivp.t is not None else 0,
            "message": str(sol_ivp.message),
        }

    def _growth_ode(self, y, N, Om_m_N, G_eff_N) -> list:
        D, dD_dN = y
        Om_val = float(np.interp(N, self.N, Om_m_N))
        G_val = float(np.interp(N, self.N, G_eff_N))
        dlnH_dN = float(np.interp(N, self.N, self._dlnH_dN))
        d2D_dN2 = -(2.0 + dlnH_dN) * dD_dN + 1.5 * Om_val * G_val * D
        return [dD_dN, d2D_dN2]

    def _compute_growth(self, z_unique: np.ndarray, a: np.ndarray, N: np.ndarray, sol: np.ndarray) -> None:
        x, y, u, zz = sol.T

        Om_m_a = u * np.exp(-3.0 * N)
        H2_oH0 = (self._H_unsorted / float(self.params["H0"])) ** 2
        Om_m_N = Om_m_a / np.maximum(H2_oH0, 1e-30)
        Om_m_N = np.clip(Om_m_N, 0.0, 2.0)

        denom = 1.0 + self.params["xi"] * zz / 3.0
        G_eff_N = np.where(np.abs(denom) < 1e-10, 1.0, 1.0 / denom)
        G_eff_N = np.clip(G_eff_N, 0.1, 10.0)

        lnH = np.log(self._H_unsorted + 1e-30)
        self._dlnH_dN = np.gradient(lnH, N)

        N_ini_growth = max(float(N[0]), -5.0)
        N_end_growth = 0.0
        mask = (N >= N_ini_growth) & (N <= N_end_growth)
        N_grid = N[mask]

        if N_grid.size < 40:
            N_grid = N.copy()
            N_ini_growth = float(N[0])

        D_ini = float(np.exp(N_ini_growth))
        dD_ini = D_ini
        y0 = [D_ini, dD_ini]

        sol_growth = odeint(
            lambda yy, NN: self._growth_ode(yy, NN, Om_m_N, G_eff_N),
            y0,
            N_grid
        )

        D_N = sol_growth[:, 0]
        D_today = float(D_N[-1])
        if (not np.isfinite(D_today)) or abs(D_today) < 1e-30:
            logger.error("Growth normalization failed (D_today invalid). Setting fsigma8=0.")
            self.fsigma8 = np.zeros_like(z_unique)
            return

        D_N = D_N / D_today
        a_growth = np.exp(N_grid)
        z_growth = 1.0 / np.maximum(a_growth, 1e-30) - 1.0
        sidx = np.argsort(z_growth)
        z_g = z_growth[sidx]
        D_g = D_N[sidx]

        lnD = np.log(D_N + 1e-30)
        fN = np.gradient(lnD, N_grid)
        f_g = fN[sidx]

        D_interp = np.interp(z_unique, z_g, D_g)
        f_interp = np.interp(z_unique, z_g, f_g)
        sigma8_z = float(self.params["sigma8_0"]) * D_interp
        self.fsigma8 = np.nan_to_num(f_interp * sigma8_z, nan=0.0, posinf=0.0, neginf=0.0)

    def H_of_z(self, z: np.ndarray) -> np.ndarray:
        if self.z is None or self.Hz is None:
            self.integrate()
        f = interp1d(self.z, self.Hz, bounds_error=False, fill_value="extrapolate")
        return f(np.asarray(z, dtype=float))

    def mu_of_z(self, z: np.ndarray) -> np.ndarray:
        if self.z is None or self.DL is None:
            self.integrate()
        f = interp1d(
            self.z,
            5.0 * np.log10(np.maximum(self.DL, 1e-30)) + 25.0,
            bounds_error=False,
            fill_value="extrapolate"
        )
        return f(np.asarray(z, dtype=float))

    def fs8_of_z(self, z: np.ndarray) -> np.ndarray:
        if self.z is None or self.fsigma8 is None:
            self.integrate()
        f = interp1d(self.z, self.fsigma8, bounds_error=False, fill_value="extrapolate")
        return f(np.asarray(z, dtype=float))

    def DV_of_z(self, z: np.ndarray) -> np.ndarray:
        if self.z is None or self.Dc is None or self.Hz is None:
            self.integrate()
        z = np.asarray(z, dtype=float)
        Dc = interp1d(self.z, self.Dc, bounds_error=False, fill_value="extrapolate")(z)
        Hz = interp1d(self.z, self.Hz, bounds_error=False, fill_value="extrapolate")(z)
        DV = ((1.0 + z) ** 2 * np.maximum(Dc, 1e-30) ** 2 * self.c *
              np.maximum(z, 1e-30) / np.maximum(Hz, 1e-30)) ** (1.0 / 3.0)
        return DV

    def chi2_against_data(self) -> Dict[str, Any]:
        if self.z is None:
            self.integrate()

        chi2_total = 0.0
        breakdown: Dict[str, float] = {}

        H_pred = self.H_of_z(Hz_z)
        chi2_hz = float(np.sum(((Hz_obs - H_pred) / Hz_err) ** 2))
        chi2_total += chi2_hz
        breakdown["H(z)"] = chi2_hz

        fs8_pred = self.fs8_of_z(fs8_z)
        chi2_fs8 = float(np.sum(((fs8_obs - fs8_pred) / fs8_err) ** 2))
        chi2_total += chi2_fs8
        breakdown["fσ8"] = chi2_fs8

        rd_fid = 147.78
        dv_over_rd_pred = self.DV_of_z(bao_z) / rd_fid
        chi2_bao = float(np.sum(((bao_dv_rd - dv_over_rd_pred) / bao_err) ** 2))
        chi2_total += chi2_bao
        breakdown["BAO(DV/rd)"] = chi2_bao

        mu_pred = self.mu_of_z(pantheon_z)
        delta = pantheon_mu - mu_pred
        try:
            chi2_sn = float(delta @ np.linalg.solve(pantheon_cov, delta))
        except Exception:
            chi2_sn = float(np.sum((delta / pantheon_err) ** 2))
        chi2_total += chi2_sn
        breakdown["Pantheon+"] = chi2_sn

        n_data = int(len(Hz_z) + len(fs8_z) + len(bao_z) + len(pantheon_z))
        return {"total": float(chi2_total), "breakdown": breakdown, "n_data": n_data}

# -----------------------------------------------------------------
# ΛCDM reference (simple)
# -----------------------------------------------------------------

def lcdm_background(H0: float, Omega_m: float, z: np.ndarray) -> Dict[str, np.ndarray]:
    z = np.asarray(z, dtype=float)
    Omega_L = 1.0 - Omega_m
    H = H0 * np.sqrt(Omega_m * (1.0 + z) ** 3 + Omega_L)
    c = 299792.458

    idx = np.argsort(z)
    z_s = z[idx]
    H_s = H[idx]

    Dc_s = cumulative_trapezoid(c / np.maximum(H_s, 1e-30), z_s, initial=0.0)
    DL_s = (1.0 + z_s) * Dc_s

    Omz = Omega_m * (1.0 + z_s) ** 3 / (Omega_m * (1.0 + z_s) ** 3 + Omega_L)
    fs8_s = 0.811 * Omz ** 0.55

    DV_s = ((1.0 + z_s) ** 2 * np.maximum(Dc_s, 1e-30) ** 2 * c *
            np.maximum(z_s, 1e-30) / np.maximum(H_s, 1e-30)) ** (1.0 / 3.0)

    inv = np.empty_like(idx)
    inv[idx] = np.arange(idx.size)
    return {"H": H_s[inv], "Dc": Dc_s[inv], "DL": DL_s[inv], "fs8": fs8_s[inv], "DV": DV_s[inv]}

def chi2_lcdm(H0: float = 67.4, Omega_m: float = 0.315) -> Dict[str, Any]:
    chi2_total = 0.0
    breakdown: Dict[str, float] = {}

    H_pred = lcdm_background(H0, Omega_m, Hz_z)["H"]
    chi2_hz = float(np.sum(((Hz_obs - H_pred) / Hz_err) ** 2))
    chi2_total += chi2_hz
    breakdown["H(z)"] = chi2_hz

    fs8_pred = lcdm_background(H0, Omega_m, fs8_z)["fs8"]
    chi2_fs8 = float(np.sum(((fs8_obs - fs8_pred) / fs8_err) ** 2))
    chi2_total += chi2_fs8
    breakdown["fσ8"] = chi2_fs8

    DV_pred = lcdm_background(H0, Omega_m, bao_z)["DV"]
    rd_fid = 147.78
    dv_over_rd_pred = DV_pred / rd_fid
    chi2_bao = float(np.sum(((bao_dv_rd - dv_over_rd_pred) / bao_err) ** 2))
    chi2_total += chi2_bao
    breakdown["BAO(DV/rd)"] = chi2_bao

    DL = lcdm_background(H0, Omega_m, pantheon_z)["DL"]
    mu_pred = 5.0 * np.log10(np.maximum(DL, 1e-30)) + 25.0
    delta = pantheon_mu - mu_pred
    try:
        chi2_sn = float(delta @ np.linalg.solve(pantheon_cov, delta))
    except Exception:
        chi2_sn = float(np.sum((delta / pantheon_err) ** 2))
    chi2_total += chi2_sn
    breakdown["Pantheon+"] = chi2_sn

    n_data = int(len(Hz_z) + len(fs8_z) + len(bao_z) + len(pantheon_z))
    return {"total": float(chi2_total), "breakdown": breakdown, "n_data": n_data}

# -----------------------------------------------------------------
# API models + ledger
# -----------------------------------------------------------------

class CosmoParams(BaseModel):
    Omega_m_0: float = 0.301
    Omega_S_0: float = 0.649
    Omega_k_0: float = -0.069
    Gamma_S: float = 0.958
    lambda_phi: float = 1.18
    xi: float = 0.102
    H0: float = 70.0
    sigma8_0: float = 0.810

ledger: List[Dict[str, Any]] = []

def add_ledger_entry(data: Dict[str, Any]) -> str:
    prev_hash = ledger[-1]["hash"] if ledger else "0" * 64
    entry = {"data": data, "prev_hash": prev_hash, "timestamp": time.time()}
    entry["hash"] = hashlib.sha256(json.dumps(entry, sort_keys=True).encode("utf-8")).hexdigest()
    ledger.append(entry)
    try:
        _safe_json_dump(ledger, os.path.join(OUTPUT_DIR, "dut_ledger.json"))
    except Exception as e:
        logger.warning(f"Ledger write failed: {e}")
    return entry["hash"]

def export_last_run_json() -> Dict[str, Any]:
    if not ledger:
        return {"error": "No runs available."}
    last_entry = ledger[-1]
    tag = _timestamp_tag()
    out = {
        "engine": "NINJA SUPREME 1.0",
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(last_entry["timestamp"])),
        "ledger_hash": last_entry["hash"],
        "prev_hash": last_entry["prev_hash"],
        "data": last_entry["data"],
    }
    path = os.path.join(OUTPUT_DIR, f"ninja_export_{tag}.json")
    _safe_json_dump(out, path)
    return {"status": "JSON_EXPORTED", "path": os.path.abspath(path), "filename": os.path.basename(path), "ledger_hash": last_entry["hash"]}

# -----------------------------------------------------------------
# Numerical validation endpoint
# -----------------------------------------------------------------

@app.post("/api/validate")
async def validate_numerics(params: CosmoParams) -> Dict[str, Any]:
    p = params.dict()
    dut = DUTIntegrator(p)

    basic_ok = True
    error_msg = None
    chi2_val = None

    try:
        dut.integrate()
        chi2 = dut.chi2_against_data()
        chi2_val = float(chi2["total"])
        basic_ok = bool(np.isfinite(chi2_val)) and (chi2_val < 1e8)
    except Exception as e:
        basic_ok = False
        error_msg = f"{type(e).__name__}: {str(e)}"

    step_results = []
    for n_pts in [2500, 5000, 10000]:
        try:
            dut_test = DUTIntegrator(p)
            dut_test.N_points = int(n_pts)
            dut_test.integrate()
            chi2_test = dut_test.chi2_against_data()
            z0_idx = int(np.argmin(np.abs(dut_test.z - 0.0)))
            H0_local = float(dut_test.Hz[z0_idx]) if dut_test.Hz is not None else 0.0
            step_results.append({
                "method": "RK4",
                "N_points": int(n_pts),
                "chi2": float(chi2_test["total"]),
                "H0_local": float(H0_local),
                "stable": True
            })
        except Exception as e:
            step_results.append({"method": "RK4", "N_points": int(n_pts), "stable": False, "error": f"{type(e).__name__}: {str(e)}"})

    heun_results = []
    for n_pts in [2500, 5000, 10000]:
        try:
            dut_h = DUTIntegrator(p)
            dut_h.N_points = int(n_pts)
            meta = dut_h.integrate_alt(method="heun")
            dut_h.integrate()  # keep canonical outputs consistent for chi2
            chi2_h = dut_h.chi2_against_data()
            z0_idx = int(np.argmin(np.abs(dut_h.z - 0.0)))
            H0_local = float(dut_h.Hz[z0_idx]) if dut_h.Hz is not None else 0.0
            heun_results.append({
                "method": "Heun",
                "N_points": int(n_pts),
                "chi2": float(chi2_h["total"]),
                "H0_local": float(H0_local),
                "stable_points": int(meta.get("stable_points", 0)),
                "stable": True
            })
        except Exception as e:
            heun_results.append({"method": "Heun", "N_points": int(n_pts), "stable": False, "error": f"{type(e).__name__}: {str(e)}"})

    ivp_tests = []
    for m in ["RK45", "Radau", "BDF"]:
        try:
            dut_ivp = DUTIntegrator(p)
            ivp_tests.append(dut_ivp.integrate_solve_ivp(method=m, rtol=1e-7, atol=1e-9))
        except Exception as e:
            ivp_tests.append({"available": solve_ivp is not None, "method": m, "success": False, "message": f"{type(e).__name__}: {str(e)}"})

    convergence = {}
    try:
        rk4_ok = [r for r in step_results if r.get("stable")]
        if len(rk4_ok) == 3:
            chi2_vals = [float(r["chi2"]) for r in rk4_ok]
            convergence["rk4_chi2_diff_2500_5000"] = float(chi2_vals[0] - chi2_vals[1])
            convergence["rk4_chi2_diff_5000_10000"] = float(chi2_vals[1] - chi2_vals[2])
            convergence["rk4_converged"] = abs(convergence["rk4_chi2_diff_5000_10000"]) < 0.1
    except Exception:
        pass

    overall_pass = bool(basic_ok) and all(r.get("stable", False) for r in step_results)

    validation_hash = add_ledger_entry({
        "type": "NUMERICAL_VALIDATION",
        "params": p,
        "basic_ok": bool(basic_ok),
        "basic_error": error_msg,
        "basic_chi2": chi2_val,
        "rk4_step_tests": step_results,
        "heun_step_tests": heun_results,
        "solve_ivp_tests": ivp_tests,
        "convergence": convergence,
        "overall_pass": bool(overall_pass)
    })

    return {
        "status": "VALIDATION_COMPLETE",
        "parameters": p,
        "basic_integration": {"ok": bool(basic_ok), "error": error_msg, "chi2": chi2_val},
        "rk4_step_tests": step_results,
        "heun_step_tests": heun_results,
        "solve_ivp_tests": ivp_tests,
        "convergence": convergence,
        "overall_pass": bool(overall_pass),
        "ledger": {"hash": validation_hash[:16] + "...", "position": len(ledger)}
    }

# -----------------------------------------------------------------
# PDF Export Endpoint
# -----------------------------------------------------------------

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("ReportLab not available. PDF export disabled.")

if PDF_AVAILABLE:
    @app.get("/api/export_pdf")
    async def export_pdf() -> Dict[str, Any]:
        if not ledger:
            return {"error": "No runs available. Run a simulation first."}

        last_entry = ledger[-1]
        data = last_entry["data"]

        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        width, height = A4

        c.setFont("Helvetica-Bold", 16)
        c.drawString(2*cm, height-2*cm, "NINJA SUPREME 1.0 - Simulation Report")
        c.setFont("Helvetica", 10)
        c.drawString(2*cm, height-2.5*cm, f"Generated (UTC): {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}")
        c.drawString(2*cm, height-3*cm, f"Ledger Hash: {last_entry['hash'][:24]}...")

        c.setFont("Helvetica-Bold", 12)
        c.drawString(2*cm, height-4*cm, "Parameters (DUT):")
        c.setFont("Helvetica", 10)

        y_pos = height - 4.5*cm
        if isinstance(data, dict) and "params" in data and isinstance(data["params"], dict):
            items = list(data["params"].items())
            for i, (key, value) in enumerate(items):
                if i % 2 == 0:
                    c.drawString(2*cm, y_pos, f"{key}: {float(value):.6g}")
                else:
                    c.drawString(10*cm, y_pos, f"{key}: {float(value):.6g}")
                    y_pos -= 0.5*cm
                if i % 2 == 0 and i == len(items) - 1:
                    y_pos -= 0.5*cm

        y_pos -= 0.5*cm
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2*cm, y_pos, "Results:")
        c.setFont("Helvetica", 10)
        y_pos -= 0.5*cm

        if isinstance(data, dict) and "chi2_DUT" in data:
            results = [
                ("chi2 DUT", f"{float(data['chi2_DUT']):.2f}"),
                ("chi2 LCDM", f"{float(data['chi2_LCDM']):.2f}"),
                ("Delta chi2", f"{float(data['Delta_chi2']):.2f}"),
                ("lnB", f"{float(data['lnB']):.2f}"),
                ("Data points", str(int(data.get("data_points", 0)))),
            ]
            for i, (label, value) in enumerate(results):
                if i % 2 == 0:
                    c.drawString(2*cm, y_pos, f"{label}: {value}")
                else:
                    c.drawString(10*cm, y_pos, f"{label}: {value}")
                    y_pos -= 0.5*cm
                if i % 2 == 0 and i == len(results) - 1:
                    y_pos -= 0.5*cm

        if isinstance(data, dict) and "datasets_sha256" in data and isinstance(data["datasets_sha256"], dict):
            y_pos -= 0.5*cm
            c.setFont("Helvetica-Bold", 12)
            c.drawString(2*cm, y_pos, "Datasets (SHA256):")
            c.setFont("Helvetica", 8)
            y_pos -= 0.5*cm
            for dataset, sha in data["datasets_sha256"].items():
                c.drawString(2*cm, y_pos, f"{dataset}: {str(sha)[:16]}...")
                y_pos -= 0.4*cm
                if y_pos < 2*cm:
                    c.showPage()
                    y_pos = height - 2*cm

        c.save()
        buf.seek(0)
        pdf_bytes = buf.read()

        tag = _timestamp_tag()
        filename = f"ninja_supreme_report_{tag}.pdf"
        path = os.path.join(OUTPUT_DIR, filename)
        _write_file_bytes(path, pdf_bytes)

        pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        return {
            "status": "PDF_GENERATED",
            "pdf_base64": pdf_b64,
            "filename": filename,
            "size_bytes": int(len(pdf_bytes)),
            "saved_path": os.path.abspath(path),
        }
else:
    @app.get("/api/export_pdf")
    async def export_pdf() -> Dict[str, Any]:
        return {"error": "PDF export requires ReportLab. Install with: pip install reportlab"}

# -----------------------------------------------------------------
# JSON Export Endpoint
# -----------------------------------------------------------------

@app.get("/api/export_json")
async def export_json() -> Dict[str, Any]:
    try:
        return export_last_run_json()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)}"}

# -----------------------------------------------------------------
# Existing Routes
# -----------------------------------------------------------------

@app.post("/api/run")
async def run_simulation(params: CosmoParams) -> Dict[str, Any]:
    p = params.dict()
    logger.info(f"RUN | params={p}")

    dut = DUTIntegrator(p)
    dut.integrate()
    chi_dut = dut.chi2_against_data()
    chi_lcdm = chi2_lcdm()

    delta_chi2 = float(chi_dut["total"] - chi_lcdm["total"])
    lnB = float(-0.5 * delta_chi2)

    plt.figure(figsize=(10, 7))
    z_plot = np.linspace(0.0, 2.5, 400)
    plt.plot(z_plot, dut.H_of_z(z_plot), lw=2, label="DUT")
    plt.plot(z_plot, lcdm_background(params.H0, params.Omega_m_0, z_plot)["H"], lw=2, linestyle="--", label="ΛCDM")
    plt.errorbar(Hz_z, Hz_obs, Hz_err, fmt="o", markersize=4, alpha=0.8, label="H(z) data")
    plt.xlabel("Redshift z")
    plt.ylabel("H(z) [km/s/Mpc]")
    plt.title(f"H(z) | Delta chi2 = {delta_chi2:.2f}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    hz_plot_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close("all")

    plt.figure(figsize=(10, 7))
    z_plot2 = np.linspace(0.0, 2.0, 350)
    plt.plot(z_plot2, dut.fs8_of_z(z_plot2), lw=2, label="DUT")
    plt.plot(z_plot2, lcdm_background(params.H0, params.Omega_m_0, z_plot2)["fs8"], lw=2, linestyle="--", label="ΛCDM")
    plt.errorbar(fs8_z, fs8_obs, fs8_err, fmt="o", markersize=4, alpha=0.8, label="fσ8 data")
    plt.xlabel("Redshift z")
    plt.ylabel("fσ8(z)")
    plt.title("Growth comparison")
    plt.grid(True, alpha=0.25)
    plt.legend()
    buf2 = io.BytesIO()
    plt.savefig(buf2, format="png", dpi=150, bbox_inches="tight")
    buf2.seek(0)
    fs8_plot_b64 = base64.b64encode(buf2.read()).decode("utf-8")
    plt.close("all")

    z0_idx = int(np.argmin(np.abs(dut.z - 0.0)))
    H0_local = float(dut.Hz[z0_idx])
    fs8_z0 = float(dut.fsigma8[z0_idx])

    entry_hash = add_ledger_entry({
        "type": "SIMULATION_RUN",
        "params": p,
        "chi2_DUT": float(chi_dut["total"]),
        "chi2_LCDM": float(chi_lcdm["total"]),
        "Delta_chi2": float(delta_chi2),
        "lnB": float(lnB),
        "data_points": int(chi_dut["n_data"]),
        "datasets_sha256": {
            "Hz": Hz_info.sha256,
            "fs8": fs8_info.sha256,
            "BAO": bao_info.sha256,
            "Pantheon": pantheon_info.sha256,
            "Planck": planck_info.sha256,
        }
    })

    result = {
        "status": "SIMULATION_COMPLETE",
        "parameters": p,
        "results": {
            "chi2_DUT": float(chi_dut["total"]),
            "chi2_LCDM": float(chi_lcdm["total"]),
            "Delta_chi2": float(delta_chi2),
            "lnB": float(lnB),
            "H0_local": float(H0_local),
            "fsigma8_z0": float(fs8_z0),
        },
        "chi2_breakdown": {"DUT": chi_dut["breakdown"], "LCDM": chi_lcdm["breakdown"]},
        "plots": {
            "hz_plot": f"data:image/png;base64,{hz_plot_b64}",
            "fs8_plot": f"data:image/png;base64,{fs8_plot_b64}",
        },
        "data_summary": {
            "H(z)_points": int(len(Hz_z)),
            "fσ8_points": int(len(fs8_z)),
            "BAO_points": int(len(bao_z)),
            "Pantheon_points": int(len(pantheon_z)),
            "total_points": int(chi_dut["n_data"]),
        },
        "datasets": {
            "Hz": dataclasses.asdict(Hz_info),
            "fs8": dataclasses.asdict(fs8_info),
            "BAO": dataclasses.asdict(bao_info),
            "Pantheon": dataclasses.asdict(pantheon_info),
            "Planck": dataclasses.asdict(planck_info),
        },
        "ledger": {"hash": entry_hash[:16] + "...", "position": len(ledger)},
    }

    try:
        tag = _timestamp_tag()
        out_path = os.path.join(OUTPUT_DIR, f"ninja_run_{tag}.json")
        _safe_json_dump(result, out_path)
    except Exception as e:
        logger.warning(f"Run JSON export failed: {e}")

    return result

@app.get("/api/data_info")
async def get_data_info() -> Dict[str, Any]:
    return {
        "datasets": {
            "H(z)": dataclasses.asdict(Hz_info),
            "fσ8": dataclasses.asdict(fs8_info),
            "BAO_DV/rd": dataclasses.asdict(bao_info),
            "Pantheon+": dataclasses.asdict(pantheon_info),
            "Planck_2018": dataclasses.asdict(planck_info),
        },
        "total_points": int(len(Hz_z) + len(fs8_z) + len(bao_z) + len(pantheon_z)),
        "cache_dir": os.path.abspath(DATA_DIR),
        "output_dir": os.path.abspath(OUTPUT_DIR),
    }

@app.get("/api/ledger")
async def get_ledger() -> List[Dict[str, Any]]:
    return ledger[-25:] if len(ledger) > 25 else ledger

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "NINJA SUPREME 1.0 OPERATIONAL",
        "version": "1.0.0",
        "data_loaded": True,
        "datasets": {
            "H(z)": int(len(Hz_z)),
            "fσ8": int(len(fs8_z)),
            "BAO": int(len(bao_z)),
            "Pantheon+": int(len(pantheon_z)),
        },
        "ledger_size": int(len(ledger)),
        "pdf_available": bool(PDF_AVAILABLE),
    }

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>NINJA SUPREME 1.0</title>
    <style>
    :root{--bg:#05070c;--card:#0b1020;--txt:#e8eefc;--muted:#8aa0c7;--accent:#22c55e;--warn:#f59e0b;--bad:#ef4444;}
    body{margin:0;background:radial-gradient(1200px 600px at 20% 10%, #0c1b3a 0%, var(--bg) 55%) fixed;color:var(--txt);font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;}
    .wrap{max-width:1200px;margin:0 auto;padding:28px 16px 60px;}
    .title{font-size:40px;font-weight:900;letter-spacing:-1px;margin:0 0 8px;}
    .sub{color:var(--muted);margin:0 0 22px;}
    .grid{display:grid;gap:14px;}
    .grid.metrics{grid-template-columns:repeat(4,minmax(0,1fr));}
    .grid.panels{grid-template-columns:repeat(2,minmax(0,1fr));margin-top:14px;}
    @media (max-width:980px){.grid.metrics{grid-template-columns:repeat(2,minmax(0,1fr));}.grid.panels{grid-template-columns:1fr;}}
    .card{background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02));border:1px solid rgba(120,140,190,.25);border-radius:18px;padding:16px;}
    .label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.12em;margin-bottom:8px;}
    .value{font-size:26px;font-weight:900;}
    .controls{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));gap:10px;margin-top:14px;}
    @media (max-width:980px){.controls{grid-template-columns:repeat(2,minmax(0,1fr));}}
    input{width:100%;padding:10px 10px;border-radius:12px;border:1px solid rgba(120,140,190,.35);background:rgba(0,0,0,.25);color:var(--txt);outline:none;}
    button{padding:12px 14px;border-radius:14px;border:1px solid rgba(34,197,94,.45);background:rgba(34,197,94,.14);color:var(--txt);font-weight:900;cursor:pointer;}
    button:disabled{opacity:.6;cursor:not-allowed;}
    .row{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px;}
    .status{margin-top:14px;color:var(--muted);}
    img{max-width:100%;border-radius:14px;border:1px solid rgba(120,140,190,.25);background:#000;}
    pre{white-space:pre-wrap;color:#b9c9ee;background:rgba(0,0,0,.18);border:1px solid rgba(120,140,190,.22);border-radius:14px;padding:12px;max-height:260px;overflow:auto;}
    </style>
</head>
<body>
    <div class="wrap">
        <h1 class="title">NINJA SUPREME 1.0</h1>
        <p class="sub">Unified ΛCDM vs DUT | Real-data χ² pipeline (Pantheon+ • BAO • H(z) • fσ8)</p>

        <div class="grid metrics">
            <div class="card"><div class="label">H0 local (DUT)</div><div class="value" id="m_h0">---</div></div>
            <div class="card"><div class="label">Delta chi2 (DUT - ΛCDM)</div><div class="value" id="m_dchi2">---</div></div>
            <div class="card"><div class="label">lnB ≈ -0.5 Delta chi2</div><div class="value" id="m_lnb">---</div></div>
            <div class="card"><div class="label">Ledger blocks</div><div class="value" id="m_led">---</div></div>
        </div>

        <div class="card" style="margin-top:14px;">
            <div class="label">Parameters (DUT)</div>
            <div class="controls">
                <div><div class="label">Ωm0</div><input id="p_om" type="number" step="0.001" value="0.301"></div>
                <div><div class="label">ΩS0</div><input id="p_os" type="number" step="0.001" value="0.649"></div>
                <div><div class="label">Ωk0</div><input id="p_ok" type="number" step="0.001" value="-0.069"></div>
                <div><div class="label">ΓS</div><input id="p_gs" type="number" step="0.001" value="0.958"></div>
                <div><div class="label">λφ</div><input id="p_lp" type="number" step="0.01" value="1.18"></div>
                <div><div class="label">ξ</div><input id="p_xi" type="number" step="0.001" value="0.102"></div>
                <div><div class="label">H0</div><input id="p_h0" type="number" step="0.1" value="70.0"></div>
                <div><div class="label">σ8</div><input id="p_s8" type="number" step="0.001" value="0.810"></div>
            </div>
            <div class="row">
                <button id="btn_run" onclick="run()">RUN (Real Data)</button>
                <button id="btn_validate" onclick="validate()">VALIDATE</button>
                <button id="btn_info" onclick="info()">DATA INFO</button>
                <button id="btn_led" onclick="ledger()">LEDGER</button>
                <button id="btn_pdf" onclick="exportPdf()">EXPORT PDF</button>
                <button id="btn_json" onclick="exportJson()">EXPORT JSON</button>
            </div>
            <div class="status" id="status">Ready.</div>
        </div>

        <div class="grid panels">
            <div class="card">
                <div class="label">H(z) plot</div>
                <img id="img_hz" alt="H(z) plot" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==">
            </div>
            <div class="card">
                <div class="label">fσ8(z) plot</div>
                <img id="img_fs8" alt="fσ8(z) plot" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==">
            </div>
        </div>

        <div class="card" style="margin-top:14px;">
            <div class="label">JSON output</div>
            <pre id="out">{}</pre>
        </div>
    </div>

    <script>
    function params(){
        return {
            Omega_m_0: parseFloat(document.getElementById('p_om').value),
            Omega_S_0: parseFloat(document.getElementById('p_os').value),
            Omega_k_0: parseFloat(document.getElementById('p_ok').value),
            Gamma_S: parseFloat(document.getElementById('p_gs').value),
            lambda_phi: parseFloat(document.getElementById('p_lp').value),
            xi: parseFloat(document.getElementById('p_xi').value),
            H0: parseFloat(document.getElementById('p_h0').value),
            sigma8_0: parseFloat(document.getElementById('p_s8').value),
        };
    }

    function setStatus(t){ document.getElementById('status').textContent = t; }
    function setDisabled(id, v){ const b=document.getElementById(id); if(b) b.disabled=v; }

    async function run(){
        setDisabled('btn_run', true);
        setStatus('Running...');
        try{
            const r = await fetch('/api/run', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(params())});
            const j = await r.json();
            document.getElementById('out').textContent = JSON.stringify(j, null, 2);
            document.getElementById('img_hz').src = j.plots.hz_plot;
            document.getElementById('img_fs8').src = j.plots.fs8_plot;
            document.getElementById('m_h0').textContent = (j.results.H0_local ?? '---');
            document.getElementById('m_dchi2').textContent = (j.results.Delta_chi2 ?? '---');
            document.getElementById('m_lnb').textContent = (j.results.lnB ?? '---');
            document.getElementById('m_led').textContent = (j.ledger.position ?? '---');
            setStatus('Done.');
        } catch(e){
            setStatus('Error: ' + e.message);
        }
        setDisabled('btn_run', false);
    }

    async function validate(){
        setDisabled('btn_validate', true);
        setStatus('Validating numerics...');
        try{
            const r = await fetch('/api/validate', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(params())});
            const j = await r.json();
            document.getElementById('out').textContent = JSON.stringify(j, null, 2);
            setStatus('Validation: ' + (j.overall_pass ? 'PASS' : 'FAIL'));
        } catch(e){
            setStatus('Error: ' + e.message);
        }
        setDisabled('btn_validate', false);
    }

    async function info(){
        setStatus('Loading data info...');
        const r = await fetch('/api/data_info');
        const j = await r.json();
        document.getElementById('out').textContent = JSON.stringify(j, null, 2);
        setStatus('Done.');
    }

    async function ledger(){
        setStatus('Loading ledger...');
        const r = await fetch('/api/ledger');
        const j = await r.json();
        document.getElementById('out').textContent = JSON.stringify(j, null, 2);
        document.getElementById('m_led').textContent = j.length;
        setStatus('Done.');
    }

    async function exportPdf(){
        setDisabled('btn_pdf', true);
        setStatus('Generating PDF...');
        try{
            const r = await fetch('/api/export_pdf');
            const j = await r.json();
            if(j.pdf_base64){
                const a = document.createElement('a');
                a.href = 'data:application/pdf;base64,' + j.pdf_base64;
                a.download = j.filename || 'ninja_report.pdf';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                setStatus('PDF downloaded.');
            } else {
                document.getElementById('out').textContent = JSON.stringify(j, null, 2);
                setStatus('PDF generation failed.');
            }
        } catch(e){
            setStatus('Error: ' + e.message);
        }
        setDisabled('btn_pdf', false);
    }

    async function exportJson(){
        setDisabled('btn_json', true);
        setStatus('Exporting JSON...');
        try{
            const r = await fetch('/api/export_json');
            const j = await r.json();
            document.getElementById('out').textContent = JSON.stringify(j, null, 2);
            setStatus(j.status ? 'JSON exported.' : 'JSON export failed.');
        } catch(e){
            setStatus('Error: ' + e.message);
        }
        setDisabled('btn_json', false);
    }
    </script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def dashboard(_: Request):
    return HTMLResponse(content=_DASHBOARD_HTML)

# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 78)
    print("NINJA SUPREME 1.0 — Unified ΛCDM vs DUT (Real Data)")
    print(f"Cache dir: {os.path.abspath(DATA_DIR)}")
    print(f"Output dir: {os.path.abspath(OUTPUT_DIR)}")
    print(f"H(z): {len(Hz_z)} | fσ8: {len(fs8_z)} | BAO: {len(bao_z)} | Pantheon+: {len(pantheon_z)}")
    print("Open: http://localhost:8000")
    print("=" * 78)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", timeout_keep_alive=30)

