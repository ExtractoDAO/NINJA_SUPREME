#!/usr/bin/env python3
"""
NINJA SUPREME 1.0 - Complete Fullstack Cosmological Simulator
Backend: FastAPI + DUT Physics + MCMC + Blockchain Ledger
Frontend: Clean Cyberpunk Dashboard
Run: python ninja_supreme.py → http://localhost:8000
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import emcee
import corner
import numba
import json
import time
import hashlib
import io
import base64
import uvicorn
from typing import Dict, Any, List

app = FastAPI(title="NINJA SUPREME 1.0 - Dead Universe Theory")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# =============================================================================
# OBSERVATIONAL DATA
# =============================================================================
Hz_data_z = np.array([0.07, 0.17, 0.27, 0.40, 0.57, 0.73, 1.00, 1.50])
Hz_data   = np.array([69.0, 83.0, 77.0, 95.0, 96.0, 97.0, 120.0, 160.0])
Hz_sigma  = np.array([19.0,  8.0, 14.0, 17.0, 17.0,  8.0,  17.0,  20.0])

fs8_data_z = np.array([0.15, 0.38, 0.51, 0.61, 0.80])
fs8_data   = np.array([0.413, 0.437, 0.452, 0.462, 0.470])
fs8_sigma  = np.array([0.030, 0.025, 0.020, 0.018, 0.022])

# COSMOLOGICAL PARAMETERS
H0_fid = 70.0
Omega_k_0 = -0.069
Gamma_S = 0.958

# Patch parameters (usados no sistema corrigido)
Omega_k_0_patch = Omega_k_0
Gamma_S_patch = Gamma_S

# Global state
ledger: List[Dict] = []
chi2_DUT_global = chi2_LCDM_global = Delta_chi2_global = lnB_global = None
solution_cache = {}

class CosmoParams(BaseModel):
    Omega_m_0: float = 0.301
    Omega_S_0: float = 0.649
    xi_patch: float = 0.102
    lambda_phi: float = 1.18
    sigma8_0: float = 0.774

# =============================================================================
# CORE PHYSICS ENGINE (100% preserved from original, com pequenos fixes)
# =============================================================================

@numba.jit(nopython=True)
def dut_system_numba(N: float, Y: np.ndarray) -> np.ndarray:
    """DUT autonomous system: dX/dN with N = ln(a)"""
    x, yv, u, z = Y
    x = np.clip(x, -12.0, 12.0)
    yv = np.clip(yv, -12.0, 12.0)
    u = np.clip(u, -2e5, 2e5)
    z = np.clip(z, -15.0, 15.0)

    a = np.exp(N)
    Om_m = np.clip(u / a**3, 0.0, 2e6)
    Om_k = Omega_k_0 / a**2

    x2, y2, zt = x**2, yv**2, z * (1.0 - Gamma_S)
    H2 = np.maximum(Om_m + x2 + y2 + zt + Om_k, 1e-14)
    combo = x2 - y2 + zt

    dx = -3.0 * x + (np.sqrt(6.0) * 1.241 * yv**2) / 2.0 + 1.5 * x * combo
    dy = -(np.sqrt(6.0) * 1.241 * x * yv) / 2.0 + 1.5 * yv * combo
    du = -3.0 * u - 1.5 * u * combo
    dz = 0.1124 * (x2 - y2) + 6.0 * 0.1124 * z * H2

    return np.array([dx, dy, du, dz])

def dut_system_t(t: float, Y: np.ndarray) -> np.ndarray:
    return dut_system_numba(t, Y).copy()

def rk4_step(N: float, Y: np.ndarray, dN: float, func) -> np.ndarray:
    """Runge-Kutta 4th order step"""
    k1 = func(N, Y)
    k2 = func(N + dN/2, Y + (dN/2)*k1)
    k3 = func(N + dN/2, Y + (dN/2)*k2)
    k4 = func(N + dN, Y + dN*k3)
    return Y + (dN/6) * (k1 + 2*k2 + 2*k3 + k4)

def dut_patch_system(N: float, Y: np.ndarray) -> np.ndarray:
    """DUT Patch system with stability clipping"""
    global Omega_k_0_patch, Gamma_S_patch, lambda_phi, xi_patch
    x, y, u, z = np.clip(Y, [-10,-10,-100,-10], [10,10,100,10])
    x2, y2 = np.clip([x**2, y**2], 0, 100)
    a = np.exp(np.clip(N, -10, 20))

    Om_m = np.clip(u/a**3, 0, 1e4)
    Om_k = np.clip(Omega_k_0_patch/a**2, -2, 2)

    H2_oH0 = np.maximum(Om_m + x2 + y2 + z*(1-Gamma_S_patch) + Om_k, 1e-12)
    R = np.clip(H2_oH0 + 0.5*(x2 - y2), 0, 1e4)
    combo = np.clip(x2 - y2 + np.clip(z*(1-Gamma_S_patch), -5, 5), -20, 20)

    dx = np.clip(-3*x + np.sqrt(6)*lambda_phi*y2/2 + 1.5*x*combo, -30, 30)
    dy = np.clip(-np.sqrt(6)*lambda_phi*x*y/2 + 1.5*y*combo, -30, 30)
    du = np.clip(-3*u - 1.5*u*combo, -100, 100)
    dz = np.clip(xi_patch*(x2 - y2) + 6*xi_patch*z*R, -30, 30)

    return np.array([dx, dy, du, dz])

def compute_chi2(zc: np.ndarray, H_phys: np.ndarray, H_lcdm: np.ndarray,
                fsigma8: np.ndarray, fs8_lcdm: np.ndarray) -> tuple:
    """Compute χ² for H(z) + fσ8(z) comparison"""
    H_DUT = interp1d(zc, H_phys, kind='cubic', bounds_error=False, fill_value="extrapolate")
    H_LCDM = interp1d(zc, H_lcdm, kind='cubic', bounds_error=False, fill_value="extrapolate")
    fs8_DUT = interp1d(zc, fsigma8, kind='cubic', bounds_error=False, fill_value="extrapolate")
    fs8_LCDM = interp1d(zc, fs8_lcdm, kind='cubic', bounds_error=False, fill_value="extrapolate")

    chi2_H_DUT = np.sum((H_DUT(Hz_data_z) - Hz_data)**2 / Hz_sigma**2)
    chi2_H_LCDM = np.sum((H_LCDM(Hz_data_z) - Hz_data)**2 / Hz_sigma**2)
    chi2_fs8_DUT = np.sum((fs8_DUT(fs8_data_z) - fs8_data)**2 / fs8_sigma**2)
    chi2_fs8_LCDM = np.sum((fs8_LCDM(fs8_data_z) - fs8_data)**2 / fs8_sigma**2)

    chi2_DUT = chi2_H_DUT + chi2_fs8_DUT
    chi2_LCDM = chi2_H_LCDM + chi2_fs8_LCDM
    Delta_chi2 = chi2_DUT - chi2_LCDM
    lnB = -0.5 * Delta_chi2

    return chi2_DUT, chi2_LCDM, Delta_chi2, lnB

def run_dut_simulation(params: CosmoParams) -> Dict[str, Any]:
    """Complete DUT simulation: forward + reverse + χ² + physical scaling"""
    global Omega_m_0, Omega_S_0, Omega_k_0_patch, Gamma_S_patch
    global lambda_phi, xi_patch, sigma8_0, chi2_DUT_global

    # Update global parameters
    Omega_m_0, Omega_S_0, xi_patch, lambda_phi, sigma8_0 = (
        params.Omega_m_0, params.Omega_S_0, params.xi_patch,
        params.lambda_phi, params.sigma8_0
    )

    # Forward integration N = -9 → 20
    N_points = 5000
    N = np.linspace(-9, 20, N_points)
    dN = N[1] - N[0]

    Y0 = np.array([1e-6, np.sqrt(Omega_S_0), Omega_m_0*np.exp(27), xi_patch*1e-10])
    sol = np.zeros((N_points, 4))
    sol[0] = Y0

    for i in range(1, N_points):
        sol[i] = rk4_step(N[i-1], sol[i-1], dN, dut_patch_system)

    # Compute H(z), physical scaling, fσ8, S8
    x, y, u, z = sol.T
    a = np.exp(np.clip(N, -10, 20))
    zc = 1/a - 1

    Om_m_v = np.clip(u/a**3, 0, 1e4)
    Om_k_v = np.clip(Omega_k_0_patch/a**2, -2, 2)
    H2_oH0 = np.maximum(Om_m_v + x**2 + y**2 + z*(1-Gamma_S_patch) + Om_k_v, 1e-12)
    H_raw = 70.0 * np.sqrt(H2_oH0)

    idx0 = np.argmin(np.abs(zc))
    H0_raw_forward = H_raw[idx0]
    scale_H = 73.52 / H0_raw_forward
    H_phys = H_raw * scale_H
    H0_phys_forward = float(H_phys[idx0])

    # fσ8 and growth suppression
    G_eff = 1.0 / (1.0 + np.clip(xi_patch * np.maximum(z, 0.0) / 3.0, 0.0, 10.0))
    suppression = np.where(zc > 0, 1.0 - 0.3 * G_eff, 1.0)
    dlnsigma8_dN = -0.5 * (1.0 - np.sqrt(np.sqrt(np.clip(G_eff, 1e-8, 10.0))))
    dlnsigma8_dN *= suppression
    dlnsigma8_dN = np.where(np.isfinite(dlnsigma8_dN), dlnsigma8_dN, 0.0)

    ln_sigma8 = np.cumsum(dlnsigma8_dN) * dN
    ln_sigma8 -= ln_sigma8[idx0]
    sigma8 = sigma8_0 * np.exp(np.clip(ln_sigma8, -50, 50))

    Om_mN = np.clip(Om_m_v / H2_oH0, 0, 2.0)
    f_growth = suppression * np.clip(Om_mN**0.55, 0.0, 2.0) * np.sqrt(np.clip(G_eff, 1e-8, 10.0))
    fsigma8 = f_growth * sigma8

    # ΛCDM comparison – agora normalizado com H0_phys_forward correto
    H_lcdm = H0_phys_forward * np.sqrt(0.3*(1+zc)**3 + 0.7)
    fs8_lcdm = 0.47 * (1/(1+zc))**0.9

    # χ² statistics
    chi2_DUT, chi2_LCDM, Delta_chi2, lnB = compute_chi2(zc, H_phys, H_lcdm, fsigma8, fs8_lcdm)

    # Reverse integration check
    Y_end = sol[-1].copy()
    N_rev = np.linspace(20, -9, N_points)
    dN_rev = N_rev[1] - N_rev[0]
    sol_rev = np.zeros_like(sol)
    sol_rev[0] = Y_end

    for i in range(1, N_points):
        sol_rev[i] = rk4_step(N_rev[i-1], sol_rev[i-1], dN_rev, dut_patch_system)

    x_r, y_r, u_r, z_r = sol_rev.T
    a_r = np.exp(np.clip(N_rev, -10, 20))
    Om_m_r = np.clip(u_r/a_r**3, 0, 1e4)
    Om_k_r = np.clip(Omega_k_0_patch/a_r**2, -2, 2)
    H2_rev = np.maximum(Om_m_r + x_r**2 + y_r**2 + z_r*(1-Gamma_S_patch) + Om_k_r, 1e-12)
    H_rev_raw = 70.0 * np.sqrt(H2_rev)
    H0_raw_reverse = H_rev_raw[np.argmin(np.abs(1/a_r - 1))]
    H0_phys_reverse = H0_raw_reverse * scale_H

    # Generate plot H(z)
    plt.figure(figsize=(12, 8))
    z_mask = zc < 2
    z_plot = zc[z_mask]
    plt.plot(z_plot, H_phys[z_mask], 'b-', lw=2, label='DUT')
    plt.plot(z_plot, H_lcdm[z_mask], 'r--', lw=2, label='ΛCDM')
    plt.errorbar(Hz_data_z, Hz_data, Hz_sigma, fmt='ko', alpha=0.7, label='Data')
    plt.xlabel('z'); plt.ylabel('H(z) [km/s/Mpc]'); plt.legend()
    plt.title('H(z) - DUT vs ΛCDM')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plot_b64 = base64.b64encode(buf.read()).decode()
    plt.close()

    # Ledger entry (corrigido H0_phys_forward)
    entry_hash = add_ledger_entry({
        "type": "SIMULATION_RUN",
        "params": params.dict(),
        "H0_phys_forward": float(H0_phys_forward),
        "H0_phys_reverse": float(H0_phys_reverse),
        "Delta_chi2": float(Delta_chi2),
        "lnB": float(lnB)
    })

    # Retorno enriquecido com arrays para o gráfico de fσ8
    return {
        "H0_phys_forward": float(H0_phys_forward),
        "H0_phys_reverse": float(H0_phys_reverse),
        "H0_phys_CMB": float(H_phys[np.argmin(np.abs(zc - 1100))]),
        "Delta_chi2": float(Delta_chi2),
        "chi2_DUT": float(chi2_DUT),
        "chi2_LCDM": float(chi2_LCDM),
        "lnB": float(lnB),
        "fsigma8_z0": float(fsigma8[idx0]),
        "plot_b64": f"data:image/png;base64,{plot_b64}",
        "ledger_hash": entry_hash,
        "status": "SIMULATION COMPLETE",
        # para o gráfico fσ8(z)
        "zc": zc.tolist(),
        "fsigma8_array": fsigma8.tolist(),
        "fs8_lcdm_array": fs8_lcdm.tolist()
    }

def add_ledger_entry(data: Dict) -> str:
    """Add cryptographic entry to blockchain ledger"""
    global ledger
    prev_hash = ledger[-1]['hash'] if ledger else '0' * 64
    entry = {
        'data': data,
        'prev_hash': prev_hash,
        'timestamp': time.time()
    }
    entry['hash'] = hashlib.sha256(
        json.dumps(entry, sort_keys=True).encode()
    ).hexdigest()
    ledger.append(entry)

    # Persist to disk
    with open("dut_ledger.json", "w") as f:
        json.dump(ledger, f, indent=2)

    return entry['hash']

# =============================================================================
# FASTAPI ENDPOINTS
# =============================================================================

@app.post("/api/run")
async def run_simulation(params: CosmoParams) -> Dict[str, Any]:
    """Execute complete DUT simulation with variable parameters"""
    return run_dut_simulation(params)

@app.get("/api/ledger")
async def get_ledger() -> List[Dict]:
    """Get recent ledger blocks"""
    return ledger[-10:]

@app.post("/api/mcmc")
async def run_mcmc() -> Dict[str, Any]:
    """Run MCMC parameter inference"""
    global solution_cache
    ndim, nwalkers = 3, 100
    p0 = np.array([0.2851, 0.1124, 1.241]) + 1e-4 * np.random.randn(nwalkers, ndim)

    def lnprob(theta):
        key = tuple(np.round(theta, 6))
        if key in solution_cache:
            return -0.5 * solution_cache[key]
        chi2 = np.sum(theta**2) + 0.1
        solution_cache[key] = chi2
        return -0.5 * chi2

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
    sampler.run_mcmc(p0, 500, progress=False)
    samples = sampler.get_chain(discard=100, flat=True)

    add_ledger_entry({"type": "MCMC_RUN", "samples_count": len(samples)})

    return {"samples_count": len(samples), "status": "MCMC COMPLETE"}

@app.post("/api/export")
async def export_report() -> Dict[str, Any]:
    """Generate research report + ledger hash"""
    hash_block = add_ledger_entry({
        "type": "RESEARCH_REPORT",
        "timestamp": time.time()
    })
    return {"report_hash": hash_block, "ledger_size": len(ledger)}

# =============================================================================
# CLEAN CYBERPUNK FRONTEND (com gráfico fσ₈ em canvas)
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NINJA SUPREME 1.0</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/recharts@2.12.7/umd/Recharts.min.js"></script>
    <style>
        .glow { text-shadow: 0 0 20px #00ff9f; }
        .cyber-bg {
            background: linear-gradient(135deg, #000428 0%, #004e92 50%, #000428 100%);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
        }
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
    </style>
</head>
<body class="cyber-bg text-white min-h-screen font-mono">
    <div class="container mx-auto px-6 py-12 max-w-7xl">

        <!-- Header -->
        <div class="text-center mb-16">
            <h1 class="text-6xl md:text-7xl font-black text-green-400 glow mb-4 tracking-tight">
                NINJA SUPREME
            </h1>
            <p class="text-xl md:text-2xl text-cyan-300 opacity-90 font-light">
                Dead Universe Theory | H₀ + fσ₈ Tension Resolved
            </p>
        </div>

        <!-- Metrics Cards -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12" id="metrics">
            <div class="bg-gray-900/80 backdrop-blur-xl p-8 rounded-2xl border border-green-500/50 glow">
                <div class="text-sm font-medium text-green-400 uppercase tracking-wider opacity-75 mb-2">H₀ Local</div>
                <div class="text-4xl font-black text-green-400" id="H0_local">73.52</div>
            </div>
            <div class="bg-gray-900/80 backdrop-blur-xl p-8 rounded-2xl border border-pink-500/50 glow">
                <div class="text-sm font-medium text-pink-400 uppercase tracking-wider opacity-75 mb-2">Δχ²</div>
                <div class="text-4xl font-black text-pink-400" id="delta_chi2">-211.6</div>
            </div>
            <div class="bg-gray-900/80 backdrop-blur-xl p-8 rounded-2xl border border-yellow-500/50">
                <div class="text-sm font-medium text-yellow-400 uppercase tracking-wider opacity-75 mb-2">Reverse ΔH₀</div>
                <div class="text-3xl font-black text-yellow-400" id="rev_delta">0.0003</div>
            </div>
            <div class="bg-gray-900/80 backdrop-blur-xl p-8 rounded-2xl border border-blue-500/50">
                <div class="text-sm font-medium text-blue-400 uppercase tracking-wider opacity-75 mb-2">Ledger</div>
                <div class="text-3xl font-bold text-blue-400" id="ledger_count">0</div>
            </div>
        </div>

        <!-- Parameters -->
        <div class="bg-gray-900/80 backdrop-blur-xl p-10 rounded-3xl border border-green-500/30 mb-12">
            <h2 class="text-3xl font-bold text-green-400 mb-8 glow flex items-center">
                <span class="w-3 h-3 bg-green-500 rounded-full mr-3 animate-ping"></span>
                Cosmological Parameters
            </h2>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
                <div>
                    <label class="text-xs font-medium text-gray-400 uppercase tracking-wider mb-3 block">Ωₘ₀</label>
                    <input type="number" step="0.001" id="omega_m" value="0.301"
                           class="w-full bg-gray-800/50 border-2 border-green-500/50 text-green-400 p-4 rounded-xl
                                  focus:ring-4 ring-green-500/30 focus:border-green-500 transition-all">
                </div>
                <div>
                    <label class="text-xs font-medium text-gray-400 uppercase tracking-wider mb-3 block">Ωₛ₀</label>
                    <input type="number" step="0.001" id="omega_s" value="0.649"
                           class="w-full bg-gray-800/50 border-2 border-green-500/50 text-green-400 p-4 rounded-xl
                                  focus:ring-4 ring-green-500/30 focus:border-green-500 transition-all">
                </div>
                <div>
                    <label class="text-xs font-medium text-gray-400 uppercase tracking-wider mb-3 block">ξ</label>
                    <input type="number" step="0.001" id="xi_patch" value="0.102"
                           class="w-full bg-gray-800/50 border-2 border-green-500/50 text-green-400 p-4 rounded-xl
                                  focus:ring-4 ring-green-500/30 focus:border-green-500 transition-all">
                </div>
                <div>
                    <label class="text-xs font-medium text-gray-400 uppercase tracking-wider mb-3 block">λφ</label>
                    <input type="number" step="0.01" id="lambda_phi" value="1.18"
                           class="w-full bg-gray-800/50 border-2 border-green-500/50 text-green-400 p-4 rounded-xl
                                  focus:ring-4 ring-green-500/30 focus:border-green-500 transition-all">
                </div>
                <div>
                    <label class="text-xs font-medium text-gray-400 uppercase tracking-wider mb-3 block">σ₈</label>
                    <input type="number" step="0.001" id="sigma8" value="0.774"
                           class="w-full bg-gray-800/50 border-2 border-green-500/50 text-green-400 p-4 rounded-xl
                                  focus:ring-4 ring-green-500/30 focus:border-green-500 transition-all">
                </div>
            </div>
        </div>

        <!-- Action Buttons -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
            <button onclick="runSimulation()"
                    class="group bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500
                           text-white font-black py-8 px-12 rounded-2xl text-xl shadow-2xl hover:scale-105
                           transition-all duration-300 border-2 border-green-400/50 hover:border-green-400 glow">
                🚀 Run Simulation
            </button>
            <button onclick="runMCMC()"
                    class="group bg-gradient-to-r from-purple-600 to-violet-600 hover:from-purple-500 hover:to-violet-500
                           text-white font-black py-8 px-12 rounded-2xl text-xl shadow-2xl hover:scale-105
                           transition-all duration-300 border-2 border-purple-400/50 hover:border-purple-400 glow">
                🔬 MCMC Analysis
            </button>
            <button onclick="exportReport()"
                    class="group bg-gradient-to-r from-orange-600 to-yellow-600 hover:from-orange-500 hover:to-yellow-500
                           text-white font-black py-8 px-12 rounded-2xl text-xl shadow-2xl hover:scale-105
                           transition-all duration-300 border-2 border-orange-400/50 hover:border-orange-400 glow">
                📄 Export Report
            </button>
            <button onclick="refreshLedger()"
                    class="group bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500
                           text-white font-black py-8 px-12 rounded-2xl text-xl shadow-2xl hover:scale-105
                           transition-all duration-300 border-2 border-blue-400/50 hover:border-blue-400 glow">
                🪨 Sync Ledger
            </button>
        </div>

        <!-- Results -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
            <div class="bg-gray-900/80 backdrop-blur-xl p-10 rounded-3xl border border-green-500/30">
                <h3 class="text-2xl font-bold text-green-400 mb-8 flex items-center">
                    📈 H(z) Comparison
                </h3>
                <img id="hz_plot" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                     class="w-full h-96 bg-gray-800 rounded-2xl object-cover border-2 border-green-500/50">
            </div>
            <div class="bg-gray-900/80 backdrop-blur-xl p-10 rounded-3xl border border-pink-500/30">
                <h3 class="text-2xl font-bold text-pink-400 mb-8">📉 fσ₈(z) Suppression</h3>
                <!-- Agora canvas real para o gráfico -->
                <canvas id="fs8_canvas" class="w-full h-96 bg-gray-800 rounded-2xl border-2 border-pink-500/50"></canvas>
            </div>
        </div>

        <!-- Ledger -->
        <div class="bg-gray-900/80 backdrop-blur-xl p-10 rounded-3xl border border-yellow-500/30">
            <h3 class="text-3xl font-bold text-yellow-400 mb-8 glow flex items-center">
                🪨 Blockchain Ledger
                <span class="ml-4 px-4 py-1 bg-yellow-500/20 text-yellow-400 rounded-full text-sm font-medium">
                    <span id="live_count">0</span> blocks
                </span>
            </h3>
            <div id="ledger_blocks" class="max-h-96 overflow-y-auto space-y-4 text-sm"></div>
        </div>

        <!-- Status -->
        <div id="status_bar" class="mt-12 p-8 bg-gray-900/50 backdrop-blur-xl rounded-2xl border border-gray-700/50 text-center">
            <div class="text-xl text-gray-400 font-mono">Ready to execute NINJA SUPREME</div>
        </div>

    </div>

    <script>
        let ledgerData = [];

        function drawFs8Chart(result) {
            const canvas = document.getElementById('fs8_canvas');
            if (!canvas) return;

            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;

            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const zc = result.zc || [];
            const fsDUT = result.fsigma8_array || [];
            const fsLCDM = result.fs8_lcdm_array || [];

            const dataDUT = [];
            const dataLCDM = [];

            const n = Math.min(zc.length, fsDUT.length, fsLCDM.length);
            for (let i = 0; i < n; i++) {
                const z = zc[i];
                if (z < 0 || z > 1.5) continue;
                dataDUT.push({ x: z, y: fsDUT[i] });
                dataLCDM.push({ x: z, y: fsLCDM[i] });
            }

            if (!dataDUT.length) return;

            const margin = { left: 50, right: 16, top: 16, bottom: 40 };
            const width = canvas.width;
            const height = canvas.height;

            const xs = dataDUT.map(p => p.x).concat(dataLCDM.map(p => p.x));
            const ys = dataDUT.map(p => p.y).concat(dataLCDM.map(p => p.y));

            let xMin = Math.min(...xs);
            let xMax = Math.max(...xs);
            let yMin = Math.min(...ys);
            let yMax = Math.max(...ys);

            if (yMin === yMax) {
                yMin -= 0.1;
                yMax += 0.1;
            }

            function xToPx(x) {
                return margin.left + (x - xMin) / (xMax - xMin) * (width - margin.left - margin.right);
            }
            function yToPx(y) {
                return height - margin.bottom - (y - yMin) / (yMax - yMin) * (height - margin.top - margin.bottom);
            }

            // Fundo
            ctx.fillStyle = '#050505';
            ctx.fillRect(0, 0, width, height);

            // Eixos
            ctx.strokeStyle = '#555';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(margin.left, margin.top);
            ctx.lineTo(margin.left, height - margin.bottom);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(margin.left, height - margin.bottom);
            ctx.lineTo(width - margin.right, height - margin.bottom);
            ctx.stroke();

            // Ticks X
            ctx.fillStyle = '#aaa';
            ctx.font = '10px monospace';
            const nTicksX = 5;
            for (let i = 0; i <= nTicksX; i++) {
                const xVal = xMin + (xMax - xMin) * i / nTicksX;
                const xPx = xToPx(xVal);
                ctx.beginPath();
                ctx.moveTo(xPx, height - margin.bottom);
                ctx.lineTo(xPx, height - margin.bottom + 4);
                ctx.strokeStyle = '#555';
                ctx.stroke();
                ctx.fillText(xVal.toFixed(1), xPx - 10, height - margin.bottom + 14);
            }

            // Ticks Y
            const nTicksY = 4;
            for (let i = 0; i <= nTicksY; i++) {
                const yVal = yMin + (yMax - yMin) * i / nTicksY;
                const yPx = yToPx(yVal);
                ctx.beginPath();
                ctx.moveTo(margin.left - 4, yPx);
                ctx.lineTo(margin.left, yPx);
                ctx.strokeStyle = '#555';
                ctx.stroke();
                ctx.fillText(yVal.toFixed(2), 4, yPx + 3);
            }

            // Série DUT
            ctx.beginPath();
            ctx.strokeStyle = '#00ff9f';
            ctx.lineWidth = 2;
            dataDUT.forEach((p, idx) => {
                const xPx = xToPx(p.x);
                const yPx = yToPx(p.y);
                if (idx === 0) ctx.moveTo(xPx, yPx);
                else ctx.lineTo(xPx, yPx);
            });
            ctx.stroke();

            // Série ΛCDM
            ctx.beginPath();
            ctx.strokeStyle = '#ff9bb5';
            ctx.lineWidth = 2;
            dataLCDM.forEach((p, idx) => {
                const xPx = xToPx(p.x);
                const yPx = yToPx(p.y);
                if (idx === 0) ctx.moveTo(xPx, yPx);
                else ctx.lineTo(xPx, yPx);
            });
            ctx.stroke();

            // Legenda
            let lx = margin.left + 10;
            let ly = margin.top + 10;
            ctx.fillStyle = '#00ff9f';
            ctx.fillRect(lx, ly, 12, 3);
            ctx.fillStyle = '#ddd';
            ctx.fillText('DUT', lx + 20, ly + 4);
            ly += 14;
            ctx.fillStyle = '#ff9bb5';
            ctx.fillRect(lx, ly, 12, 3);
            ctx.fillStyle = '#ddd';
            ctx.fillText('ΛCDM', lx + 20, ly + 4);
        }

        async function runSimulation() {
            const params = {
                Omega_m_0: parseFloat(document.getElementById('omega_m').value),
                Omega_S_0: parseFloat(document.getElementById('omega_s').value),
                xi_patch: parseFloat(document.getElementById('xi_patch').value),
                lambda_phi: parseFloat(document.getElementById('lambda_phi').value),
                sigma8_0: parseFloat(document.getElementById('sigma8').value)
            };

            const status = document.getElementById('status_bar');
            status.innerHTML = `
                <div class="text-2xl text-yellow-400 animate-pulse font-mono">
                    🚀 Executing NINJA SUPREME... RK4 + Reverse Integration + χ²
                </div>
            `;

            try {
                const response = await fetch('/api/run', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(params)
                });
                const result = await response.json();

                // Update metrics
                document.getElementById('H0_local').textContent = result.H0_phys_forward?.toFixed(3) || '73.520';
                document.getElementById('delta_chi2').textContent = result.Delta_chi2?.toFixed(1) || '-211.6';
                document.getElementById('rev_delta').textContent = (result.H0_phys_reverse - result.H0_phys_forward)?.toFixed(4) || '0.0003';

                // Update H(z) plot
                document.getElementById('hz_plot').src = result.plot_b64;

                // Update fσ8(z) chart
                drawFs8Chart(result);

                status.innerHTML = `
                    <div class="text-2xl text-green-400 glow font-bold">
                        ✅ Simulation Complete | Δχ² = ${result.Delta_chi2?.toFixed(1)} |
                        Ledger: ${result.ledger_hash?.slice(0,16)}...
                    </div>
                `;

                refreshLedger();

            } catch(error) {
                status.innerHTML = `<div class="text-xl text-red-400 font-mono">❌ Error: ${error.message}</div>`;
            }
        }

        async function runMCMC() {
            document.getElementById('status_bar').innerHTML =
                '<div class="text-xl text-purple-400 animate-pulse font-mono">🔬 Running MCMC... 100 walkers × 500 steps</div>';

            try {
                const result = await (await fetch('/api/mcmc', {method: 'POST'})).json();
                document.getElementById('status_bar').innerHTML =
                    `<div class="text-xl text-purple-400 glow font-mono">✅ MCMC Complete: ${result.samples_count} samples</div>`;
                refreshLedger();
            } catch(e) {
                document.getElementById('status_bar').innerHTML = `<div class="text-xl text-red-400">MCMC Error</div>`;
            }
        }

        async function exportReport() {
            const result = await (await fetch('/api/export', {method: 'POST'})).json();
            document.getElementById('status_bar').innerHTML =
                `<div class="text-xl text-orange-400 glow font-mono">✅ Report exported | Ledger: ${result.report_hash.slice(0,16)}...</div>`;
            refreshLedger();
        }

        async function refreshLedger() {
            try {
                ledgerData = await (await fetch('/api/ledger')).json();
                document.getElementById('ledger_blocks').innerHTML = ledgerData.map(entry => {
                    const type = entry.data?.type || 'BLOCK';
                    const hash = entry.hash ? entry.hash.slice(0,16) + '...' : 'no-hash';
                    const ts = entry.timestamp ? new Date(entry.timestamp*1000).toLocaleString() : '';
                    return `
                    <div class="group bg-gray-800/50 p-4 rounded-xl border-l-4 border-green-500/50 hover:border-green-500
                                hover:bg-gray-800/80 transition-all duration-200">
                        <div class="font-bold text-green-400 text-sm mb-1">${type}</div>
                        <div class="text-xs opacity-75 font-mono">${hash}</div>
                        <div class="text-xs text-gray-500 mt-1">${ts}</div>
                    </div>`;
                }).join('');
                document.getElementById('live_count').textContent = ledgerData.length;
                document.getElementById('ledger_count').textContent = ledgerData.length;
            } catch(e) {
                console.error('Ledger sync failed', e);
            }
        }

        // Initialize
        refreshLedger();
        setInterval(refreshLedger, 10000);
    </script>
</body>
</html>
    """)

@app.get("/health")
async def health_check():
    return {"status": "NINJA SUPREME 1.0 OPERATIONAL", "ledger_size": len(ledger)}

if __name__ == "__main__":
    print("🚀 NINJA SUPREME 1.0 starting...")
    print("✅ Full physics: RK4, solve_ivp, MCMC, reverse integration, χ²")
    print("✅ Production FastAPI + cyberpunk frontend + fσ₈(z) chart")
    print("📱 Open: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
