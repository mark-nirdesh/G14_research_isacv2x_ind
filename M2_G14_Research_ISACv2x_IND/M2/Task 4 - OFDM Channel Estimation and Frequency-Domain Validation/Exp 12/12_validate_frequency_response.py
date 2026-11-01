
import mitsuba as mi

# Select CPU/LLVM before importing Sionna RT
mi.set_variant("llvm_ad_mono_polarized")

import sionna.rt
from sionna.rt import (
    load_scene,
    Transmitter,
    Receiver,
    PlanarArray,
    PathSolver,
)

import numpy as np
import drjit as dr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 1. CONFIGURATION
# ============================================================
C = 299_792_458.0
FC = 5.9e9

FS = 500e6             # 500 MHz sampling rate
DT = 1.0 / FS          # 2 ns sample interval
BANDWIDTH = 100e6      # Evaluate over +/-50 MHz
MAX_DELAY = 400e-9
KERNEL_RADIUS = 16
NUM_FREQ = 1001

OUTPUT_FILE = "frequency_response_validation.png"


def to_numpy(value):
    if hasattr(value, "numpy"):
        return value.numpy()
    return dr.to_numpy(value)


# ============================================================
# 2. BUILD THE SIONNA RT SCENE
# ============================================================
print("Loading Sionna RT scene...")

scene = load_scene(sionna.rt.scene.simple_street_canyon)
scene.frequency = FC

scene.tx_array = PlanarArray(
    num_rows=1,
    num_cols=1,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="iso",
    polarization="V",
)

scene.rx_array = PlanarArray(
    num_rows=1,
    num_cols=1,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="iso",
    polarization="V",
)

tx = Transmitter(
    name="tx",
    position=[-20, 0, 2],
    orientation=[0, 0, 0],
)

rx = Receiver(
    name="rx",
    position=[20, 0, 2],
    orientation=[0, 0, 0],
)

scene.add(tx)
scene.add(rx)
tx.look_at(rx)

# ============================================================
# 3. COMPUTE PROPAGATION PATHS
# ============================================================
print("Computing propagation paths...")

paths = PathSolver()(scene)

tau = to_numpy(paths.tau).reshape(-1)
a_real = to_numpy(paths.a[0]).reshape(-1)
a_imag = to_numpy(paths.a[1]).reshape(-1)
valid = to_numpy(paths.valid).astype(bool).reshape(-1)

tau = tau[valid]
a = a_real[valid] + 1j * a_imag[valid]

order = np.argsort(tau)
tau = tau[order]
a = a[order]

if len(tau) == 0:
    raise RuntimeError("No valid propagation paths were returned.")

print(f"Backend: {mi.variant()}")
print(f"Carrier frequency: {FC / 1e9:.2f} GHz")
print(f"Valid paths: {len(tau)}")

# ============================================================
# 4. DIRECT RAY-BASED FREQUENCY RESPONSE
#
# H(f) = sum_i a_i * exp(-j*2*pi*f*tau_i)
#
# Frequencies are baseband offsets from the 5.9 GHz carrier.
# ============================================================
f = np.linspace(
    -BANDWIDTH / 2,
    BANDWIDTH / 2,
    NUM_FREQ,
)

H_ray = np.exp(
    -2j * np.pi * f[:, None] * tau[None, :]
) @ a

# ============================================================
# 5. FRACTIONAL-DELAY SAMPLED CIR
#
# Use a finite Lanczos-windowed sinc kernel to distribute
# each path over neighbouring time samples.
# ============================================================
num_samples = int(np.ceil(MAX_DELAY / DT))
t = np.arange(num_samples) * DT
h = np.zeros(num_samples, dtype=np.complex128)

for delay, coefficient in zip(tau, a):
    fractional_index = delay / DT
    center = int(np.floor(fractional_index))

    indices = np.arange(
        center - KERNEL_RADIUS,
        center + KERNEL_RADIUS + 2,
    )

    offset = indices - fractional_index

    # Lanczos-windowed sinc interpolation
    weights = (
        np.sinc(offset)
        * np.sinc(offset / (KERNEL_RADIUS + 1))
    )

    keep = (
        (np.abs(offset) < KERNEL_RADIUS + 1)
        & (indices >= 0)
        & (indices < num_samples)
    )

    h[indices[keep]] += coefficient * weights[keep]

# ============================================================
# 6. FREQUENCY RESPONSE OF SAMPLED CIR
#
# H_sample(f) = sum_n h[n] * exp(-j*2*pi*f*n*DT)
# ============================================================
H_sample = np.exp(
    -2j * np.pi * f[:, None] * t[None, :]
) @ h

# ============================================================
# 7. FULL-BAND AND CENTRAL-BAND ERROR METRICS
# ============================================================
reference_rms = np.sqrt(np.mean(np.abs(H_ray) ** 2))
error_rms = np.sqrt(np.mean(np.abs(H_sample - H_ray) ** 2))

relative_rmse = error_rms / max(reference_rms, 1e-15)

# Central evaluation region: +/-40 MHz
central = np.abs(f) <= 40e6

central_reference_rms = np.sqrt(
    np.mean(np.abs(H_ray[central]) ** 2)
)
central_error_rms = np.sqrt(
    np.mean(
        np.abs(H_sample[central] - H_ray[central]) ** 2
    )
)

central_relative_rmse = (
    central_error_rms / max(central_reference_rms, 1e-15)
)

# Magnitude normalized to the maximum of the direct ray model
scale = max(np.max(np.abs(H_ray)), 1e-15)

ray_db = 20 * np.log10(
    np.maximum(np.abs(H_ray) / scale, 1e-8)
)
sample_db = 20 * np.log10(
    np.maximum(np.abs(H_sample) / scale, 1e-8)
)

# Wrapped phase difference in degrees
phase_difference = np.angle(
    H_sample * np.conj(H_ray),
    deg=True,
)

# Exclude deep nulls where phase is unstable
phase_mask = np.abs(H_ray) > 0.05 * scale

if np.any(phase_mask):
    phase_rmse = np.sqrt(
        np.mean(phase_difference[phase_mask] ** 2)
    )
else:
    phase_rmse = float("nan")

normalized_error = np.abs(H_sample - H_ray) / scale

# ============================================================
# 8. PRINT RESULTS
# ============================================================
print("\n========== VALIDATION RESULTS ==========")
print(f"Sampling rate: {FS / 1e6:.1f} MHz")
print(f"Sample interval: {DT * 1e9:.3f} ns")
print(f"Evaluation bandwidth: {BANDWIDTH / 1e6:.1f} MHz")
print(f"Sampled CIR length: {num_samples} samples")
print(f"Delay window: {MAX_DELAY * 1e9:.1f} ns")

print("\nFull-band metrics:")
print(f"Relative complex RMSE: {relative_rmse:.6f}")
print(f"Relative complex RMSE: {relative_rmse * 100:.3f}%")

print("\nCentral +/-40 MHz metrics:")
print(f"Relative complex RMSE: {central_relative_rmse:.6f}")
print(f"Relative complex RMSE: {central_relative_rmse * 100:.3f}%")

print("\nPhase metric:")
print(f"Phase RMSE away from deep nulls: {phase_rmse:.3f} degrees")

print("\nPath delay range:")
print(f"Minimum: {tau.min() * 1e9:.3f} ns")
print(f"Maximum: {tau.max() * 1e9:.3f} ns")

# ============================================================
# 9. PLOT FREQUENCY RESPONSE AND ERRORS
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(11, 10))

# Magnitude
axes[0].plot(
    f / 1e6,
    ray_db,
    label="Direct ray model",
)
axes[0].plot(
    f / 1e6,
    sample_db,
    "--",
    label="Sampled CIR",
)
axes[0].set_ylabel("Relative magnitude (dB)")
axes[0].set_title(
    "Frequency Response Validation at 5.9 GHz"
)
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Phase difference
axes[1].plot(f / 1e6, phase_difference)
axes[1].set_ylabel("Phase difference (degrees)")
axes[1].grid(True, alpha=0.3)

# Normalized absolute complex error
axes[2].plot(f / 1e6, normalized_error)
axes[2].axvspan(
    -40, 40,
    alpha=0.12,
    label="Central +/-40 MHz",
)
axes[2].set_xlabel("Baseband frequency offset (MHz)")
axes[2].set_ylabel("Normalized absolute error")
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=160)
plt.close()

print(f"\nSaved graph: {OUTPUT_FILE}")
print("Frequency-response validation completed.")
