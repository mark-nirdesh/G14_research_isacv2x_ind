
import mitsuba as mi
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
# EXPERIMENT 3: OFDM WAVEFORM AND CHANNEL ESTIMATION
# ============================================================

C = 299_792_458.0
FC = 5.9e9

FS = 100e6
NFFT = 1024
CP_LEN = 128

MAX_DELAY = 600e-9
KERNEL_RADIUS = 16
RNG_SEED = 2026

OUTPUT_FIGURE = "experiment13_ofdm_channel_estimation.png"


def to_numpy(value):
    if hasattr(value, "numpy"):
        return value.numpy()
    return dr.to_numpy(value)


# ============================================================
# 1. RECREATE THE VALIDATED SIONNA RT SCENE
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
# 2. COMPUTE THE PROPAGATION PATHS
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
# 3. CONSTRUCT A FRACTIONAL-DELAY DISCRETE CHANNEL
#
# h[n] approximates the ray-tracing channel using a
# Lanczos-windowed sinc interpolation kernel.
# ============================================================

dt = 1.0 / FS
num_samples = int(np.ceil(MAX_DELAY / dt))

time = np.arange(num_samples) * dt
h = np.zeros(num_samples, dtype=np.complex128)

for delay, coefficient in zip(tau, a):
    fractional_index = delay / dt
    center = int(np.floor(fractional_index))

    indices = np.arange(
        center - KERNEL_RADIUS,
        center + KERNEL_RADIUS + 2,
    )

    offset = indices - fractional_index

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

if np.max(np.abs(h)) == 0:
    raise RuntimeError("The constructed discrete channel is zero.")

# ============================================================
# 4. DEFINE THE OFDM WAVEFORM
# ============================================================

subcarrier_spacing = FS / NFFT
useful_symbol_duration = NFFT / FS
cp_duration = CP_LEN / FS

# FFT bin frequencies in NumPy's standard FFT ordering
frequencies = np.fft.fftfreq(NFFT, d=1.0 / FS)

# Reference channel response of the constructed discrete CIR
H_reference = np.fft.fft(h, n=NFFT)

# Known QPSK symbols on all subcarriers
rng = np.random.default_rng(RNG_SEED)

bits_i = rng.integers(0, 2, NFFT)
bits_q = rng.integers(0, 2, NFFT)

X = (
    (1 - 2 * bits_i)
    + 1j * (1 - 2 * bits_q)
) / np.sqrt(2)

# OFDM modulation: frequency domain -> time domain
x = np.fft.ifft(X, n=NFFT)

# Add cyclic prefix
x_cp = np.concatenate((x[-CP_LEN:], x))

# ============================================================
# 5. PASS THE OFDM WAVEFORM THROUGH THE DISCRETE CHANNEL
#
# Linear convolution models the sampled multipath channel.
# CP removal and FFT recover the received subcarriers.
# ============================================================

y_full = np.convolve(x_cp, h, mode="full")

# The CP protects the useful symbol from previous-symbol
# channel memory when CP_LEN >= len(h)-1.
if CP_LEN < len(h) - 1:
    raise RuntimeError(
        "Cyclic prefix is too short for this channel. "
        "Increase CP_LEN or shorten the channel support."
    )

start = CP_LEN
y_useful = y_full[start:start + NFFT]

if len(y_useful) != NFFT:
    raise RuntimeError("Incorrect received OFDM symbol length.")

Y = np.fft.fft(y_useful, n=NFFT)

# ============================================================
# 6. ESTIMATE THE CHANNEL USING KNOWN PILOT SYMBOLS
#
# Y[k] = H[k] X[k]
# H_est[k] = Y[k] / X[k]
# ============================================================

H_estimated = Y / X

error = H_estimated - H_reference

reference_rms = np.sqrt(np.mean(np.abs(H_reference) ** 2))
error_rms = np.sqrt(np.mean(np.abs(error) ** 2))

relative_rmse = error_rms / max(reference_rms, 1e-15)

magnitude_scale = max(np.max(np.abs(H_reference)), 1e-15)

H_reference_db = 20 * np.log10(
    np.maximum(np.abs(H_reference) / magnitude_scale, 1e-8)
)

H_estimated_db = 20 * np.log10(
    np.maximum(np.abs(H_estimated) / magnitude_scale, 1e-8)
)

# Phase difference, with reference deep nulls excluded
phase_error = np.angle(
    H_estimated * np.conj(H_reference),
    deg=True,
)

phase_mask = np.abs(H_reference) > 0.05 * magnitude_scale

if np.any(phase_mask):
    phase_rmse = np.sqrt(
        np.mean(phase_error[phase_mask] ** 2)
    )
else:
    phase_rmse = float("nan")

# ============================================================
# 7. PRINT EXPERIMENT RESULTS
# ============================================================

print("\n========== OFDM CONFIGURATION ==========")
print(f"Sampling rate: {FS / 1e6:.2f} MHz")
print(f"FFT size: {NFFT}")
print(f"Subcarrier spacing: {subcarrier_spacing / 1e3:.6f} kHz")
print(f"Useful symbol duration: {useful_symbol_duration * 1e6:.3f} us")
print(f"Cyclic prefix: {CP_LEN} samples")
print(f"Cyclic prefix duration: {cp_duration * 1e6:.3f} us")
print(f"Discrete channel length: {len(h)} samples")
print(f"Valid ray entries: {len(tau)}")

print("\n========== CHANNEL ESTIMATION RESULTS ==========")
print(f"Reference channel RMS: {reference_rms:.6e}")
print(f"Estimation error RMS: {error_rms:.6e}")
print(f"Relative complex RMSE: {relative_rmse:.8e}")
print(f"Relative complex RMSE: {relative_rmse * 100:.6f}%")
print(f"Phase RMSE away from deep nulls: {phase_rmse:.8f} degrees")

print("\nNoise: disabled")
print("Modulation: QPSK")
print("Channel: static Sionna RT-derived discrete channel")

# ============================================================
# 8. PLOT CHANNEL ESTIMATION RESULTS
# ============================================================

freq_mhz = frequencies / 1e6

fig, axes = plt.subplots(3, 1, figsize=(12, 11))

# Channel magnitude
axes[0].plot(
    freq_mhz,
    H_reference_db,
    label="Reference channel",
    linewidth=2,
)

axes[0].plot(
    freq_mhz,
    H_estimated_db,
    "--",
    label="Estimated channel",
)

axes[0].set_title("Experiment 3: OFDM Channel Estimation")
axes[0].set_ylabel("Relative magnitude (dB)")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Phase difference
axes[1].plot(freq_mhz, phase_error)
axes[1].set_ylabel("Phase error (degrees)")
axes[1].grid(True, alpha=0.3)

# Complex estimation error
axes[2].plot(
    freq_mhz,
    np.abs(error) / magnitude_scale,
)

axes[2].set_xlabel("Baseband subcarrier frequency (MHz)")
axes[2].set_ylabel("Normalized absolute error")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_FIGURE, dpi=160)
plt.close()

print(f"\nSaved figure: {OUTPUT_FIGURE}")
print("Experiment 3 completed.")
