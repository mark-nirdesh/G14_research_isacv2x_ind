
import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")

import sionna.rt
from sionna.rt import (
    load_scene,
    Transmitter,
    Receiver,
    PlanarArray,
    PathSolver
)

import numpy as np
import drjit as dr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# EXPERIMENT 14: OFDM DELAY-PROFILE ESTIMATION
# ============================================================

C = 299_792_458.0

FC = 5.9e9
FS = 100e6

NFFT = 1024
CP_LEN = 128

MAX_DELAY = 600e-9
KERNEL_RADIUS = 16

RNG_SEED = 2026

OUTPUT_FIGURE = "experiment14_ofdm_delay_profile.png"

# Peak detection configuration
PEAK_THRESHOLD_DB = -25.0
MIN_PEAK_DISTANCE = 1


# ============================================================
# CONVERSION FUNCTION
# ============================================================

def to_numpy(value):
    if hasattr(value, "numpy"):
        return value.numpy()
    return dr.to_numpy(value)


# ============================================================
# STEP 1: LOAD SIONNA RT SCENE
# ============================================================

print("\nLoading Sionna RT scene...")

scene = load_scene(sionna.rt.scene.simple_street_canyon)
scene.frequency = FC

scene.tx_array = PlanarArray(
    num_rows=1,
    num_cols=1,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="iso",
    polarization="V"
)

scene.rx_array = PlanarArray(
    num_rows=1,
    num_cols=1,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="iso",
    polarization="V"
)

tx = Transmitter(
    name="tx",
    position=[-20, 0, 2],
    orientation=[0, 0, 0]
)

rx = Receiver(
    name="rx",
    position=[20, 0, 2],
    orientation=[0, 0, 0]
)

scene.add(tx)
scene.add(rx)

tx.look_at(rx)


# ============================================================
# STEP 2: COMPUTE RAY-TRACING PATHS
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

NUM_PATHS = len(tau)

print("\n========== RAY-TRACING RESULTS ==========")
print(f"Backend: {mi.variant()}")
print(f"Carrier frequency: {FC / 1e9:.2f} GHz")
print(f"Valid propagation paths: {NUM_PATHS}")
print(f"Earliest path: {np.min(tau) * 1e9:.3f} ns")
print(f"Latest path: {np.max(tau) * 1e9:.3f} ns")


# ============================================================
# STEP 3: CONSTRUCT DISCRETE-TIME CHANNEL
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
        center + KERNEL_RADIUS + 2
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


# ============================================================
# STEP 4: GENERATE QPSK OFDM SYMBOL
# ============================================================

print("\nGenerating OFDM signal...")

rng = np.random.default_rng(RNG_SEED)

bits_i = rng.integers(0, 2, NFFT)
bits_q = rng.integers(0, 2, NFFT)

X = (
    (1 - 2 * bits_i)
    + 1j * (1 - 2 * bits_q)
) / np.sqrt(2)

x = np.fft.ifft(X, n=NFFT)

x_cp = np.concatenate((
    x[-CP_LEN:],
    x
))


# ============================================================
# STEP 5: PROPAGATE THROUGH CHANNEL
# ============================================================

print("Propagating OFDM signal through channel...")

y_full = np.convolve(x_cp, h, mode="full")

if CP_LEN < len(h) - 1:
    raise RuntimeError(
        "Cyclic prefix is shorter than channel memory."
    )

start = CP_LEN

y_useful = y_full[start:start + NFFT]

Y = np.fft.fft(y_useful, n=NFFT)


# ============================================================
# STEP 6: ESTIMATE FREQUENCY-DOMAIN CHANNEL
# ============================================================

H_estimated = Y / X

H_reference = np.fft.fft(h, n=NFFT)

frequency = np.fft.fftfreq(NFFT, d=dt)


# ============================================================
# STEP 7: CONVERT FREQUENCY RESPONSE TO DELAY DOMAIN
# ============================================================

print("Calculating delay profiles...")

# IFFT of estimated frequency response
h_estimated = np.fft.ifft(H_estimated, n=NFFT)

# IFFT of reference frequency response
h_reference = np.fft.ifft(H_reference, n=NFFT)

# Delay-axis samples
delay_axis = np.arange(NFFT) / FS

# Magnitude
estimated_magnitude = np.abs(h_estimated)
reference_magnitude = np.abs(h_reference)

# Normalize both against reference maximum
normalization = np.max(reference_magnitude)

estimated_normalized = (
    estimated_magnitude / max(normalization, 1e-15)
)

reference_normalized = (
    reference_magnitude / max(normalization, 1e-15)
)

# Convert to dB
estimated_db = 20 * np.log10(
    np.maximum(estimated_normalized, 1e-12)
)

reference_db = 20 * np.log10(
    np.maximum(reference_normalized, 1e-12)
)


# ============================================================
# STEP 8: DETECT LOCAL DELAY-DOMAIN PEAKS
# ============================================================

def detect_peaks(signal_db, threshold_db=-25.0, min_distance=1):

    candidates = []

    for i in range(1, len(signal_db) - 1):

        is_local_maximum = (
            signal_db[i] > signal_db[i - 1]
            and signal_db[i] >= signal_db[i + 1]
        )

        above_threshold = signal_db[i] >= threshold_db

        if is_local_maximum and above_threshold:
            candidates.append(i)

    # Keep stronger peaks when candidates are too close
    candidates = sorted(
        candidates,
        key=lambda i: signal_db[i],
        reverse=True
    )

    selected = []

    for candidate in candidates:

        if all(
            abs(candidate - existing) >= min_distance
            for existing in selected
        ):
            selected.append(candidate)

    return np.array(sorted(selected), dtype=int)


detected_indices = detect_peaks(
    estimated_db,
    threshold_db=PEAK_THRESHOLD_DB,
    min_distance=MIN_PEAK_DISTANCE
)

detected_delays = delay_axis[detected_indices]

detected_levels = estimated_db[detected_indices]


# ============================================================
# STEP 9: COMPARE WITH SIONNA RT RAY DELAYS
# ============================================================

print("\n========== DELAY-PROFILE RESULTS ==========")

print(f"Sampling interval: {dt * 1e9:.3f} ns")
print(f"Delay window: {NFFT / FS * 1e6:.3f} us")
print(f"Delay sample spacing: {1 / FS * 1e9:.3f} ns")

print(f"\nSionna RT ray delays: {NUM_PATHS}")
print(f"Detected local peaks: {len(detected_indices)}")

print("\nSIONNA RT RAY DELAYS")

print("-" * 55)
print(f"{'Path':<10}{'Delay (ns)':<20}{'Sample index':<15}")
print("-" * 55)

for i, delay in enumerate(tau):

    fractional_index = delay * FS

    print(
        f"{i + 1:<10}"
        f"{delay * 1e9:<20.3f}"
        f"{fractional_index:<15.3f}"
    )


print("\nDETECTED DELAY-DOMAIN PEAKS")

print("-" * 55)
print(f"{'Peak':<10}{'Delay (ns)':<20}{'Magnitude (dB)':<15}")
print("-" * 55)

for i, index in enumerate(detected_indices):

    print(
        f"{i + 1:<10}"
        f"{delay_axis[index] * 1e9:<20.3f}"
        f"{estimated_db[index]:<15.3f}"
    )


# ============================================================
# STEP 10: CHANNEL RECONSTRUCTION ERROR
# ============================================================

reconstruction_error = h_estimated[:len(h)] - h

error_rms = np.sqrt(
    np.mean(np.abs(reconstruction_error) ** 2)
)

reference_rms = np.sqrt(
    np.mean(np.abs(h) ** 2)
)

relative_rmse = error_rms / max(reference_rms, 1e-15)

print("\n========== RECONSTRUCTION VALIDATION ==========")

print(f"Reference channel RMS: {reference_rms:.6e}")
print(f"Reconstruction error RMS: {error_rms:.6e}")
print(f"Relative complex RMSE: {relative_rmse:.6e}")


# ============================================================
# STEP 11: PLOT DELAY PROFILE
# ============================================================

print("\nGenerating delay-profile figure...")

fig, axes = plt.subplots(
    2,
    1,
    figsize=(14, 9),
    constrained_layout=True
)

# ------------------------------------------------------------
# Plot 1: Delay profile and Sionna RT ray delays
# ------------------------------------------------------------

axes[0].plot(
    delay_axis * 1e9,
    reference_db,
    label="Reference delay profile",
    linewidth=1.8
)

axes[0].plot(
    delay_axis * 1e9,
    estimated_db,
    "--",
    label="Estimated delay profile",
    linewidth=1.5
)

# Mark actual ray delays
for i, delay in enumerate(tau):

    axes[0].axvline(
        delay * 1e9,
        color="green",
        linestyle=":",
        alpha=0.55,
        linewidth=1
    )

# Mark detected local maxima
axes[0].scatter(
    detected_delays * 1e9,
    detected_levels,
    marker="x",
    s=65,
    label="Detected local peaks",
    zorder=5
)

axes[0].set_xlim(0, 400)
axes[0].set_ylim(-60, 5)

axes[0].set_title(
    "Experiment 14: OFDM Delay-Profile Estimation"
)

axes[0].set_xlabel("Propagation delay (ns)")
axes[0].set_ylabel("Normalized magnitude (dB)")

axes[0].grid(True, alpha=0.3)
axes[0].legend()


# ------------------------------------------------------------
# Plot 2: Linear normalized delay profile
# ------------------------------------------------------------

axes[1].plot(
    delay_axis * 1e9,
    reference_normalized,
    label="Reference profile",
    linewidth=1.6
)

axes[1].plot(
    delay_axis * 1e9,
    estimated_normalized,
    "--",
    label="Estimated profile",
    linewidth=1.4
)

axes[1].set_xlim(0, 400)

axes[1].set_xlabel("Propagation delay (ns)")
axes[1].set_ylabel("Normalized magnitude")

axes[1].set_title(
    "Linear Magnitude of the Delay Profile"
)

axes[1].grid(True, alpha=0.3)
axes[1].legend()


plt.savefig(
    OUTPUT_FIGURE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"\nSaved figure: {OUTPUT_FIGURE}")

print("\nExperiment 4 completed.")
