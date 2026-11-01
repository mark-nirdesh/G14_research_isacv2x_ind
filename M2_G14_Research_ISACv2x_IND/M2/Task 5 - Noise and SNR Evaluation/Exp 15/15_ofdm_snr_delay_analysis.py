
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
import csv


# ============================================================
# EXPERIMENT 15: REFINED SNR AND DELAY ESTIMATION ANALYSIS
# ============================================================

FC = 5.9e9
FS = 100e6

NFFT = 1024
CP_LEN = 128

MAX_DELAY = 600e-9
KERNEL_RADIUS = 16

SNR_VALUES = [-5, 0, 5, 10, 15, 20]

NUM_TRIALS = 100
RNG_SEED = 2026

PEAK_THRESHOLD_DB = -25.0

# Matching tolerance: one sample = 10 ns
MATCH_TOLERANCE_SAMPLES = 1.0

OUTPUT_FIGURE = "experiment15_refined_snr_analysis.png"
OUTPUT_CSV = "experiment15_refined_snr_results.csv"
OUTPUT_PEAK_CSV = "experiment15_per_peak_detection.csv"


# ============================================================
# CONVERSION FUNCTION
# ============================================================

def to_numpy(value):
    if hasattr(value, "numpy"):
        return value.numpy()
    return dr.to_numpy(value)


# ============================================================
# PEAK DETECTION
# ============================================================

def detect_peaks(signal_db, threshold_db):

    peaks = []

    for i in range(1, len(signal_db) - 1):

        local_maximum = (
            signal_db[i] > signal_db[i - 1]
            and signal_db[i] >= signal_db[i + 1]
        )

        above_threshold = signal_db[i] >= threshold_db

        if local_maximum and above_threshold:
            peaks.append(i)

    return np.array(peaks, dtype=int)


# ============================================================
# FRACTIONAL-SAMPLE PARABOLIC INTERPOLATION
# ============================================================

def refine_peak_position(signal_db, peak_index):

    if peak_index <= 0 or peak_index >= len(signal_db) - 1:
        return float(peak_index)

    left = signal_db[peak_index - 1]
    center = signal_db[peak_index]
    right = signal_db[peak_index + 1]

    denominator = left - 2.0 * center + right

    if abs(denominator) < 1e-12:
        return float(peak_index)

    offset = 0.5 * (left - right) / denominator

    # A parabolic maximum should remain within adjacent samples.
    offset = np.clip(offset, -0.5, 0.5)

    return float(peak_index) + offset


# ============================================================
# ONE-TO-ONE PEAK MATCHING
# ============================================================

def match_peaks(reference_positions, detected_positions, tolerance):

    candidates = []

    for ref_i, ref_position in enumerate(reference_positions):

        for det_i, det_position in enumerate(detected_positions):

            error = abs(det_position - ref_position)

            if error <= tolerance:
                candidates.append((error, ref_i, det_i))

    # Assign the closest pairs first.
    candidates.sort(key=lambda item: item[0])

    matched_reference = set()
    matched_detected = set()
    errors = []

    for error, ref_i, det_i in candidates:

        if ref_i in matched_reference:
            continue

        if det_i in matched_detected:
            continue

        matched_reference.add(ref_i)
        matched_detected.add(det_i)

        errors.append(
            detected_positions[det_i] - reference_positions[ref_i]
        )

    false_peaks = len(detected_positions) - len(matched_detected)

    missed_peaks = len(reference_positions) - len(matched_reference)

    return (
        matched_reference,
        matched_detected,
        np.asarray(errors, dtype=float),
        false_peaks,
        missed_peaks
    )


# ============================================================
# CONFIDENCE INTERVAL
# ============================================================

def mean_confidence_interval(values):

    values = np.asarray(values, dtype=float)

    mean = np.mean(values)

    if len(values) < 2:
        return mean, mean, mean

    standard_error = np.std(values, ddof=1) / np.sqrt(len(values))

    lower = mean - 1.96 * standard_error
    upper = mean + 1.96 * standard_error

    return mean, max(0.0, lower), min(1.0, upper)


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

print("\n========== RAY-TRACING RESULTS ==========")
print(f"Backend: {mi.variant()}")
print(f"Carrier frequency: {FC / 1e9:.2f} GHz")
print(f"Valid ray entries: {len(tau)}")


# ============================================================
# STEP 3: CONSTRUCT DISCRETE CHANNEL
# ============================================================

dt = 1.0 / FS

num_samples = int(np.ceil(MAX_DELAY / dt))

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
# STEP 4: GENERATE QPSK OFDM SIGNAL
# ============================================================

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

print("Generating noise-free received signal...")

y_full = np.convolve(x_cp, h, mode="full")

if CP_LEN < len(h) - 1:
    raise RuntimeError("Cyclic prefix is shorter than channel memory.")

y_clean = y_full[CP_LEN:CP_LEN + NFFT]

signal_power = np.mean(np.abs(y_clean) ** 2)

print(f"Received signal power: {signal_power:.6e}")


# ============================================================
# STEP 6: GENERATE REFERENCE DELAY PROFILE
# ============================================================

H_reference = np.fft.fft(h, n=NFFT)

h_reference = np.fft.ifft(H_reference, n=NFFT)

delay_axis = np.arange(NFFT) / FS

reference_magnitude = np.abs(h_reference)

reference_max = np.max(reference_magnitude)

reference_db = 20 * np.log10(
    np.maximum(
        reference_magnitude / max(reference_max, 1e-15),
        1e-12
    )
)

analysis_length = int(np.ceil(400e-9 * FS)) + 1

reference_window_db = reference_db[:analysis_length]

reference_integer_peaks = detect_peaks(
    reference_window_db,
    PEAK_THRESHOLD_DB
)

reference_positions = np.array([
    refine_peak_position(reference_window_db, p)
    for p in reference_integer_peaks
])

print("\n========== REFERENCE PEAKS ==========")

for i, position in enumerate(reference_positions):

    print(
        f"Peak {i + 1}: "
        f"{position * dt * 1e9:.3f} ns"
    )

print(f"Reference peak count: {len(reference_positions)}")


# ============================================================
# STEP 7: MONTE CARLO SNR ANALYSIS
# ============================================================

results = []
per_peak_results = []

example_profiles = {}

print("\n========== REFINED MONTE CARLO ANALYSIS ==========")

for snr_db in SNR_VALUES:

    print(f"\nProcessing SNR = {snr_db} dB")

    trial_detection_rates = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    all_delay_errors_ns = []

    peak_detection_counts = np.zeros(
        len(reference_positions),
        dtype=int
    )

    false_peaks_per_trial = []

    for trial in range(NUM_TRIALS):

        # ----------------------------------------------------
        # ADD COMPLEX AWGN
        # ----------------------------------------------------

        snr_linear = 10 ** (snr_db / 10)

        noise_variance = signal_power / snr_linear

        noise = np.sqrt(noise_variance / 2) * (
            rng.standard_normal(NFFT)
            + 1j * rng.standard_normal(NFFT)
        )

        y_noisy = y_clean + noise

        # ----------------------------------------------------
        # CHANNEL ESTIMATION
        # ----------------------------------------------------

        Y = np.fft.fft(y_noisy, n=NFFT)

        H_estimated = Y / X

        # ----------------------------------------------------
        # DELAY PROFILE
        # ----------------------------------------------------

        h_estimated = np.fft.ifft(H_estimated, n=NFFT)

        estimated_magnitude = np.abs(h_estimated)

        estimated_db = 20 * np.log10(
            np.maximum(
                estimated_magnitude / max(reference_max, 1e-15),
                1e-12
            )
        )

        estimated_window_db = estimated_db[:analysis_length]

        # ----------------------------------------------------
        # DETECT AND REFINE PEAKS
        # ----------------------------------------------------

        detected_integer_peaks = detect_peaks(
            estimated_window_db,
            PEAK_THRESHOLD_DB
        )

        detected_positions = np.array([
            refine_peak_position(estimated_window_db, p)
            for p in detected_integer_peaks
        ])

        # ----------------------------------------------------
        # ONE-TO-ONE MATCHING
        # ----------------------------------------------------

        matched_ref, matched_det, errors, fp, fn = match_peaks(
            reference_positions,
            detected_positions,
            MATCH_TOLERANCE_SAMPLES
        )

        tp = len(matched_ref)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        trial_detection_rates.append(
            tp / max(len(reference_positions), 1)
        )

        false_peaks_per_trial.append(fp)

        all_delay_errors_ns.extend(errors * dt * 1e9)

        for ref_index in matched_ref:
            peak_detection_counts[ref_index] += 1

        if trial == 0:
            example_profiles[snr_db] = estimated_db.copy()

    # --------------------------------------------------------
    # AGGREGATE PERFORMANCE METRICS
    # --------------------------------------------------------

    total_reference_opportunities = (
        NUM_TRIALS * len(reference_positions)
    )

    detection_probability = (
        total_tp / max(total_reference_opportunities, 1)
    )

    precision = total_tp / max(total_tp + total_fp, 1)

    recall = total_tp / max(total_tp + total_fn, 1)

    f1_score = (
        2 * precision * recall / max(precision + recall, 1e-15)
    )

    false_alarm_rate = (
        total_fp / max(NUM_TRIALS * len(reference_positions), 1)
    )

    average_false_peaks = np.mean(false_peaks_per_trial)

    if len(all_delay_errors_ns) > 0:

        errors_array = np.asarray(all_delay_errors_ns)

        delay_mae = np.mean(np.abs(errors_array))

        delay_rmse = np.sqrt(np.mean(errors_array ** 2))

        delay_bias = np.mean(errors_array)

    else:

        delay_mae = np.nan
        delay_rmse = np.nan
        delay_bias = np.nan

    # Confidence interval based on trial-level detection rates
    mean_detection, ci_lower, ci_upper = mean_confidence_interval(
        trial_detection_rates
    )

    result = {
        "SNR_dB": snr_db,
        "Detection_probability_percent": detection_probability * 100,
        "Precision_percent": precision * 100,
        "Recall_percent": recall * 100,
        "F1_score": f1_score,
        "False_peaks_per_trial": average_false_peaks,
        "False_alarm_rate": false_alarm_rate,
        "Delay_MAE_ns": delay_mae,
        "Delay_RMSE_ns": delay_rmse,
        "Delay_bias_ns": delay_bias,
        "Detection_CI_lower_percent": ci_lower * 100,
        "Detection_CI_upper_percent": ci_upper * 100
    }

    results.append(result)

    for i, position in enumerate(reference_positions):

        per_peak_results.append({
            "SNR_dB": snr_db,
            "Reference_peak": i + 1,
            "Reference_delay_ns": position * dt * 1e9,
            "Detection_probability_percent":
                peak_detection_counts[i] / NUM_TRIALS * 100
        })

    print(f"TP: {total_tp}")
    print(f"FP: {total_fp}")
    print(f"FN: {total_fn}")

    print(f"Detection probability: {detection_probability * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1_score:.4f}")

    print(f"False peaks per trial: {average_false_peaks:.3f}")

    print(f"Delay MAE: {delay_mae:.4f} ns")
    print(f"Delay RMSE: {delay_rmse:.4f} ns")
    print(f"Delay bias: {delay_bias:.4f} ns")

    print(
        f"95% detection CI: "
        f"{ci_lower * 100:.2f}% - {ci_upper * 100:.2f}%"
    )


# ============================================================
# STEP 8: SAVE CSV RESULTS
# ============================================================

with open(OUTPUT_CSV, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(results[0].keys())
    )

    writer.writeheader()
    writer.writerows(results)


with open(OUTPUT_PEAK_CSV, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(per_peak_results[0].keys())
    )

    writer.writeheader()
    writer.writerows(per_peak_results)


# ============================================================
# STEP 9: PRINT SUMMARY
# ============================================================

print("\n========== REFINED SNR SUMMARY ==========")

print(
    f"{'SNR':<8}"
    f"{'Detection':<13}"
    f"{'Precision':<13}"
    f"{'Recall':<12}"
    f"{'F1':<10}"
    f"{'False/trial':<13}"
    f"{'RMSE(ns)':<12}"
)

print("-" * 81)

for r in results:

    print(
        f"{r['SNR_dB']:<8}"
        f"{r['Detection_probability_percent']:<13.2f}"
        f"{r['Precision_percent']:<13.2f}"
        f"{r['Recall_percent']:<12.2f}"
        f"{r['F1_score']:<10.3f}"
        f"{r['False_peaks_per_trial']:<13.3f}"
        f"{r['Delay_RMSE_ns']:<12.3f}"
    )


# ============================================================
# STEP 10: GENERATE PERFORMANCE FIGURES
# ============================================================

snr_array = np.array(SNR_VALUES)

detection = np.array([
    r["Detection_probability_percent"] for r in results
])

precision = np.array([
    r["Precision_percent"] for r in results
])

recall = np.array([
    r["Recall_percent"] for r in results
])

f1 = np.array([
    r["F1_score"] for r in results
])

false_peaks = np.array([
    r["False_peaks_per_trial"] for r in results
])

rmse = np.array([
    r["Delay_RMSE_ns"] for r in results
])

mae = np.array([
    r["Delay_MAE_ns"] for r in results
])

ci_lower = np.array([
    r["Detection_CI_lower_percent"] for r in results
])

ci_upper = np.array([
    r["Detection_CI_upper_percent"] for r in results
])

fig, axes = plt.subplots(
    2, 2,
    figsize=(15, 10),
    constrained_layout=True
)

# ------------------------------------------------------------
# Detection probability with confidence intervals
# ------------------------------------------------------------

axes[0, 0].errorbar(
    snr_array,
    detection,
    yerr=[
        detection - ci_lower,
        ci_upper - detection
    ],
    marker="o",
    capsize=4
)

axes[0, 0].set_title("Detection Probability")
axes[0, 0].set_xlabel("SNR (dB)")
axes[0, 0].set_ylabel("Detection probability (%)")
axes[0, 0].set_ylim(0, 105)
axes[0, 0].grid(True, alpha=0.3)


# ------------------------------------------------------------
# Precision, recall and F1
# ------------------------------------------------------------

axes[0, 1].plot(
    snr_array, precision,
    marker="o", label="Precision"
)

axes[0, 1].plot(
    snr_array, recall,
    marker="s", label="Recall"
)

axes[0, 1].plot(
    snr_array, f1 * 100,
    marker="^", label="F1-score (%)"
)

axes[0, 1].set_title("Detection Quality Metrics")
axes[0, 1].set_xlabel("SNR (dB)")
axes[0, 1].set_ylabel("Percentage (%)")
axes[0, 1].set_ylim(0, 105)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()


# ------------------------------------------------------------
# False peaks
# ------------------------------------------------------------

axes[1, 0].plot(
    snr_array,
    false_peaks,
    marker="s"
)

axes[1, 0].set_title("Average False Peaks per Trial")
axes[1, 0].set_xlabel("SNR (dB)")
axes[1, 0].set_ylabel("False peaks per trial")
axes[1, 0].grid(True, alpha=0.3)


# ------------------------------------------------------------
# Delay estimation error
# ------------------------------------------------------------

axes[1, 1].plot(
    snr_array,
    rmse,
    marker="o",
    label="RMSE"
)

axes[1, 1].plot(
    snr_array,
    mae,
    marker="^",
    label="MAE"
)

axes[1, 1].set_title("Fractional Delay Estimation Error")
axes[1, 1].set_xlabel("SNR (dB)")
axes[1, 1].set_ylabel("Delay error (ns)")
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()


plt.savefig(
    OUTPUT_FIGURE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# STEP 11: PLOT EXAMPLE DELAY PROFILES
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))

for snr_db in [0, 10, 20]:

    ax.plot(
        delay_axis[:analysis_length] * 1e9,
        example_profiles[snr_db][:analysis_length],
        label=f"{snr_db} dB"
    )

ax.plot(
    delay_axis[:analysis_length] * 1e9,
    reference_db[:analysis_length],
    "--",
    label="Noise-free reference"
)

ax.set_title("Delay Profiles under Different SNR Conditions")
ax.set_xlabel("Propagation delay (ns)")
ax.set_ylabel("Normalized magnitude (dB)")
ax.set_ylim(-60, 10)
ax.grid(True, alpha=0.3)
ax.legend()

plt.savefig(
    "experiment15_refined_delay_profiles.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\n========== OUTPUT FILES ==========")
print(OUTPUT_FIGURE)
print("experiment15_refined_delay_profiles.png")
print(OUTPUT_CSV)
print(OUTPUT_PEAK_CSV)

print("\nRefined Experiment 15 completed.")
