
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
# EXPERIMENT 16: ADAPTIVE DETECTION THRESHOLD ANALYSIS
# ============================================================

FC = 5.9e9
FS = 100e6

NFFT = 1024
CP_LEN = 128

MAX_DELAY = 600e-9
KERNEL_RADIUS = 16

SNR_VALUES = [-5, 0, 5, 10, 15, 20]
THRESHOLDS_DB = [-30, -25, -20]

NUM_TRIALS = 100
RNG_SEED = 2026

MATCH_TOLERANCE_SAMPLES = 1.0

OUTPUT_CSV = "experiment16_threshold_results.csv"
OUTPUT_PEAK_CSV = "experiment16_per_peak_results.csv"
OUTPUT_FIGURE = "experiment16_threshold_analysis.png"
OUTPUT_PROFILE = "experiment16_threshold_delay_profiles.png"


# Fixed noise-free reference peaks from Experiment 15.
# These must remain unchanged across threshold conditions.

REFERENCE_PEAKS_NS = np.array([
    131.150812,
    149.600922,
    176.494451,
    230.029754,
    281.620815
])


# ============================================================
# CONVERSION
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

        is_local_maximum = (
            signal_db[i] > signal_db[i - 1]
            and signal_db[i] >= signal_db[i + 1]
        )

        above_threshold = signal_db[i] >= threshold_db

        if is_local_maximum and above_threshold:
            peaks.append(i)

    return np.asarray(peaks, dtype=int)


# ============================================================
# FRACTIONAL-SAMPLE INTERPOLATION
# ============================================================

def refine_peak_position(signal_db, peak_index):

    if peak_index <= 0 or peak_index >= len(signal_db) - 1:
        return float(peak_index)

    left = signal_db[peak_index - 1]
    centre = signal_db[peak_index]
    right = signal_db[peak_index + 1]

    denominator = left - 2.0 * centre + right

    if abs(denominator) < 1e-12:
        return float(peak_index)

    offset = 0.5 * (left - right) / denominator

    offset = np.clip(offset, -0.5, 0.5)

    return float(peak_index) + offset


# ============================================================
# ONE-TO-ONE PEAK MATCHING
# ============================================================

def match_peaks(reference, detected, tolerance):

    candidates = []

    for ref_i, ref_position in enumerate(reference):

        for det_i, det_position in enumerate(detected):

            error = abs(det_position - ref_position)

            if error <= tolerance:
                candidates.append((error, ref_i, det_i))

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
            detected[det_i] - reference[ref_i]
        )

    false_peaks = len(detected) - len(matched_detected)

    missed_peaks = len(reference) - len(matched_reference)

    return (
        matched_reference,
        matched_detected,
        np.asarray(errors, dtype=float),
        false_peaks,
        missed_peaks
    )


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
# STEP 2: COMPUTE RAY-TRACING CHANNEL
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

    centre = int(np.floor(fractional_index))

    indices = np.arange(
        centre - KERNEL_RADIUS,
        centre + KERNEL_RADIUS + 2
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
# STEP 4: GENERATE OFDM SIGNAL
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
# STEP 6: REFERENCE DELAY PROFILE
# ============================================================

H_reference = np.fft.fft(h, n=NFFT)

h_reference = np.fft.ifft(H_reference, n=NFFT)

reference_magnitude = np.abs(h_reference)

reference_max = np.max(reference_magnitude)

analysis_length = int(np.ceil(400e-9 * FS)) + 1

reference_db = 20 * np.log10(
    np.maximum(
        reference_magnitude / max(reference_max, 1e-15),
        1e-12
    )
)

reference_positions = REFERENCE_PEAKS_NS / (dt * 1e9)

print("\n========== FIXED REFERENCE PEAKS ==========")

for i, delay in enumerate(REFERENCE_PEAKS_NS):

    print(f"Peak {i + 1}: {delay:.3f} ns")

print("Reference peaks: 5")
print("Reference threshold-independent positions retained.")


# ============================================================
# STEP 7: MONTE CARLO THRESHOLD ANALYSIS
# ============================================================

results = []
per_peak_results = []

example_profiles = {}

print("\n========== THRESHOLD SENSITIVITY ANALYSIS ==========")

for snr_db in SNR_VALUES:

    print(f"\nProcessing SNR = {snr_db} dB")

    snr_linear = 10 ** (snr_db / 10)

    noise_variance = signal_power / snr_linear

    # Separate accumulators for each threshold
    accumulators = {}

    for threshold in THRESHOLDS_DB:

        accumulators[threshold] = {
            "TP": 0,
            "FP": 0,
            "FN": 0,
            "errors_ns": [],
            "false_per_trial": [],
            "peak_counts": np.zeros(
                len(reference_positions),
                dtype=int
            )
        }

    # Generate one noise realisation per trial and SNR.
    # Reuse it across all three thresholds for fair comparison.

    for trial in range(NUM_TRIALS):

        noise = np.sqrt(noise_variance / 2) * (
            rng.standard_normal(NFFT)
            + 1j * rng.standard_normal(NFFT)
        )

        y_noisy = y_clean + noise

        Y = np.fft.fft(y_noisy, n=NFFT)

        H_estimated = Y / X

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
        # TEST ALL THRESHOLDS USING THE SAME NOISY PROFILE
        # ----------------------------------------------------

        for threshold in THRESHOLDS_DB:

            detected_indices = detect_peaks(
                estimated_window_db,
                threshold
            )

            detected_positions = np.array([
                refine_peak_position(estimated_window_db, p)
                for p in detected_indices
            ])

            (
                matched_ref,
                matched_det,
                errors,
                fp,
                fn
            ) = match_peaks(
                reference_positions,
                detected_positions,
                MATCH_TOLERANCE_SAMPLES
            )

            acc = accumulators[threshold]

            acc["TP"] += len(matched_ref)
            acc["FP"] += fp
            acc["FN"] += fn

            acc["errors_ns"].extend(errors * dt * 1e9)

            acc["false_per_trial"].append(fp)

            for ref_index in matched_ref:
                acc["peak_counts"][ref_index] += 1

            if trial == 0:
                example_profiles[(snr_db, threshold)] = (
                    estimated_db.copy()
                )

    # --------------------------------------------------------
    # CALCULATE METRICS FOR EACH THRESHOLD
    # --------------------------------------------------------

    for threshold in THRESHOLDS_DB:

        acc = accumulators[threshold]

        tp = acc["TP"]
        fp = acc["FP"]
        fn = acc["FN"]

        total_reference = NUM_TRIALS * len(reference_positions)

        detection = tp / max(total_reference, 1)

        precision = tp / max(tp + fp, 1)

        recall = tp / max(tp + fn, 1)

        f1 = (
            2 * precision * recall
            / max(precision + recall, 1e-15)
        )

        false_per_trial = np.mean(acc["false_per_trial"])

        errors = np.asarray(acc["errors_ns"], dtype=float)

        if len(errors) > 0:

            mae = np.mean(np.abs(errors))

            rmse = np.sqrt(np.mean(errors ** 2))

            bias = np.mean(errors)

        else:

            mae = np.nan
            rmse = np.nan
            bias = np.nan

        result = {
            "SNR_dB": snr_db,
            "Threshold_dB": threshold,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Detection_probability_percent": detection * 100,
            "Precision_percent": precision * 100,
            "Recall_percent": recall * 100,
            "F1_score": f1,
            "False_peaks_per_trial": false_per_trial,
            "Delay_MAE_ns": mae,
            "Delay_RMSE_ns": rmse,
            "Delay_bias_ns": bias
        }

        results.append(result)

        for i, position in enumerate(reference_positions):

            per_peak_results.append({
                "SNR_dB": snr_db,
                "Threshold_dB": threshold,
                "Reference_peak": i + 1,
                "Reference_delay_ns": REFERENCE_PEAKS_NS[i],
                "Detection_probability_percent":
                    acc["peak_counts"][i] / NUM_TRIALS * 100
            })

        print(
            f"Threshold {threshold:>4} dB | "
            f"Detection: {detection * 100:6.2f}% | "
            f"Precision: {precision * 100:6.2f}% | "
            f"F1: {f1:.3f} | "
            f"False/trial: {false_per_trial:.3f} | "
            f"RMSE: {rmse:.4f} ns"
        )


# ============================================================
# STEP 8: SAVE CSV FILES
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
# STEP 9: GENERATE SUMMARY FIGURES
# ============================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(15, 10),
    constrained_layout=True
)

metrics = [
    ("Detection_probability_percent", "Detection Probability (%)"),
    ("Precision_percent", "Precision (%)"),
    ("False_peaks_per_trial", "False Peaks per Trial"),
    ("Delay_RMSE_ns", "Delay RMSE (ns)")
]

for ax, (metric, title) in zip(axes.flat, metrics):

    for threshold in THRESHOLDS_DB:

        selected = [
            r for r in results
            if r["Threshold_dB"] == threshold
        ]

        values = [r[metric] for r in selected]

        ax.plot(
            SNR_VALUES,
            values,
            marker="o",
            label=f"{threshold} dB threshold"
        )

    ax.set_title(title)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    ax.legend()


fig.suptitle(
    "Experiment 16: Detection Threshold Sensitivity",
    fontsize=15
)

plt.savefig(
    OUTPUT_FIGURE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# STEP 10: PLOT DELAY PROFILES AT 10 dB
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))

for threshold in THRESHOLDS_DB:

    profile = example_profiles[(10, threshold)]

    ax.plot(
        np.arange(analysis_length) * dt * 1e9,
        profile[:analysis_length],
        label=f"{threshold} dB threshold"
    )

ax.plot(
    np.arange(analysis_length) * dt * 1e9,
    reference_db[:analysis_length],
    "--",
    label="Noise-free reference"
)

ax.set_title("Delay Profiles at 10 dB under Different Thresholds")
ax.set_xlabel("Propagation delay (ns)")
ax.set_ylabel("Normalized magnitude (dB)")
ax.set_ylim(-60, 10)
ax.grid(True, alpha=0.3)
ax.legend()

plt.savefig(
    OUTPUT_PROFILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# STEP 11: PRINT FINAL SUMMARY
# ============================================================

print("\n========== EXPERIMENT 16 SUMMARY ==========")

print(
    f"{'SNR':<7}"
    f"{'Threshold':<12}"
    f"{'Detection':<12}"
    f"{'Precision':<12}"
    f"{'F1':<9}"
    f"{'False/trial':<13}"
    f"{'RMSE(ns)':<12}"
)

print("-" * 77)

for r in results:

    print(
        f"{r['SNR_dB']:<7}"
        f"{r['Threshold_dB']:<12}"
        f"{r['Detection_probability_percent']:<12.2f}"
        f"{r['Precision_percent']:<12.2f}"
        f"{r['F1_score']:<9.3f}"
        f"{r['False_peaks_per_trial']:<13.3f}"
        f"{r['Delay_RMSE_ns']:<12.3f}"
    )


print("\n========== OUTPUT FILES ==========")
print(OUTPUT_CSV)
print(OUTPUT_PEAK_CSV)
print(OUTPUT_FIGURE)
print(OUTPUT_PROFILE)

print("\nExperiment 16 completed.")
