
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
# EXPERIMENT 17
# NOISE-FLOOR-ADAPTIVE MULTIPATH DETECTION
# ============================================================

FC = 5.9e9
FS = 100e6

NFFT = 1024
CP_LEN = 128

MAX_DELAY = 600e-9
KERNEL_RADIUS = 16

SNR_VALUES = [-5, 0, 5, 10, 15, 20]

PFA_VALUES = [1e-2, 1e-3, 1e-4]

FIXED_THRESHOLD_DB = -25.0

NUM_TRIALS = 100
RNG_SEED = 2027

MATCH_TOLERANCE_NS = 10.0

OUTPUT_CSV = "experiment17_adaptive_detection_results.csv"
OUTPUT_PEAK_CSV = "experiment17_per_peak_results.csv"
OUTPUT_FIGURE = "experiment17_adaptive_detection_analysis.png"
OUTPUT_PROFILE = "experiment17_adaptive_delay_profiles.png"

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
# LOCAL MAXIMUM DETECTION
# ============================================================

def detect_peaks(magnitude, threshold):

    peaks = []

    for i in range(1, len(magnitude) - 1):

        local_maximum = (
            magnitude[i] > magnitude[i - 1]
            and magnitude[i] >= magnitude[i + 1]
        )

        above_threshold = magnitude[i] >= threshold

        if local_maximum and above_threshold:
            peaks.append(i)

    return np.asarray(peaks, dtype=int)


# ============================================================
# FRACTIONAL PEAK INTERPOLATION
# ============================================================

def refine_peak_position(magnitude, peak_index):

    if peak_index <= 0 or peak_index >= len(magnitude) - 1:
        return float(peak_index)

    # Interpolate in logarithmic magnitude
    eps = 1e-15

    left = 20 * np.log10(max(magnitude[peak_index - 1], eps))
    centre = 20 * np.log10(max(magnitude[peak_index], eps))
    right = 20 * np.log10(max(magnitude[peak_index + 1], eps))

    denominator = left - 2 * centre + right

    if abs(denominator) < 1e-12:
        return float(peak_index)

    offset = 0.5 * (left - right) / denominator

    offset = np.clip(offset, -0.5, 0.5)

    return float(peak_index) + offset


# ============================================================
# ONE-TO-ONE PEAK MATCHING
# ============================================================

def match_peaks(reference_ns, detected_ns):

    candidates = []

    for ref_i, ref_delay in enumerate(reference_ns):

        for det_i, det_delay in enumerate(detected_ns):

            error = abs(det_delay - ref_delay)

            if error <= MATCH_TOLERANCE_NS:
                candidates.append((error, ref_i, det_i))

    candidates.sort(key=lambda x: x[0])

    matched_ref = set()
    matched_det = set()
    errors = []

    for error, ref_i, det_i in candidates:

        if ref_i in matched_ref or det_i in matched_det:
            continue

        matched_ref.add(ref_i)
        matched_det.add(det_i)

        errors.append(
            detected_ns[det_i] - reference_ns[ref_i]
        )

    fp = len(detected_ns) - len(matched_det)

    fn = len(reference_ns) - len(matched_ref)

    return matched_ref, matched_det, np.asarray(errors), fp, fn


# ============================================================
# ADAPTIVE NOISE ESTIMATION
# ============================================================

def estimate_noise_variance(delay_response, noise_start_index):

    noise_region = delay_response[noise_start_index:]

    noise_power = np.abs(noise_region) ** 2

    # Median-based estimator for complex Gaussian noise.
    # For exponential power: median = sigma^2 * ln(2).

    sigma_squared = np.median(noise_power) / np.log(2)

    return max(float(sigma_squared), 1e-30)


# ============================================================
# ADAPTIVE THRESHOLD CALCULATION
# ============================================================

def calculate_threshold(sigma_squared, pfa):

    threshold_power = -sigma_squared * np.log(pfa)

    threshold_amplitude = np.sqrt(threshold_power)

    return threshold_amplitude


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
# STEP 2: COMPUTE PROPAGATION PATHS
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
print(f"Valid propagation paths: {len(tau)}")


# ============================================================
# STEP 3: CONSTRUCT DISCRETE CHANNEL
# ============================================================

dt = 1.0 / FS

num_channel_samples = int(np.ceil(MAX_DELAY / dt))

h = np.zeros(
    num_channel_samples,
    dtype=np.complex128
)

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
        & (indices < num_channel_samples)
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

print("Generating reference received signal...")

y_full = np.convolve(x_cp, h, mode="full")

if CP_LEN < len(h) - 1:
    raise RuntimeError(
        "Cyclic prefix is shorter than channel memory."
    )

y_clean = y_full[CP_LEN:CP_LEN + NFFT]

signal_power = np.mean(np.abs(y_clean) ** 2)

print(f"Received signal power: {signal_power:.6e}")


# ============================================================
# STEP 6: REFERENCE DELAY PROFILE
# ============================================================

H_reference = np.fft.fft(h, n=NFFT)

h_reference = np.fft.ifft(H_reference, n=NFFT)

reference_max = np.max(np.abs(h_reference))

reference_indices = (
    REFERENCE_PEAKS_NS / (dt * 1e9)
)

delay_axis_ns = np.arange(NFFT) * dt * 1e9

# Only search for peaks in the physical delay region.
analysis_length = int(np.ceil(400e-9 * FS)) + 1

# Estimate noise using bins after the 600 ns
# simulated channel support.

noise_start_index = int(np.ceil(MAX_DELAY * FS))

if noise_start_index >= NFFT:
    raise RuntimeError("Invalid noise-estimation region.")


# ============================================================
# STEP 7: MONTE CARLO ANALYSIS
# ============================================================

detector_names = [
    "Fixed -25 dB",
    "Adaptive PFA 1e-2",
    "Adaptive PFA 1e-3",
    "Adaptive PFA 1e-4"
]

results = []
per_peak_results = []

example_profiles = {}
example_thresholds = {}

print("\n========== ADAPTIVE DETECTION ANALYSIS ==========")

for snr_db in SNR_VALUES:

    print(f"\nProcessing SNR = {snr_db} dB")

    snr_linear = 10 ** (snr_db / 10)

    noise_variance = signal_power / snr_linear

    accumulators = {}

    for detector in detector_names:

        accumulators[detector] = {
            "TP": 0,
            "FP": 0,
            "FN": 0,
            "errors_ns": [],
            "false_per_trial": [],
            "peak_counts": np.zeros(
                len(REFERENCE_PEAKS_NS),
                dtype=int
            ),
            "thresholds_db": []
        }

    for trial in range(NUM_TRIALS):

        # Same noisy signal is evaluated by every detector.

        noise = np.sqrt(noise_variance / 2) * (
            rng.standard_normal(NFFT)
            + 1j * rng.standard_normal(NFFT)
        )

        y_noisy = y_clean + noise

        Y = np.fft.fft(y_noisy, n=NFFT)

        H_estimated = Y / X

        h_estimated = np.fft.ifft(H_estimated, n=NFFT)

        magnitude = np.abs(h_estimated)

        # Noise estimate from the noise-only delay region.
        sigma_hat_squared = estimate_noise_variance(
            h_estimated,
            noise_start_index
        )

        adaptive_thresholds = {
             f"Adaptive PFA {pfa:.0e}".replace("e-0", "e-"):
                calculate_threshold(sigma_hat_squared, pfa)
            for pfa in PFA_VALUES
        }

        # Fixed threshold is normalized to the same
        # noise-free reference amplitude.

        fixed_threshold = (
            reference_max
            * 10 ** (FIXED_THRESHOLD_DB / 20)
        )

        threshold_values = {
            "Fixed -25 dB": fixed_threshold,
            **adaptive_thresholds
        }

        for detector in detector_names:

            threshold = threshold_values[detector]

            detected_indices = detect_peaks(
                magnitude[:analysis_length],
                threshold
            )

            detected_positions = np.array([
                refine_peak_position(magnitude, p)
                for p in detected_indices
            ])

            detected_delays_ns = (
                detected_positions * dt * 1e9
            )

            (
                matched_ref,
                matched_det,
                errors,
                fp,
                fn
            ) = match_peaks(
                REFERENCE_PEAKS_NS,
                detected_delays_ns
            )

            acc = accumulators[detector]

            acc["TP"] += len(matched_ref)
            acc["FP"] += fp
            acc["FN"] += fn

            acc["errors_ns"].extend(errors)

            acc["false_per_trial"].append(fp)

            acc["thresholds_db"].append(
                20 * np.log10(
                    max(threshold / reference_max, 1e-15)
                )
            )

            for ref_index in matched_ref:
                acc["peak_counts"][ref_index] += 1

            if trial == 0:

                example_profiles[(snr_db, detector)] = (
                    magnitude.copy()
                )

                example_thresholds[(snr_db, detector)] = threshold

    # ========================================================
    # CALCULATE PERFORMANCE METRICS
    # ========================================================

    for detector in detector_names:

        acc = accumulators[detector]

        tp = acc["TP"]
        fp = acc["FP"]
        fn = acc["FN"]

        total_reference = NUM_TRIALS * len(REFERENCE_PEAKS_NS)

        detection = tp / max(total_reference, 1)

        precision = tp / max(tp + fp, 1)

        recall = tp / max(tp + fn, 1)

        f1 = (
            2 * precision * recall
            / max(precision + recall, 1e-15)
        )

        false_per_trial = np.mean(acc["false_per_trial"])

        errors = np.asarray(acc["errors_ns"])

        if len(errors):

            mae = np.mean(np.abs(errors))

            rmse = np.sqrt(np.mean(errors ** 2))

            bias = np.mean(errors)

        else:

            mae = np.nan
            rmse = np.nan
            bias = np.nan

        average_threshold_db = np.mean(
            acc["thresholds_db"]
        )

        result = {
            "SNR_dB": snr_db,
            "Detector": detector,
            "Average_threshold_dB": average_threshold_db,
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
            "Delay_bias_ns": bias,
            "Matched_peak_estimates": len(errors)
        }

        results.append(result)

        for i, delay in enumerate(REFERENCE_PEAKS_NS):

            per_peak_results.append({
                "SNR_dB": snr_db,
                "Detector": detector,
                "Peak_number": i + 1,
                "Reference_delay_ns": delay,
                "Detection_probability_percent":
                    acc["peak_counts"][i] / NUM_TRIALS * 100
            })

        print(
            f"{detector:<22} "
            f"Detection: {detection * 100:6.2f}% | "
            f"Precision: {precision * 100:6.2f}% | "
            f"F1: {f1:.3f} | "
            f"False/trial: {false_per_trial:.3f} | "
            f"RMSE: {rmse:.4f} ns | "
            f"Threshold: {average_threshold_db:.2f} dB"
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
# STEP 9: PERFORMANCE GRAPHS
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

    for detector in detector_names:

        selected = [
            r for r in results
            if r["Detector"] == detector
        ]

        values = [r[metric] for r in selected]

        ax.plot(
            SNR_VALUES,
            values,
            marker="o",
            label=detector
        )

    ax.set_title(title)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


fig.suptitle(
    "Experiment 17: Adaptive Multipath Detection",
    fontsize=15
)

plt.savefig(
    OUTPUT_FIGURE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# STEP 10: DELAY PROFILE AND THRESHOLD VISUALIZATION
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))

snr_to_plot = 10

profile = example_profiles[
    (snr_to_plot, "Adaptive PFA 1e-3")
]

profile_db = 20 * np.log10(
    np.maximum(
        profile[:analysis_length] / reference_max,
        1e-12
    )
)

ax.plot(
    delay_axis_ns[:analysis_length],
    profile_db,
    label="Noisy delay profile"
)

reference_db = 20 * np.log10(
    np.maximum(
        np.abs(h_reference[:analysis_length]) / reference_max,
        1e-12
    )
)

ax.plot(
    delay_axis_ns[:analysis_length],
    reference_db,
    "--",
    label="Noise-free reference"
)

for detector in detector_names:

    threshold = example_thresholds[
        (snr_to_plot, detector)
    ]

    threshold_db = 20 * np.log10(
        max(threshold / reference_max, 1e-15)
    )

    ax.axhline(
        threshold_db,
        linestyle=":",
        label=f"{detector}: {threshold_db:.1f} dB"
    )

ax.set_title(
    f"Adaptive Thresholds at {snr_to_plot} dB SNR"
)

ax.set_xlabel("Propagation delay (ns)")
ax.set_ylabel("Normalized magnitude (dB)")
ax.set_ylim(-80, 10)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

plt.savefig(
    OUTPUT_PROFILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# STEP 11: FINAL SUMMARY
# ============================================================

print("\n========== EXPERIMENT 17 SUMMARY ==========")

for snr_db in SNR_VALUES:

    print(f"\nSNR = {snr_db} dB")

    for detector in detector_names:

        r = next(
            item for item in results
            if item["SNR_dB"] == snr_db
            and item["Detector"] == detector
        )

        print(
            f"{detector:<22} "
            f"Detection={r['Detection_probability_percent']:.2f}% "
            f"Precision={r['Precision_percent']:.2f}% "
            f"F1={r['F1_score']:.3f} "
            f"FP/trial={r['False_peaks_per_trial']:.3f} "
            f"RMSE={r['Delay_RMSE_ns']:.4f} ns"
        )


print("\n========== OUTPUT FILES ==========")

print(OUTPUT_CSV)
print(OUTPUT_PEAK_CSV)
print(OUTPUT_FIGURE)
print(OUTPUT_PROFILE)

print("\nExperiment 17 completed.")
