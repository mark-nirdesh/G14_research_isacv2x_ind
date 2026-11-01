
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
# EXPERIMENT 17A
# FALSE ALARM CALIBRATION AND NOISE ESTIMATOR VALIDATION
# ============================================================

FC = 5.9e9
FS = 100e6

NFFT = 1024
CP_LEN = 128

MAX_DELAY = 600e-9
KERNEL_RADIUS = 16

SNR_VALUES = [-5, 0, 5, 10, 15, 20]
PFA_VALUES = [1e-2, 1e-3, 1e-4]

NUM_NOISE_TRIALS = 10000
NUM_CHANNEL_TRIALS = 500
BATCH_SIZE = 250

RNG_SEED = 2028

MATCH_TOLERANCE_NS = 10.0
SEARCH_MAX_DELAY_NS = 400.0

REFERENCE_PEAKS_NS = np.array([
    131.150812,
    149.600922,
    176.494451,
    230.029754,
    281.620815
])

OUTPUT_NOISE_CSV = "experiment17A_noise_calibration.csv"
OUTPUT_CHANNEL_CSV = "experiment17A_channel_validation.csv"
OUTPUT_FIGURE = "experiment17A_calibration_analysis.png"
OUTPUT_PROFILE = "experiment17A_noise_profile.png"


# ============================================================
# CONVERSION
# ============================================================

def to_numpy(value):

    if hasattr(value, "numpy"):
        return value.numpy()

    return dr.to_numpy(value)


# ============================================================
# NOISE VARIANCE ESTIMATION
# ============================================================

def estimate_noise_variance_batch(h_estimated, noise_start):

    noise_region = h_estimated[:, noise_start:]

    noise_power = np.abs(noise_region) ** 2

    sigma_squared = (
        np.median(noise_power, axis=1) / np.log(2)
    )

    return np.maximum(sigma_squared, 1e-30)


# ============================================================
# ADAPTIVE THRESHOLD
# ============================================================

def calculate_threshold_batch(sigma_squared, pfa):

    return np.sqrt(
        -sigma_squared * np.log(pfa)
    )


# ============================================================
# LOCAL MAXIMUM DETECTION
# ============================================================

def detect_peaks(magnitude, threshold):

    indices = []

    for i in range(1, len(magnitude) - 1):

        if (
            magnitude[i] >= threshold
            and magnitude[i] > magnitude[i - 1]
            and magnitude[i] >= magnitude[i + 1]
        ):
            indices.append(i)

    return np.asarray(indices, dtype=int)


# ============================================================
# FRACTIONAL DELAY REFINEMENT
# ============================================================

def refine_peak_position(magnitude, peak_index):

    if peak_index <= 0 or peak_index >= len(magnitude) - 1:
        return float(peak_index)

    eps = 1e-15

    left = 20 * np.log10(
        max(magnitude[peak_index - 1], eps)
    )

    centre = 20 * np.log10(
        max(magnitude[peak_index], eps)
    )

    right = 20 * np.log10(
        max(magnitude[peak_index + 1], eps)
    )

    denominator = left - 2 * centre + right

    if abs(denominator) < 1e-12:
        return float(peak_index)

    offset = (
        0.5 * (left - right) / denominator
    )

    return float(peak_index) + np.clip(
        offset, -0.5, 0.5
    )


# ============================================================
# ONE-TO-ONE REFERENCE MATCHING
# ============================================================

def match_peaks(reference_ns, detected_ns):

    candidates = []

    for ref_i, ref_delay in enumerate(reference_ns):

        for det_i, det_delay in enumerate(detected_ns):

            error = abs(det_delay - ref_delay)

            if error <= MATCH_TOLERANCE_NS:
                candidates.append((error, ref_i, det_i))

    candidates.sort(key=lambda item: item[0])

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

    false_peaks = len(detected_ns) - len(matched_det)

    missed_peaks = len(reference_ns) - len(matched_ref)

    return matched_ref, errors, false_peaks, missed_peaks


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
# STEP 2: CONSTRUCT DISCRETE CHANNEL
# ============================================================

dt = 1.0 / FS

channel_length = int(np.ceil(MAX_DELAY / dt))

h = np.zeros(
    channel_length,
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
        & (indices < channel_length)
    )

    h[indices[keep]] += coefficient * weights[keep]


# ============================================================
# STEP 3: GENERATE OFDM SIGNAL
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
# STEP 4: REFERENCE RECEIVED SIGNAL
# ============================================================

y_full = np.convolve(x_cp, h, mode="full")

if CP_LEN < len(h) - 1:
    raise RuntimeError("Cyclic prefix is insufficient.")

y_clean = y_full[CP_LEN:CP_LEN + NFFT]

signal_power = np.mean(np.abs(y_clean) ** 2)

print(f"Received signal power: {signal_power:.6e}")


# ============================================================
# STEP 5: DELAY-DOMAIN CONFIGURATION
# ============================================================

noise_start = int(np.ceil(MAX_DELAY * FS))

search_end = int(
    np.floor(SEARCH_MAX_DELAY_NS / (dt * 1e9))
) + 1

if noise_start >= NFFT:
    raise RuntimeError("Noise estimation region is empty.")

if search_end >= noise_start:
    raise RuntimeError(
        "Search region overlaps noise estimation region."
    )

# Exclude boundary bins when calculating local maxima.
test_indices = np.arange(1, search_end - 1)

num_test_bins = len(test_indices)

print(f"Noise estimation starts at sample: {noise_start}")
print(f"Noise-only bins available: {NFFT - noise_start}")
print(f"False-alarm test bins: {num_test_bins}")


# ============================================================
# STEP 6: NOISE-ONLY MONTE CARLO CALIBRATION
# ============================================================

print("\n========== NOISE-ONLY CALIBRATION ==========")

noise_results = []

example_noise_profile = None
example_noise_thresholds = None

for snr_db in SNR_VALUES:

    snr_linear = 10 ** (snr_db / 10)

    noise_variance = signal_power / snr_linear

    print(f"\nSNR = {snr_db} dB")
    print(f"Actual noise variance: {noise_variance:.6e}")

    for pfa in PFA_VALUES:

        total_exceedances = 0
        total_test_cells = 0
        profile_false_alarms = 0
        total_local_false_peaks = 0

        variance_estimates = []

        completed = 0

        while completed < NUM_NOISE_TRIALS:

            batch = min(
                BATCH_SIZE,
                NUM_NOISE_TRIALS - completed
            )

            # Generate noise in the received time domain.
            noise = np.sqrt(noise_variance / 2) * (
                rng.standard_normal((batch, NFFT))
                + 1j * rng.standard_normal((batch, NFFT))
            )

            # Apply the same OFDM channel-estimation chain.
            Y_noise = np.fft.fft(noise, axis=1)

            H_noise = Y_noise / X[None, :]

            h_noise = np.fft.ifft(H_noise, axis=1)

            sigma_hat = estimate_noise_variance_batch(
                h_noise,
                noise_start
            )

            thresholds = calculate_threshold_batch(
                sigma_hat,
                pfa
            )

            test_power = (
                np.abs(h_noise[:, test_indices]) ** 2
            )

            # Per-cell false-alarm probability.
            exceedances = (
                test_power >= thresholds[:, None] ** 2
            )

            total_exceedances += int(
                np.sum(exceedances)
            )

            total_test_cells += batch * num_test_bins

            # Profile-level false alarms and local maxima.
            for row in range(batch):

                magnitude = np.abs(h_noise[row])

                peaks = detect_peaks(
                    magnitude[:search_end],
                    thresholds[row]
                )

                total_local_false_peaks += len(peaks)

                if len(peaks) > 0:
                    profile_false_alarms += 1

            variance_estimates.extend(sigma_hat.tolist())

            if example_noise_profile is None and snr_db == 10:

                example_noise_profile = np.abs(
                    h_noise[0]
                ).copy()

                example_noise_thresholds = {
                    p: float(
                        calculate_threshold_batch(
                            sigma_hat[:1], p
                        )[0]
                    )
                    for p in PFA_VALUES
                }

            completed += batch

        empirical_cell_pfa = (
            total_exceedances / total_test_cells
        )

        empirical_profile_pfa = (
            profile_false_alarms / NUM_NOISE_TRIALS
        )

        average_false_peaks = (
            total_local_false_peaks / NUM_NOISE_TRIALS
        )

        variance_estimates = np.asarray(
            variance_estimates
        )

        mean_estimated_variance = np.mean(
            variance_estimates
        )

        variance_ratio = (
            mean_estimated_variance / noise_variance
        )

        # Independent-bin theoretical profile probability.
        theoretical_profile_pfa = (
            1 - (1 - pfa) ** num_test_bins
        )

        result = {
            "SNR_dB": snr_db,
            "PFA_target": pfa,
            "Actual_noise_variance": noise_variance,
            "Mean_estimated_noise_variance":
                mean_estimated_variance,
            "Variance_estimation_ratio": variance_ratio,
            "Empirical_cell_PFA": empirical_cell_pfa,
            "Theoretical_cell_PFA": pfa,
            "Empirical_profile_PFA":
                empirical_profile_pfa,
            "Theoretical_profile_PFA":
                theoretical_profile_pfa,
            "Average_false_local_peaks":
                average_false_peaks,
            "Noise_trials": NUM_NOISE_TRIALS,
            "Test_bins_per_trial": num_test_bins
        }

        noise_results.append(result)

        print(
            f"PFA={pfa:.0e} | "
            f"Variance ratio={variance_ratio:.4f} | "
            f"Cell PFA={empirical_cell_pfa:.6e} | "
            f"Profile PFA={empirical_profile_pfa:.4f} | "
            f"False peaks/trial={average_false_peaks:.4f}"
        )


# ============================================================
# STEP 7: CHANNEL + NOISE VALIDATION
# ============================================================

print("\n========== CHANNEL + NOISE VALIDATION ==========")

channel_results = []

for snr_db in SNR_VALUES:

    snr_linear = 10 ** (snr_db / 10)

    noise_variance = signal_power / snr_linear

    accumulators = {}

    for pfa in PFA_VALUES:

        accumulators[pfa] = {
            "TP": 0,
            "FP": 0,
            "FN": 0,
            "errors": [],
            "thresholds": []
        }

    for trial in range(NUM_CHANNEL_TRIALS):

        noise = np.sqrt(noise_variance / 2) * (
            rng.standard_normal(NFFT)
            + 1j * rng.standard_normal(NFFT)
        )

        y_noisy = y_clean + noise

        Y = np.fft.fft(y_noisy)

        H_estimated = Y / X

        h_estimated = np.fft.ifft(H_estimated)

        magnitude = np.abs(h_estimated)

        sigma_hat = estimate_noise_variance_batch(
            h_estimated[None, :],
            noise_start
        )[0]

        for pfa in PFA_VALUES:

            threshold = calculate_threshold_batch(
                np.array([sigma_hat]),
                pfa
            )[0]

            peaks = detect_peaks(
                magnitude[:search_end],
                threshold
            )

            refined_positions = np.array([
                refine_peak_position(magnitude, peak)
                for peak in peaks
            ])

            detected_delays_ns = (
                refined_positions * dt * 1e9
            )

            matched_ref, errors, fp, fn = match_peaks(
                REFERENCE_PEAKS_NS,
                detected_delays_ns
            )

            acc = accumulators[pfa]

            acc["TP"] += len(matched_ref)
            acc["FP"] += fp
            acc["FN"] += fn

            acc["errors"].extend(errors)

            acc["thresholds"].append(threshold)

    for pfa in PFA_VALUES:

        acc = accumulators[pfa]

        tp = acc["TP"]
        fp = acc["FP"]
        fn = acc["FN"]

        total_reference = (
            NUM_CHANNEL_TRIALS * len(REFERENCE_PEAKS_NS)
        )

        detection = tp / total_reference

        precision = tp / max(tp + fp, 1)

        recall = tp / max(tp + fn, 1)

        f1 = (
            2 * precision * recall
            / max(precision + recall, 1e-15)
        )

        errors = np.asarray(acc["errors"])

        rmse = (
            np.sqrt(np.mean(errors ** 2))
            if len(errors) else np.nan
        )

        mae = (
            np.mean(np.abs(errors))
            if len(errors) else np.nan
        )

        result = {
            "SNR_dB": snr_db,
            "PFA_target": pfa,
            "Detection_probability_percent": detection * 100,
            "Precision_percent": precision * 100,
            "Recall_percent": recall * 100,
            "F1_score": f1,
            "False_peaks_per_trial": fp / NUM_CHANNEL_TRIALS,
            "Delay_MAE_ns": mae,
            "Delay_RMSE_ns": rmse,
            "Mean_threshold_amplitude":
                np.mean(acc["thresholds"])
        }

        channel_results.append(result)

        print(
            f"SNR={snr_db:>3} dB | "
            f"PFA={pfa:.0e} | "
            f"Detection={detection * 100:.2f}% | "
            f"Precision={precision * 100:.2f}% | "
            f"F1={f1:.3f} | "
            f"False/trial={fp / NUM_CHANNEL_TRIALS:.3f} | "
            f"RMSE={rmse:.4f} ns"
        )


# ============================================================
# STEP 8: SAVE CSV FILES
# ============================================================

with open(OUTPUT_NOISE_CSV, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(noise_results[0].keys())
    )

    writer.writeheader()
    writer.writerows(noise_results)


with open(OUTPUT_CHANNEL_CSV, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(channel_results[0].keys())
    )

    writer.writeheader()
    writer.writerows(channel_results)


# ============================================================
# STEP 9: PLOT NOISE CALIBRATION RESULTS
# ============================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(15, 10),
    constrained_layout=True
)

for pfa in PFA_VALUES:

    selected = [
        r for r in noise_results
        if r["PFA_target"] == pfa
    ]

    snrs = [r["SNR_dB"] for r in selected]

    cell_values = [
        r["Empirical_cell_PFA"] for r in selected
    ]

    profile_values = [
        r["Empirical_profile_PFA"] for r in selected
    ]

    axes[0, 0].plot(
        snrs, cell_values,
        marker="o",
        label=f"PFA {pfa:.0e}"
    )

    axes[0, 1].plot(
        snrs, profile_values,
        marker="o",
        label=f"PFA {pfa:.0e}"
    )

axes[0, 0].set_yscale("log")
axes[0, 0].set_title("Empirical Per-Cell PFA")
axes[0, 0].set_xlabel("SNR (dB)")
axes[0, 0].set_ylabel("False-alarm probability")
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

axes[0, 1].set_title("Empirical Profile-Level PFA")
axes[0, 1].set_xlabel("SNR (dB)")
axes[0, 1].set_ylabel("Probability of at least one false peak")
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

for pfa in PFA_VALUES:

    selected = [
        r for r in noise_results
        if r["PFA_target"] == pfa
    ]

    snrs = [r["SNR_dB"] for r in selected]

    ratios = [
        r["Variance_estimation_ratio"]
        for r in selected
    ]

    axes[1, 0].plot(
        snrs, ratios,
        marker="o",
        label=f"PFA {pfa:.0e}"
    )

axes[1, 0].axhline(
    1.0,
    linestyle="--",
    label="Ideal variance ratio"
)

axes[1, 0].set_title("Noise Variance Estimation Accuracy")
axes[1, 0].set_xlabel("SNR (dB)")
axes[1, 0].set_ylabel("Estimated / Actual variance")
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

for pfa in PFA_VALUES:

    selected = [
        r for r in channel_results
        if r["PFA_target"] == pfa
    ]

    snrs = [r["SNR_dB"] for r in selected]

    false_peaks = [
        r["False_peaks_per_trial"] for r in selected
    ]

    axes[1, 1].plot(
        snrs, false_peaks,
        marker="o",
        label=f"PFA {pfa:.0e}"
    )

axes[1, 1].set_title("Channel + Noise: False Peaks")
axes[1, 1].set_xlabel("SNR (dB)")
axes[1, 1].set_ylabel("False peaks per trial")
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

fig.suptitle(
    "Experiment 17A: False Alarm Calibration",
    fontsize=15
)

plt.savefig(
    OUTPUT_FIGURE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# STEP 10: NOISE-ONLY PROFILE VISUALIZATION
# ============================================================

if example_noise_profile is not None:

    delay_axis_ns = (
        np.arange(NFFT) * dt * 1e9
    )

    normalized_profile = (
        example_noise_profile
        / max(np.max(example_noise_profile), 1e-15)
    )

    profile_db = 20 * np.log10(
        np.maximum(normalized_profile, 1e-12)
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        delay_axis_ns,
        profile_db,
        label="Noise-only delay profile"
    )

    for pfa, threshold in example_noise_thresholds.items():

        threshold_db = 20 * np.log10(
            threshold / max(
                np.max(example_noise_profile),
                1e-15
            )
        )

        ax.axhline(
            threshold_db,
            linestyle=":",
            label=f"PFA {pfa:.0e}"
        )

    ax.axvline(
        MAX_DELAY * 1e9,
        linestyle="--",
        label="Noise estimation boundary"
    )

    ax.axvspan(
        0,
        SEARCH_MAX_DELAY_NS,
        alpha=0.08,
        label="False-alarm test region"
    )

    ax.set_title("Noise-Only Delay Profile and Adaptive Thresholds")
    ax.set_xlabel("Delay (ns)")
    ax.set_ylabel("Normalized magnitude (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.savefig(
        OUTPUT_PROFILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# STEP 11: FINAL SUMMARY
# ============================================================

print("\n========== EXPERIMENT 17A COMPLETED ==========")

print("\nNOISE CALIBRATION RESULTS")

for result in noise_results:

    print(
        f"SNR={result['SNR_dB']:>3} dB | "
        f"PFA={result['PFA_target']:.0e} | "
        f"Variance ratio="
        f"{result['Variance_estimation_ratio']:.4f} | "
        f"Cell PFA={result['Empirical_cell_PFA']:.6e} | "
        f"Profile PFA={result['Empirical_profile_PFA']:.4f}"
    )

print("\nCHANNEL VALIDATION RESULTS")

for result in channel_results:

    print(
        f"SNR={result['SNR_dB']:>3} dB | "
        f"PFA={result['PFA_target']:.0e} | "
        f"Detection="
        f"{result['Detection_probability_percent']:.2f}% | "
        f"Precision="
        f"{result['Precision_percent']:.2f}% | "
        f"False/trial="
        f"{result['False_peaks_per_trial']:.3f}"
    )

print("\nOUTPUT FILES")

print(OUTPUT_NOISE_CSV)
print(OUTPUT_CHANNEL_CSV)
print(OUTPUT_FIGURE)
print(OUTPUT_PROFILE)

print("\nExperiment 17A completed successfully.")
