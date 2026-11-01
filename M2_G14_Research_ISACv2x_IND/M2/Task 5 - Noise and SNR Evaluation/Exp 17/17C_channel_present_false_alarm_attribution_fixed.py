import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")

import sionna.rt
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver

import numpy as np
import drjit as dr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

# ============================================================
# EXPERIMENT 17C
# CHANNEL-PRESENT FALSE-ALARM ATTRIBUTION
# ============================================================

FC = 5.9e9
FS = 100e6
NFFT = 1024
CP_LEN = 128
MAX_DELAY = 600e-9
KERNEL_RADIUS = 16

SNR_VALUES = [-5, 0, 5, 10, 15, 20]
PFA_VALUES = [1e-2, 1e-3, 1e-4]

NUM_CHANNEL_TRIALS = 1000
BATCH_SIZE = 250
RNG_SEED = 2031

SEARCH_MAX_DELAY_NS = 400.0
RAY_GROUP_GAP_NS = 10.0
MATCH_TOLERANCE_NS = 10.0
PROFILE_PEAK_FLOOR_DB = -45.0

# Secondary deterministic features are searched at a lower floor.
# They are only labelled deterministic sidelobes if they are stable local
# maxima in the noise-free composite response and are not ray-group supported.
SIDELobe_FLOOR_DB = -60.0
SIDELobe_MATCH_TOLERANCE_NS = 3.0

OUTPUT_RAYS_CSV = "experiment17C_ray_ground_truth.csv"
OUTPUT_PROFILE_CSV = "experiment17C_profile_ground_truth.csv"
OUTPUT_RESULTS_CSV = "experiment17C_detection_attribution.csv"
OUTPUT_SUMMARY_CSV = "experiment17C_false_alarm_summary.csv"
OUTPUT_FIGURE = "experiment17C_false_alarm_attribution.png"
OUTPUT_PROFILE_FIGURE = "experiment17C_reference_features.png"


def to_numpy(value):
    if hasattr(value, "numpy"):
        return value.numpy()
    return dr.to_numpy(value)


def estimate_noise_variance(h_estimated, noise_start):
    noise_region = h_estimated[noise_start:]
    noise_power = np.abs(noise_region) ** 2
    return max(float(np.median(noise_power) / np.log(2)), 1e-30)


def calculate_threshold(sigma_squared, pfa):
    return np.sqrt(-sigma_squared * np.log(pfa))


def detect_peaks(magnitude, threshold, end_index=None):
    if end_index is None:
        end_index = len(magnitude)
    end_index = min(end_index, len(magnitude))
    indices = []
    for i in range(1, end_index - 1):
        if (magnitude[i] >= threshold
                and magnitude[i] > magnitude[i - 1]
                and magnitude[i] >= magnitude[i + 1]):
            indices.append(i)
    return np.asarray(indices, dtype=int)


def refine_peak_position(magnitude, peak_index):
    if peak_index <= 0 or peak_index >= len(magnitude) - 1:
        return float(peak_index)

    eps = 1e-15
    left = 20 * np.log10(max(magnitude[peak_index - 1], eps))
    centre = 20 * np.log10(max(magnitude[peak_index], eps))
    right = 20 * np.log10(max(magnitude[peak_index + 1], eps))
    denominator = left - 2 * centre + right

    if abs(denominator) < 1e-12:
        return float(peak_index)

    offset = 0.5 * (left - right) / denominator
    return float(peak_index) + float(np.clip(offset, -0.5, 0.5))


def greedy_one_to_one_match(reference_delays, detected_delays, tolerance_ns):
    """Return one-to-one (reference index, detection index, signed error) matches."""
    candidates = []
    for ref_i, ref_delay in enumerate(reference_delays):
        for det_i, det_delay in enumerate(detected_delays):
            error = float(det_delay - ref_delay)
            if abs(error) <= tolerance_ns:
                candidates.append((abs(error), ref_i, det_i, error))

    candidates.sort(key=lambda item: item[0])
    used_ref, used_det, matches = set(), set(), []

    for _, ref_i, det_i, error in candidates:
        if ref_i not in used_ref and det_i not in used_det:
            used_ref.add(ref_i)
            used_det.add(det_i)
            matches.append((ref_i, det_i, error))

    return matches, used_ref, used_det


def group_rays(ray_delays_ns, ray_coefficients, max_gap_ns):
    """Cluster adjacent ray delays when the inter-ray gap is <= max_gap_ns."""
    order = np.argsort(ray_delays_ns)
    delays = np.asarray(ray_delays_ns)[order]
    coeffs = np.asarray(ray_coefficients)[order]

    groups = []
    current = [0]
    for i in range(1, len(delays)):
        if delays[i] - delays[i - 1] <= max_gap_ns:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)

    output = []
    for group_id, members in enumerate(groups, start=1):
        d = delays[members]
        a = coeffs[members]
        weights = np.abs(a) ** 2
        if np.sum(weights) > 0:
            weighted_delay = float(np.sum(d * weights) / np.sum(weights))
        else:
            weighted_delay = float(np.mean(d))

        output.append({
            "group_id": group_id,
            "ray_count": len(members),
            "ray_delays_ns": d.tolist(),
            "group_delay_ns": weighted_delay,
            "group_power_sum": float(np.sum(weights))
        })
    return output


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 1. LOAD SIONNA RT SCENE AND EXTRACT ALL VALID RAYS
# ============================================================

print("\nLoading Sionna RT scene...")
scene = load_scene(sionna.rt.scene.simple_street_canyon)
scene.frequency = FC

scene.tx_array = PlanarArray(
    num_rows=1, num_cols=1,
    vertical_spacing=0.5, horizontal_spacing=0.5,
    pattern="iso", polarization="V"
)
scene.rx_array = PlanarArray(
    num_rows=1, num_cols=1,
    vertical_spacing=0.5, horizontal_spacing=0.5,
    pattern="iso", polarization="V"
)

tx = Transmitter(name="tx", position=[-20, 0, 2], orientation=[0, 0, 0])
rx = Receiver(name="rx", position=[20, 0, 2], orientation=[0, 0, 0])
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
coefficients = a_real[valid] + 1j * a_imag[valid]
order = np.argsort(tau)
tau = tau[order]
coefficients = coefficients[order]
ray_delays_ns = tau * 1e9

print(f"Backend: {mi.variant()}")
print(f"Carrier frequency: {FC / 1e9:.2f} GHz")
print(f"Valid ray entries: {len(ray_delays_ns)}")

ray_rows = []
for i, (delay, coeff) in enumerate(zip(ray_delays_ns, coefficients), start=1):
    ray_rows.append({
        "Ray_ID": i,
        "Delay_ns": float(delay),
        "Coefficient_real": float(np.real(coeff)),
        "Coefficient_imag": float(np.imag(coeff)),
        "Coefficient_magnitude": float(np.abs(coeff)),
        "Coefficient_phase_deg": float(np.angle(coeff, deg=True))
    })

ray_groups = group_rays(ray_delays_ns, coefficients, RAY_GROUP_GAP_NS)
print(f"Ray groups using {RAY_GROUP_GAP_NS:.1f} ns adjacent-gap rule: {len(ray_groups)}")

group_rows = []
for group in ray_groups:
    group_rows.append({
        "Group_ID": group["group_id"],
        "Ray_count": group["ray_count"],
        "Ray_delays_ns": ";".join(f"{x:.6f}" for x in group["ray_delays_ns"]),
        "Power_weighted_group_delay_ns": group["group_delay_ns"],
        "Sum_ray_power": group["group_power_sum"]
    })

# ============================================================
# 2. CONSTRUCT THE SAME DISCRETE CHANNEL AS EXPERIMENT 17A
# ============================================================

dt = 1.0 / FS
channel_length = int(np.ceil(MAX_DELAY / dt))
h = np.zeros(channel_length, dtype=np.complex128)

for delay, coefficient in zip(tau, coefficients):
    fractional_index = delay / dt
    centre = int(np.floor(fractional_index))
    indices = np.arange(
        centre - KERNEL_RADIUS,
        centre + KERNEL_RADIUS + 2
    )
    offset = indices - fractional_index
    weights = np.sinc(offset) * np.sinc(offset / (KERNEL_RADIUS + 1))
    keep = (
        (np.abs(offset) < KERNEL_RADIUS + 1)
        & (indices >= 0)
        & (indices < channel_length)
    )
    h[indices[keep]] += coefficient * weights[keep]

# ============================================================
# 3. GENERATE OFDM REFERENCE SIGNAL AND CHANNEL RESPONSE
# ============================================================

rng = np.random.default_rng(RNG_SEED)
bits_i = rng.integers(0, 2, NFFT)
bits_q = rng.integers(0, 2, NFFT)
X = ((1 - 2 * bits_i) + 1j * (1 - 2 * bits_q)) / np.sqrt(2)

x = np.fft.ifft(X, n=NFFT)
x_cp = np.concatenate((x[-CP_LEN:], x))
y_clean_full = np.convolve(x_cp, h, mode="full")

if CP_LEN < len(h) - 1:
    raise RuntimeError("Cyclic prefix is insufficient for the discrete channel.")

y_clean = y_clean_full[CP_LEN:CP_LEN + NFFT]
signal_power = float(np.mean(np.abs(y_clean) ** 2))

H_clean = np.fft.fft(y_clean) / X
h_clean_estimate = np.fft.ifft(H_clean)
noise_start = int(np.ceil(MAX_DELAY * FS))

search_end = int(np.floor(SEARCH_MAX_DELAY_NS / (dt * 1e9))) + 1
delay_axis_ns = np.arange(NFFT) * dt * 1e9

if search_end >= noise_start:
    raise RuntimeError("Search region overlaps the noise-estimation region.")

print(f"Received signal power: {signal_power:.6e}")
print(f"Noise estimation starts at sample: {noise_start}")
print(f"Search interval: 0 to {SEARCH_MAX_DELAY_NS:.1f} ns")

# ============================================================
# 4. BUILD NOISE-FREE COMPOSITE-PROFILE GROUND TRUTH
# ============================================================

profile_magnitude = np.abs(h_clean_estimate)
search_profile = profile_magnitude[:search_end]
profile_peak_floor = np.max(search_profile) * 10 ** (PROFILE_PEAK_FLOOR_DB / 20)
secondary_peak_floor = np.max(search_profile) * 10 ** (SIDELobe_FLOOR_DB / 20)

# Primary peaks: used for multipath detection probability and delay accuracy.
profile_peak_indices = detect_peaks(
    profile_magnitude, profile_peak_floor, end_index=search_end
)
profile_peak_positions = np.array([
    refine_peak_position(profile_magnitude, idx)
    for idx in profile_peak_indices
])
profile_peak_delays_ns = profile_peak_positions * dt * 1e9
profile_peak_amplitudes = profile_magnitude[profile_peak_indices]

# Expanded noise-free feature list: lower-floor stable local maxima are
# candidate deterministic secondary features/sidelobes. They are not counted
# as primary multipath targets unless supported by a ray group.
secondary_peak_indices = detect_peaks(
    profile_magnitude, secondary_peak_floor, end_index=search_end
)
secondary_peak_positions = np.array([
    refine_peak_position(profile_magnitude, idx)
    for idx in secondary_peak_indices
])
secondary_peak_delays_ns = secondary_peak_positions * dt * 1e9
secondary_peak_amplitudes = profile_magnitude[secondary_peak_indices]

# Build a feature catalogue with explicit ray-supported / deterministic-secondary
# labels. A secondary feature must be a noise-free local maximum and not lie
# within SIDELobe_MATCH_TOLERANCE_NS of a ray-group delay.
feature_rows = []
feature_delays_ns = []
feature_types = []
for feature_id, (delay_ns, amplitude) in enumerate(
        zip(secondary_peak_delays_ns, secondary_peak_amplitudes), start=1):
    distances = np.array([
        abs(delay_ns - group["group_delay_ns"]) for group in ray_groups
    ])
    nearest_group_idx = int(np.argmin(distances))
    nearest_distance = float(distances[nearest_group_idx])
    if nearest_distance <= MATCH_TOLERANCE_NS:
        feature_type = "ray-supported-primary"
        group_id = ray_groups[nearest_group_idx]["group_id"]
    else:
        feature_type = "deterministic-secondary-feature"
        group_id = ""
    feature_delays_ns.append(float(delay_ns))
    feature_types.append(feature_type)
    feature_rows.append({
        "Feature_ID": feature_id,
        "Delay_ns": float(delay_ns),
        "Magnitude": float(amplitude),
        "Relative_magnitude_dB": float(
            20 * np.log10(max(amplitude / np.max(search_profile), 1e-15))
        ),
        "Classification": feature_type,
        "Nearest_ray_group_ID": group_id,
        "Distance_to_nearest_ray_group_ns": nearest_distance
    })

# Primary target catalogue is the 17B -45 dB reference, independently matched
# to ray groups. Secondary features are tracked separately and never inflate TP.
profile_rows = [r for r in feature_rows if r["Relative_magnitude_dB"] >= PROFILE_PEAK_FLOOR_DB]
profile_peak_delays_ns = np.array([r["Delay_ns"] for r in profile_rows], dtype=float)
profile_peak_types = [r["Classification"] for r in profile_rows]
profile_peak_amplitudes = np.array([r["Magnitude"] for r in profile_rows], dtype=float)

# Delay regions around all noise-free local maxima are used to attribute noisy
# detections. Match to a primary feature first, then a secondary deterministic
# feature; anything else is noise/unmodelled. This avoids calling every
# unmatched detection a noise false alarm without checking the clean channel.
print("\n========== GROUND-TRUTH SUMMARY ==========")
print(f"Valid ray entries: {len(ray_delays_ns)}")
print(f"Ray groups: {len(ray_groups)}")
print(f"Noise-free composite profile peaks above "
      f"{PROFILE_PEAK_FLOOR_DB:.1f} dB floor: {len(profile_peak_delays_ns)}")
print(f"Expanded local maxima above {SIDELobe_FLOOR_DB:.1f} dB floor: {len(secondary_peak_delays_ns)}")

print("\nRAY GROUPS")
for group in ray_groups:
    print(
        f"Group {group['group_id']}: "
        f"rays={group['ray_count']}, "
        f"delays={group['ray_delays_ns']}, "
        f"weighted delay={group['group_delay_ns']:.3f} ns"
    )

print("\nNOISE-FREE COMPOSITE PROFILE PEAKS")
for row in profile_rows:
    print(
        f"Peak {row['Feature_ID']}: "
        f"{row['Delay_ns']:.3f} ns | "
        f"{row['Relative_magnitude_dB']:.2f} dB | "
        f"{row['Classification']}"
    )

# ============================================================
# 5. CHANNEL-PRESENT MONTE CARLO CLASSIFICATION
# ============================================================

print("\n========== CHANNEL-PRESENT CLASSIFICATION ==========")

trial_rows = []
summary_rows = []

for snr_db in SNR_VALUES:
    noise_variance = signal_power / (10 ** (snr_db / 10))
    print(f"\nProcessing SNR = {snr_db} dB")

    for pfa in PFA_VALUES:
        total_detected = 0
        profile_tp = 0
        profile_fn = 0
        ray_supported_detected = 0
        unmatched_detected = 0
        deterministic_secondary_detected = 0
        noise_or_unmodelled_detected = 0
        delay_errors = []
        thresholds = []

        for trial in range(NUM_CHANNEL_TRIALS):
            noise = np.sqrt(noise_variance / 2) * (
                rng.standard_normal(NFFT)
                + 1j * rng.standard_normal(NFFT)
            )
            y_noisy = y_clean + noise
            H_estimated = np.fft.fft(y_noisy) / X
            h_estimated = np.fft.ifft(H_estimated)
            magnitude = np.abs(h_estimated)

            sigma_hat = estimate_noise_variance(
                h_estimated, noise_start
            )
            threshold = calculate_threshold(sigma_hat, pfa)
            thresholds.append(float(threshold))

            peak_indices = detect_peaks(
                magnitude, threshold, end_index=search_end
            )
            refined_positions = np.array([
                refine_peak_position(magnitude, idx)
                for idx in peak_indices
            ])
            detected_delays_ns = refined_positions * dt * 1e9

            matches, matched_profile, matched_detections = (
                greedy_one_to_one_match(
                    profile_peak_delays_ns,
                    detected_delays_ns,
                    MATCH_TOLERANCE_NS
                )
            )

            trial_profile_tp = len(matches)
            trial_profile_fn = len(profile_peak_delays_ns) - trial_profile_tp
            trial_unmatched = len(detected_delays_ns) - len(matched_detections)

            trial_ray_supported = 0
            trial_secondary = 0
            trial_noise_unmodelled = 0
            trial_errors = []
            primary_detection_indices = set(matched_detections)
            unmatched_indices = [i for i in range(len(detected_delays_ns))
                                 if i not in primary_detection_indices]
            secondary_feature_indices = [i for i, t in enumerate(feature_types)
                                          if t == "deterministic-secondary-feature"]
            secondary_feature_delays = secondary_peak_delays_ns[secondary_feature_indices]
            unmatched_delays = detected_delays_ns[unmatched_indices]
            secondary_matches, _, secondary_local_detection_indices = greedy_one_to_one_match(
                secondary_feature_delays, unmatched_delays, SIDELobe_MATCH_TOLERANCE_NS
            )
            secondary_detection_indices = {
                unmatched_indices[local_i] for local_i in secondary_local_detection_indices
            }
            secondary_detection_to_feature = {
                unmatched_indices[det_i]: secondary_feature_indices[ref_i]
                for ref_i, det_i, _ in secondary_matches
            }
            trial_secondary = len(secondary_detection_indices)
            trial_noise_unmodelled = len(detected_delays_ns) - len(primary_detection_indices) - trial_secondary

            for ref_i, det_i, error in matches:
                if profile_peak_types[ref_i] == "ray-supported-primary":
                    trial_ray_supported += 1
                trial_errors.append(error)

            profile_tp += trial_profile_tp
            profile_fn += trial_profile_fn
            ray_supported_detected += trial_ray_supported
            deterministic_secondary_detected += trial_secondary
            noise_or_unmodelled_detected += trial_noise_unmodelled
            unmatched_detected += trial_unmatched
            total_detected += len(detected_delays_ns)
            delay_errors.extend(trial_errors)

            # Store per-trial classifications for auditability.
            for det_i, delay_ns in enumerate(detected_delays_ns):
                matched_ref_i = next(
                    (ref_i for ref_i, d_i, _ in matches if d_i == det_i),
                    None
                )
                if matched_ref_i is None:
                    if det_i in secondary_detection_indices:
                        classification = "deterministic-secondary-feature"
                        matched_profile_delay = float(secondary_peak_delays_ns[
                            secondary_detection_to_feature[det_i]
                        ])
                    else:
                        classification = "noise-or-unmodelled-channel-detection"
                        matched_profile_delay = ""
                    error_ns = ""
                else:
                    classification = profile_peak_types[matched_ref_i]
                    matched_profile_delay = float(profile_peak_delays_ns[matched_ref_i])
                    error_ns = float(delay_ns - profile_peak_delays_ns[matched_ref_i])

                trial_rows.append({
                    "SNR_dB": snr_db,
                    "Target_PFA": pfa,
                    "Trial": trial + 1,
                    "Detection_ID": det_i + 1,
                    "Detected_delay_ns": float(delay_ns),
                    "Detection_magnitude": float(magnitude[peak_indices[det_i]]),
                    "Adaptive_threshold": float(threshold),
                    "Classification": classification,
                    "Matched_profile_peak_delay_ns": matched_profile_delay,
                    "Delay_error_ns": error_ns
                })

        total_profile_references = NUM_CHANNEL_TRIALS * len(profile_peak_delays_ns)
        profile_detection_probability = (
            profile_tp / total_profile_references
            if total_profile_references else np.nan
        )
        profile_precision = profile_tp / max(profile_tp + unmatched_detected, 1)
        profile_recall = profile_tp / max(profile_tp + profile_fn, 1)
        profile_f1 = (
            2 * profile_precision * profile_recall
            / max(profile_precision + profile_recall, 1e-15)
        )

        ray_group_recall = (
            ray_supported_detected
            / max(NUM_CHANNEL_TRIALS * len(ray_groups), 1)
        )

        errors = np.asarray(delay_errors, dtype=float)
        delay_mae = float(np.mean(np.abs(errors))) if len(errors) else np.nan
        delay_rmse = float(np.sqrt(np.mean(errors ** 2))) if len(errors) else np.nan

        false_unmatched_per_trial = unmatched_detected / NUM_CHANNEL_TRIALS
        ray_supported_per_trial = ray_supported_detected / NUM_CHANNEL_TRIALS

        result = {
            "SNR_dB": snr_db,
            "Target_PFA": pfa,
            "Noise_variance": float(noise_variance),
            "Mean_adaptive_threshold": float(np.mean(thresholds)),
            "Noise_free_profile_reference_peaks": len(profile_peak_delays_ns),
            "Ray_groups": len(ray_groups),
            "Total_detections": total_detected,
            "Profile_reference_TP": profile_tp,
            "Profile_reference_FN": profile_fn,
            "Unmatched_detections": unmatched_detected,
            "Ray_supported_detections_per_trial": ray_supported_per_trial,
            "Deterministic_secondary_detections_per_trial": deterministic_secondary_detected / NUM_CHANNEL_TRIALS,
            "Noise_or_unmodelled_detections_per_trial": noise_or_unmodelled_detected / NUM_CHANNEL_TRIALS,
            "Unmatched_detections_per_trial": false_unmatched_per_trial,
            "Profile_detection_probability_percent":
                profile_detection_probability * 100,
            "Profile_precision_percent": profile_precision * 100,
            "Profile_recall_percent": profile_recall * 100,
            "Profile_F1_score": profile_f1,
            "Ray_group_recall_percent": ray_group_recall * 100,
            "Delay_MAE_ns": delay_mae,
            "Delay_RMSE_ns": delay_rmse,
            "Channel_trials": NUM_CHANNEL_TRIALS,
            "Match_tolerance_ns": MATCH_TOLERANCE_NS,
            "Profile_peak_floor_dB": PROFILE_PEAK_FLOOR_DB
        }
        summary_rows.append(result)

        print(
            f"PFA={pfa:.0e} | "
            f"Profile detection={profile_detection_probability*100:.2f}% | "
            f"Precision={profile_precision*100:.2f}% | "
            f"F1={profile_f1:.3f} | "
            f"Ray-supported/trial={ray_supported_per_trial:.3f} | "
            f"Secondary/trial={deterministic_secondary_detected/NUM_CHANNEL_TRIALS:.3f} | "
            f"Noise/unmodelled/trial={noise_or_unmodelled_detected/NUM_CHANNEL_TRIALS:.3f} | "
            f"RMSE={delay_rmse:.4f} ns"
        )

# ============================================================
# 6. SAVE GROUND TRUTH AND DETECTION RESULTS
# ============================================================

write_csv(OUTPUT_RAYS_CSV, ray_rows)
write_csv(OUTPUT_PROFILE_CSV, feature_rows)
write_csv(OUTPUT_RESULTS_CSV, trial_rows)
write_csv(OUTPUT_SUMMARY_CSV, summary_rows)

# ============================================================
# 7. PLOT NOISE-FREE GROUND-TRUTH PROFILE
# ============================================================

profile_db = 20 * np.log10(
    np.maximum(profile_magnitude / max(np.max(search_profile), 1e-15), 1e-12)
)

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(
    delay_axis_ns[:search_end],
    profile_db[:search_end],
    label="Noise-free composite channel"
)

for group in ray_groups:
    ax.axvline(
        group["group_delay_ns"],
        linestyle=":",
        alpha=0.65,
        label=f"Ray group {group['group_id']}"
    )

for row in feature_rows:
    ax.scatter(
        row["Delay_ns"],
        row["Relative_magnitude_dB"],
        marker="x" if row["Classification"] == "ray-supported-primary" else "s",
        s=65,
        label=row["Classification"] if row["Feature_ID"] == 1 else None
    )

ax.set_title("Experiment 17C: Noise-Free Reference Feature Catalogue")
ax.set_xlabel("Propagation delay (ns)")
ax.set_ylabel("Normalized magnitude (dB)")
ax.set_xlim(0, SEARCH_MAX_DELAY_NS)
ax.set_ylim(PROFILE_PEAK_FLOOR_DB - 5, 3)
ax.grid(True, alpha=0.3)
ax.legend(ncol=2, fontsize=8)
fig.tight_layout()
fig.savefig(OUTPUT_PROFILE_FIGURE, dpi=300, bbox_inches="tight")
plt.close(fig)

# ============================================================
# 8. PLOT CLASSIFICATION METRICS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

for pfa in PFA_VALUES:
    selected = [r for r in summary_rows if r["Target_PFA"] == pfa]
    snrs = [r["SNR_dB"] for r in selected]

    axes[0, 0].plot(
        snrs,
        [r["Profile_detection_probability_percent"] for r in selected],
        marker="o", label=f"PFA {pfa:.0e}"
    )
    axes[0, 1].plot(
        snrs,
        [r["Profile_precision_percent"] for r in selected],
        marker="o", label=f"PFA {pfa:.0e}"
    )
    axes[1, 0].plot(
        snrs,
        [r["Unmatched_detections_per_trial"] for r in selected],
        marker="o", label=f"PFA {pfa:.0e}"
    )
    axes[1, 1].plot(
        snrs,
        [r["Delay_RMSE_ns"] for r in selected],
        marker="o", label=f"PFA {pfa:.0e}"
    )

axes[0, 0].set_title("Composite-Profile Detection Probability")
axes[0, 0].set_ylabel("Detection probability (%)")
axes[0, 1].set_title("Composite-Profile Precision")
axes[0, 1].set_ylabel("Precision (%)")
axes[1, 0].set_title("Unmatched Detections per Trial")
axes[1, 0].set_ylabel("Detections per trial")
axes[1, 1].set_title("Delay Estimation RMSE")
axes[1, 1].set_ylabel("RMSE (ns)")

for ax in axes.flat:
    ax.set_xlabel("SNR (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend()

fig.suptitle("Experiment 17C: Channel-Present False-Alarm Attribution", fontsize=15)
fig.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")
plt.close(fig)

# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n========== EXPERIMENT 17C COMPLETED ==========")
print(f"Ray entries: {len(ray_delays_ns)}")
print(f"Ray groups: {len(ray_groups)}")
print(f"Primary noise-free peaks: {len(profile_peak_delays_ns)}")
print(f"Expanded noise-free local maxima: {len(secondary_peak_delays_ns)}")
print(f"Deterministic secondary features: {sum(t == 'deterministic-secondary-feature' for t in feature_types)}")
print("\nSNR | PFA | Profile detection | Precision | F1 | "
      "Ray/trial | Sidelobe/trial | Unmatched/trial | RMSE(ns)")

for r in summary_rows:
    print(
        f"{r['SNR_dB']:>3} | {r['Target_PFA']:.0e} | "
        f"{r['Profile_detection_probability_percent']:6.2f}% | "
        f"{r['Profile_precision_percent']:6.2f}% | "
        f"{r['Profile_F1_score']:.3f} | "
        f"{r['Ray_supported_detections_per_trial']:.3f} | "
        f"{r['Deterministic_secondary_detections_per_trial']:.3f} | "
        f"{r['Noise_or_unmodelled_detections_per_trial']:.3f} | "
        f"{r['Delay_RMSE_ns']:.4f}"
    )

print("\nOUTPUT FILES")
for filename in [
    OUTPUT_RAYS_CSV,
    OUTPUT_PROFILE_CSV,
    OUTPUT_RESULTS_CSV,
    OUTPUT_SUMMARY_CSV,
    OUTPUT_FIGURE,
    OUTPUT_PROFILE_FIGURE
]:
    print(filename)

print("\nExperiment 17C completed successfully.")
