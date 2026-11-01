
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

# --------------------------------------------------
# Configuration
# --------------------------------------------------
C = 299_792_458.0
FC = 5.9e9
FS = 100e6
DT = 1.0 / FS
MAX_DELAY = 400e-9
KERNEL_RADIUS = 8  # Samples on either side of each fractional delay

def to_numpy(value):
    if hasattr(value, "numpy"):
        return value.numpy()
    return dr.to_numpy(value)

# --------------------------------------------------
# 1. Recreate the ray-tracing experiment
# --------------------------------------------------
scene = load_scene(sionna.rt.scene.simple_street_canyon)
scene.frequency = FC

scene.tx_array = PlanarArray(
    num_rows=1, num_cols=1,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="iso",
    polarization="V",
)
scene.rx_array = PlanarArray(
    num_rows=1, num_cols=1,
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

paths = PathSolver()(scene)

# --------------------------------------------------
# 2. Extract path delays and complex coefficients
# --------------------------------------------------
tau = to_numpy(paths.tau).reshape(-1)
a_real = to_numpy(paths.a[0]).reshape(-1)
a_imag = to_numpy(paths.a[1]).reshape(-1)
valid = to_numpy(paths.valid).astype(bool).reshape(-1)

tau = tau[valid]
a = a_real[valid] + 1j * a_imag[valid]

# Sort by delay
order = np.argsort(tau)
tau = tau[order]
a = a[order]

# --------------------------------------------------
# 3. Ideal ray-delay profile
# --------------------------------------------------
# Plot each ray at its actual delay. No delay rounding.
ray_magnitude = np.abs(a)
ray_db = 20 * np.log10(
    np.maximum(ray_magnitude / np.max(ray_magnitude), 1e-12)
)

# --------------------------------------------------
# 4. Fractional-delay sampled approximation
# --------------------------------------------------
num_samples = int(np.ceil(MAX_DELAY / DT))
time = np.arange(num_samples) * DT
h = np.zeros(num_samples, dtype=np.complex128)

for delay, coefficient in zip(tau, a):
    # Fractional sample position; do not round the path delay
    u = delay / DT
    center = int(np.floor(u))

    indices = np.arange(
        center - KERNEL_RADIUS,
        center + KERNEL_RADIUS + 2,
    )

    # Lanczos-windowed sinc interpolation kernel
    offset = indices - u
    weights = np.sinc(offset) * np.sinc(
        offset / (KERNEL_RADIUS + 1)
    )

    inside_kernel = np.abs(offset) < KERNEL_RADIUS + 1
    inside_grid = (indices >= 0) & (indices < num_samples)
    keep = inside_kernel & inside_grid

    h[indices[keep]] += coefficient * weights[keep]

# Relative magnitude for display
h_max = np.max(np.abs(h))
if h_max == 0:
    raise RuntimeError("The sampled channel is zero.")

h_db = 20 * np.log10(
    np.maximum(np.abs(h) / h_max, 1e-4)
)

# --------------------------------------------------
# 5. Plot both representations
# --------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(11, 8))

axes[0].stem(tau * 1e9, ray_db)
axes[0].set_title("Ideal Ray-Delay Profile — 5.9 GHz")
axes[0].set_xlabel("Actual propagation delay (ns)")
axes[0].set_ylabel("Relative individual-path magnitude (dB)")
axes[0].set_xlim(0, MAX_DELAY * 1e9)
axes[0].set_ylim(-80, 5)
axes[0].grid(True, alpha=0.3)

axes[1].stem(time * 1e9, h_db)
axes[1].set_title("Fractional-Delay Sampled CIR Approximation")
axes[1].set_xlabel("Time (ns)")
axes[1].set_ylabel("Relative sampled magnitude (dB)")
axes[1].set_xlim(0, MAX_DELAY * 1e9)
axes[1].set_ylim(-80, 5)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("corrected_cir_5p9GHz.png", dpi=160)
plt.close()

# --------------------------------------------------
# 6. Report numerical checks
# --------------------------------------------------
print(f"Carrier frequency: {FC / 1e9:.2f} GHz")
print(f"Sampling rate: {FS / 1e6:.1f} MHz")
print(f"Sample interval: {DT * 1e9:.2f} ns")
print(f"Valid ray entries: {len(tau)}")
print(f"Delay range: {tau.min()*1e9:.3f} to "
      f"{tau.max()*1e9:.3f} ns")
print(f"Sampled CIR grid: {num_samples} samples")

print("\nFirst five paths:")
for i in range(min(5, len(tau))):
    print(
        f"  {i+1}: delay={tau[i]*1e9:.3f} ns, "
        f"magnitude={abs(a[i]):.6e}, "
        f"fractional sample={tau[i]/DT:.3f}"
    )

print("\nSaved: corrected_cir_5p9GHz.png")
print("Note: the sampled CIR uses a finite Lanczos-windowed sinc kernel.")
