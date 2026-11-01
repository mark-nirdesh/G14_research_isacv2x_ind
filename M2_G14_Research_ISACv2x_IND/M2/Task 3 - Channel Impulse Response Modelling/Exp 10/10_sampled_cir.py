
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

C = 299_792_458.0
SAMPLE_RATE = 100e6  # 100 MHz, for an illustrative sampled channel
MAX_DELAY = 400e-9   # 400 ns

def to_numpy(value):
    if hasattr(value, "numpy"):
        return value.numpy()
    return dr.to_numpy(value)

# Recreate the baseline scene and radio link
scene = load_scene(sionna.rt.scene.simple_street_canyon)
scene.frequency = 5.9e9

scene.tx_array = PlanarArray(
    num_rows=1, num_cols=1,
    vertical_spacing=0.5, horizontal_spacing=0.5,
    pattern="iso", polarization="V",
)
scene.rx_array = PlanarArray(
    num_rows=1, num_cols=1,
    vertical_spacing=0.5, horizontal_spacing=0.5,
    pattern="iso", polarization="V",
)

tx = Transmitter("tx", [-20, 0, 2], [0, 0, 0])
rx = Receiver("rx", [20, 0, 2], [0, 0, 0])
scene.add(tx)
scene.add(rx)
tx.look_at(rx)

paths = PathSolver()(scene)

# Extract the path delays and complex coefficients
tau = to_numpy(paths.tau).reshape(-1)
a_real = to_numpy(paths.a[0]).reshape(-1)
a_imag = to_numpy(paths.a[1]).reshape(-1)
valid = to_numpy(paths.valid).astype(bool).reshape(-1)

tau = tau[valid]
a = a_real[valid] + 1j * a_imag[valid]

# Build a discrete time grid
dt = 1.0 / SAMPLE_RATE
num_samples = int(np.ceil(MAX_DELAY / dt))
time = np.arange(num_samples) * dt
h = np.zeros(num_samples, dtype=np.complex128)

# Bin each path to the nearest sample.
# This is a simple approximation, not a fractional-delay channel model.
indices = np.rint(tau / dt).astype(int)
keep = (indices >= 0) & (indices < num_samples)
np.add.at(h, indices[keep], a[keep])

# Plot relative sampled CIR magnitude
h_db = 20 * np.log10(
    np.maximum(np.abs(h) / max(np.max(np.abs(h)), 1e-15), 1e-12)
)

plt.figure(figsize=(10, 5))
plt.stem(time * 1e9, h_db)
plt.xlabel("Delay (ns)")
plt.ylabel("Relative sampled CIR magnitude (dB)")
plt.title("Approximate Sampled CIR at 5.9 GHz")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("sampled_cir_5p9GHz.png", dpi=160)
plt.close()

print("Carrier frequency: 5.9 GHz")
print(f"Sample rate: {SAMPLE_RATE/1e6:.1f} MHz")
print(f"Sample interval: {dt*1e9:.2f} ns")
print(f"Valid path entries: {len(tau)}")
print(f"Occupied CIR samples: {np.count_nonzero(h)}")
print("Saved: sampled_cir_5p9GHz.png")
print("Note: nearest-sample binning approximates fractional path delays.")
