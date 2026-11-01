
import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")

import sionna.rt
from sionna.rt import (
    load_scene, Transmitter, Receiver, PlanarArray, PathSolver
)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Set up the scene
scene = load_scene(sionna.rt.scene.simple_street_canyon)
scene.frequency = 5.9e9

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

tx = Transmitter(
    name="tx", position=[-20, 0, 2], orientation=[0, 0, 0]
)
rx = Receiver(
    name="rx", position=[20, 0, 2], orientation=[0, 0, 0]
)
scene.add(tx)
scene.add(rx)
tx.look_at(rx)

paths = PathSolver()(scene)

def to_numpy(value):
    if hasattr(value, "numpy"):
        return value.numpy()
    import drjit as dr
    return dr.to_numpy(value)

# Extract delays, coefficients, and validity
tau = to_numpy(paths.tau).reshape(-1)
a_real = to_numpy(paths.a[0]).reshape(-1)
a_imag = to_numpy(paths.a[1]).reshape(-1)
valid = to_numpy(paths.valid).astype(bool).reshape(-1)

tau_ns = tau[valid] * 1e9
a_complex = a_real[valid] + 1j * a_imag[valid]
magnitude = np.abs(a_complex)

# Sort paths by delay
order = np.argsort(tau_ns)
tau_ns = tau_ns[order]
magnitude = magnitude[order]

# Convert magnitude to relative dB
magnitude_db = 20 * np.log10(
    np.maximum(magnitude / np.max(magnitude), 1e-12)
)

# Plot path magnitudes at their delays
plt.figure(figsize=(10, 5))
plt.stem(tau_ns, magnitude_db)
plt.xlabel("Propagation delay (ns)")
plt.ylabel("Relative path magnitude (dB)")
plt.title("Sionna RT Channel Impulse Response — 5.9 GHz")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("cir_5p9GHz.png", dpi=160)
plt.close()

print("Valid paths:", len(tau_ns))
print("Strongest path delay:", tau_ns[np.argmax(magnitude)], "ns")
print("Saved: cir_5p9GHz.png")
