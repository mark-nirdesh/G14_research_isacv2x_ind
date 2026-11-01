
import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")

import sionna.rt
from sionna.rt import (
    load_scene, Transmitter, Receiver, PlanarArray, PathSolver
)
import numpy as np

C = 299_792_458.0  # Speed of light (m/s)

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
    """Convert supported tensor values to NumPy for inspection."""
    if hasattr(value, "numpy"):
        return value.numpy()
    try:
        import drjit as dr
        return dr.to_numpy(value)
    except Exception:
        return np.asarray(value)

# Path delays
tau = to_numpy(paths.tau).astype(float).reshape(-1)
distance = C * tau

print(f"Carrier frequency: {float(scene.frequency[0])/1e9:.2f} GHz")
print(f"Path entries: {len(tau)}")
print("\nPath delay and equivalent propagation distance:")

for i, (delay, dist) in enumerate(zip(tau, distance), start=1):
    print(
        f"Path {i:02d}: delay = {delay*1e9:9.3f} ns, "
        f"c × delay = {dist:8.3f} m"
    )

print("\nCoefficient structure:")
a = paths.a
print("Type:", type(a).__name__)

if isinstance(a, tuple):
    print("Tuple length:", len(a))
    for i, component in enumerate(a):
        print(f"Component {i}: type={type(component).__name__}")
        try:
            arr = to_numpy(component)
            print("  Shape:", arr.shape)
            print("  First values:", arr.reshape(-1)[:8])
        except Exception as exc:
            print("  Conversion error:", exc)
else:
    try:
        arr = to_numpy(a)
        print("Shape:", arr.shape)
        print("First values:", arr.reshape(-1)[:8])
    except Exception as exc:
        print("Conversion error:", exc)

# Inspect path validity
try:
    valid = to_numpy(paths.valid)
    print("\nValidity tensor shape:", valid.shape)
    print("Valid entries:", int(np.count_nonzero(valid)))
except Exception as exc:
    print("\nCould not inspect validity tensor:", exc)
