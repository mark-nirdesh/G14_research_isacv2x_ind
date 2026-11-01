
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

C = 299_792_458.0

def to_numpy(value):
    if isinstance(value, (tuple, list)):
        return [to_numpy(v) for v in value]
    if hasattr(value, "numpy"):
        return value.numpy()
    try:
        return dr.to_numpy(value)
    except Exception:
        return np.asarray(value)

# Rebuild the same experiment
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

tx = Transmitter(
    name="tx", position=[-20, 0, 2],
    orientation=[0, 0, 0],
)
rx = Receiver(
    name="rx", position=[20, 0, 2],
    orientation=[0, 0, 0],
)
scene.add(tx)
scene.add(rx)
tx.look_at(rx)

paths = PathSolver()(scene)

tau = to_numpy(paths.tau).reshape(-1)
valid = to_numpy(paths.valid).astype(bool).reshape(-1)

# Inspect the path metadata before interpreting it
for name in ("interactions", "vertices", "objects", "primitives"):
    print(f"\n--- {name} ---")
    try:
        value = to_numpy(getattr(paths, name))
        if isinstance(value, list):
            for i, item in enumerate(value):
                arr = np.asarray(item)
                print(f"Component {i}: shape={arr.shape}")
                print("Sample:", arr.reshape(-1)[:15])
        else:
            arr = np.asarray(value)
            print("Shape:", arr.shape)
            print("Sample:", arr.reshape(-1)[:30])
    except Exception as exc:
        print("Inspection error:", repr(exc))

print("\n--- Path delay table ---")
order = np.argsort(tau)

for idx in order:
    if not valid[idx]:
        continue
    delay = tau[idx]
    print(
        f"Path entry {idx}: "
        f"delay={delay*1e9:.3f} ns, "
        f"equivalent distance={C*delay:.3f} m"
    )
