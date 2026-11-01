
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

# Load scene
scene = load_scene(sionna.rt.scene.simple_street_canyon)
scene.frequency = 5.9e9

# Antenna configuration
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

# Radio link
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

# Solve propagation paths
paths = PathSolver()(scene)

print("\nPath object attributes:")
print([name for name in dir(paths) if not name.startswith("_")])

# Inspect common path quantities if available
for name in ("tau", "a", "theta_t", "phi_t", "theta_r", "phi_r"):
    try:
        value = getattr(paths, name)
        array = value.numpy()
        print(f"\n{name}:")
        print("Shape:", array.shape)
        print("Sample:", array.flatten()[:10])
    except Exception as exc:
        print(f"{name}: could not convert ({exc})")
