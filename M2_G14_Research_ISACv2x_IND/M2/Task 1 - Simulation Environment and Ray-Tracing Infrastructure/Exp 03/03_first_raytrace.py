
import mitsuba as mi

# Select CPU/LLVM before importing Sionna RT
mi.set_variant("llvm_ad_mono_polarized")

import sionna.rt
from sionna.rt import (
    load_scene,
    Transmitter,
    Receiver,
    PlanarArray,
    PathSolver,
)

print("Backend:", mi.variant())

# 1. Load scene
scene = load_scene(sionna.rt.scene.simple_street_canyon)
scene.frequency = 5.9e9
print("Scene loaded at 5.9 GHz")

# 2. Configure antennas
scene.tx_array = PlanarArray(
    num_rows=1,
    num_cols=1,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="iso",
    polarization="V",
)

scene.rx_array = PlanarArray(
    num_rows=1,
    num_cols=1,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="iso",
    polarization="V",
)

# 3. Add transmitter and receiver
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

print("Transmitter and receiver added")

# 4. Compute propagation paths
solver = PathSolver()
paths = solver(scene)

print("Ray tracing completed successfully")
print("Path object:", paths)
print("Experiment 1 completed")
