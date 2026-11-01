import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")

import sionna.rt
from sionna.rt import (
    load_scene, Transmitter, Receiver, PlanarArray, PathSolver
)
import numpy as np
import drjit as dr

def arr(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return dr.to_numpy(x)

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

tx = Transmitter("tx", [-20, 0, 2], [0, 0, 0])
rx = Receiver("rx", [20, 0, 2], [0, 0, 0])
scene.add(tx)
scene.add(rx)
tx.look_at(rx)

paths = PathSolver()(scene)

tau = arr(paths.tau).reshape(-1)
valid = arr(paths.valid).astype(bool).reshape(-1)
interactions = arr(paths.interactions)
vertices = arr(paths.vertices)

print("Interaction array shape:", interactions.shape)
print("Vertex array shape:", vertices.shape)

for p in np.argsort(tau):
    if not valid[p]:
        continue

    print(f"\nPath index {p}: delay = {tau[p]*1e9:.3f} ns")

    # Layout: interaction-depth, TX, RX, path
    codes = interactions[:, 0, 0, p]
    print("Interaction values by depth:", codes.tolist())

    # Layout: depth, TX, RX, path, XYZ
    coords = vertices[:, 0, 0, p, :]
    print("Vertex coordinates by depth:")
    for depth, xyz in enumerate(coords):
        print(f"  depth {depth}: {xyz.tolist()}")
