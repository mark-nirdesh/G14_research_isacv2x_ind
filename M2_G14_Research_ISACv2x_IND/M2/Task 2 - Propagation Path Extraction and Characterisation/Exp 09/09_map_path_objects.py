import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")

import sionna.rt
from sionna.rt import (
    load_scene, Transmitter, Receiver, PlanarArray, PathSolver
)
import numpy as np
import drjit as dr

def to_numpy(x):
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

tau = to_numpy(paths.tau).reshape(-1)
valid = to_numpy(paths.valid).astype(bool).reshape(-1)
interactions = to_numpy(paths.interactions)
objects = to_numpy(paths.objects)

# Map numeric scene-object IDs to names
object_names = {}
for obj in scene.objects.values():
    object_names[int(obj.object_id)] = getattr(obj, "name", str(obj))

interaction_names = {
    0: "None",
    1: "Specular reflection",
    2: "Diffuse reflection",
    4: "Refraction",
}

print("Scene objects:")
for obj_id, name in object_names.items():
    print(f"  ID {obj_id}: {name}")

print("\nPath interaction/object table:")
for p in np.argsort(tau):
    if not valid[p]:
        continue

    print(f"\nPath {p}: delay={tau[p]*1e9:.3f} ns")

    for depth in range(interactions.shape[0]):
        code = int(interactions[depth, 0, 0, p])
        obj_id = int(objects[depth, 0, 0, p])

        if code == 0:
            continue

        name = object_names.get(obj_id, "Unknown/unmapped object")
        kind = interaction_names.get(code, f"Unknown code {code}")
        print(
            f"  Depth {depth}: {kind}; "
            f"object ID={obj_id}; object={name}"
        )
