import sionna.rt
from sionna.rt import (
    load_scene,
    Transmitter,
    Receiver,
    PlanarArray,
    PathSolver,
)

# Load Sionna's built-in example scene
scene = load_scene(sionna.rt.scene.simple_street_canyon)

# Set the V2X carrier frequency
scene.frequency = 5.9e9

# Single-element antenna arrays for the first test
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

# Add transmitter and receiver
tx = Transmitter(
    name="tx",
    position=[-10, 0, 5],
)

rx = Receiver(
    name="rx",
    position=[10, 0, 1.5],
)

scene.add(tx)
scene.add(rx)

print("Scene loaded:", sionna.rt.scene.simple_street_canyon)
print("Carrier frequency:", scene.frequency / 1e9, "GHz")
print("Computing propagation paths...")

solver = PathSolver()

paths = solver(
    scene=scene,
    max_depth=3,
    los=True,
    specular_reflection=True,
    diffuse_reflection=False,
    refraction=False,
    seed=7,
)

print("Path computation completed.")
print("Paths object:", type(paths))

# Convert paths into a channel impulse response
a, tau = paths.cir()

print("Channel coefficient shape:", a.shape)
print("Path delay shape:", tau.shape)
print("First ray-tracing test completed successfully.")
