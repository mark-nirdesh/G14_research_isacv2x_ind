import mitsuba as mi

# Select the CPU/LLVM backend explicitly.
mi.set_variant("llvm_ad_mono_polarized")

import sionna.rt
from sionna.rt import load_scene

print("Sionna RT imported")
print("Mitsuba variant:", mi.variant())

scene = load_scene(sionna.rt.scene.simple_street_canyon)
scene.frequency = 5.9e9

print("Scene loaded successfully")
print("Carrier frequency:", scene.frequency / 1e9, "GHz")
print("Scene objects:", len(scene.objects))
print("LLVM scene-loading test passed")
