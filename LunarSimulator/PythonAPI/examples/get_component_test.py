import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

client = carla.Client('localhost', 2000)
world = client.get_world()

location = carla.Location(200.0, 200.0, 200.0)
rotation = carla.Rotation(0.0, 0.0, 0.0)
transform = carla.Transform(location, rotation)

bp_library = world.get_blueprint_library()
bp_lander = bp_library.find('static.blueprint.lander')
lander = world.spawn_actor(bp_lander, transform)

component_transform = lander.get_component_world_transform('charging-port')
print(component_transform)
