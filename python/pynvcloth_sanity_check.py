import os
import sys

# You may need to supply a value for PYTHONPATH if this is not your build
# directory
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../build/bin/'))
import pynvcloth as nvc  # noqa: E402

import numpy as np
import trimesh
from trimesh.viewer import SceneViewer
import pyglet

# The NvCloth environment must be setup (and destroyed) explicitly.
# daniel: This doesn't seem very python-like. Can we do this implicitly?
nvc.allocate_env()

# Cloth factories can be created via the appropriate constructor.
null_factory = nvc.Factory()
print(type(null_factory))

factory = nvc.create_factory_cpu()
print(type(factory))

# Note: resources need to be explicitly freed. We could also control this with function scoping
del null_factory

del factory


def scoped_alloc():
    another = nvc.create_factory_cpu()
    print(type(another))


scoped_alloc()

cmd = nvc.ClothMeshDesc()
cmd.set_to_default()
print(cmd.is_valid(), flush=True)

rows = 30
cols = 30
dx = 1.0 / rows
dy = 1.0 / cols
vertices = []
faces = []
inv_masses = []
for r in range(rows):
    for c in range(cols):
        x = c * dx
        y = -r * dy
        z = np.sin(np.deg2rad((x + y) * 360 * 5)) * 0.01
        vertices.append([x, y, z])
        inv_masses.append(1.0)
        if r > 0 and c > 0:
            p00 = (r - 1) * cols + (c - 1)
            p01 = (r - 1) * cols + (c - 0)
            p10 = (r - 0) * cols + (c - 1)
            p11 = (r - 0) * cols + (c - 0)
            faces.append([p00, p11, p01])
            faces.append([p00, p10, p11])
inv_masses[0] = 0.0  # anchor particle 0

render_mesh = trimesh.Trimesh(vertices, faces, process=False)

# Note: these must be declared as named variables so they stay in scope while the mesh is created.
triangles = nvc.VectorTri([nvc.Triangle(f[0], f[1], f[2]) for f in faces])
points = nvc.VectorNx3([nvc.Vec3(v[0], v[1], v[2]) for v in vertices])
inv_masses = nvc.VectorFloat(inv_masses)
point_masses = nvc.VectorNx4([nvc.Vec4(v, w) for (v, w) in zip(points, inv_masses)])

cmd.triangles = nvc.as_bounded_data(triangles)
cmd.points = nvc.as_bounded_data(points)
cmd.inv_masses = nvc.as_bounded_data(inv_masses)

print(cmd.is_valid(), flush=True)

factory = nvc.create_factory_cpu()
gravity = nvc.Vec3(0., -9.8, 0.)
fabric = nvc.cook_fabric_from_mesh(factory, cmd, gravity, False)
cloth = factory.create_cloth(point_masses, fabric)
solver = factory.create_solver()
solver.add_cloth(cloth)

cloth.set_solver_frequency(300)
cloth.set_gravity(gravity)
cloth.set_lift_coefficient(0.1)
cloth.set_drag_coefficient(0.1)
cloth.set_friction(0.1)


def gen_simulation():
    while True:
        cur_particles = cloth.get_current_particles()
        particles_as_py = [[p.x, p.y, p.z, p.w] for p in cur_particles]
        solver.simulate(1. / 60.)
        print('updated!')
        yield np.array(particles_as_py)[:, :3]


simul_gen = gen_simulation()


def viewer_callback(scene):
    verts = next(simul_gen)
    render_mesh.vertices = verts


v = SceneViewer(trimesh.Scene([render_mesh]), start_loop=False, callback=viewer_callback)
pyglet.app.run()

print('clearing up!', flush=True)
solver.remove_cloth(cloth)
del solver
del cloth
del fabric
del factory

nvc.free_env()
