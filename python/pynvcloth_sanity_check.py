import os
import sys

# You may need to supply a value for PYTHONPATH if this is not your build
# directory
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../build/bin/'))
import pynvcloth as nvc

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

# Note: these must be declared as named variables so they stay in scope while the mesh is created.
triangles = nvc.VectorTri([nvc.Triangle(0, 1, 3), nvc.Triangle(0, 2, 3), nvc.Triangle(2, 3, 5), nvc.Triangle(2, 4, 5)])
points = nvc.VectorNx3([
    nvc.Vec3(0.0, 0.0, 0.0),
    nvc.Vec3(0.0, 1.0, 0.0),
    nvc.Vec3(1.0, 0.0, 0.0),
    nvc.Vec3(1.0, 1.0, 0.0),
    nvc.Vec3(2.0, 0.0, 0.0),
    nvc.Vec3(2.0, 1.0, 0.0),
])
inv_masses = nvc.VectorFloat([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
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
for _ in range(3):
    cur_particles = cloth.get_current_particles()
    particles_as_py = [[p.x, p.y, p.z, p.w] for p in cur_particles]
    print(particles_as_py)
    solver.simulate(1. / 60.)

solver.remove_cloth(cloth)
del solver
del cloth
del fabric
del factory

nvc.free_env()
