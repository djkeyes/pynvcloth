
#include <iostream>
#include <memory>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include <NvCloth/Cloth.h>
#include <NvCloth/Fabric.h>
#include <NvCloth/Factory.h>
#include <NvCloth/Solver.h>
#include <NvClothExt/ClothFabricCooker.h>
#include <d3d11.h>
#include <d3dcommon.h>

#include "CallbackImplementations.h"

using std::unique_ptr;
using std::vector;

using nv::cloth::BoundedData;
using nv::cloth::ClothMeshDesc;

namespace py = pybind11;

struct FactoryDeleter {
  void operator()(nv::cloth::Factory* const factory) const {
    NvClothDestroyFactory(factory);
  }
};

using Factory = unique_ptr<nv::cloth::Factory, FactoryDeleter>;

struct FabricDeleter {
  void operator()(nv::cloth::Fabric* const f) const { f->decRefCount(); }
};

using Fabric = unique_ptr<nv::cloth::Fabric, FabricDeleter>;

struct ClothDeleter {
  void operator()(nv::cloth::Cloth* const c) const { NV_CLOTH_DELETE(c); }
};

using Cloth = unique_ptr<nv::cloth::Cloth, ClothDeleter>;

struct SolverDeleter {
  void operator()(nv::cloth::Solver* const s) const { NV_CLOTH_DELETE(s); }
};

using Solver = unique_ptr<nv::cloth::Solver, SolverDeleter>;

// We are using unique_ptr to enforce custom deleters, so don't let pybind11 try
// to extract out the raw pointers.
PYBIND11_MAKE_OPAQUE(Factory);
PYBIND11_MAKE_OPAQUE(Fabric);
PYBIND11_MAKE_OPAQUE(Cloth);
PYBIND11_MAKE_OPAQUE(Solver);
PYBIND11_MAKE_OPAQUE(std::unique_ptr<DxContextManagerCallbackImpl>);

struct Triangle {
  Triangle() : a(0), b(0), c(0) {}

  Triangle(const uint32_t a, const uint32_t b, const uint32_t c)
      : a(a), b(b), c(c) {}

  uint32_t a, b, c;
};

struct Quad {
  Quad() : a(0), b(0), c(0), d(0) {}

  Quad(const uint32_t a, const uint32_t b, const uint32_t c, const uint32_t d)
      : a(a), b(b), c(c), d(d) {}

  uint32_t a, b, c, d;
};

auto create_factory_cpu() {
  return Factory(NvClothCreateFactoryCPU());
}

auto create_dx11_context_manager() {
  // This is a little sloppy -- for directx, we need something that owns both
  // the device and the factory.
  ID3D11Device* device;
  ID3D11DeviceContext* context;
  const auto feature_level = D3D_FEATURE_LEVEL_11_0;
  D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
                    &feature_level, 1, D3D11_SDK_VERSION, &device, nullptr,
                    &context);
  return std::make_unique<DxContextManagerCallbackImpl>(device, false);
}

auto create_factory_dx11(
    const std::unique_ptr<DxContextManagerCallbackImpl>& context_manager) {
  return Factory(NvClothCreateFactoryDX11(context_manager.get()));
}

// We need to be able to pass in arrays with C++-managed memory (since
// BoundedData requires a long-lived pointer to contiguous memory).
// Therefore, create opaque types for vector, and use bind_vector later to
// emulate vector functions in python
PYBIND11_MAKE_OPAQUE(vector<Triangle>);
PYBIND11_MAKE_OPAQUE(vector<Quad>);
PYBIND11_MAKE_OPAQUE(vector<float>);
PYBIND11_MAKE_OPAQUE(vector<int32_t>);
PYBIND11_MAKE_OPAQUE(vector<physx::PxVec3>);
PYBIND11_MAKE_OPAQUE(vector<physx::PxVec4>);

template <typename T>
auto bounded_data_view(const vector<T>& vec) {
  BoundedData result;
  result.count = static_cast<physx::PxU32>(vec.size());
  result.stride = sizeof(T);
  result.data = vec.data();
  return result;
}

Fabric cook_fabric_from_mesh(const Factory& factory,
                             const ClothMeshDesc& desc,
                             const physx::PxVec3& gravity,
                             const bool use_geodesic_tether) {
  // TODO(daniel): is phaseTypes important? currently just throwing it away.
  return Fabric(NvClothCookFabricFromMesh(factory.get(), desc, gravity, nullptr,
                                          use_geodesic_tether));
}

void set_collision_mesh(Cloth& c,
                        const vector<Triangle>& triangles,
                        const vector<physx::PxVec3>& vertices) {
  // NvCloth expects separate triangles, so we really do have to make some
  // copies here.
  vector<physx::PxVec3> split_triangles(triangles.size() * 3);
  for (size_t i = 0; i < triangles.size(); ++i) {
    const auto& tri = triangles[i];
    split_triangles[3ULL * i + 0ULL] = vertices[tri.a];
    split_triangles[3ULL * i + 1ULL] = vertices[tri.b];
    split_triangles[3ULL * i + 2ULL] = vertices[tri.c];
  }
  const nv::cloth::Range<physx::PxVec3> range(
      split_triangles.data(), split_triangles.data() + split_triangles.size());
  c->setTriangles(range, 0, c->getNumTriangles());
}

PYBIND11_MODULE(pynvcloth, m) {
  m.doc() = "A python wrapper around NvCloth";

  m.def("allocate_env", &NvClothEnvironment::AllocateEnv,
        "Initialize the NvCloth library and register necessary handlers.");
  m.def("free_env", &NvClothEnvironment::FreeEnv,
        "De-initialize the NvCloth library.");

  py::class_<Factory>(m, "Factory")
      .def(py::init<>())
      .def("create_cloth",
           [](Factory& factory, vector<physx::PxVec4>& particle_positions,
              Fabric& fabric) {
             Cloth cloth(factory->createCloth(
                 nv::cloth::Range<physx::PxVec4>(
                     particle_positions.data(),
                     particle_positions.data() + particle_positions.size()),
                 *fabric));

             // TODO(daniel): how should users take advantage of this?
             vector<nv::cloth::PhaseConfig> phases(fabric->getNumPhases());
             for (uint32_t i = 0; i < fabric->getNumPhases(); i++) {
               phases[i].mPhaseIndex = i;
               phases[i].mStiffness = 1.0f;
               phases[i].mStiffnessMultiplier = 1.0f;
               phases[i].mCompressionLimit = 1.0f;
               phases[i].mStretchLimit = 1.0f;
             }
             cloth->setPhaseConfig(nv::cloth::Range<nv::cloth::PhaseConfig>(
                 phases.data(), phases.data() + fabric->getNumPhases()));
             return cloth;
           })
      .def("create_solver",
           [](Factory& factory) { return Solver(factory->createSolver()); });

  m.def("create_factory_cpu", &create_factory_cpu);

  (void)py::class_<std::unique_ptr<DxContextManagerCallbackImpl>>(
      m, "DirectX11ContextManager");

  m.def("create_dx11_context_manager", &create_dx11_context_manager);
  m.def("create_factory_dx11", &create_factory_dx11);

  py::class_<Fabric>(m, "Fabric").def(py::init<>());
  py::class_<Cloth>(m, "Cloth")
      .def(py::init<>())
      .def("set_solver_frequency",
           [](Cloth& c, const float t) { c->setSolverFrequency(t); })
      .def("set_gravity",
           [](Cloth& c, const physx::PxVec3& gravity) {
             c->setGravity(gravity);
           })
      .def("set_lift_coefficient",
           [](Cloth& c, const float coeff) { c->setLiftCoefficient(coeff); })
      .def("set_drag_coefficient",
           [](Cloth& c, const float coeff) { c->setDragCoefficient(coeff); })
      .def("set_friction",
           [](Cloth& c, const float coeff) { c->setFriction(coeff); })
      .def("set_collision_mesh", &set_collision_mesh)
      .def("clear_inertia", [](Cloth& c) { c->clearInertia(); })
      .def("set_damping",
           [](Cloth& c, const physx::PxVec3& v) { c->setDamping(v); })
      .def("set_linear_drag",
           [](Cloth& c, const physx::PxVec3& v) { c->setLinearDrag(v); })
      .def("set_angular_drag",
           [](Cloth& c, const physx::PxVec3& v) { c->setAngularDrag(v); })
      .def("set_linear_inertia",
           [](Cloth& c, const physx::PxVec3& v) { c->setLinearInertia(v); })
      .def("set_angular_inertia",
           [](Cloth& c, const physx::PxVec3& v) { c->setAngularInertia(v); })
      .def(
          "set_centrifugal_inertia",
          [](Cloth& c, const physx::PxVec3& v) { c->setCentrifugalInertia(v); })
      .def("set_stiffness_frequency",
           [](Cloth& c, const float f) { c->setStiffnessFrequency(f); })
      .def("enable_continuous_collision",
           [](Cloth& c, const bool b) { c->enableContinuousCollision(b); })
      .def("set_collision_mass_scale",
           [](Cloth& c, const float f) { c->setCollisionMassScale(f); })
      .def("clear_particle_accelerations",
           [](Cloth& c) { c->clearParticleAccelerations(); })
      .def("get_current_particles", [](Cloth& c) {
        const auto particles = c->getCurrentParticles();
        // copy and return result
        return vector<physx::PxVec4>(particles.begin(), particles.end());
      });
  py::class_<Solver>(m, "Solver")
      .def(py::init<>())
      .def("add_cloth", [](Solver& s, Cloth& c) { s->addCloth(c.get()); })
      .def("remove_cloth", [](Solver& s, Cloth& c) { s->removeCloth(c.get()); })
      .def("simulate", [](Solver& s, const float dt) {
        s->beginSimulation(dt);
        for (int i = 0; i < s->getSimulationChunkCount(); i++) {
          s->simulateChunk(i);
        }
        s->endSimulation();
      });

  py::class_<Triangle>(m, "Triangle")
      .def(py::init<>())
      .def(py::init<uint32_t, uint32_t, uint32_t>())
      .def_readwrite("a", &Triangle::a)
      .def_readwrite("b", &Triangle::b)
      .def_readwrite("c", &Triangle::c);

  py::class_<Quad>(m, "Quad")
      .def(py::init<>())
      .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t>())
      .def_readwrite("a", &Quad::a)
      .def_readwrite("b", &Quad::b)
      .def_readwrite("c", &Quad::c)
      .def_readwrite("c", &Quad::c);

  py::class_<physx::PxVec3>(m, "Vec3")
      .def(py::init<>())
      .def(py::init<float, float, float>())
      .def_readwrite("x", &physx::PxVec3::x)
      .def_readwrite("y", &physx::PxVec3::y)
      .def_readwrite("z", &physx::PxVec3::z);

  py::class_<physx::PxVec4>(m, "Vec4")
      .def(py::init<>())
      .def(py::init<float, float, float, float>())
      .def(py::init<physx::PxVec3, float>())
      .def_readwrite("x", &physx::PxVec4::x)
      .def_readwrite("y", &physx::PxVec4::y)
      .def_readwrite("z", &physx::PxVec4::z)
      .def_readwrite("w", &physx::PxVec4::w);

  py::class_<BoundedData>(m, "BoundedData").def(py::init<>());

  // (void) to suppress warning C26444 on MSVC
  (void)py::bind_vector<vector<Triangle>>(m, "VectorTri");
  (void)py::bind_vector<vector<Quad>>(m, "VectorQuad");
  (void)py::bind_vector<vector<float>>(m, "VectorFloat");
  (void)py::bind_vector<vector<int32_t>>(m, "VectorInt");
  (void)py::bind_vector<vector<physx::PxVec3>>(m, "VectorNx3");
  (void)py::bind_vector<vector<physx::PxVec4>>(m, "VectorNx4");

  m.def("as_bounded_data", &bounded_data_view<Triangle>);
  m.def("as_bounded_data", &bounded_data_view<Quad>);
  m.def("as_bounded_data", &bounded_data_view<float>);
  m.def("as_bounded_data", &bounded_data_view<int32_t>);
  m.def("as_bounded_data", &bounded_data_view<physx::PxVec3>);
  m.def("as_bounded_data", &bounded_data_view<physx::PxVec4>);

  py::class_<ClothMeshDesc>(m, "ClothMeshDesc")
      .def(py::init<>())
      .def_readwrite("points", &ClothMeshDesc::points)
      .def_readwrite("points_stiffness", &ClothMeshDesc::pointsStiffness)
      .def_readwrite("inv_masses", &ClothMeshDesc::invMasses)
      .def_readwrite("triangles", &ClothMeshDesc::triangles)
      .def_readwrite("quads", &ClothMeshDesc::quads)
      .def_readwrite("flags", &ClothMeshDesc::flags)
      .def("set_to_default", &ClothMeshDesc::setToDefault)
      .def("is_valid", &ClothMeshDesc::isValid);

  m.def("cook_fabric_from_mesh", &cook_fabric_from_mesh);
}
