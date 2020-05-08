
#include <iostream>
#include <memory>
#include <vector>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <Eigen/Core>

#include <NvCloth/Cloth.h>
#include <NvCloth/Fabric.h>
#include <NvCloth/Factory.h>
#include <NvCloth/Solver.h>
#include <NvClothExt/ClothFabricCooker.h>
#include <d3d11.h>
#include <d3dcommon.h>

#include "CallbackImplementations.h"

using std::make_shared;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;

using nv::cloth::BoundedData;
using nv::cloth::ClothMeshDesc;

namespace py = pybind11;

using MatX4f = Eigen::Matrix<float, Eigen::Dynamic, 4, Eigen::RowMajor>;
using MatX3f = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
using MatXf =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatX4i = Eigen::Matrix<int32_t, Eigen::Dynamic, 4, Eigen::RowMajor>;
using MatX3i = Eigen::Matrix<int32_t, Eigen::Dynamic, 3, Eigen::RowMajor>;
using MatXi =
    Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/**
 * A note on resource ownership:
 *
 * Since this interfaces with Python, which uses a garbage-collected memory
 * manager, it's difficult to apply strict rules for single-object ownership
 * using, e.g., unique_ptr. Instead, most functions in this API return a
 * reference-counted shared_ptr, and and deallocation is handled by the last
 * owner of a resource. For this to work, we need to ensure there are no cycles
 * in ownership. Here is a DAG describing ownership among types. Arrows denote
 * has-a relationships, and * arrows denote has-many.
 *
 * NvClothEnv <- Factory <- Fabric <- Cloth <*- Solver
 *
 * DirectX Context <- DirectX Factory
 *
 * For example, a Fabric keeps a handle to its factory, and does not allow its
 * factory to be released until the fabric itself is released.
 */

/**
 * RAII wrapper for environment setup and teardown
 */
struct NvClothEnv {
  NvClothEnv() { NvClothEnvironment::AllocateEnv(); }
  NvClothEnv(const NvClothEnv&) = delete;
  NvClothEnv(NvClothEnv&&) = delete;
  NvClothEnv& operator=(const NvClothEnv&) = delete;
  NvClothEnv& operator=(NvClothEnv&&) = delete;
  ~NvClothEnv() { NvClothEnvironment::FreeEnv(); }
};

// manage the environment via reference counting
static shared_ptr<NvClothEnv> nv_cloth_env;

void allocate_env() {
  nv_cloth_env.reset(new NvClothEnv);
}

void free_env() {
  nv_cloth_env.reset();
}

class DxContextManager {
 public:
  DxContextManager() : impl_(nullptr) {}
  explicit DxContextManager(DxContextManagerCallbackImpl* const impl)
      : impl_(impl) {}

  auto get() const { return impl_.get(); }

 private:
  unique_ptr<DxContextManagerCallbackImpl> impl_;
};

auto create_dx11_context_manager() {
  // This is a little sloppy -- for directx, we need something that owns both
  // the device and the factory.
  ID3D11Device* device;
  ID3D11DeviceContext* context;
  const auto feature_level = D3D_FEATURE_LEVEL_11_0;
  D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
                    &feature_level, 1, D3D11_SDK_VERSION, &device, nullptr,
                    &context);
  return make_shared<DxContextManager>(
      new DxContextManagerCallbackImpl(device, false));
}

struct FactoryDeleter {
  void operator()(nv::cloth::Factory* const factory) const {
    NvClothDestroyFactory(factory);
  }
};

class Cloth;
class Fabric;
class Solver;

class Factory {
 public:
  Factory() : directx_context_manager_(nullptr), impl_(nullptr) {}

  static auto create_factory_cpu() {
    return make_shared<Factory>(NvClothCreateFactoryCPU());
  }

  static auto create_factory_dx11(
      shared_ptr<DxContextManager> context_manager) {
    const auto f = NvClothCreateFactoryDX11(context_manager->get());
    return make_shared<Factory>(f, std::move(context_manager));
  }

  shared_ptr<Cloth> create_cloth(MatX4f& particle_positions,
                                 shared_ptr<Fabric> fabric);
  shared_ptr<Solver> create_solver();

  nv::cloth::Factory* get() const { return impl_.get(); };

  explicit Factory(nv::cloth::Factory* const impl)
      : directx_context_manager_(nullptr), impl_(impl) {}
  explicit Factory(nv::cloth::Factory* const impl,
                   shared_ptr<DxContextManager> context_manager)
      : directx_context_manager_(std::move(context_manager)), impl_(impl) {}

 private:
  shared_ptr<NvClothEnv> env_;
  shared_ptr<DxContextManager> directx_context_manager_;

  unique_ptr<nv::cloth::Factory, FactoryDeleter> impl_;
};

struct FabricDeleter {
  void operator()(nv::cloth::Fabric* const f) const { f->decRefCount(); }
};

class Fabric {
 public:
  Fabric() : factory_(nullptr), impl_(nullptr) {}
  Fabric(nv::cloth::Fabric* const impl, shared_ptr<Factory> factory)
      : factory_(std::move(factory)), impl_(impl) {}

  auto get() const { return impl_.get(); }

 private:
  shared_ptr<Factory> factory_;

  unique_ptr<nv::cloth::Fabric, FabricDeleter> impl_;
};

struct ClothDeleter {
  void operator()(nv::cloth::Cloth* const c) const { NV_CLOTH_DELETE(c); }
};

class Cloth {
 public:
  Cloth() : fabric_(nullptr), impl_(nullptr) {}
  Cloth(nv::cloth::Cloth* const impl, shared_ptr<Fabric> fabric)
      : fabric_(std::move(fabric)), impl_(impl) {}

  void set_solver_frequency(const float t) { impl_->setSolverFrequency(t); }
  void set_gravity(const physx::PxVec3& gravity) { impl_->setGravity(gravity); }
  void set_lift_coefficient(const float coeff) {
    impl_->setLiftCoefficient(coeff);
  }
  void set_drag_coefficient(const float coeff) {
    impl_->setDragCoefficient(coeff);
  }
  void set_friction(const float coeff) { impl_->setFriction(coeff); }
  void clear_inertia() { impl_->clearInertia(); }
  void set_damping(const physx::PxVec3& v) { impl_->setDamping(v); }
  void set_linear_drag(const physx::PxVec3& v) { impl_->setLinearDrag(v); }
  void set_angular_drag(const physx::PxVec3& v) { impl_->setAngularDrag(v); }
  void set_linear_inertia(const physx::PxVec3& v) {
    impl_->setLinearInertia(v);
  }
  void set_angular_inertia(const physx::PxVec3& v) {
    impl_->setAngularInertia(v);
  }
  void set_centrifugal_inertia(const physx::PxVec3& v) {
    impl_->setCentrifugalInertia(v);
  }
  void set_stiffness_frequency(const float f) {
    impl_->setStiffnessFrequency(f);
  }
  void enable_continuous_collision(const bool b) {
    impl_->enableContinuousCollision(b);
  }
  void set_collision_mass_scale(const float f) {
    impl_->setCollisionMassScale(f);
  }
  void clear_particle_accelerations() { impl_->clearParticleAccelerations(); }
  auto get_current_particles() {
    const auto particles = impl_->getCurrentParticles();
    Eigen::Map<MatX4f> result(reinterpret_cast<float*>(particles.begin()),
                              particles.size(), 4);
    // copy and return result
    return result;
  }

  auto get() const { return impl_.get(); }

 private:
  shared_ptr<Fabric> fabric_;

  unique_ptr<nv::cloth::Cloth, ClothDeleter> impl_;
};

struct SolverDeleter {
  void operator()(nv::cloth::Solver* const s) const { NV_CLOTH_DELETE(s); }
};

class Solver {
 public:
  Solver() : impl_(nullptr) {}
  explicit Solver(nv::cloth::Solver* const impl) : impl_(impl) {}

  Solver(const Solver&) = delete;
  Solver(Solver&&) = default;
  Solver& operator=(const Solver&) = delete;
  Solver& operator=(Solver&&) = default;
  ~Solver() {
    // Automatically remove dangling cloths, in case users forgot
    for (const auto cloth : cloths_) {
      impl_->removeCloth(cloth->get());
    }
  }

  auto get() const { return impl_.get(); }

  void add_cloth(shared_ptr<Cloth> cloth) {
    impl_->addCloth(cloth->get());
    cloths_.emplace(std::move(cloth));
  }
  void remove_cloth(const shared_ptr<Cloth>& cloth) {
    impl_->removeCloth(cloth->get());
    cloths_.erase(cloth);
  }

  void simulate(const float dt) {
    impl_->beginSimulation(dt);
    for (int i = 0; i < impl_->getSimulationChunkCount(); i++) {
      impl_->simulateChunk(i);
    }
    impl_->endSimulation();
  }

 private:
  unique_ptr<nv::cloth::Solver, SolverDeleter> impl_;

  std::unordered_set<shared_ptr<Cloth>> cloths_;
};

shared_ptr<Cloth> Factory::create_cloth(MatX4f& particle_positions,
                                        shared_ptr<Fabric> fabric) {
  const auto begin = particle_positions.data();
  const auto end = begin + particle_positions.size();
  const nv::cloth::Range<physx::PxVec4> range(
      reinterpret_cast<physx::PxVec4*>(begin),
      reinterpret_cast<physx::PxVec4*>(end));

  auto cloth =
      make_shared<Cloth>(impl_->createCloth(range, *fabric->get()), fabric);

  // TODO(daniel): how should users take advantage of this?
  vector<nv::cloth::PhaseConfig> phases(fabric->get()->getNumPhases());
  for (uint32_t i = 0; i < fabric->get()->getNumPhases(); i++) {
    phases[i].mPhaseIndex = i;
    phases[i].mStiffness = 1.0f;
    phases[i].mStiffnessMultiplier = 1.0f;
    phases[i].mCompressionLimit = 1.0f;
    phases[i].mStretchLimit = 1.0f;
  }
  cloth->get()->setPhaseConfig(nv::cloth::Range<nv::cloth::PhaseConfig>(
      phases.data(), phases.data() + fabric->get()->getNumPhases()));
  return cloth;
}

shared_ptr<Solver> Factory::create_solver() {
  return make_shared<Solver>(impl_->createSolver());
}

template <typename MatrixT>
auto bounded_data_view(const Eigen::Ref<MatrixT>& mat) {
  // Note: this assumes a row-major ordering
  BoundedData result;
  result.count = static_cast<physx::PxU32>(mat.rows());
  result.stride = static_cast<physx::PxU32>(mat.rowStride() *
                                            sizeof(typename MatrixT::Scalar));
  result.data = mat.data();
  return result;
}

Fabric cook_fabric_from_mesh(shared_ptr<Factory> factory,
                             const ClothMeshDesc& desc,
                             const physx::PxVec3& gravity,
                             const bool use_geodesic_tether) {
  // TODO(daniel): is phaseTypes important? currently just throwing it away.
  const auto ptr = factory->get();
  return Fabric(NvClothCookFabricFromMesh(ptr, desc, gravity, nullptr,
                                          use_geodesic_tether),
                std::move(factory));
}

void set_collision_mesh(Cloth& c,
                        const Eigen::Ref<MatX3i>& triangles,
                        const Eigen::Ref<MatX3f>& vertices) {
  // NvCloth expects separate triangles, so we really do have to make some
  // copies here.
  MatX3f split_triangles(triangles.size() * 3, 3);
  for (int i = 0; i < triangles.rows(); ++i) {
    const auto& tri = triangles.row(i);
    split_triangles.row(3ULL * i + 0ULL) = vertices.row(tri[0]);
    split_triangles.row(3ULL * i + 1ULL) = vertices.row(tri[1]);
    split_triangles.row(3ULL * i + 2ULL) = vertices.row(tri[2]);
  }
  const auto begin = split_triangles.data();
  const auto end = begin + split_triangles.size();
  const nv::cloth::Range<physx::PxVec3> range(
      reinterpret_cast<physx::PxVec3*>(begin),
      reinterpret_cast<physx::PxVec3*>(end));

  c.get()->setTriangles(range, 0, c.get()->getNumTriangles());
}

PYBIND11_MODULE(pynvcloth, m) {
  m.doc() = "A python wrapper around NvCloth";

  m.def("allocate_env", &::allocate_env,
        "Initialize the NvCloth library and register necessary handlers.");
  m.def("free_env", &::free_env, "De-initialize the NvCloth library.");

  py::class_<Factory, shared_ptr<Factory>>(m, "Factory")
      .def(py::init<>())
      .def("create_cloth", &Factory::create_cloth)
      .def("create_solver", &Factory::create_solver);

  m.def("create_factory_cpu", &Factory::create_factory_cpu);

  (void)py::class_<DxContextManager, shared_ptr<DxContextManager>>(
      m, "DirectX11ContextManager");

  m.def("create_dx11_context_manager", &create_dx11_context_manager);
  m.def("create_factory_dx11", &Factory::create_factory_dx11);

  py::class_<Fabric, shared_ptr<Fabric>>(m, "Fabric").def(py::init<>());
  py::class_<Cloth, shared_ptr<Cloth>>(m, "Cloth")
      .def(py::init<>())
      .def("set_solver_frequency", &Cloth::set_solver_frequency)
      .def("set_gravity", &Cloth::set_gravity)
      .def("set_lift_coefficient", &Cloth::set_lift_coefficient)
      .def("set_drag_coefficient", &Cloth::set_drag_coefficient)
      .def("set_friction", &Cloth::set_friction)
      .def("set_collision_mesh", &set_collision_mesh)
      .def("clear_inertia", &Cloth::clear_inertia)
      .def("set_damping", &Cloth::set_damping)
      .def("set_linear_drag", &Cloth::set_linear_drag)
      .def("set_angular_drag", &Cloth::set_angular_drag)
      .def("set_linear_inertia", &Cloth::set_linear_inertia)
      .def("set_angular_inertia", &Cloth::set_angular_inertia)
      .def("set_centrifugal_inertia", &Cloth::set_centrifugal_inertia)
      .def("set_stiffness_frequency", &Cloth::set_stiffness_frequency)
      .def("enable_continuous_collision", &Cloth::enable_continuous_collision)
      .def("set_collision_mass_scale", &Cloth::set_collision_mass_scale)
      .def("clear_particle_accelerations", &Cloth::clear_particle_accelerations)
      .def("get_current_particles", &Cloth::get_current_particles);

  py::class_<Solver, shared_ptr<Solver>>(m, "Solver")
      .def(py::init<>())
      .def("add_cloth", &Solver::add_cloth)
      .def("remove_cloth", &Solver::remove_cloth)
      .def("simulate", &Solver::simulate);

  py::class_<physx::PxVec3>(m, "Vec3")
      .def(py::init<>())
      .def(py::init<float, float, float>())
      .def_readwrite("x", &physx::PxVec3::x)
      .def_readwrite("y", &physx::PxVec3::y)
      .def_readwrite("z", &physx::PxVec3::z);

  py::class_<BoundedData>(m, "BoundedData").def(py::init<>());

  m.def("as_bounded_data", &bounded_data_view<MatXf>);
  m.def("as_bounded_data", &bounded_data_view<MatXi>);

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
