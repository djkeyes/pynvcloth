
#include <iostream>
#include <memory>
#include <optional>

#include <NvCloth/Cloth.h>
#include <NvCloth/Factory.h>
#include <NvClothExt/ClothFabricCooker.h>
#include <PxVec3.h>
#include <d3d11.h>

#include "CallbackImplementations.h"

using std::cout;
using std::endl;
using std::unique_ptr;

using Factory =
    unique_ptr<nv::cloth::Factory, decltype(&NvClothDestroyFactory)>;

struct FabricDeleter {
  void operator()(nv::cloth::Fabric* const f) { f->decRefCount(); }
};

using Fabric = unique_ptr<nv::cloth::Fabric, FabricDeleter>;

struct ClothDeleter {
  void operator()(nv::cloth::Cloth* const c) { NV_CLOTH_DELETE(c); }
};

using Cloth = unique_ptr<nv::cloth::Cloth, ClothDeleter>;

template <typename T>
static nv::cloth::BoundedData as_bounded_data(T& vec) {
  nv::cloth::BoundedData d;
  d.data = &vec[0];
  d.stride = sizeof(vec[0]);
  d.count = static_cast<physx::PxU32>(vec.size());

  return d;
}

struct Triangle {
  Triangle() {}
  Triangle(uint32_t _a, uint32_t _b, uint32_t _c) : a(_a), b(_b), c(_c) {}
  uint32_t a, b, c;
};

void run_simple_simulation(bool use_directx) {
  cout << "hello nvcloth!" << endl;

  std::optional<DxContextManagerCallbackImpl> device_context_manager;

  use_directx = use_directx && NvClothCompiledWithDxSupport();

  if (use_directx) {
    // This is a little sloppy -- for directx, we need something that owns both
    // the device and the factory.
    ID3D11Device* device;
    ID3D11DeviceContext* context;
    auto feature_level = D3D_FEATURE_LEVEL_11_0;
    auto result = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                                    0, &feature_level, 1, D3D11_SDK_VERSION,
                                    &device, nullptr, &context);
    if (result != S_OK) {
      exit(-1);
    }
    device_context_manager.emplace(device);
  }

  Factory factory(nullptr, nullptr);
  if (!use_directx) {
    factory = Factory(NvClothCreateFactoryCPU(), NvClothDestroyFactory);
  } else {
    factory = Factory(NvClothCreateFactoryDX11(&*device_context_manager),
                      NvClothDestroyFactory);
  }
  nv::cloth::Vector<int32_t>::Type phase_types;

  nv::cloth::ClothMeshDesc mesh_desc;
  mesh_desc.setToDefault();
  // make a 10x10 grid of points
  constexpr int rows = 10;
  constexpr int cols = 10;

  const auto num_points = rows * cols;
  std::vector<physx::PxVec3> points(num_points);
  std::vector<float> inv_masses(num_points);
  std::vector<physx::PxVec4> particle_positions(num_points);
  std::vector<Triangle> triangles(2 * (rows - 1) * (cols - 1));

  float default_mass = 1.0f;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const auto idx = r * cols + c;
      inv_masses[idx] = 1.f / default_mass;
      points[idx] = physx::PxVec3(1.f * r, 1.f * c, 0.0f);
      particle_positions[idx] = physx::PxVec4(points[idx], inv_masses[idx]);

      if (r > 0 && c > 0) {
        const uint32_t p00 = (r - 1) * cols + (c - 1);
        const uint32_t p01 = (r - 1) * cols + (c - 0);
        const uint32_t p10 = (r - 0) * cols + (c - 1);
        const uint32_t p11 = (r - 0) * cols + (c - 0);
        const size_t tri_idx1 = 2 * (r - 1) * (cols - 1) + 2 * (c - 1);

        const size_t tri_idx2 = tri_idx1 + 1;
        // TODO: does nvcloth have requirements on the winding order?
        triangles[tri_idx1] = Triangle(p00, p11, p01);
        triangles[tri_idx2] = Triangle(p00, p10, p11);
      }
    }
  }

  mesh_desc.points.data = points.data();
  mesh_desc.points.stride = sizeof(points[0]);
  mesh_desc.points.count = static_cast<physx::PxU32>(points.size());

  mesh_desc.invMasses.data = inv_masses.data();
  mesh_desc.invMasses.stride = sizeof(inv_masses[0]);
  mesh_desc.invMasses.count = static_cast<physx::PxU32>(inv_masses.size());

  mesh_desc.triangles.data = triangles.data();
  mesh_desc.triangles.stride = sizeof(triangles[0]);
  mesh_desc.triangles.count = static_cast<physx::PxU32>(triangles.size());

  physx::PxVec3 gravity(0.f, -9.8f, 0.f);
  Fabric fabric(NvClothCookFabricFromMesh(factory.get(), mesh_desc, gravity,
                                          &phase_types, false));

  Cloth cloth(factory->createCloth(
      nv::cloth::Range<physx::PxVec4>(
          &particle_positions[0],
          &particle_positions[0] + particle_positions.size()),
      *fabric));
  // particlePositions can be freed here.
}
int main(int argc, char** argv) {
  NvClothEnvironment::AllocateEnv();

  run_simple_simulation(false);
  run_simple_simulation(true);

  NvClothEnvironment::FreeEnv();
  return 0;
}