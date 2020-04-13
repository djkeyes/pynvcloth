#include <pybind11/pybind11.h>

#include <NvCloth/Factory.h>

#include "CallbackImplementations.h"

using std::unique_ptr;

namespace py = pybind11;

struct FactoryDeleter {
	void operator()(nv::cloth::Factory* const factory) const {
		NvClothDestroyFactory(factory);
	}
};

using Factory = unique_ptr<nv::cloth::Factory, FactoryDeleter>;
// We are using a unique_ptr to enforce a custom deleter, so don't let pybind11
// try to extract out the raw pointer.
PYBIND11_MAKE_OPAQUE(Factory);

auto create_factory_cpu()
{
	return Factory(NvClothCreateFactoryCPU());
}

PYBIND11_MODULE(pynvcloth, m) {
	m.doc() = "A python wrapper around NvCloth";

	m.def("allocate_env", &NvClothEnvironment::AllocateEnv, "Initialize the NvCloth library and register necessary handlers.");
	m.def("free_env", &NvClothEnvironment::FreeEnv, "De-initialize the NvCloth library.");

	py::class_<Factory>(m, "Factory").def(py::init<>());

	m.def("create_factory_cpu", &create_factory_cpu);
}
