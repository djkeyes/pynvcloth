#include <pybind11/pybind11.h>

#include "CallbackImplementations.h"

PYBIND11_MODULE(pynvcloth, m) {
	m.doc() = "A python wrapper around NvCloth";

	m.def("allocate_env", &NvClothEnvironment::AllocateEnv, "Initialize the NvCloth library and register necessary handlers.");
	m.def("free_env", &NvClothEnvironment::FreeEnv, "De-initialize the NvCloth library.");
}
