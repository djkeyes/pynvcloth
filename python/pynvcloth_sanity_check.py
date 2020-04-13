
import sys
import os
# You may need to supply a value for PYTHONPATH if this is not your build 
# directory
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../build/bin/'))
import pynvcloth

# The NvCloth environment must be setup (and destroyed) explicitly.
# daniel: This doesn't seem very python-like. Can we do this implicitly?
pynvcloth.allocate_env()

# Cloth factories can be created via the appropriate constructor.
null_factory = pynvcloth.Factory()
print(type(null_factory))

factory = pynvcloth.create_factory_cpu()
print(type(factory))

# Note: resources need to be explicitly freed. We could also control this with function scoping
del null_factory

del factory

def scoped_alloc():
    another = pynvcloth.create_factory_cpu()
    print(type(another))

scoped_alloc()

pynvcloth.free_env()