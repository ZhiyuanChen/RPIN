import importlib
from glob import glob
module_names = glob('./rpin/models/*.py')
module_names = [module_name.replace("\\", "/") for module_name in module_names]
for module_name in module_names:
    name = module_name.split('/')[-1].split('.')[0]
    if name == '__init__':
        continue
    importlib.import_module('rpin.models.' + name)
