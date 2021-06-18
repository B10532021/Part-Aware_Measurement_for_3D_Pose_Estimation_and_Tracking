import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)

utils_path = osp.join(this_dir, 'utils')
add_path(utils_path)

eval_path = osp.join(this_dir, 'eval')
add_path(eval_path)