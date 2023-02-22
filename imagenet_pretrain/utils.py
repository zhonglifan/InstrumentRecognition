from __future__ import annotations
import importlib.util
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import types


def import_source_file(fname: str | Path, modname: str) -> "types.ModuleType":
    """
    Import a Python source file and return the loaded module.

    Args:
      fname: The full path to the source file.  It may container characters like `.`
          or `-`.
      modname: The name for the loaded module.  It may contain `.` and even characters
          that would normally not be allowed (e.g., `-`).
    Return:
      The imported module

    Raises:
      ImportError: If the file cannot be imported (e.g, if it's not a `.py` file or if
          it does not exist).
      Exception: Any exception that is raised while executing the module (e.g.,
          :exc:`SyntaxError).  These are errors made by the author of the module!
    """
    # https://stackoverflow.com/questions/19009932/import-arbitrary-python-source-file-python-3-3
    # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    spec = importlib.util.spec_from_file_location(modname, fname)
    if spec is None:
        raise ImportError(f"Could not load spec for module '{modname}' at: {fname}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    try:
        spec.loader.exec_module(module)
    except FileNotFoundError as e:
        raise ImportError(f"{e.strerror}: {fname}") from e
    return module
    # import_source_file(Path("/tmp/my_mod.py"), "my_mod")


class AverageMeter(object):
    """
    Computes and stores the average and current value
    https://github.com/gpleiss/efficient_densenet_pytorch/blob/master/demo.py
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AutoDoor(object):
    def __init__(self, metric_name):
        self.name = metric_name
        self.els = 0
        if self.name == 'acc':
            self.val = 0
        elif self.name == 'loss':
            self.val = float('inf')
        else:
            raise NotImplementedError

    def update(self, val):
        self.save = False
        if (self.name == 'acc' and val > self.val) or (self.name == 'loss' and val < self.val):
            self.val = val
            self.save = True
            self.els = 0
        else:
            self.els += 1
