import inspect
import types
from typing import List


def get_registry(package, exclude_substr: List[str]=None):
    if exclude_substr is None:
        exclude_substr = ['Dummy', 'Abstract']
    registry = {}
    for entry_name in dir(package):
        if entry_name.startswith('__'):
            continue
        entry = getattr(package, entry_name)
        if inspect.isabstract(entry):
            continue
        if any((substring in entry_name for substring in exclude_substr)):
            continue
        if not isinstance(entry, types.ModuleType):
            registry[entry_name] = entry
    return registry


class LazyClass(object):
    def __getattr__(self, attr):
        try:
            return object.__getattr__(attr)
        except AttributeError:
            return self.do_nothing

    @staticmethod
    def do_nothing(*args, **kwargs):
        pass
