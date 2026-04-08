import importlib
import pkgutil
from langchain_core.tools import BaseTool


def load_all_tools() -> list:
    tools = []
    package = importlib.import_module("tools")
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        if module_name == "__init__":
            continue
        module = importlib.import_module(f"tools.{module_name}")
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, BaseTool):
                tools.append(attr)
    return tools
