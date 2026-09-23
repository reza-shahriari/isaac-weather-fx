"""Run in Isaac Sim's Script Editor. Prints every render setting containing 'fog'.

Compare the output with backends/viewport/rtx_settings.py and fix any path that differs.
"""
import carb.settings


def walk(node, prefix):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, f"{prefix}/{key}")
    else:
        yield prefix, node


settings = carb.settings.get_settings()
found = [(p, v) for p, v in walk(settings.get("/rtx") or {}, "/rtx") if "fog" in p.lower()]
for path, value in sorted(found):
    print(f"{path} = {value!r}")
print(f"{len(found)} fog-related settings found")
