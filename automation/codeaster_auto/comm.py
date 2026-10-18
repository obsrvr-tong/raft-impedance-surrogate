"""Fill a .comm template.

Templates use Python's string.Template placeholders: $NAME or ${NAME}.
Every placeholder must get a value, and every value must be used, so a typo
in either place fails loudly instead of producing a silently wrong model.
"""
import re
from string import Template

_PLACEHOLDER = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


def placeholders(text):
    return {a or b for a, b in _PLACEHOLDER.findall(text)}


def render_comm(template_path, values):
    with open(template_path, encoding="utf-8") as f:
        text = f.read()
    needed = placeholders(text)
    missing = needed - set(values)
    unused = set(values) - needed
    if missing:
        raise KeyError(f"{template_path}: no value for {sorted(missing)}")
    if unused:
        raise KeyError(f"{template_path}: values not used by template {sorted(unused)}")
    return Template(text).substitute({k: str(v) for k, v in values.items()})


def write_comm(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
