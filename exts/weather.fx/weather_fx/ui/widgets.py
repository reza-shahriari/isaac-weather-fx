"""Widgets generated from dataclass field metadata."""
from __future__ import annotations

import math

import omni.ui as ui

LABEL_WIDTH = 150
ROW_HEIGHT = 22


class UpdateGuard:
    """Blocks UI -> state commits while the UI is being refreshed from state."""

    def __init__(self):
        self.active = False

    def __enter__(self):
        self.active = True
        return self

    def __exit__(self, *exc):
        self.active = False


def _fmt(value, unit):
    text = f"{value:.4g}" if isinstance(value, float) else str(value)
    return f"{text} {unit}".strip()


class _Binding:
    def __init__(self, meta, commit, guard):
        self.meta = meta
        self._commit = commit
        self._guard = guard

    def commit(self, value):
        if not self._guard.active:
            self._commit(value)

    def refresh(self, value):
        raise NotImplementedError


class BoolBinding(_Binding):
    def __init__(self, meta, value, commit, guard):
        super().__init__(meta, commit, guard)
        self.model = ui.SimpleBoolModel(bool(value))
        ui.CheckBox(model=self.model, tooltip=meta.get("tooltip") or "")
        self.model.add_value_changed_fn(lambda m: self.commit(m.as_bool))

    def refresh(self, value):
        self.model.set_value(bool(value))


class FloatBinding(_Binding):
    """Linear or log-scale slider with a live value label."""

    def __init__(self, meta, value, commit, guard):
        super().__init__(meta, commit, guard)
        self.log = bool(meta.get("log_scale"))
        lo, hi = meta.get("min"), meta.get("max")
        tip = meta.get("tooltip") or ""
        self.model = ui.SimpleFloatModel(self._to_ui(value))
        if lo is not None and hi is not None:
            ui.FloatSlider(model=self.model, min=self._to_ui(lo), max=self._to_ui(hi), tooltip=tip)
        else:
            ui.FloatDrag(model=self.model, tooltip=tip)
        self.label = ui.Label(_fmt(float(value), meta.get("unit", "")), width=90)
        self.model.add_value_changed_fn(self._changed)

    def _to_ui(self, v):
        return math.log10(max(float(v), 1e-12)) if self.log else float(v)

    def _from_ui(self, v):
        return 10.0 ** v if self.log else v

    def _changed(self, model):
        value = self._from_ui(model.as_float)
        self.label.text = _fmt(value, self.meta.get("unit", ""))
        self.commit(value)

    def refresh(self, value):
        self.model.set_value(self._to_ui(value))
        self.label.text = _fmt(float(value), self.meta.get("unit", ""))


class IntBinding(_Binding):
    def __init__(self, meta, value, commit, guard):
        super().__init__(meta, commit, guard)
        self.model = ui.SimpleIntModel(int(value))
        lo, hi = meta.get("min"), meta.get("max")
        ui.IntDrag(model=self.model, min=lo if lo is not None else 0, max=hi if hi is not None else 10**9,
                   tooltip=meta.get("tooltip") or "")
        self.model.add_value_changed_fn(lambda m: self.commit(m.as_int))

    def refresh(self, value):
        self.model.set_value(int(value))


class Vec3Binding(_Binding):
    def __init__(self, meta, value, commit, guard):
        super().__init__(meta, commit, guard)
        names = ("R", "G", "B") if meta.get("widget") == "color" else ("X", "Y", "Z")
        lo = meta.get("min") if meta.get("min") is not None else -1e6
        hi = meta.get("max") if meta.get("max") is not None else 1e6
        self.models = []
        for name, v in zip(names, value):
            ui.Label(name, width=12)
            model = ui.SimpleFloatModel(float(v))
            ui.FloatDrag(model=model, min=lo, max=hi, step=0.01)
            model.add_value_changed_fn(lambda _m: self.commit(tuple(x.as_float for x in self.models)))
            self.models.append(model)

    def refresh(self, value):
        for model, v in zip(self.models, value):
            model.set_value(float(v))


class ChoiceBinding(_Binding):
    def __init__(self, meta, value, commit, guard):
        super().__init__(meta, commit, guard)
        self.choices = list(meta["choices"])
        self.combo = ui.ComboBox(self.choices.index(value), *self.choices)
        self.combo.model.add_item_changed_fn(self._changed)

    def _changed(self, model, _item):
        self.commit(self.choices[model.get_item_value_model().as_int])

    def refresh(self, value):
        self.combo.model.get_item_value_model().set_value(self.choices.index(value))


class StringBinding(_Binding):
    def __init__(self, meta, value, commit, guard):
        super().__init__(meta, commit, guard)
        self.model = ui.SimpleStringModel(str(value))
        ui.StringField(model=self.model, tooltip=meta.get("tooltip") or "")
        self.model.add_end_edit_fn(lambda m: self.commit(m.as_string))

    def refresh(self, value):
        self.model.set_value(str(value))


def make_binding(f, value, commit, guard):
    """Build one labelled row for dataclass field ``f``."""
    meta = dict(f.metadata)
    default = f.default
    with ui.HStack(height=ROW_HEIGHT, spacing=4):
        ui.Label(meta.get("label", f.name), width=LABEL_WIDTH, tooltip=meta.get("tooltip") or "")
        if isinstance(default, bool):
            return BoolBinding(meta, value, commit, guard)
        if isinstance(default, int):
            return IntBinding(meta, value, commit, guard)
        if isinstance(default, float):
            return FloatBinding(meta, value, commit, guard)
        if isinstance(default, tuple):
            return Vec3Binding(meta, value, commit, guard)
        if meta.get("choices"):
            return ChoiceBinding(meta, value, commit, guard)
        return StringBinding(meta, value, commit, guard)
