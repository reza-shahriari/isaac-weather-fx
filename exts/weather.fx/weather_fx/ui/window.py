"""Weather FX panel. Built entirely from the state dataclasses."""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import fields
from functools import partial

import carb
import omni.kit.app
import omni.ui as ui

from ..core.state import SECTION_TYPES
from .widgets import UpdateGuard, make_binding

SECTION_TITLES = {
    "general": "General",
    "sky": "Sky, sun and moon",
    "clouds": "Clouds",
    "fog": "Fog",
    "wind": "Wind",
    "rain": "Rain",
    "snow": "Snow",
    "lighting": "Lighting",
}


class WeatherWindow:
    def __init__(self, controller):
        self._c = controller
        self._bindings = {}
        self._guard = UpdateGuard()
        self._show_advanced = False
        self._stats_label = None
        self._preset_combo = None
        self._path_model = ui.SimpleStringModel(os.path.join(os.path.expanduser("~"), "weather_fx.json"))
        self._window = ui.Window("Weather FX", width=460, height=760)
        self._window.frame.set_build_fn(self._build)
        self._token = controller.on_change(self._on_state_changed)

    # ------------------------------------------------------------ public
    @property
    def visible(self):
        return self._window.visible

    @visible.setter
    def visible(self, value):
        self._window.visible = value

    def destroy(self):
        self._c.remove_on_change(self._token)
        self._bindings = {}
        self._window.destroy()
        self._window = None

    # ------------------------------------------------------------ build
    def _build(self):
        self._bindings = {}
        state = self._c.state
        with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF):
            with ui.VStack(spacing=6, height=0):
                self._build_toolbar()
                for name in SECTION_TYPES:
                    self._build_section(name, getattr(state, name))
                self._build_io()
                ui.Separator()
                self._stats_label = ui.Label("", word_wrap=True, height=0)
        self._update_stats()

    def _build_toolbar(self):
        with ui.HStack(height=24, spacing=4):
            ui.Label("Preset", width=50)
            self._preset_combo = ui.ComboBox(0, *self._c.list_presets())
            ui.Button("Apply", width=70, clicked_fn=self._apply_preset)
        with ui.HStack(height=24, spacing=4):
            ui.Button("Clear all", clicked_fn=self._c.clear)
            ui.Button("Randomize",
                      tooltip="A coherent random day: regime first, then the parameters within "
                              "it, so the cloud base matches the dew point and the rain comes "
                              "with the deck that is producing it. Leaves General alone.",
                      clicked_fn=self._randomize)
            ui.Button("Step 1/30 s", tooltip="Advance once (manual time source)",
                      clicked_fn=lambda: self._c.step(1.0 / 30.0))
            advanced = ui.SimpleBoolModel(self._show_advanced)
            ui.CheckBox(model=advanced, width=20)
            ui.Label("Advanced", width=70)
            advanced.add_value_changed_fn(self._toggle_advanced)

    def _build_section(self, name, values):
        with ui.CollapsableFrame(SECTION_TITLES.get(name, name), collapsed=(name == "general")):
            with ui.VStack(spacing=3, height=0):
                for f in fields(type(values)):
                    if f.metadata.get("advanced") and not self._show_advanced:
                        continue
                    binding = make_binding(f, getattr(values, f.name),
                                           partial(self._commit, name, f.name), self._guard)
                    self._bindings[(name, f.name)] = binding

    def _randomize(self):
        """Draw a new day. Unseeded on purpose -- the button is for exploring, and a caller that
        needs a reproducible one passes a seed to `controller.randomize(seed)`."""
        self._c.randomize()

    def _build_io(self):
        with ui.CollapsableFrame("Save / Load", collapsed=True):
            with ui.HStack(height=24, spacing=4):
                ui.StringField(model=self._path_model)
                ui.Button("Save", width=50, clicked_fn=self._save)
                ui.Button("Load", width=50, clicked_fn=self._load)

    # ------------------------------------------------------------ callbacks
    def _commit(self, section, key, value):
        try:
            self._c.configure(**{section: {key: value}})
        except Exception as exc:
            carb.log_warn(f"weather_fx: {section}.{key}: {exc}")

    def _on_state_changed(self, state, changed):
        with self._guard:
            for (section, key), binding in self._bindings.items():
                if section in changed:
                    binding.refresh(getattr(getattr(state, section), key))
        self._update_stats()

    def _apply_preset(self):
        names = self._c.list_presets()
        index = self._preset_combo.model.get_item_value_model().as_int
        self._c.apply_preset(names[index])

    def _toggle_advanced(self, model):
        self._show_advanced = model.as_bool

        async def rebuild():
            await omni.kit.app.get_app().next_update_async()
            if self._window is not None:
                self._window.frame.rebuild()

        asyncio.ensure_future(rebuild())

    def _save(self):
        path = self._path_model.as_string
        try:
            self._c.save(path)
            carb.log_info(f"weather_fx: saved {path}")
        except OSError as exc:
            carb.log_error(f"weather_fx: save failed: {exc}")

    def _load(self):
        path = self._path_model.as_string
        try:
            self._c.load(path)
        except (OSError, ValueError, KeyError) as exc:
            carb.log_error(f"weather_fx: load failed: {exc}")

    def _update_stats(self):
        if self._stats_label is None:
            return
        try:
            self._stats_label.text = "Stats: " + json.dumps(self._c.stats().get("viewport", {}))
        except Exception:
            pass
