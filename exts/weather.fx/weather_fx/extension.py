"""Kit entry point: menu item + window, sharing the API controller."""
import carb
import carb.settings
import omni.ext

from . import api

SETTING_OPEN_ON_STARTUP = "/exts/weather.fx/open_window_on_startup"
MENU_PARENT = "Window"


class WeatherFxExtension(omni.ext.IExt):
    def on_startup(self, ext_id):
        self._controller = api.get_controller()
        self._window = None
        self._menu_items = []
        try:
            from omni.kit.menu.utils import MenuItemDescription, add_menu_items

            self._menu_items = [MenuItemDescription(name="Weather FX", onclick_fn=self.show_window)]
            add_menu_items(self._menu_items, MENU_PARENT)
        except Exception as exc:  # headless apps may have no menu
            carb.log_info(f"weather_fx: no menu ({exc})")
        if carb.settings.get_settings().get(SETTING_OPEN_ON_STARTUP):
            self.show_window()

    def show_window(self, *_):
        if self._window is None:
            from .ui.window import WeatherWindow

            self._window = WeatherWindow(self._controller)
        self._window.visible = True

    def on_shutdown(self):
        if self._menu_items:
            try:
                from omni.kit.menu.utils import remove_menu_items

                remove_menu_items(self._menu_items, MENU_PARENT)
            except Exception:
                pass
        if self._window is not None:
            self._window.destroy()
            self._window = None
        api.shutdown_controller()
