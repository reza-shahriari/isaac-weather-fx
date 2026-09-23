"""Paste into Window > Script Editor with the Weather FX extension enabled."""
from weather_fx import api as weather

wx = weather.get_controller()

# One call, several sections
wx.configure(
    fog={"enabled": True, "visibility_m": 250},
    rain={"enabled": True, "rate_mm_h": 25},
    wind={"speed_mps": 5, "direction_deg": 30, "gust_strength": 0.3},
    lighting={"enabled": True, "light_scale": 0.5},
)

print(wx.stats())

# Later:
# wx.apply_preset("blizzard")
# wx.follow("/World/Robot/camera_link/Camera")   # particles follow a robot camera
# wx.save("/tmp/my_weather.json")
# wx.clear()
