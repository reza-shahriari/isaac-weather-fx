"""The test scene shared by the demo and the capture script (Z-up, meters).

A ground slab, red boxes at increasing distance to show fog falloff, a sun,
an HDR sky dome and a camera looking down the row of boxes.
"""
from pxr import Gf, UsdGeom, UsdLux

CAMERA_PATH = "/World/Camera"
DEFAULT_SKY = "/NVIDIA/Assets/Skies/Cloudy/kloofendal_48d_partly_cloudy_4k.hdr"


def resolve_sky(sky: str) -> str:
    """'auto' -> the default Isaac Sim sky, 'none' -> '', a path or URL is used as is."""
    if sky == "none":
        return ""
    if sky != "auto":
        return sky
    try:
        from isaacsim.storage.native import get_assets_root_path

        root = get_assets_root_path()
    except Exception as exc:
        print(f"[scene] no Isaac Sim assets root ({exc}); using a plain sky color")
        return ""
    return root + DEFAULT_SKY if root else ""


def build_scene(stage, sky: str = "auto") -> None:
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.AddScaleOp().Set(Gf.Vec3f(50, 50, 0.05))
    for i, x in enumerate(range(5, 105, 20)):
        box = UsdGeom.Cube.Define(stage, f"/World/Box_{i}")
        box.AddTranslateOp().Set(Gf.Vec3d(x, 0, 1))
        box.GetDisplayColorAttr().Set([Gf.Vec3f(0.8, 0.2, 0.1)])

    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(3000)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-45, 0, 30))

    dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
    dome.CreateIntensityAttr(1000)
    texture = resolve_sky(sky)
    if texture:
        dome.CreateTextureFileAttr(texture)
        dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
    else:
        dome.CreateColorAttr(Gf.Vec3f(0.45, 0.6, 0.85))

    camera = UsdGeom.Camera.Define(stage, CAMERA_PATH)
    UsdGeom.XformCommonAPI(camera).SetTranslate(Gf.Vec3d(-8, 0, 2))
    UsdGeom.XformCommonAPI(camera).SetRotate(Gf.Vec3f(90, 0, -90))
