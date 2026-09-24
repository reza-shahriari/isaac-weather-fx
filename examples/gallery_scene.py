"""The showcase scene for the README gallery: a horizon, and nothing that lights it.

Deliberately different from :mod:`demo_scene`, which carries its own sun and its own HDR dome so
the fog and rain effects have something to be seen against. This scene has **no lights at all**.
Every photon in a gallery frame comes from the sky weather-fx authored -- the Perez dome, the sun
and the moon under ``/WeatherFX/Sky`` -- because a showcase that quietly leaves a default
`DistantLight` in the stage is showing you two suns and crediting one.

**The cameras follow the light.** A scenario is a place, a date and an hour, so the sun is wherever
the ephemeris puts it, which on one afternoon is behind the camera and on another in front of it.
Aiming the cameras at a fixed compass bearing would therefore photograph the sunset from behind,
and the gallery would show a model that cannot do sunsets. :func:`aim_sky_camera` and
:func:`pose_subject` take the bearing of whichever body is lighting the scene and frame it.
"""
import math

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

SKY_CAMERA = "/World/CameraSky"
SUBJECT_CAMERA = "/World/CameraSubject"
SUBJECT_ROOT = "/World/Subject"

#: Close to ``SkyParams.ground_albedo``. The dome bake hazes the lower hemisphere toward this
#: value, so a ground plane of a different colour makes the horizon a visible seam.
GROUND_COLOUR = (0.19, 0.20, 0.15)

#: Eye height, and how far up the wide camera tilts. Two thirds sky, one third ground.
EYE_M = (0.0, 0.0, 1.7)
SKY_TILT_DEG = 22.0
SKY_FOCAL_MM = 18.0
#: Degrees the wide camera looks round from the light. Straight into the sun backlights every
#: cloud in frame and flattens the sky to one hue; straight away from it is a flat blue wall.
#: Just over a quarter turn puts the warm side and the blue side in the same photograph, which is
#: how a sunset is usually shot and the only framing that shows this model doing both at once.
SKY_OFFSET_DEG = 58.0

#: The subject frame: **a short lens close in**, not a long lens far away, and that is not a style
#: choice. The sky behind the subject is a lat-long dome texture, so the background's sharpness is
#: set by how much the lens magnifies it. An 85 mm lens eight metres back frames a 0.45 m aircraft
#: the same way and magnifies the sky six times more, turning a cloud into visible texels.
SUBJECT_FOCAL_MM = 35.0
SUBJECT_RANGE_M = 3.4
SUBJECT_ELEVATION_DEG = 26.0
#: Degrees the subject sits round from the light, so it is lit across the frame rather than from
#: directly behind the camera. Flat frontal light on a white aircraft is a cut-out.
SUBJECT_OFFSET_DEG = 42.0


def _direction(azimuth_deg: float, elevation_deg: float) -> Gf.Vec3d:
    """ENU, Z-up: azimuth from north (+Y) toward east (+X)."""
    azimuth = math.radians(float(azimuth_deg))
    elevation = math.radians(float(elevation_deg))
    return Gf.Vec3d(
        math.cos(elevation) * math.sin(azimuth),
        math.cos(elevation) * math.cos(azimuth),
        math.sin(elevation),
    )


def _look_at(prim: object, eye: Gf.Vec3d, target: Gf.Vec3d) -> None:
    """Aim a camera down its own -Z at ``target``, Z-up."""
    view = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0.0, 0.0, 1.0))
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(view.GetInverse())


def _camera(stage: object, path: str, focal_mm: float) -> object:
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr(float(focal_mm))
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(11.787)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1_000_000.0))
    return camera


def _matte(stage: object, path: str, colour: tuple) -> object:
    """A plain rough dielectric. **Not a display colour**: RTX shades an unbound prim with a
    default near-white material and ignores `primvars:displayColor`, so a ground authored at a
    0.19 albedo renders four times too bright and the sky it is supposed to sit under looks dim
    by comparison. One bound material is the difference between a horizon and a light box.
    """
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.92)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(0.05)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def build_scene(stage: object, *, ground_km: float = 12.0, silhouettes: bool = True) -> None:
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.DefinePrim("/World", "Xform")

    half = 0.5 * ground_km * 1000.0
    ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
    ground.CreatePointsAttr(
        [
            Gf.Vec3f(-half, -half, 0.0),
            Gf.Vec3f(half, -half, 0.0),
            Gf.Vec3f(half, half, 0.0),
            Gf.Vec3f(-half, half, 0.0),
        ]
    )
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)] * 4)
    ground.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    ground.CreateDisplayColorAttr([Gf.Vec3f(*GROUND_COLOUR)])
    ground.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(
        _matte(stage, "/World/Looks/Ground", GROUND_COLOUR)
    )

    if silhouettes:
        # A treeline all the way round, because the camera turns to follow the light and a row
        # laid out along one bearing would be behind it half the time. Haze and fog are *depth*
        # cues: with nothing at distance in the frame, 300 m of visibility and 30 km look alike.
        stage.DefinePrim("/World/Treeline", "Xform")
        stand_material = _matte(stage, "/World/Looks/Treeline", (0.055, 0.075, 0.045))
        index = 0
        for ring, (distance, height, width) in enumerate(
            ((260.0, 9.0, 34.0), (620.0, 13.0, 70.0), (1250.0, 19.0, 130.0), (2400.0, 28.0, 240.0))
        ):
            for step in range(12):
                bearing = math.radians(step * 30.0 + ring * 11.0)
                jitter = 1.0 + 0.35 * math.sin(step * 2.3 + ring)
                box = UsdGeom.Cube.Define(stage, f"/World/Treeline/Stand_{index}")
                UsdGeom.XformCommonAPI(box).SetTranslate(
                    Gf.Vec3d(
                        distance * math.sin(bearing),
                        distance * math.cos(bearing),
                        0.5 * height * jitter,
                    )
                )
                UsdGeom.XformCommonAPI(box).SetScale(
                    Gf.Vec3f(0.5 * width, 0.5 * width, 0.5 * height * jitter)
                )
                UsdGeom.XformCommonAPI(box).SetRotate(Gf.Vec3f(0.0, 0.0, -math.degrees(bearing)))
                box.CreateDisplayColorAttr([Gf.Vec3f(0.06, 0.08, 0.05)])
                UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(stand_material)
                index += 1

    _camera(stage, SKY_CAMERA, SKY_FOCAL_MM)
    _camera(stage, SUBJECT_CAMERA, SUBJECT_FOCAL_MM)
    stage.DefinePrim(SUBJECT_ROOT, "Xform")
    aim_sky_camera(stage, 180.0)


def aim_sky_camera(stage: object, azimuth_deg: float, *, tilt_deg: float = SKY_TILT_DEG,
                   offset_deg: float = SKY_OFFSET_DEG) -> None:
    """Point the wide camera off the light's bearing by ``offset_deg``, tilted up."""
    eye = Gf.Vec3d(*EYE_M)
    bearing = float(azimuth_deg) + float(offset_deg)
    _look_at(stage.GetPrimAtPath(SKY_CAMERA), eye, eye + 3000.0 * _direction(bearing, tilt_deg))


def reference_subject(stage: object, usd_path: str) -> None:
    """Bring a prepared asset in under :data:`SUBJECT_ROOT`."""
    root = stage.GetPrimAtPath(SUBJECT_ROOT) or stage.DefinePrim(SUBJECT_ROOT, "Xform")
    root.GetReferences().ClearReferences()
    root.GetReferences().AddReference(usd_path)


def pose_subject(
    stage: object,
    azimuth_deg: float,
    *,
    range_m: float = SUBJECT_RANGE_M,
    elevation_deg: float = SUBJECT_ELEVATION_DEG,
    bank_deg: float = 13.0,
    pitch_deg: float = -7.0,
    heading_offset_deg: float = 205.0,
) -> object:
    """Put the subject on a bearing from the camera, bank it, and frame it.

    The attitude is not decoration. An aircraft photographed dead level reads as a model on a
    stand; a few degrees of bank and a nose-down pitch read as one that is flying, and the frame
    is then about the sky it is flying in.

    ``azimuth_deg`` is the bearing of the *light*; the subject is placed
    :data:`SUBJECT_OFFSET_DEG` round from it so the light crosses the airframe.
    """
    bearing = float(azimuth_deg) + SUBJECT_OFFSET_DEG
    eye = Gf.Vec3d(*EYE_M)
    position = eye + float(range_m) * _direction(bearing, elevation_deg)

    root = stage.GetPrimAtPath(SUBJECT_ROOT)
    api = UsdGeom.XformCommonAPI(root)
    api.SetRotate(Gf.Vec3f(float(pitch_deg), float(bank_deg), bearing + heading_offset_deg))
    api.SetTranslate(position)
    # Recentre on the archive's own centroid, so the framing does not depend on where whichever
    # exporter produced it happened to leave the origin.
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    box = cache.ComputeWorldBound(root).ComputeAlignedRange()
    mid = box.GetMidpoint()
    api.SetTranslate(position + (position - Gf.Vec3d(mid[0], mid[1], mid[2])))

    _look_at(stage.GetPrimAtPath(SUBJECT_CAMERA), eye, position)
    return box.GetSize()
