"""RTX "Simple Fog" setting paths, kept in one place.

These are the paths the Render Settings > Common > Simple Fog panel is
expected to use. Verify them once on your install with
``tools/dump_rtx_fog_settings.py`` and edit this map if they differ.
Paths that do not exist at runtime are skipped with a single warning.
"""
FOG_SETTINGS = {
    "enabled": "/rtx/fog/enabled",
    "color": "/rtx/fog/fogColor",
    "color_intensity": "/rtx/fog/fogColorIntensity",
    "z_up": "/rtx/fog/fogZup/enabled",
    "start_distance": "/rtx/fog/fogStartDist",
    "end_distance": "/rtx/fog/fogEndDist",
    "distance_density": "/rtx/fog/fogDistanceDensity",
    "start_height": "/rtx/fog/fogStartHeight",
    "height_density": "/rtx/fog/fogHeightDensity",
    "height_falloff": "/rtx/fog/fogHeightFalloff",
}
