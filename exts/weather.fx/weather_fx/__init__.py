"""Weather FX: fog, rain and snow for Isaac Sim.

The ``core`` package is plain Python + numpy and can be imported anywhere
(tests, tooling). Everything that touches Omniverse lives in ``runtime``,
``backends``, ``ui`` and ``api`` and is only imported inside Kit.
"""
__version__ = "0.1.0"

try:
    import omni.ext  # noqa: F401
except ImportError:  # plain Python: core only
    pass
else:
    from .extension import WeatherFxExtension  # noqa: F401
