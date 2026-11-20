"""Power-grid final engine (accepted scoring core, explicit inputs)."""

from terag.engines.grid.config_factory import make_grid_config
from terag.engines.grid.engine import GridEngine

__all__ = ["GridEngine", "make_grid_config"]
