"""H-search reliability gating layer (post-hoc score transforms over the
frozen engine's channel outputs; see gating.py)."""
from terag.engines.hgate.gating import predict_gated, merge_hg, HG_DEFAULT, HGError

__all__ = ['predict_gated', 'merge_hg', 'HG_DEFAULT', 'HGError']
