"""Optional, manifest-gated native GGUF runtime.

The ordinary plugin needs no native bundle. A prepared bundle opts the exact
tested checkpoint into the native implementation; other models retain Humming.
"""

from .policy import configure_engine_args
from .runtime import activate_for_model, install_scheduler_hook

__all__ = ["activate_for_model", "configure_engine_args", "install_scheduler_hook"]
