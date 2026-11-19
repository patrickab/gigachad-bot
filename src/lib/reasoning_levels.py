"""Which `reasoning_effort` levels a model accepts, read from the gateway's `/v1/models` `thinking_levels`."""

from typing import Any

from lib import pi_models

# The app's "off": a chat request at this level omits `reasoning_effort` entirely.
NONE = "none"
PI_OFF = "off"


def levels(model: str) -> list[str]:
    """Effort levels of stored selector `model`, Pi's `off` as the app's `none`, empty when the gateway lacks it or is down."""
    gateway_levels = pi_models.thinking_levels(pi_models.gateway_model_id(model)) or []
    return [NONE if level == PI_OFF else level for level in gateway_levels]


def effort_kwargs(effort: str | None) -> dict[str, Any]:
    """The completion kwargs carrying `effort`, empty when it is unset or off (the gateway clamps the rest)."""
    if effort and effort not in (NONE, PI_OFF):
        return {"reasoning_effort": effort}
    return {}
