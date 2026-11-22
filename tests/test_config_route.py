"""`GET /api/config` hands the LearnHarness frame URL to host logins only."""

from types import SimpleNamespace

import pytest

from backend.identity import get_request_identity
from backend.routes.config import router
from backend.routes.deps import get_model_provider_store, get_user_preferences
from tests.conftest import request_identity, route_client

FRAME_URL = "https://gigachad-backend.tail.ts.net:8444"


@pytest.mark.parametrize(("has_host_access", "expected"), [(True, FRAME_URL), (False, None)])
def test_learnharness_url_is_the_frame_url_for_host_logins_only(monkeypatch, has_host_access, expected):
    monkeypatch.setenv("GIGACHAD_LEARNHARNESS_FRAME_URL", FRAME_URL)
    store = SimpleNamespace(load_defaults=lambda: {})
    overrides = {
        get_model_provider_store: store,
        get_request_identity: request_identity(has_host_access),
        get_user_preferences: SimpleNamespace(animated_background=lambda: True),
    }
    client = route_client(router, overrides=overrides)

    assert client.get("/api/config").json()["learnharness_url"] == expected


def test_animated_background_is_read_and_stored_per_user():
    state = {"on": False}
    preferences = SimpleNamespace(
        animated_background=lambda: state["on"],
        set_animated_background=lambda enabled: state.update(on=enabled),
    )
    overrides = {
        get_model_provider_store: SimpleNamespace(load_defaults=lambda: {}),
        get_request_identity: request_identity(False),
        get_user_preferences: preferences,
    }
    client = route_client(router, overrides=overrides)

    assert client.get("/api/config").json()["animated_background"] is False
    assert client.put("/api/config/animated-background", json={"enabled": True}).status_code == 204
    assert client.get("/api/config").json()["animated_background"] is True
