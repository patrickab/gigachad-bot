"""`GET /api/config` hands the LearnHarness frame URL to host logins only."""

from types import SimpleNamespace

import pytest

from backend.identity import get_request_identity
from backend.routes.config import router
from backend.routes.deps import get_model_provider_store
from tests.conftest import request_identity, route_client

FRAME_URL = "https://gigachad-backend.tail.ts.net:8444"


@pytest.mark.parametrize(("has_host_access", "expected"), [(True, FRAME_URL), (False, None)])
def test_learnharness_url_is_the_frame_url_for_host_logins_only(monkeypatch, has_host_access, expected):
    monkeypatch.setenv("GIGACHAD_LEARNHARNESS_FRAME_URL", FRAME_URL)
    store = SimpleNamespace(load_defaults=lambda: {})
    client = route_client(router, overrides={get_model_provider_store: store, get_request_identity: request_identity(has_host_access)})

    assert client.get("/api/config").json()["learnharness_url"] == expected
