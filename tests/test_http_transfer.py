"""TransferMiddleware: revalidation and gzip for whole JSON bodies, untouched streams and ranges."""

from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from backend.http_plumbing import TransferMiddleware

BIG = {"points": [[i * 0.123456789, i * 1.5] for i in range(2000)]}

app = FastAPI()
app.add_middleware(TransferMiddleware)


@app.get("/json")
def big_json() -> dict:
    return BIG


@app.post("/json")
def posted_json() -> dict:
    return BIG


@app.get("/stream")
def stream() -> StreamingResponse:
    return StreamingResponse(iter([b"data: a\n\n", b"data: b\n\n"]), media_type="text/event-stream")


@app.get("/range")
def ranged() -> Response:
    return Response(b"x" * 4096, status_code=206, media_type="text/plain", headers={"Content-Range": "bytes 0-4095/9000"})


client = TestClient(app)


def test_unchanged_json_revalidates_to_an_empty_304():
    first = client.get("/json")
    assert first.json() == BIG
    assert first.headers["cache-control"] == "private, no-cache"

    reopened = client.get("/json", headers={"If-None-Match": first.headers["etag"]})
    assert (reopened.status_code, reopened.content, reopened.headers["etag"]) == (304, b"", first.headers["etag"])
    assert client.get("/json", headers={"If-None-Match": 'W/"stale"'}).status_code == 200


def test_json_is_gzipped_only_for_clients_that_accept_it():
    zipped = client.get("/json", headers={"Accept-Encoding": "gzip"})
    assert zipped.headers["content-encoding"] == "gzip"
    assert zipped.json() == BIG
    assert int(zipped.headers["content-length"]) < len(zipped.content) // 2

    plain = client.get("/json", headers={"Accept-Encoding": "identity"})
    assert "content-encoding" not in plain.headers
    assert plain.headers["etag"] == zipped.headers["etag"]


def test_non_get_responses_are_compressed_but_never_validated():
    posted = client.post("/json", headers={"Accept-Encoding": "gzip"})
    assert posted.headers["content-encoding"] == "gzip"
    assert "etag" not in posted.headers


def test_streams_and_ranges_pass_through_unchanged():
    streamed = client.get("/stream", headers={"Accept-Encoding": "gzip"})
    assert streamed.content == b"data: a\n\ndata: b\n\n"
    assert "content-encoding" not in streamed.headers

    ranged_response = client.get("/range", headers={"Accept-Encoding": "gzip"})
    assert (ranged_response.status_code, ranged_response.content) == (206, b"x" * 4096)
    assert "content-encoding" not in ranged_response.headers
    assert "etag" not in ranged_response.headers
