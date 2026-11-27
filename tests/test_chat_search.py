"""Listing and searching every chat of a login: `GET /api/chat-histories/all` and `/search` over the stored chat documents."""

import json

from psycopg_pool import ConnectionPool
import pytest

from backend.routes import histories
from backend.routes.deps import get_user_scope
from lib.changes import UserScope
from lib.chat_store import SEARCH_MATCH_CAP, ChatStore
from lib.postgres_data_store import PostgresDataStore
from tests.conftest import create_user, route_client

pytestmark = pytest.mark.usefixtures("clean_database")


def pair(question: str, answer: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def write(store: PostgresDataStore, key: str, data: object) -> None:
    store.write_bytes(key, json.dumps(data).encode())


def client_for(pool: ConnectionPool, user_id):
    return route_client(histories.router, overrides={get_user_scope: UserScope(pool, user_id)})


@pytest.fixture
def alice(postgres_pool: ConnectionPool):
    return create_user(postgres_pool, "alice@example.test")


@pytest.fixture
def alice_store(postgres_pool: ConnectionPool, alice) -> PostgresDataStore:
    return PostgresDataStore(UserScope(postgres_pool, alice))


def test_all_lists_every_chat_across_projects_newest_first(postgres_pool, alice, alice_store) -> None:
    write(alice_store, "chat/paper/project.json", {"name": "Paper", "tabs": [{"filename": "draft.json", "name": "Draft tab"}]})
    write(alice_store, "chat/loose.json", {"messages": pair("q", "a"), "title": "Loose chat"})
    write(alice_store, "chat/paper/draft.json", {"messages": pair("q", "a"), "title": "Stored title"})
    write(alice_store, "chat/paper/untitled.json", [*pair("q", "a")])
    write(alice_store, "chat/paper/_uploads/x.json", {"messages": []})
    write(alice_store, "chat/folder/inner.json", {"messages": [], "title": "Inner"})

    chats = client_for(postgres_pool, alice).get("/api/chat-histories/all").json()["chats"]

    assert [(c["historyFile"], c["projectSlug"], c["title"]) for c in chats] == [
        ("folder/inner.json", None, "Inner"),
        ("paper/untitled.json", "paper", "untitled"),
        ("paper/draft.json", "paper", "Draft tab"),
        ("loose.json", None, "Loose chat"),
    ]
    assert all(c["updatedAt"] for c in chats)
    assert chats[0]["updatedAt"] >= chats[-1]["updatedAt"]


def test_search_finds_pairs_with_qa_index_role_and_ordinal(postgres_pool, alice, alice_store) -> None:
    write(alice_store, "chat/paper/project.json", {"name": "Paper", "tabs": []})
    messages = [
        *pair("first question", "nothing here"),
        *pair("about Kafka", "kafka and KAFKA again,\nkafka"),
        *pair("other", "no"),
    ]
    write(alice_store, "chat/paper/notes.json", {"messages": messages, "title": "Notes"})
    write(alice_store, "chat/plain.json", {"messages": pair("unrelated", "text")})

    response = client_for(postgres_pool, alice).get("/api/chat-histories/search", params={"q": "kafka"})

    assert response.status_code == 200
    [hit] = response.json()["hits"]
    assert (hit["historyFile"], hit["projectSlug"], hit["title"]) == ("paper/notes.json", "paper", "Notes")
    assert [(m["qaIndex"], m["role"], m["ordinal"]) for m in hit["matches"]] == [
        (1, "user", 0),
        (1, "assistant", 0),
        (1, "assistant", 1),
        (1, "assistant", 2),
    ]
    assert hit["matches"][3]["snippet"] == "kafka and KAFKA again, kafka"


def test_search_counts_pairs_as_the_frontend_does(alice_store) -> None:
    # A leading system message shifts the pairs: only a user message at an even index starts one.
    messages = [{"role": "system", "content": "needle"}, *pair("skipped needle", "x"), "junk", *pair("q", "needle")]
    write(alice_store, "chat/odd.json", {"messages": [*pair("needle", "a"), *pair("b", "needle")]})
    write(alice_store, "chat/shifted.json", {"messages": messages})

    hits = {hit["historyFile"]: hit["matches"] for hit in ChatStore(data_store=alice_store).search("needle", 20)}

    assert [(m["qaIndex"], m["role"]) for m in hits["odd.json"]] == [(0, "user"), (1, "assistant")]
    assert [(m["qaIndex"], m["role"]) for m in hits["shifted.json"]] == [(0, "assistant")]


def test_snippet_is_one_bounded_line(alice_store) -> None:
    text = "a" * 500 + "\n\nNeedle\t" + "b" * 500
    write(alice_store, "chat/long.json", {"messages": pair("q", text)})

    [hit] = ChatStore(data_store=alice_store).search("needle", 20)
    snippet = hit["matches"][0]["snippet"]

    assert "\n" not in snippet and "Needle" in snippet
    assert snippet.startswith("…") and snippet.endswith("…") and len(snippet) < 140


def test_another_logins_chats_are_never_returned(postgres_pool, alice, alice_store) -> None:
    bob = create_user(postgres_pool, "bob@example.test")
    bob_store = PostgresDataStore(UserScope(postgres_pool, bob))
    write(bob_store, "chat/bob.json", {"messages": pair("secret needle", "a"), "title": "Bob"})
    write(alice_store, "chat/alice.json", {"messages": pair("q", "a"), "title": "Alice"})

    client = client_for(postgres_pool, alice)

    assert [c["historyFile"] for c in client.get("/api/chat-histories/all").json()["chats"]] == ["alice.json"]
    assert client.get("/api/chat-histories/search", params={"q": "needle"}).json() == {"hits": []}


def test_limit_bounds_chats_and_matches_are_capped(postgres_pool, alice, alice_store) -> None:
    for index in range(5):
        write(alice_store, f"chat/c{index}.json", {"messages": pair("needle " * 30, "needle")})
    client = client_for(postgres_pool, alice)

    hits = client.get("/api/chat-histories/search", params={"q": "needle", "limit": 3}).json()["hits"]

    assert [hit["historyFile"] for hit in hits] == ["c4.json", "c3.json", "c2.json"]
    assert all(len(hit["matches"]) == SEARCH_MATCH_CAP for hit in hits)
    assert client.get("/api/chat-histories/search", params={"q": "needle", "limit": 0}).status_code == 422
    assert client.get("/api/chat-histories/search", params={"q": "needle", "limit": 101}).status_code == 422


@pytest.mark.parametrize("params", [{"q": ""}, {"q": "   "}, {}])
def test_empty_query_is_refused(postgres_pool, alice, params) -> None:
    response = client_for(postgres_pool, alice).get("/api/chat-histories/search", params=params)

    assert response.status_code == 400


def test_query_over_200_characters_is_refused(postgres_pool, alice) -> None:
    client = client_for(postgres_pool, alice)

    assert client.get("/api/chat-histories/search", params={"q": "x" * 201}).status_code == 422
    assert client.get("/api/chat-histories/search", params={"q": "x" * 200}).status_code == 200
