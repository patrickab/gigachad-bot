/** `removeTurn` on small literal entry lists: one test per rule of the tree. */
import { describe, expect, test } from "bun:test";
import { pathToRoot } from "../src/host/entries.ts";
import { removeTurn, type TurnEntry } from "../src/host/local/turns.ts";

const user = (id: string, parentId: string | null): TurnEntry => ({ type: "message", id, parentId, message: { role: "user", content: `ask ${id}` } });
const answer = (id: string, parentId: string): TurnEntry => ({ type: "message", id, parentId, message: { role: "assistant", content: `answer ${id}` } });
const tool = (id: string, parentId: string): TurnEntry => ({ type: "message", id, parentId, message: { role: "toolResult", content: `output ${id}` } });
const entry = (type: string, id: string, parentId: string | null, fields: Record<string, unknown> = {}): TurnEntry => ({
  type,
  id,
  parentId,
  ...fields,
});

/** Three turns: m (model) → u1 a1 → u2 t2 a2 → u3 a3. */
function threeTurns(): TurnEntry[] {
  return [
    entry("model_change", "m", null),
    user("u1", "m"),
    answer("a1", "u1"),
    user("u2", "a1"),
    tool("t2", "u2"),
    answer("a2", "t2"),
    user("u3", "a2"),
    answer("a3", "u3"),
  ];
}

function removed(entries: TurnEntry[], id: string, through: "turn" | "end" = "turn") {
  const result = removeTurn(entries, id, through);
  if (!result.ok) throw new Error(`refused: ${result.reason}`);
  return result;
}

const ids = (entries: TurnEntry[]) => entries.map((e) => e.id);
const parents = (entries: TurnEntry[]) => Object.fromEntries(entries.map((e) => [e.id, e.parentId]));

describe("removeTurn", () => {
  test("turn: removes the prompt up to the next prompt, the next entry takes the range's parent", () => {
    const result = removed(threeTurns(), "u2");
    expect(ids(result.entries)).toEqual(["m", "u1", "a1", "u3", "a3"]);
    expect(parents(result.entries).u3).toBe("a1");
    expect(ids(result.removed)).toEqual(["u2", "t2", "a2"]);
    expect(result.leafId).toBe("a3");
  });

  test("turn: the last turn leaves its parent as leaf", () => {
    const result = removed(threeTurns(), "u3");
    expect(ids(result.entries)).toEqual(["m", "u1", "a1", "u2", "t2", "a2"]);
    expect(result.leafId).toBe("a2");
  });

  test("turn: entries it keeps are the same objects, the input is not changed", () => {
    const entries = threeTurns();
    const copy = structuredClone(entries);
    const result = removed(entries, "u2");
    expect(result.entries[0]).toBe(entries[0]!);
    expect(entries).toEqual(copy);
  });

  test("end: removes the prompt and everything after it, the entry before it becomes the leaf", () => {
    const result = removed(threeTurns(), "u2", "end");
    expect(ids(result.entries)).toEqual(["m", "u1", "a1"]);
    expect(ids(result.removed)).toEqual(["u2", "t2", "a2", "u3", "a3"]);
    expect(result.leafId).toBe("a1");
  });

  test("turn stops at a compaction", () => {
    const entries = [
      ...threeTurns().slice(0, 6),
      entry("compaction", "c", "a2", { firstKeptEntryId: "u2", summary: "u1" }),
      user("u3", "c"),
    ];
    const result = removed(entries, "u2");
    expect(ids(result.entries)).toEqual(["m", "u1", "a1", "c", "u3"]);
    expect(result.entries[3]).toMatchObject({ parentId: "a1", firstKeptEntryId: "c" });
  });

  test("side branches rooted in the range go with it, those before it stay", () => {
    const entries = [
      ...threeTurns(),
      // a retry of u2's answer, off the active path
      answer("a2b", "u2"),
      user("u2c", "a2b"),
      // a sibling of u2, rooted before the range
      user("u2x", "a1"),
    ];
    // keep u3's answer the leaf: the last line of the file
    entries.push(answer("a3b", "a3"));
    const result = removed(entries, "u2");
    expect(ids(result.entries)).toEqual(["m", "u1", "a1", "u3", "a3", "u2x", "a3b"]);
    expect(ids(result.removed)).toEqual(["u2", "t2", "a2", "a2b", "u2c"]);
  });

  test("turn: model, thinking and session info changes in the range stay, re-linked", () => {
    const entries = [
      user("u1", null),
      answer("a1", "u1"),
      user("u2", "a1"),
      entry("model_change", "mc", "u2"),
      entry("thinking_level_change", "tl", "mc"),
      answer("a2", "tl"),
      entry("session_info", "si", "a2", { name: "n" }),
      entry("custom", "cu", "si"),
      entry("custom_message", "cm", "cu"),
      entry("branch_summary", "bs", "cm"),
      user("u3", "bs"),
    ];
    const result = removed(entries, "u2");
    expect(ids(result.entries)).toEqual(["u1", "a1", "mc", "tl", "si", "u3"]);
    expect(parents(result.entries)).toEqual({ u1: null, a1: "u1", mc: "a1", tl: "mc", si: "tl", u3: "si" });
    expect(ids(result.removed)).toEqual(["u2", "a2", "cu", "cm", "bs"]);
  });

  test("end: state changes after the prompt go too", () => {
    const entries = [user("u1", null), answer("a1", "u1"), user("u2", "a1"), entry("model_change", "mc", "u2"), answer("a2", "mc")];
    expect(ids(removed(entries, "u2", "end").entries)).toEqual(["u1", "a1"]);
  });

  test("labels and context edits whose target is gone go, the others stay", () => {
    const entries = [
      ...threeTurns(),
      entry("label", "l1", "a3", { targetId: "a2", label: "x" }),
      entry("label", "l2", "l1", { targetId: "a1", label: "y" }),
      entry("context_edit", "e1", "l2", { targetId: "t2", replacement: null }),
      entry("context_edit", "e2", "e1", { targetId: "a3", replacement: null }),
      // a label of the removed label
      entry("label", "l3", "e2", { targetId: "l1", label: "z" }),
    ];
    const result = removed(entries, "u2");
    expect(ids(result.entries)).toEqual(["m", "u1", "a1", "u3", "a3", "l2", "e2"]);
    expect(parents(result.entries)).toMatchObject({ l2: "a3", e2: "l2" });
    expect(result.leafId).toBe("e2");
  });

  test("turn: a label in the range whose target stays is kept, re-linked", () => {
    const entries = threeTurns();
    entries.splice(5, 0, entry("label", "l", "t2", { targetId: "u1", label: "x" }));
    entries[6] = answer("a2", "l");
    const result = removed(entries, "u2");
    expect(ids(result.entries)).toEqual(["m", "u1", "a1", "l", "u3", "a3"]);
    expect(parents(result.entries)).toMatchObject({ l: "a1", u3: "l" });
  });

  test("a compaction's first kept entry in the range moves to the next surviving entry", () => {
    const entries = [
      ...threeTurns(),
      entry("compaction", "c", "a3", { firstKeptEntryId: "u2", summary: "u1 only" }),
      user("u4", "c"),
    ];
    const result = removed(entries, "u2");
    expect(result.entries.find((e) => e.id === "c")?.firstKeptEntryId).toBe("u3");
  });

  test("a first kept entry with nothing surviving before the compaction moves to the compaction", () => {
    const entries = [
      user("u1", null),
      answer("a1", "u1"),
      entry("compaction", "c", "a1", { firstKeptEntryId: "u1", summary: "none" }),
      user("u2", "c"),
    ];
    const result = removed(entries, "u1");
    expect(ids(result.entries)).toEqual(["c", "u2"]);
    expect(result.entries[0]).toMatchObject({ parentId: null, firstKeptEntryId: "c" });
  });

  test("end: the leaf moves to the last line when a side branch follows it in the file", () => {
    const entries = [
      user("u1", null),
      answer("a1", "u1"),
      user("u2x", "a1"),
      answer("a2x", "u2x"),
      user("u2", "a1"),
      answer("a2", "u2"),
    ];
    const result = removed(entries, "u2", "end");
    expect(ids(result.entries)).toEqual(["u1", "u2x", "a2x", "a1"]);
    expect(result.leafId).toBe("a1");
  });

  describe("refuses", () => {
    test("an unknown id", () => {
      expect(removeTurn(threeTurns(), "nope", "turn")).toEqual({ ok: false, reason: "unknown" });
    });

    test("an entry that is no user prompt", () => {
      expect(removeTurn(threeTurns(), "a2", "turn")).toEqual({ ok: false, reason: "not_prompt" });
      expect(removeTurn(threeTurns(), "m", "end")).toEqual({ ok: false, reason: "not_prompt" });
    });

    test("a prompt off the active path", () => {
      const entries = [user("u1", null), answer("a1", "u1"), user("u2x", "a1"), user("u2", "a1"), answer("a2", "u2")];
      expect(removeTurn(entries, "u2x", "turn")).toEqual({ ok: false, reason: "off_branch" });
    });

    test("a turn a surviving compaction summarized", () => {
      const entries = [
        ...threeTurns(),
        entry("compaction", "c", "a3", { firstKeptEntryId: "u3", summary: "u1 and u2" }),
        user("u4", "c"),
      ];
      expect(removeTurn(entries, "u2", "turn")).toEqual({ ok: false, reason: "compacted" });
      expect(removeTurn(entries, "u1", "turn")).toEqual({ ok: false, reason: "compacted" });
      // the kept turn and the turn after the compaction can go
      expect(removeTurn(entries, "u3", "turn").ok).toBe(true);
      expect(removeTurn(entries, "u4", "turn").ok).toBe(true);
    });

    test("not when the compaction goes with the edit", () => {
      const entries = [
        ...threeTurns(),
        entry("compaction", "c", "a3", { firstKeptEntryId: "u3", summary: "u1 and u2" }),
        user("u4", "c"),
      ];
      const result = removed(entries, "u3", "end");
      expect(ids(result.entries)).toEqual(["m", "u1", "a1", "u2", "t2", "a2"]);
    });
  });
});

test("pathToRoot ends at a repeated id on a cyclic parent chain", () => {
  const entries = [user("u1", "a1"), answer("a1", "u1")];
  const path = pathToRoot(new Map(entries.map((e) => [e.id, e])), "a1");
  expect(ids(path)).toEqual(["u1", "a1"]);
});
