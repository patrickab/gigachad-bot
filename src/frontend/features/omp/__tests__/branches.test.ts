import { describe, expect, it } from "vitest"
import { sessionBranches } from "@/features/omp/branches"
import type { BranchNode } from "@gigachad/chat-ui/protocol"

function entry(id: string, role: "user" | "assistant" | undefined, preview: string, children: BranchNode[] = []): BranchNode {
  return { id, type: role ? "message" : "model_change", role, preview, children }
}

describe("sessionBranches", () => {
  it("labels each branch by its first prompt after the fork and counts the prompts before it", () => {
    const tree = [entry("model", undefined, "", [
      entry("u1", "user", "plan the feature", [
        entry("a1", "assistant", "here is a plan", [
          entry("u2", "user", "use polars", [entry("a2", "assistant", "done")]),
          entry("u3", "user", "use duckdb", [entry("a3", "assistant", "")]),
        ]),
      ]),
    ])]
    expect(sessionBranches(tree)).toEqual([
      { leafId: "a2", label: "use polars", branchPoint: 1 },
      { leafId: "a3", label: "use duckdb", branchPoint: 1 },
    ])
  })

  it("keeps a straight session as one branch and survives very deep sessions", () => {
    let deep = entry("last", "assistant", "final answer")
    for (let index = 0; index < 20_000; index++) deep = entry(`e${index}`, index % 2 ? "user" : "assistant", "", [deep])
    expect(sessionBranches([deep])).toEqual([{ leafId: "last", label: "final answer", branchPoint: 0 }])
  })
})
