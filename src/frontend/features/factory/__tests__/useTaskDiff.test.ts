import { beforeEach, describe, expect, it, vi } from "vitest"
import { renderHook, waitFor } from "@testing-library/react"
import type { FactoryTaskDiff } from "@/lib/domain/factory"

const core = vi.hoisted(() => ({ diff: vi.fn<(id: string) => Promise<FactoryTaskDiff>>() }))
const api = vi.hoisted(() => ({
  callFactory: vi.fn(async (action: string, params: Record<string, unknown>) => {
    if (action !== "tasks.diff") throw new Error(`unexpected ${action}`)
    return core.diff(params.id as string)
  }),
}))

vi.mock("@/lib/backend/api", () => api)

import { parseTaskDiff, useTaskDiff } from "@/features/factory/useTaskDiff"

const MODIFIED = `diff --git a/src/a.ts b/src/a.ts
index 1111111..2222222 100644
--- a/src/a.ts
+++ b/src/a.ts
@@ -1,3 +1,3 @@
 const one = 1
-const two = 3
+const two = 2
 const three = 3
`
const ADDED = `diff --git a/docs/new file.md b/docs/new file.md
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ b/docs/new file.md
@@ -0,0 +1,2 @@
+# New
+text
`
const DELETED = `diff --git a/old.py b/old.py
deleted file mode 100644
index 4444444..0000000
--- a/old.py
+++ /dev/null
@@ -1 +0,0 @@
-print("old")
`
const BINARY = `diff --git a/img/logo.png b/img/logo.png
new file mode 100644
index 0000000..5555555
Binary files /dev/null and b/img/logo.png differ
`
beforeEach(() => {
  core.diff.mockReset()
})

type CoreFile = FactoryTaskDiff["files"][number]
const counts = (path: string, added: number, removed: number): CoreFile => ({ path, added, removed })
const A = counts("src/a.ts", 1, 1)
const NEW = counts("docs/new file.md", 2, 0)
const OLD = counts("old.py", 0, 1)
const LOGO = counts("img/logo.png", 0, 0)

describe("parseTaskDiff", () => {
  it("shows the core's files in its order with its counts, each with its section's status and lines", () => {
    const files = parseTaskDiff({ diff: BINARY + DELETED + ADDED + MODIFIED, files: [A, NEW, OLD, LOGO] })

    expect(files.map((file) => [file.path, file.status, file.added, file.removed])).toEqual([
      ["src/a.ts", "modified", 1, 1],
      ["docs/new file.md", "added", 2, 0],
      ["old.py", "deleted", 0, 1],
      ["img/logo.png", "added", 0, 0],
    ])
    expect(files[0].patch.lines!.map((line) => line.kind)).toEqual(["hunk", "context", "removed", "added", "context"])
    expect(files[3].patch.lines).toBeNull()
  })

  it("keeps a file the diff was cut before, without lines", () => {
    const files = parseTaskDiff({ diff: MODIFIED, files: [A, counts("package-lock.json", 900, 20)] })

    expect(files.map((file) => [file.path, file.added, file.removed, file.patch.lines])).toEqual([
      ["src/a.ts", 1, 1, expect.any(Array)],
      ["package-lock.json", 900, 20, null],
    ])
  })

  it("has no files for an empty answer", () => {
    expect(parseTaskDiff({ diff: "", files: [] })).toEqual([])
  })

  it("keeps the parsed file of an unchanged one", () => {
    const before = parseTaskDiff({ diff: MODIFIED + ADDED, files: [A, NEW] })
    const after = parseTaskDiff({ diff: MODIFIED + ADDED.replace("+text", "+more"), files: [A, NEW] }, before)

    expect(after[0]).toBe(before[0])
    expect(after[1]).not.toBe(before[1])
    expect(after[1].patch.lines!.at(-1)!.text).toBe("more")
  })
})

/** A core answer with one file per section of `text`. */
const diff = (text: string): FactoryTaskDiff => ({ base: "abc", diff: text, files: [...text.matchAll(/^diff --git a\/(\S+) /gm)].map(([, path]) => counts(path, 1, 1)) })

describe("useTaskDiff", () => {
  it("reads the diff on mount and again when the revision changes, keeping the shown files meanwhile", async () => {
    core.diff.mockResolvedValueOnce(diff(MODIFIED))
    const hook = renderHook(({ revision }) => useTaskDiff("t1", revision, true), { initialProps: { revision: 0 } })
    expect(hook.result.current.files).toBeUndefined()
    await waitFor(() => expect(hook.result.current.files).toHaveLength(1))
    expect(api.callFactory).toHaveBeenLastCalledWith("tasks.diff", { id: "t1" })
    expect(core.diff).toHaveBeenCalledTimes(1)

    let answer: (answered: FactoryTaskDiff) => void = () => {}
    core.diff.mockReturnValueOnce(new Promise((resolve) => { answer = resolve }))
    hook.rerender({ revision: 4 })
    await waitFor(() => expect(core.diff).toHaveBeenCalledTimes(2))
    expect(hook.result.current.files).toHaveLength(1)

    answer(diff(MODIFIED + ADDED))
    await waitFor(() => expect(hook.result.current.files).toHaveLength(2))

    hook.rerender({ revision: 4 })
    expect(core.diff).toHaveBeenCalledTimes(2)
  })

  it("has no files, no read and no error for a task that has not started, and reads once it has", async () => {
    core.diff.mockResolvedValue(diff(MODIFIED))
    const hook = renderHook(({ started }) => useTaskDiff("t1", 0, started), { initialProps: { started: false } })

    expect(hook.result.current.files).toEqual([])
    expect(hook.result.current.error).toBe("")
    expect(core.diff).not.toHaveBeenCalled()

    hook.rerender({ started: true })
    await waitFor(() => expect(hook.result.current.files).toHaveLength(1))
  })

  it("reports a failed read and keeps the files it has", async () => {
    core.diff.mockResolvedValueOnce(diff(MODIFIED)).mockRejectedValueOnce(new Error("the worktree of task t1 is gone (/w/t1)"))
    const hook = renderHook(({ revision }) => useTaskDiff("t1", revision, true), { initialProps: { revision: 0 } })
    await waitFor(() => expect(hook.result.current.files).toHaveLength(1))

    hook.rerender({ revision: 1 })
    await waitFor(() => expect(hook.result.current.error).toBe("the worktree of task t1 is gone (/w/t1)"))
    expect(hook.result.current.files).toHaveLength(1)
  })
})
