import { beforeEach, describe, expect, it, vi } from "vitest"
import { renderHook, waitFor } from "@testing-library/react"
import type { FactoryTaskDiff } from "@/lib/domain/factory"

const api = vi.hoisted(() => {
  class ApiError extends Error {
    constructor(message: string, readonly status: number) {
      super(message)
    }
  }
  return { ApiError, getFactoryTaskDiff: vi.fn<(id: string) => Promise<FactoryTaskDiff>>() }
})

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
const RENAMED = `diff --git a/src/before.ts b/src/after.ts
similarity index 100%
rename from src/before.ts
rename to src/after.ts
`

beforeEach(() => {
  api.getFactoryTaskDiff.mockReset()
})

describe("parseTaskDiff", () => {
  it("splits a diff into its files with path, status and line counts", () => {
    const files = parseTaskDiff(MODIFIED + ADDED + DELETED + BINARY + RENAMED)

    expect(files.map((file) => [file.path, file.status, file.added, file.removed])).toEqual([
      ["src/a.ts", "modified", 1, 1],
      ["docs/new file.md", "added", 2, 0],
      ["old.py", "deleted", 0, 1],
      ["img/logo.png", "added", null, null],
      ["src/after.ts", "renamed", null, null],
    ])
    expect(files[0].patch.lines!.map((line) => line.kind)).toEqual(["hunk", "context", "removed", "added", "context"])
    expect(files[3].patch.lines).toBeNull()
  })

  it("has no files for an empty diff", () => {
    expect(parseTaskDiff("")).toEqual([])
  })

  it("keeps the parsed file of an unchanged section", () => {
    const before = parseTaskDiff(MODIFIED + ADDED)
    const after = parseTaskDiff(MODIFIED + ADDED.replace("+text", "+more"), before)

    expect(after[0]).toBe(before[0])
    expect(after[1]).not.toBe(before[1])
    expect(after[1].patch.lines!.at(-1)!.text).toBe("more")
  })
})

describe("useTaskDiff", () => {
  it("reads the diff on mount and again when the revision changes, keeping the shown files meanwhile", async () => {
    api.getFactoryTaskDiff.mockResolvedValueOnce({ base: "abc", diff: MODIFIED })
    const hook = renderHook(({ revision }) => useTaskDiff("t1", revision), { initialProps: { revision: 0 } })
    expect(hook.result.current.files).toBeUndefined()
    await waitFor(() => expect(hook.result.current.files).toHaveLength(1))
    expect(api.getFactoryTaskDiff).toHaveBeenCalledTimes(1)

    let answer: (diff: FactoryTaskDiff) => void = () => {}
    api.getFactoryTaskDiff.mockReturnValueOnce(new Promise((resolve) => { answer = resolve }))
    hook.rerender({ revision: 4 })
    expect(api.getFactoryTaskDiff).toHaveBeenCalledTimes(2)
    expect(hook.result.current.files).toHaveLength(1)

    answer({ base: "abc", diff: MODIFIED + ADDED })
    await waitFor(() => expect(hook.result.current.files).toHaveLength(2))

    hook.rerender({ revision: 4 })
    expect(api.getFactoryTaskDiff).toHaveBeenCalledTimes(2)
  })

  it("has no files and no error for a task without a diff base yet", async () => {
    api.getFactoryTaskDiff.mockRejectedValue(new api.ApiError("Unknown task", 404))
    const hook = renderHook(() => useTaskDiff("t1", 0))

    await waitFor(() => expect(hook.result.current.files).toEqual([]))
    expect(hook.result.current.error).toBe("")
  })

  it("reports a failed read and keeps the files it has", async () => {
    api.getFactoryTaskDiff.mockResolvedValueOnce({ base: "abc", diff: MODIFIED }).mockRejectedValueOnce(new api.ApiError("git failed", 502))
    const hook = renderHook(({ revision }) => useTaskDiff("t1", revision), { initialProps: { revision: 0 } })
    await waitFor(() => expect(hook.result.current.files).toHaveLength(1))

    hook.rerender({ revision: 1 })
    await waitFor(() => expect(hook.result.current.error).toBe("git failed"))
    expect(hook.result.current.files).toHaveLength(1)
  })
})
