import { describe, expect, it } from "vitest"
import { composerAction, draftOf, fileCompletions, piCompletions } from "../composerInput"

describe("composerAction", () => {
  it.each([
    ["runs !cmd in the shell, in context", "!ls -la", [], { kind: "bash", command: "ls -la", excludeFromContext: false }],
    ["keeps !!cmd output out of the model's context", "!! git status", [], { kind: "bash", command: "git status", excludeFromContext: true }],
    ["never appends chips to a ! command", "!make test\n\nchip text", ["chip text"], { kind: "bash", command: "make test", excludeFromContext: false }],
    ["does not read a chip starting with ! as a command", "!rm -rf", ["!rm -rf"], { kind: "prompt", text: "!rm -rf" }],
    ["sends a bare ! as a prompt", "!", [], { kind: "prompt", text: "!" }],
    ["compacts with /compact", "/compact", [], { kind: "compact" }],
    ["steers /compact by the rest", "/compact keep the test plan", [], { kind: "compact", instructions: "keep the test plan" }],
    ["leaves other slash commands to Pi", "/compactor", [], { kind: "prompt", text: "/compactor" }],
    ["sends text with its chips as a prompt", "fix it\n\nchip", ["chip"], { kind: "prompt", text: "fix it\n\nchip" }],
  ])("%s", (_name, draft, chips, action) => {
    expect(composerAction(draft, chips)).toEqual(action)
  })
})

describe("draftOf", () => {
  it("strips the chips ChatInput appended", () => {
    expect(draftOf("draft\n\none\n\ntwo", ["one", "two"])).toBe("draft")
    expect(draftOf("one", ["one"])).toBe("")
    expect(draftOf("draft", [])).toBe("draft")
  })
})

describe("fileCompletions", () => {
  const files = ["src/lib/fuzzy.ts", "docs/fuzzy-notes.md", "src/frontend/features/pi/PiComposer.tsx", "README.md"]

  it("ranks the files for the @ word at the caret, a name match first", () => {
    const text = "look at @pico"
    const items = fileCompletions(text, text.length, files)
    expect(items.map((item) => item.label)).toEqual(["src/frontend/features/pi/PiComposer.tsx"])
    expect(items[0]).toMatchObject({ from: 8, to: text.length, insert: "@src/frontend/features/pi/PiComposer.tsx ", mono: true })
  })

  it("prefers the tighter match in the name", () => {
    const items = fileCompletions("@fuzzy", 6, files)
    expect(items.map((item) => item.label)).toEqual(["src/lib/fuzzy.ts", "docs/fuzzy-notes.md"])
  })

  it("lists every file for a bare @ and none without one", () => {
    expect(fileCompletions("@", 1, files)).toHaveLength(files.length)
    expect(fileCompletions("mail@host", 9, files)).toEqual([])
    expect(fileCompletions("plain", 5, files)).toEqual([])
  })

  it("completes the @ word before the caret, not the text after it", () => {
    const items = fileCompletions("@READ and more", 5, files)
    expect(items[0]).toMatchObject({ label: "README.md", from: 0, to: 5 })
  })
})

describe("piCompletions", () => {
  it("offers /compact next to Pi's commands", async () => {
    const complete = piCompletions([{ name: "review", source: "prompt" }], [])
    const labels = (await complete("/", 1)).map((item) => item.label)
    expect(labels).toEqual(["/compact", "/review"])
    expect((await complete("/comp", 5)).map((item) => item.label)).toEqual(["/compact"])
  })
})
