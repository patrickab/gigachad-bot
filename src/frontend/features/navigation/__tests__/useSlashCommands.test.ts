import { renderHook } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { useSlashCommands } from "@/features/navigation/useSlashCommands"
import type { MemorizeScope } from "@/features/memory/useMemoryCommandBar"

function setup(over: { filesEnabled?: boolean; hasProject?: boolean } = {}) {
  const deps = {
    filesEnabled: true,
    hasProject: true,
    openFilePicker: vi.fn(),
    memorize: vi.fn<(scope: MemorizeScope) => Promise<void>>(async () => {}),
    ...over,
  }
  const { result } = renderHook(() => useSlashCommands(deps))
  return { deps, result }
}

describe("useSlashCommands menu items", () => {
  it("lists every command when files and a project are available", () => {
    const { result } = setup()
    expect(result.current.menuItems.map((i) => i.command)).toEqual(["/memorize", "/memorize-global", "/memorize-project", "/files-load"])
  })

  it("hides /files-load without file roots and /memorize-project without a project", () => {
    const { result } = setup({ filesEnabled: false, hasProject: false })
    expect(result.current.menuItems.map((i) => i.command)).toEqual(["/memorize", "/memorize-global"])
  })
})

describe("runSlashCommand", () => {
  it("leaves ordinary text, unknown memorize variants and the removed /mindmap to the chat", async () => {
    const { result, deps } = setup()
    expect(await result.current.runSlashCommand("hello")).toBe(false)
    expect(await result.current.runSlashCommand("/memorize-foo")).toBe(false)
    expect(await result.current.runSlashCommand("/mindmap")).toBe(false)
    expect(deps.memorize).not.toHaveBeenCalled()
  })

  it("opens the file picker for /files-load", async () => {
    const { result, deps } = setup()
    expect(await result.current.runSlashCommand("/files-load")).toBe(true)
    expect(deps.openFilePicker).toHaveBeenCalledOnce()
  })

  it("maps each memorize variant to its scope", async () => {
    const { result, deps } = setup()
    await result.current.runSlashCommand("/memorize")
    await result.current.runSlashCommand("/memorize notes")
    await result.current.runSlashCommand("/memorize-global")
    await result.current.runSlashCommand("/memorize-project now")
    expect(deps.memorize.mock.calls.map(([scope]) => scope)).toEqual([null, null, "global", "project"])
  })
})
