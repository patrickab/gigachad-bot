import { renderHook } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { useSlashCommands } from "@/features/navigation/useSlashCommands"

function setup(over: { filesEnabled?: boolean } = {}) {
  const deps = {
    filesEnabled: true,
    openFilePicker: vi.fn(),
    memorize: vi.fn<() => Promise<void>>(async () => {}),
    ...over,
  }
  const { result } = renderHook(() => useSlashCommands(deps))
  return { deps, result }
}

describe("useSlashCommands menu items", () => {
  it("lists every command when files are available", () => {
    const { result } = setup()
    expect(result.current.menuItems.map((i) => i.command)).toEqual(["/memorize", "/files-load"])
  })

  it("hides /files-load without file roots", () => {
    const { result } = setup({ filesEnabled: false })
    expect(result.current.menuItems.map((i) => i.command)).toEqual(["/memorize"])
  })
})

describe("runSlashCommand", () => {
  it("leaves ordinary text, the removed scoped variants and the removed /mindmap to the chat", async () => {
    const { result, deps } = setup()
    expect(await result.current.runSlashCommand("hello")).toBe(false)
    expect(await result.current.runSlashCommand("/memorize-global")).toBe(false)
    expect(await result.current.runSlashCommand("/memorize-project")).toBe(false)
    expect(await result.current.runSlashCommand("/mindmap")).toBe(false)
    expect(deps.memorize).not.toHaveBeenCalled()
  })

  it("opens the file picker for /files-load", async () => {
    const { result, deps } = setup()
    expect(await result.current.runSlashCommand("/files-load")).toBe(true)
    expect(deps.openFilePicker).toHaveBeenCalledOnce()
  })

  it("memorizes for /memorize, with or without trailing text", async () => {
    const { result, deps } = setup()
    expect(await result.current.runSlashCommand("/memorize")).toBe(true)
    expect(await result.current.runSlashCommand("/memorize notes")).toBe(true)
    expect(deps.memorize).toHaveBeenCalledTimes(2)
  })
})
