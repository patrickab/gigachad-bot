import { act, renderHook } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { EnabledToolsProvider, useEnabledTools } from "@/features/chat/EnabledToolsContext"
import { TOOLS } from "@/features/chat/tools"

describe("useEnabledTools", () => {
  it("offers the default tools and toggles one without disturbing the others", () => {
    const { result } = renderHook(() => useEnabledTools(), { wrapper: EnabledToolsProvider })

    expect(result.current.enabledTools).toEqual(TOOLS.filter((tool) => tool.defaultEnabled).map((tool) => tool.name))
    expect(result.current.enabledTools).not.toContain("workspace_agent")

    act(() => result.current.toggleTool("sandbox_plot"))

    expect(result.current.enabledTools).not.toContain("sandbox_plot")
    expect(result.current.enabledTools).toContain("web_search")

    act(() => result.current.toggleTool("workspace_agent"))

    expect(result.current.enabledTools).toContain("workspace_agent")
  })
})
