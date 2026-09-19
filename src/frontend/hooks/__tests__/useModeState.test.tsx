import { act, renderHook } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { ModeProvider, SANDBOX_PLOT_TOOL, useModeState } from "@/hooks/useModeState"

describe("useModeState", () => {
  it("enables and toggles the sandbox plot tool name sent with chat requests", () => {
    const { result } = renderHook(() => useModeState(), { wrapper: ModeProvider })

    expect(result.current.enabledTools).toContain(SANDBOX_PLOT_TOOL)
    expect(result.current.enabledTools).not.toContain("plot")

    act(() => result.current.togglePlot())

    expect(result.current.enabledTools).not.toContain(SANDBOX_PLOT_TOOL)
  })
})
