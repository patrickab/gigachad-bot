import { renderHook } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"

const pressEscape = () => document.body.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }))

describe("useEscapeLayer", () => {
  it("gives Escape to the innermost active layer only, then to the outer one once it closes", () => {
    const below = vi.fn()
    document.addEventListener("keydown", below)
    const closeWindow = vi.fn()
    const cancelForm = vi.fn()
    renderHook(() => useEscapeLayer(closeWindow))
    const form = renderHook(({ editing }) => useEscapeLayer(cancelForm, editing), { initialProps: { editing: true } })

    pressEscape()
    expect(cancelForm).toHaveBeenCalledOnce()
    expect(closeWindow).not.toHaveBeenCalled()
    expect(below).not.toHaveBeenCalled()

    form.rerender({ editing: false })
    pressEscape()
    expect(closeWindow).toHaveBeenCalledOnce()
    expect(cancelForm).toHaveBeenCalledOnce()
    document.removeEventListener("keydown", below)
  })
})
