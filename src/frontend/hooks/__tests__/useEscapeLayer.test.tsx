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

  it("leaves Escape to a focused Neovim terminal, so vim can leave insert mode without closing the window", () => {
    const closeWindow = vi.fn()
    const layer = renderHook(() => useEscapeLayer(closeWindow))
    const terminal = document.createElement("div")
    terminal.dataset.neovimTerminal = ""
    const terminalInput = document.body.appendChild(terminal).appendChild(document.createElement("textarea"))

    const escape = new KeyboardEvent("keydown", { key: "Escape", bubbles: true, cancelable: true })
    terminalInput.dispatchEvent(escape)
    expect(closeWindow).not.toHaveBeenCalled()
    expect(escape.defaultPrevented).toBe(false)

    pressEscape()
    expect(closeWindow).toHaveBeenCalledOnce()
    terminal.remove()
    layer.unmount()
  })
})
