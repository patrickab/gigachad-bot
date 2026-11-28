import { afterEach, expect, it, vi } from "vitest"
import { measureSidebarInsets } from "../TabContent"

afterEach(() => vi.unstubAllGlobals())

/** `element` laid out at `box` (jsdom has no layout). */
function place(element: HTMLElement, box: { offsetLeft?: number; offsetWidth?: number; clientWidth?: number }) {
  for (const [key, value] of Object.entries(box)) Object.defineProperty(element, key, { value, configurable: true })
  return element
}

it("measures the left inset from the app's edge, past the tab stack the sidebar sits beside, and the right one from the tab's edge", () => {
  vi.stubGlobal("ResizeObserver", class {
    constructor(private readonly update: () => void) {}
    observe() { this.update() }
    disconnect() {}
  })
  const stack = place(document.createElement("div"), { offsetLeft: 300 })
  stack.setAttribute("data-tab-stack", "")
  const tab = place(document.createElement("div"), { clientWidth: 1000 })
  const main = place(document.createElement("main"), { offsetLeft: 20, offsetWidth: 640 })
  tab.append(main)
  stack.append(tab)

  measureSidebarInsets(main)

  expect(main.style.getPropertyValue("--under-left")).toBe("320px")
  expect(main.style.getPropertyValue("--under-right")).toBe("340px")
})
