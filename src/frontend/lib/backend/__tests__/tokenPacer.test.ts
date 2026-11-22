import { describe, expect, it } from "vitest"
import { createTokenPacer } from "../tokenPacer"

function setup() {
  let now = 0
  let pending: (() => void) | null = null
  const out: string[] = []
  const scheduler = {
    request: (cb: () => void) => ((pending = cb), 1),
    cancel: () => void (pending = null),
  }
  performance.now = () => now
  const pacer = createTokenPacer((t) => out.push(t), scheduler)
  const tick = (ms: number) => {
    now += ms
    const cb = pending
    pending = null
    cb?.()
  }
  return { pacer, out, tick, text: () => out.join("") }
}

describe("createTokenPacer", () => {
  it("starts with the first frame, then paces without losing text", () => {
    const { pacer, tick, text } = setup()
    pacer.push("a".repeat(200))
    tick(16)
    expect(text().length).toBeGreaterThan(0)
    for (let i = 0; i < 200; i++) tick(16)
    expect(text()).toBe("a".repeat(200))
  })

  it("drains a burst gradually, not in one emit", () => {
    const { pacer, out, tick } = setup()
    pacer.push("x".repeat(400))
    for (let i = 0; i < 20; i++) tick(16)
    expect(out.length).toBeGreaterThan(3)
  })

  it("flush emits the rest at once", () => {
    const { pacer, tick, text } = setup()
    pacer.push("hello world")
    tick(16)
    pacer.flush()
    expect(text()).toBe("hello world")
  })

  it("finish resolves after draining", async () => {
    const { pacer, tick, text } = setup()
    pacer.push("b".repeat(100))
    const done = pacer.finish()
    for (let i = 0; i < 100; i++) tick(16)
    await done
    expect(text()).toBe("b".repeat(100))
  })

  it("does not split surrogate pairs", () => {
    const { pacer, out, tick } = setup()
    pacer.push("😀".repeat(50))
    for (let i = 0; i < 300; i++) tick(16)
    expect(out.every((chunk) => !/[\ud800-\udbff]$/.test(chunk))).toBe(true)
    expect(out.join("")).toBe("😀".repeat(50))
  })
})
