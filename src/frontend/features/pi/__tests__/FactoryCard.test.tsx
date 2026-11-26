import { afterEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { FactoryCard } from "../FactoryCard"
import { CARD_KINDS, factoryCard } from "../factoryCard"
import { CARD_FIXTURES } from "./factoryCardFixtures"

afterEach(cleanup)

const cardOf = (raw: unknown) => {
  const card = factoryCard({ card: raw })
  if (!card) throw new Error("no card")
  return card
}

describe("factoryCard", () => {
  it("reads a card of every kind", () => {
    for (const kind of CARD_KINDS) expect(cardOf(CARD_FIXTURES[kind]).kind).toBe(kind)
  })

  it("ignores unknown fields and skips malformed entries", () => {
    const card = cardOf({ ...CARD_FIXTURES.contract, extra: 1, rows: [{ label: "a", text: "b", x: 1 }, { label: 2 }], findings: [{ severity: "fatal", text: "x" }] })
    expect(card).toEqual({ v: 1, kind: "contract", tone: "quiet", title: "contract cards changed", rows: [{ label: "a", text: "b" }], findings: [] })
    expect(cardOf(CARD_FIXTURES.wave)).not.toHaveProperty("reviewer")
  })

  it("is null without a card of version 1", () => {
    expect(factoryCard(undefined)).toBeNull()
    expect(factoryCard({ result: {} })).toBeNull()
    expect(factoryCard({ card: { ...CARD_FIXTURES.task, v: 2 } })).toBeNull()
    expect(factoryCard({ card: { ...CARD_FIXTURES.task, kind: "board" } })).toBeNull()
  })
})

describe("FactoryCard", () => {
  it("draws every kind with its title and facts", () => {
    for (const kind of CARD_KINDS) {
      const { unmount } = render(<FactoryCard card={cardOf(CARD_FIXTURES[kind])} />)
      expect(screen.getByText(CARD_FIXTURES[kind].title)).toBeTruthy()
      unmount()
    }
    render(<FactoryCard card={cardOf(CARD_FIXTURES.action)} />)
    expect(screen.getByText("ui/2 · merged · 3/3 · $5.12")).toBeTruthy()
  })

  it("opens an attention card in the warning ink and keeps a quiet one collapsed and muted", () => {
    render(<FactoryCard card={cardOf(CARD_FIXTURES.task)} />)
    const attention = screen.getByRole("button", { name: /a coder asks/ })
    expect(attention).toHaveAttribute("aria-expanded", "true")
    expect(attention.className).toContain("text-warning")
    expect(screen.getByText("open")).toBeTruthy()
    cleanup()

    render(<FactoryCard card={cardOf(CARD_FIXTURES.scout)} />)
    const quiet = screen.getByRole("button", { name: /scout reported/ })
    expect(quiet).toHaveAttribute("aria-expanded", "false")
    expect(quiet.className).not.toContain("text-warning")
    expect(quiet.className).toContain("text-ink-muted")
    fireEvent.click(quiet)
    expect(screen.getByText("Key facts")).toBeTruthy()
  })

  it("draws a card without a body as a static row", () => {
    render(<FactoryCard card={cardOf(CARD_FIXTURES.action)} />)
    expect(screen.queryByRole("button")).toBeNull()
    expect(screen.getByText("wave ui/2 was merged")).toBeTruthy()
  })

  it("collapses findings to counts per severity and opens them on demand", () => {
    render(<FactoryCard card={cardOf(CARD_FIXTURES.wave)} />)
    expect(screen.getByText("why")).toBeTruthy()
    const blockers = screen.getByRole("button", { name: "1 blocker" })
    screen.getByRole("button", { name: "1 major" })
    const minor = screen.getByRole("button", { name: "2 minor" })
    expect(screen.queryByText("a stale link")).toBeNull()
    fireEvent.click(minor)
    expect(screen.getByText("a stale link")).toBeTruthy()
    expect(screen.getByText("a comment repeats the code")).toBeTruthy()
    expect(blockers).toHaveAttribute("aria-expanded", "false")
  })

  it("opens a finding's file at its line", () => {
    const onOpenFile = vi.fn()
    render(<FactoryCard card={cardOf(CARD_FIXTURES.wave)} onOpenFile={onOpenFile} />)
    fireEvent.click(screen.getByRole("button", { name: "1 blocker" }))
    fireEvent.click(screen.getByRole("button", { name: "src/frontend/features/pi/PiTab.tsx:42" }))
    expect(onOpenFile).toHaveBeenCalledWith("src/frontend/features/pi/PiTab.tsx", 42)
    fireEvent.click(screen.getByRole("button", { name: "1 major" }))
    fireEvent.click(screen.getByRole("button", { name: "src/frontend/features/pi/types.ts" }))
    expect(onOpenFile).toHaveBeenLastCalledWith("src/frontend/features/pi/types.ts", undefined)
  })
})
