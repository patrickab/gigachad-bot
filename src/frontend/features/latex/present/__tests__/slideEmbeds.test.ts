import { describe, expect, it } from "vitest"
import { deckFrames, frameStep, type DeckEmbed } from "../slideEmbeds"

const embed = (page: number, url: string): DeckEmbed => ({ url, page, left: 0, top: 0, width: 1, height: 1 })

describe("deckFrames", () => {
  it("shares one frame between consecutive pages with the same address, a gap or another address starts a new one", () => {
    const frames = deckFrames([embed(5, "a"), embed(2, "a"), embed(3, "a"), embed(3, "b"), embed(4, "b"), embed(4, "c")])
    expect(frames.map(({ key, pages }) => [key, pages.map(({ page }) => page)])).toEqual([
      ["2 a", [2, 3]],
      ["3 b", [3, 4]],
      ["4 c", [4]],
      ["5 a", [5]],
    ])
  })

  it("names the shown page's step within its frame, -1 off its pages", () => {
    const [frame] = deckFrames([embed(7, "a"), embed(8, "a"), embed(9, "a")])
    expect([6, 7, 8, 9, 10].map((page) => frameStep(frame, page))).toEqual([-1, 0, 1, 2, -1])
  })
})
