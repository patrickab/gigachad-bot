import { describe, expect, it } from "vitest"
import { escapeCurrencyDollars as esc } from "../src/math.ts"

describe("escapeCurrencyDollars", () => {
  it("escapes currency", () => {
    expect(esc("costs $5 and $10")).toBe("costs \\$5 and \\$10")
    expect(esc("$5")).toBe("\\$5")
    expect(esc("pay $ 5 now")).toBe("pay \\$ 5 now")
  })
  it("keeps math", () => {
    expect(esc("$x^2$")).toBe("$x^2$")
    expect(esc("a $a+b$ b, $$\\int x$$")).toBe("a $a+b$ b, $$\\int x$$")
    expect(esc("\\(x\\) and $y$")).toBe("\\(x\\) and $y$")
  })
  it("rejects a closing dollar before a digit or after a space", () => {
    expect(esc("$x$5")).toBe("\\$x\\$5")
    expect(esc("$x $")).toBe("\\$x \\$")
  })
  it("handles mixed lines", () => {
    expect(esc("$x$ costs $5")).toBe("$x$ costs \\$5")
    expect(esc("$x^2$ is not $5 or $10")).toBe("$x^2$ is not \\$5 or \\$10")
  })
  it("leaves escaped dollars", () => {
    expect(esc("\\$5 and \\$10")).toBe("\\$5 and \\$10")
  })
  it("does not pair across paragraphs", () => {
    expect(esc("$5\n\nand $x$")).toBe("\\$5\n\nand $x$")
  })
  it("leaves code unchanged", () => {
    expect(esc("`$5 and $10` $5")).toBe("`$5 and $10` \\$5")
    expect(esc("```\n$5 and $10\n```\n$3")).toBe("```\n$5 and $10\n```\n\\$3")
    expect(esc("~~~\n$5\n~~~")).toBe("~~~\n$5\n~~~")
  })
})

it("leaves text between HTML tags", () => {
  expect(esc("<td>$5 and $10</td> $5")).toBe("<td>$5 and $10</td> \\$5")
})
