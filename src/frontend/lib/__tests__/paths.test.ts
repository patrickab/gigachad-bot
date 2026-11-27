import { describe, expect, it } from "vitest"
import { isInside, relativeIn } from "../paths"

describe("isInside", () => {
  it("matches the directory and below it, never a sibling sharing its prefix", () => {
    expect(isInside("/a/foo", "/a/foo")).toBe(true)
    expect(isInside("/a/foo/src", "/a/foo/")).toBe(true)
    expect(isInside("/a/foobar", "/a/foo")).toBe(false)
    expect(isInside("/a", "/a/foo")).toBe(false)
  })
})

describe("relativeIn", () => {
  it("shortens a path below the directory and leaves any other unchanged", () => {
    expect(relativeIn("/home/me/thesis", "/home/me/thesis/chapters/intro.tex")).toBe("chapters/intro.tex")
    expect(relativeIn("/home/me/thesis/", "/home/me/thesis/main.tex")).toBe("main.tex")
    expect(relativeIn("/home/me/thesis", "/home/me/thesis-old/main.tex")).toBe("/home/me/thesis-old/main.tex")
  })
})
