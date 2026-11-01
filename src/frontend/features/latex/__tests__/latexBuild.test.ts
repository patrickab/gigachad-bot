import { describe, expect, it, vi } from "vitest"
import { createBuildQueue, quoteSelection } from "../latexBuild"

describe("createBuildQueue", () => {
  it("runs one follow-up for any number of triggers during a build", async () => {
    const builds: Array<PromiseWithResolvers<void>> = []
    const trigger = createBuildQueue(() => {
      const build = Promise.withResolvers<void>()
      builds.push(build)
      return build.promise
    })

    void trigger()
    expect(builds).toHaveLength(1)
    void trigger()
    void trigger()
    const last = trigger()
    expect(builds).toHaveLength(1)

    builds[0].resolve()
    await vi.waitFor(() => expect(builds).toHaveLength(2))
    builds[1].resolve()
    await last
    expect(builds).toHaveLength(2)
  })

  it("starts a fresh build once idle", async () => {
    let started = 0
    const trigger = createBuildQueue(async () => { started++ })
    await trigger()
    await trigger()
    expect(started).toBe(2)
  })
})

describe("quoteSelection", () => {
  it("quotes the lines under their file name and range, leaving a blank line to type in", () => {
    const text = quoteSelection("/home/me/thesis/main.tex", { start_line: 13, end_line: 14, text: "\\section{A}\nHello" })
    expect(text).toBe("main.tex:13-14\n```latex\n\\section{A}\nHello\n```\n\n")
  })
})
