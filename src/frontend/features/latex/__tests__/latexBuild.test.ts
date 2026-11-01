import { describe, expect, it, vi } from "vitest"
import { createBuildQueue } from "../latexBuild"

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
