import { afterEach, describe, expect, it } from "vitest"

import { getApiBase, setApiBase } from "@/lib/config"

const originalApiBase = getApiBase()

afterEach(() => setApiBase(originalApiBase))

describe("API base configuration", () => {
  it("normalizes the Tauri sidecar URL", () => {
    setApiBase("http://127.0.0.1:43123/api/")

    expect(getApiBase()).toBe("http://127.0.0.1:43123/api")
  })
})
