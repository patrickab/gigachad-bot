import { afterEach, describe, expect, it } from "vitest"
import { getApiBase, setApiBase } from "@/lib/backend/config"
import { apiWsUrl } from "@/lib/backend/urls"

const originalApiBase = getApiBase()

afterEach(() => setApiBase(originalApiBase))

describe("apiWsUrl", () => {
  it("is the backend URL over ws, or wss for an https backend", () => {
    setApiBase("http://127.0.0.1:8001/api")
    expect(apiWsUrl("/jupyter/server/")).toBe("ws://127.0.0.1:8001/api/jupyter/server/")
    setApiBase("https://bot.example.net/api")
    expect(apiWsUrl("/files/watch")).toBe("wss://bot.example.net/api/files/watch")
  })
})
