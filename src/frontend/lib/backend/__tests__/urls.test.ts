import { describe, expect, it, vi } from "vitest"
import { apiWsUrl } from "@/lib/backend/urls"

const backend = vi.hoisted(() => ({ apiBase: "" }))
vi.mock("@/lib/backend/config", () => ({ getApiBase: () => backend.apiBase }))

describe("apiWsUrl", () => {
  it("is the backend URL over ws, or wss for an https backend", () => {
    backend.apiBase = "http://127.0.0.1:8001/api"
    expect(apiWsUrl("/jupyter/server/")).toBe("ws://127.0.0.1:8001/api/jupyter/server/")
    backend.apiBase = "https://bot.example.net/api"
    expect(apiWsUrl("/files/watch")).toBe("wss://bot.example.net/api/files/watch")
  })
})
