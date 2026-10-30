import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { useEffect } from "react"
import { afterEach, expect, it, vi } from "vitest"
import { LEARN_EMBED_CHANNEL, LEARN_PROTOCOL_VERSION, type LearnFrameMessage } from "@gigachad/chat-ui/learn-protocol"
import { TabContext, type Tab } from "@/contexts/TabContext"
import type { LearnView, ModelsResponse } from "@/lib/domain/types"
import type * as Settings from "@/contexts/SettingsContext"
import type * as Api from "@/lib/backend/api"
import { LearnProvider, useLearn } from "../LearnProvider"
import { LearnModelSelector, LearnReasoningSelector } from "../LearnModel"

const FRAME_URL = "http://127.0.0.1:30181"
vi.mock("@/contexts/SettingsContext", async (importOriginal) => ({
  ...await importOriginal<typeof Settings>(),
  useSettings: () => ({ hostAccess: true, learnharnessUrl: FRAME_URL }),
}))
vi.mock("@/hooks/useThemeName", () => ({ useThemeName: () => "light" }))
vi.mock("@/lib/backend/api", async (importOriginal) => ({ ...await importOriginal<typeof Api>(), fetchOmpCatalog: () => new Promise(() => {}) }))
vi.mock("../useLearnSessions", () => ({ useLearnSessions: () => ({ sessions: [], error: null, refresh: vi.fn(), claim: vi.fn() }) }))
vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} })

const CATALOG: ModelsResponse = {
  ollama: ["ollama/qwen3", "ollama/gemma4"],
  providers: [{ label: "Gemini", litellm_id: "gemini", models: ["gemini-3.5-flash", "gemini-3.1-pro"] }],
  defaults: { default_model: "", small_model: "", vision_model: "", memory_model: "", omp_model: "" },
  tab_order: [],
}
const SERVED: LearnFrameMessage = {
  type: "models",
  served: [{ provider: "custom", modelId: "gemini/gemini-3.5-flash" }, { provider: "custom", modelId: "ollama/qwen3" }],
}

type Model = Extract<LearnFrameMessage, { type: "model" }>
const modelState = (fields: Partial<Model> = {}): Model => ({
  type: "model", sessionId: null, model: { provider: "custom", modelId: "gemini/gemini-3.5-flash" },
  thinkingLevel: "medium", thinkingLevels: ["low", "medium", "high"], locked: false, unavailable: false, ...fields,
})

function Attach({ view }: { view: LearnView }) {
  const { attachFrame } = useLearn()
  useEffect(() => attachFrame({ slot: document.body, view, onViewChange: vi.fn() }), [attachFrame, view])
  return null
}

/** The learn top bar's model controls over a fake frame at `version`. */
function mountTopBar(view: LearnView, version = LEARN_PROTOCOL_VERSION, onProvidersChange = vi.fn()) {
  const tab = { tab: { chatId: "c1", projectSlug: null } as Tab, isActive: true, setModeLabel: vi.fn(), updateConfig: vi.fn(), setAppMode: vi.fn(), setWorkspaceItem: vi.fn(), setHistoryFile: vi.fn(), setTitle: vi.fn() }
  const rendered = render(
    <LearnProvider onOpenItem={vi.fn()}>
      <Attach view={view} />
      <TabContext.Provider value={tab}>
        <LearnReasoningSelector view={view} />
        <LearnModelSelector view={view} models={CATALOG} onProvidersChange={onProvidersChange} onDefaultsChange={vi.fn()} placement="below" />
      </TabContext.Provider>
    </LearnProvider>,
  )
  const frame = rendered.container.querySelector("iframe")!
  const posted = vi.spyOn(frame.contentWindow!, "postMessage").mockImplementation(() => {})
  const fromFrame = (data: LearnFrameMessage | { type: "ready"; version: number }) => act(() => {
    window.dispatchEvent(new MessageEvent("message", { data: { channel: LEARN_EMBED_CHANNEL, ...data }, origin: FRAME_URL, source: frame.contentWindow }))
  })
  fromFrame({ type: "ready", version })
  const sent = (type: string) => posted.mock.calls.map(([envelope]) => envelope as { type: string; channel?: string })
    .filter((message) => message.type === type)
    .map((message) => Object.fromEntries(Object.entries(message).filter(([key]) => key !== "channel")))
  return { fromFrame, sent }
}

const modelButton = () => screen.queryByRole("button", { name: /^Model:/ })

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

it("offers only served models and moves the draft's strong tier through the frame", () => {
  const { fromFrame, sent } = mountTopBar({ kind: "new" })
  fromFrame(SERVED)
  fromFrame(modelState())

  fireEvent.click(modelButton()!)
  expect(screen.getAllByRole("menuitemradio").map((item) => item.textContent)).toEqual(["gemini-3.5-flash"])
  fireEvent.click(screen.getByRole("button", { name: "Ollama" }))
  expect(screen.getAllByRole("menuitemradio").map((item) => item.textContent)).toEqual(["qwen3"])
  fireEvent.click(screen.getByRole("menuitemradio", { name: "qwen3" }))

  expect(sent("set-model")).toEqual([{ type: "set-model", sessionId: null, provider: "custom", modelId: "ollama/qwen3" }])

  fireEvent.click(screen.getByRole("button", { name: "Reasoning: medium" }))
  fireEvent.click(screen.getByRole("menuitemradio", { name: /high/ }))
  expect(sent("set-thinking")).toEqual([{ type: "set-thinking", sessionId: null, level: "high" }])
})

it("narrows only the pick list: Manage models edits and saves the full catalog", async () => {
  const onProvidersChange = vi.fn().mockResolvedValue(undefined)
  const { fromFrame } = mountTopBar({ kind: "new" }, LEARN_PROTOCOL_VERSION, onProvidersChange)
  fromFrame(SERVED)
  fromFrame(modelState())

  fireEvent.click(modelButton()!)
  expect(screen.getAllByRole("menuitemradio").map((item) => item.textContent)).toEqual(["gemini-3.5-flash"])
  fireEvent.click(screen.getByRole("button", { name: "Manage models" }))
  fireEvent.click(screen.getByRole("button", { name: "Expand Gemini" }))
  expect(screen.getByText("gemini-3.1-pro")).toBeInTheDocument()
  fireEvent.change(screen.getByRole("textbox", { name: "Model name" }), { target: { value: "gemini-4" } })
  fireEvent.click(screen.getByRole("button", { name: "Add model" }))

  await waitFor(() => expect(onProvidersChange).toHaveBeenCalledTimes(1))
  expect(onProvidersChange.mock.calls[0][0]).toEqual([{ label: "Gemini", litellm_id: "gemini", models: ["gemini-3.5-flash", "gemini-3.1-pro", "gemini-4"] }])
})

it("shows a session's model as a read-only pill, the selector again once it is no longer served", () => {
  const { fromFrame, sent } = mountTopBar({ kind: "session", sessionId: "s1" })
  fromFrame(SERVED)
  fromFrame(modelState({ sessionId: "s1", thinkingLevel: "high", locked: true }))

  expect(screen.getByRole("note", { name: "Session model: gemini-3.5-flash, thinking high" })).toBeInTheDocument()
  expect(modelButton()).toBeNull()
  expect(screen.queryByRole("button", { name: /^Reasoning:/ })).toBeNull()

  fromFrame(modelState({ sessionId: "s1", model: { provider: "custom", modelId: "gemini/gemini-2-flash" }, locked: true, unavailable: true }))
  fireEvent.click(modelButton()!)
  fireEvent.click(screen.getByRole("menuitemradio", { name: "gemini-3.5-flash" }))
  expect(sent("set-model")).toEqual([{ type: "set-model", sessionId: "s1", provider: "custom", modelId: "gemini/gemini-3.5-flash" }])
})

it("opens the selector when LearnHarness asks for its model menu", () => {
  const { fromFrame } = mountTopBar({ kind: "new" })
  fromFrame(SERVED)
  fromFrame(modelState())

  fromFrame({ type: "model-open", sessionId: "elsewhere" })
  expect(modelButton()).toHaveAttribute("aria-expanded", "false")
  fromFrame({ type: "model-open", sessionId: null })
  expect(modelButton()).toHaveAttribute("aria-expanded", "true")
})

it("never opens for a session's read-only pill, opens once its model is no longer served", () => {
  const { fromFrame } = mountTopBar({ kind: "session", sessionId: "s1" })
  fromFrame(SERVED)
  fromFrame(modelState({ sessionId: "s1", locked: true }))
  fromFrame({ type: "model-open", sessionId: "s1" })
  expect(modelButton()).toBeNull()

  fromFrame(modelState({ sessionId: "s1", locked: true, unavailable: true }))
  fromFrame({ type: "model-open", sessionId: "s1" })
  expect(modelButton()).toHaveAttribute("aria-expanded", "true")
})

it("shows nothing for a frame at another protocol version", () => {
  const { fromFrame } = mountTopBar({ kind: "new" }, 1)
  fromFrame(SERVED)
  fromFrame(modelState())
  fromFrame({ type: "model-open", sessionId: null })

  expect(modelButton()).toBeNull()
  expect(screen.queryByRole("button", { name: /^Reasoning:/ })).toBeNull()
  expect(screen.queryByRole("dialog")).toBeNull()
})
