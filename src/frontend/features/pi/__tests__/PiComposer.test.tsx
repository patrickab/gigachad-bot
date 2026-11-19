import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { clearChips, stageChip } from "@/features/chat/composer/composerChips"
import { PiComposer, type PiSubmit } from "../PiComposer"

vi.mock("@/features/chat/EnabledToolsContext", () => ({ useEnabledTools: () => ({ enabledTools: [], toggleTool: () => {} }) }))
vi.mock("@/contexts/SettingsContext", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/contexts/SettingsContext")>(),
  useSettings: () => ({ ocrModel: "" }),
  useNeovimSetting: () => ({ available: false, enabled: false, setPreferred: () => {} }),
}))
vi.mock("@/lib/backend/api", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/lib/backend/api")>(),
  fetchPiFiles: vi.fn(async () => ["src/app.ts"]),
}))

const SESSION = "s1"

function renderComposer({ running = false, onAbort = vi.fn(async () => [] as string[]) } = {}) {
  const onSubmit = vi.fn<(submit: PiSubmit) => void>()
  render(
    <PiComposer
      sessionKey={SESSION}
      repo="repo"
      running={running}
      disabled={false}
      commands={[]}
      dialog={undefined}
      queue={{ steering: [], followUp: [] }}
      notices={[]}
      statusLines={{}}
      editorText={null}
      onSubmit={onSubmit}
      onAbort={onAbort}
      onAnswerDialog={() => {}}
    />,
  )
  const textbox = screen.getByRole("textbox")
  const send = (text: string, keys: { altKey?: boolean } = {}) => {
    fireEvent.change(textbox, { target: { value: text } })
    fireEvent.keyDown(textbox, { key: "Enter", ...keys })
  }
  return { onSubmit, onAbort, textbox, send }
}

describe("PiComposer", () => {
  beforeEach(() => {
    const stored = new Map<string, string>()
    vi.stubGlobal("localStorage", {
      getItem: (key: string) => stored.get(key) ?? null,
      setItem: (key: string, value: string) => { stored.set(key, value) },
      removeItem: (key: string) => { stored.delete(key) },
    })
  })
  afterEach(() => {
    cleanup()
    clearChips(`pi:${SESSION}`)
    vi.unstubAllGlobals()
  })

  it("steers with Enter while a run streams", async () => {
    const { onSubmit, send } = renderComposer({ running: true })
    send("look at the tests")
    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith({ action: { kind: "prompt", text: "look at the tests" }, images: [], mode: "steer" }))
  })

  it("queues a follow-up with Alt+Enter, and the next Enter steers again", async () => {
    const { onSubmit, send } = renderComposer({ running: true })
    send("then summarize", { altKey: true })
    await waitFor(() => expect(onSubmit).toHaveBeenLastCalledWith(expect.objectContaining({ mode: "followUp" })))
    send("and this")
    await waitFor(() => expect(onSubmit).toHaveBeenLastCalledWith(expect.objectContaining({ action: { kind: "prompt", text: "and this" }, mode: "steer" })))
  })

  it("aborts with Esc while running and takes the queued messages back into the draft", async () => {
    const onAbort = vi.fn(async () => ["queued steer"])
    const { textbox } = renderComposer({ running: true, onAbort })
    fireEvent.change(textbox, { target: { value: "draft" } })
    await act(async () => {
      fireEvent.keyDown(textbox, { key: "Escape" })
    })
    expect(onAbort).toHaveBeenCalledOnce()
    await waitFor(() => expect(textbox).toHaveValue("queued steer\n\ndraft"))
  })

  it("leaves Esc alone while idle", () => {
    const { onAbort, textbox } = renderComposer()
    fireEvent.keyDown(textbox, { key: "Escape" })
    expect(onAbort).not.toHaveBeenCalled()
  })

  it("runs !cmd and !!cmd in the shell", async () => {
    const { onSubmit, send } = renderComposer()
    send("!ls")
    await waitFor(() => expect(onSubmit).toHaveBeenLastCalledWith(expect.objectContaining({ action: { kind: "bash", command: "ls", excludeFromContext: false } })))
    send("!!git status")
    await waitFor(() => expect(onSubmit).toHaveBeenLastCalledWith(expect.objectContaining({ action: { kind: "bash", command: "git status", excludeFromContext: true } })))
  })

  it("keeps staged chips out of a ! command and staged for the next message", async () => {
    const { onSubmit, send } = renderComposer()
    act(() => {
      stageChip(`pi:${SESSION}`, { label: "selection", text: "chip body" })
    })
    send("!make test")
    await waitFor(() => expect(onSubmit).toHaveBeenLastCalledWith(expect.objectContaining({ action: { kind: "bash", command: "make test", excludeFromContext: false } })))
    send("explain the failure")
    await waitFor(() => expect(onSubmit).toHaveBeenLastCalledWith(expect.objectContaining({ action: { kind: "prompt", text: "explain the failure\n\nchip body" } })))
  })

  it("compacts with /compact and its instructions", async () => {
    const { onSubmit, send } = renderComposer()
    send("/compact keep the API notes")
    await waitFor(() => expect(onSubmit).toHaveBeenLastCalledWith(expect.objectContaining({ action: { kind: "compact", instructions: "keep the API notes" } })))
  })
})
