import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { clearChips, stageChip } from "@/features/chat/composer/composerChips"
import { fetchPiFiles, pastePiVideo, type PiFileScope } from "@/lib/backend/api"
import { PiComposer, type PiInsertion, type PiSubmit } from "../PiComposer"

vi.mock("@/features/chat/EnabledToolsContext", () => ({ useEnabledTools: () => ({ enabledTools: [], toggleTool: () => {} }) }))
vi.mock("@/contexts/SettingsContext", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/contexts/SettingsContext")>(),
  useSettings: () => ({ ocrModel: "" }),
  useNeovimSetting: () => ({ available: false, enabled: false, setPreferred: () => {} }),
}))
vi.mock("@/lib/backend/api", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/lib/backend/api")>(),
  fetchPiFiles: vi.fn(async () => ["src/app.ts"]),
  pastePiVideo: vi.fn(async (video: File) => `.pi/pastes/20261119-120000-${video.name}`),
}))

const SESSION = "s1"

function renderComposer({ running = false, disabled = false, onAbort = vi.fn(async () => [] as string[]), files = { repo: "repo" } as PiFileScope, insertion = null as PiInsertion | null, onInserted = vi.fn() } = {}) {
  const onSubmit = vi.fn<(submit: PiSubmit) => void>()
  const element = (props: { disabled: boolean; insertion: PiInsertion | null }) => (
    <PiComposer
      sessionKey={SESSION}
      files={files}
      running={running}
      disabled={props.disabled}
      commands={[]}
      dialog={undefined}
      queue={{ steering: [], followUp: [] }}
      notices={[]}
      statusLines={{}}
      editorText={null}
      insertion={props.insertion}
      onInserted={onInserted}
      onSubmit={onSubmit}
      onAbort={onAbort}
      onAnswerDialog={() => {}}
    />
  )
  const { rerender } = render(element({ disabled, insertion }))
  const textbox = screen.getByRole("textbox")
  const send = (text: string, keys: { altKey?: boolean } = {}) => {
    fireEvent.change(textbox, { target: { value: text } })
    fireEvent.keyDown(textbox, { key: "Enter", ...keys })
  }
  const update = (props: { disabled: boolean; insertion: PiInsertion | null }) => rerender(element(props))
  return { onSubmit, onAbort, onInserted, textbox, send, update }
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

  it("completes @ with the files of a chat's folder", async () => {
    vi.useFakeTimers()
    try {
      const { textbox } = renderComposer({ files: { cwd: "/work/markdown" } })
      expect(fetchPiFiles).toHaveBeenLastCalledWith({ cwd: "/work/markdown" })
      await act(() => vi.advanceTimersByTimeAsync(0))
      ;(textbox as HTMLTextAreaElement).focus()
      fireEvent.change(textbox, { target: { value: "@app", selectionStart: 4 } })
      await act(() => vi.advanceTimersByTimeAsync(100))
      expect(screen.getByRole("option", { name: /src\/app\.ts/ })).toBeInTheDocument()
    } finally {
      vi.useRealTimers()
    }
  })

  it("appends an insertion to the draft once the composer is enabled, once per id", async () => {
    const insertion: PiInsertion = { id: 1, text: "notes.md:3-4\n> quoted" }
    const { textbox, onInserted, update } = renderComposer({ disabled: true, insertion })
    expect(textbox).toHaveValue("")
    expect(onInserted).not.toHaveBeenCalled()

    update({ disabled: false, insertion })
    await waitFor(() => expect(textbox).toHaveValue("notes.md:3-4\n> quoted"))
    expect(onInserted).toHaveBeenCalledOnce()
    update({ disabled: false, insertion: { ...insertion } })
    expect(textbox).toHaveValue("notes.md:3-4\n> quoted")
  })

  it("stages a labelled insertion as a chip that joins the next prompt", async () => {
    const { onSubmit, send, onInserted } = renderComposer({ insertion: { id: 2, label: "slide 3", text: "\\begin{frame}" } })
    await waitFor(() => expect(onInserted).toHaveBeenCalledOnce())
    send("tighten this")
    await waitFor(() => expect(onSubmit).toHaveBeenLastCalledWith(expect.objectContaining({ action: { kind: "prompt", text: "tighten this\n\n\\begin{frame}" } })))
  })

  it("stores a pasted, dropped or inserted video and mentions it in the draft", async () => {
    const clip = new File(["v"], "clip.mp4", { type: "video/mp4" })
    const { textbox, update } = renderComposer()
    fireEvent.change(textbox, { target: { value: "what happens in" } })
    fireEvent.paste(textbox, { clipboardData: { files: [clip], items: [], getData: () => "" } })
    await waitFor(() => expect(textbox).toHaveValue("what happens in @.pi/pastes/20261119-120000-clip.mp4 "))
    expect(pastePiVideo).toHaveBeenLastCalledWith(clip, { repo: "repo" })

    const dropped = new File(["v"], "drop.webm", { type: "video/webm" })
    fireEvent.drop(textbox, { dataTransfer: { files: [dropped], types: ["Files"] } })
    await waitFor(() => expect(textbox).toHaveValue("what happens in @.pi/pastes/20261119-120000-clip.mp4 @.pi/pastes/20261119-120000-drop.webm "))

    const inserted = new File(["v"], "screen.mov", { type: "video/quicktime" })
    update({ disabled: false, insertion: { id: 3, text: "", files: [inserted] } })
    await waitFor(() => expect((textbox as HTMLTextAreaElement).value).toMatch(/drop\.webm @\.pi\/pastes\/20261119-120000-screen\.mov $/))
  })

  it("shows why a video could not be stored", async () => {
    vi.mocked(pastePiVideo).mockRejectedValueOnce(new Error("Videos are at most 200 MB"))
    const { textbox } = renderComposer()
    fireEvent.paste(textbox, { clipboardData: { files: [new File(["v"], "big.mp4", { type: "video/mp4" })], items: [], getData: () => "" } })
    expect(await screen.findByRole("alert")).toHaveTextContent("Videos are at most 200 MB")
    expect(textbox).toHaveValue("")
  })
})
