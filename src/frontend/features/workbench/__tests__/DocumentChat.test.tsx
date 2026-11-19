import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import { useState, type ReactNode } from "react"
import type { PiInsertion } from "@/features/pi/PiComposer"
import type { PiSessionHeader } from "@/features/pi/PiTopBar"
import type { OpenedWorkbenchChat, PiSessionInfo, WorkbenchChatLink } from "@/lib/backend/api"
import { openedChat } from "./chatFixtures"

type OpenRequest = { path: string; sessionId?: string; new?: boolean; model?: { provider: string; id: string }; thinking?: string }
const api = vi.hoisted(() => ({
  openWorkbenchChat: vi.fn<(request: OpenRequest) => Promise<OpenedWorkbenchChat>>(),
  fetchWorkbenchChats: vi.fn<(path: string) => Promise<{ link: WorkbenchChatLink; sessions: PiSessionInfo[] }>>(),
  markWorkbenchChatInformed: vi.fn<(path: string, sessionId: string) => Promise<WorkbenchChatLink>>(),
  archivePiSession: vi.fn(),
  prepareWorkbench: vi.fn<(path: string) => Promise<{ path: string; cwd: string }>>(),
  ApiError: class extends Error { status = 0 },
}))
vi.mock("@/lib/backend/api", () => api)
vi.mock("@/features/editor/FilePanel", () => ({
  FilePanel: ({ actions, children }: { actions?: ReactNode; children: ReactNode }) => <section>{actions}{children}</section>,
}))
vi.mock("@/features/pi/PiTopBar", () => ({ PiPanelBar: ({ actions }: { actions?: ReactNode }) => <div>{actions}</div> }))

/** The header a running session reports: Haiku at thinking `low`. */
const HEADER: PiSessionHeader = {
  tone: "ready",
  sessionName: undefined,
  controls: {
    model: { provider: "anthropic", id: "claude-haiku-4-5", name: "Haiku", reasoning: true },
    models: [],
    thinkingLevel: "low",
    thinkingLevels: [],
    stats: null,
    onModel: () => {},
    onThinking: () => {},
  },
}

vi.mock("@/features/pi/PiTab", () => ({
  PiSessionView: ({ sessionId, cwd, files, lead, onLeadSent, insertion, renderHeader }: {
    sessionId: string
    cwd: string
    files: { cwd: string }
    lead?: string
    onLeadSent?: () => void
    insertion?: PiInsertion | null
    renderHeader: (header: PiSessionHeader) => ReactNode
  }) => <div>
    {renderHeader(HEADER)}
    <p>session {sessionId} in {cwd} files {files.cwd}</p>
    <pre data-testid="lead">{lead ?? "none"}</pre>
    {insertion && <p>insert {insertion.text || "-"} files {insertion.files?.length ?? 0}</p>}
    <button type="button" onClick={() => onLeadSent?.()}>send</button>
  </div>,
}))

import { useWorkbenchChat, WorkbenchChat } from "../WorkbenchChat"
import { FileChat } from "../FileChat"
import { canvasLead, fileLead, projectLead } from "../fileLead"

const MARKDOWN = "/work/markdown/notes.md"
const MARKDOWN_CWD = "/work/markdown"

/** A workbench's chat as its owner wires it: the item holds the shown session (starting at `sessionId`), a button quotes `quoted` into the draft. */
function Harness({ chatKey, lead, sessionId, onSessionChange = () => {}, quoted = "" }: {
  chatKey: string
  lead: (cwd: string) => string
  sessionId?: string
  onSessionChange?: (sessionId: string) => void
  quoted?: string
}) {
  const [itemSession, setItemSession] = useState(sessionId)
  const onChatSessionChange = (next: string) => {
    setItemSession(next)
    onSessionChange(next)
  }
  const props = { chatSessionId: itemSession, onChatSessionChange, drawerOpen: true, onDrawerOpenChange: () => {}, shellCommands: { names: [], run: async () => false } }
  const state = useWorkbenchChat(chatKey, lead, props)
  return <>
    <button type="button" onClick={() => state.quote(quoted)}>quote</button>
    <WorkbenchChat state={state} {...props} />
  </>
}

const leadShown = () => screen.getByTestId("lead").textContent

beforeEach(() => {
  api.openWorkbenchChat.mockResolvedValue(openedChat({ id: "s1", cwd: MARKDOWN_CWD }))
  api.prepareWorkbench.mockImplementation(async (path) => ({ path, cwd: "/unused" }))
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  vi.useRealTimers()
})

describe("Markdown workbench chat", () => {
  it("opens the file's chat, the item learns its session, and names the file to the new session once", async () => {
    const onSessionChange = vi.fn()
    render(<Harness chatKey={MARKDOWN} lead={(cwd) => fileLead(MARKDOWN, cwd)} onSessionChange={onSessionChange} />)
    expect(await screen.findByText(`session s1 in ${MARKDOWN_CWD} files ${MARKDOWN_CWD}`)).toBeTruthy()
    expect(api.openWorkbenchChat).toHaveBeenCalledWith({ path: MARKDOWN, sessionId: undefined })
    expect(onSessionChange).toHaveBeenCalledWith("s1")
    expect(leadShown()).toBe("File: notes.md")

    fireEvent.click(screen.getByRole("button", { name: "send" }))
    expect(leadShown()).toBe("none")
    expect(api.markWorkbenchChatInformed).not.toHaveBeenCalled()
  })

  it("resumes the item's session and leads nothing once it has messages", async () => {
    api.openWorkbenchChat.mockResolvedValue(openedChat({ id: "s7", cwd: MARKDOWN_CWD, messageCount: 4 }))
    render(<Harness chatKey={MARKDOWN} lead={(cwd) => fileLead(MARKDOWN, cwd)} sessionId="s7" />)
    expect(await screen.findByText(/session s7/)).toBeTruthy()
    expect(api.openWorkbenchChat).toHaveBeenCalledWith({ path: MARKDOWN, sessionId: "s7" })
    expect(leadShown()).toBe("none")
  })

  it("tells a session the file moved to its new path once, then clears the notice", async () => {
    api.openWorkbenchChat.mockResolvedValue(openedChat({ id: "s1", cwd: MARKDOWN_CWD, messageCount: 4, moved: ["s1"] }))
    api.markWorkbenchChatInformed.mockResolvedValue({ active: "s1", sessionIds: ["s1"], moved: [] })
    render(<Harness chatKey={MARKDOWN} lead={(cwd) => fileLead(MARKDOWN, cwd)} sessionId="s1" />)
    await screen.findByText(/session s1/)
    expect(leadShown()).toBe("File: notes.md")

    fireEvent.click(screen.getByRole("button", { name: "send" }))
    expect(leadShown()).toBe("none")
    await vi.waitFor(() => expect(api.markWorkbenchChatInformed).toHaveBeenCalledWith(MARKDOWN, "s1"))
  })

  it("lists the file's sessions and switches to one through the item", async () => {
    const onSessionChange = vi.fn()
    api.fetchWorkbenchChats.mockResolvedValue({
      link: { active: "s1", sessionIds: ["s1", "s2"], moved: [] },
      sessions: [
        openedChat({ id: "s1", cwd: MARKDOWN_CWD, firstMessage: "File: notes.md\n\nFix the intro" }).session,
        { ...openedChat({ id: "s2", cwd: MARKDOWN_CWD, firstMessage: "File: notes.md\n\nAdd a summary" }).session, modifiedAt: "2026-02-01T00:00:00Z" },
      ],
    })
    render(<Harness chatKey={MARKDOWN} lead={(cwd) => fileLead(MARKDOWN, cwd)} sessionId="s1" onSessionChange={onSessionChange} />)
    await screen.findByText(/session s1/)

    fireEvent.click(screen.getByRole("button", { name: "Sessions of this file" }))
    const titles = await screen.findAllByRole("button", { name: /^(Fix the intro|Add a summary)$/ })
    expect(titles.map((title) => title.textContent)).toEqual(["Add a summary", "Fix the intro"])
    expect(api.fetchWorkbenchChats).toHaveBeenCalledWith(MARKDOWN)
    api.openWorkbenchChat.mockResolvedValue(openedChat({ id: "s2", cwd: MARKDOWN_CWD, messageCount: 2 }))
    fireEvent.click(titles[0])
    expect(onSessionChange).toHaveBeenCalledWith("s2")
    expect(await screen.findByText(/session s2/)).toBeTruthy()
    expect(api.openWorkbenchChat).toHaveBeenLastCalledWith({ path: MARKDOWN, sessionId: "s2" })
  })

  it("starts a new session on the shown session's model and thinking level", async () => {
    const onSessionChange = vi.fn()
    render(<Harness chatKey={MARKDOWN} lead={(cwd) => fileLead(MARKDOWN, cwd)} sessionId="s1" onSessionChange={onSessionChange} />)
    await screen.findByText(/session s1/)
    api.openWorkbenchChat.mockResolvedValue(openedChat({ id: "s9", cwd: MARKDOWN_CWD }))
    fireEvent.click(screen.getByRole("button", { name: "New session" }))
    await vi.waitFor(() => expect(onSessionChange).toHaveBeenCalledWith("s9"))
    expect(api.openWorkbenchChat).toHaveBeenLastCalledWith({ path: MARKDOWN, new: true, model: { provider: "anthropic", id: "claude-haiku-4-5" }, thinking: "low" })
  })

  it("puts a quote into the composer draft", async () => {
    render(<Harness chatKey={MARKDOWN} lead={(cwd) => fileLead(MARKDOWN, cwd)} quoted={"notes.md:2\n```markdown\nhi\n```"} />)
    await screen.findByText(/session s1/)
    fireEvent.click(screen.getByRole("button", { name: "quote" }))
    expect(screen.getByText(/^insert notes\.md:2/).textContent).toBe("insert notes.md:2\n```markdown\nhi\n``` files 0")
  })
})

describe("LaTeX workbench chat", () => {
  it("links the project folder and names the project and its root", async () => {
    const folder = "/home/me/talk"
    api.openWorkbenchChat.mockResolvedValue(openedChat({ id: "t1", cwd: folder, kind: "latex" }))
    render(<Harness chatKey={folder} lead={(cwd) => projectLead(folder, `${folder}/main.tex`, cwd)} />)
    await screen.findByText(`session t1 in ${folder} files ${folder}`)
    expect(api.openWorkbenchChat).toHaveBeenCalledWith({ path: folder, sessionId: undefined })
    expect(leadShown()).toBe(`Project: ${folder}, root: main.tex`)
  })
})

describe("PDF chat", () => {
  const PDF = "/home/me/papers/attention.pdf"
  const PDF_CWD = "/work/pdf"

  it("waits for the extracted text, then names the PDF and its text to the new session", async () => {
    const { promise, resolve: ready } = Promise.withResolvers<OpenedWorkbenchChat>()
    api.openWorkbenchChat
      .mockResolvedValueOnce(openedChat({ id: "p1", cwd: PDF_CWD, kind: "pdf", pdfText: { state: "pending" } }))
      .mockReturnValueOnce(promise)
    render(<FileChat path={PDF} width={320} onClose={() => {}} />)
    expect(await screen.findByText("Extracting the PDF text…")).toBeTruthy()
    expect(screen.queryByText(/^session/)).toBeNull()
    expect(api.prepareWorkbench).toHaveBeenCalledWith(PDF)
    await vi.waitFor(() => expect(api.openWorkbenchChat).toHaveBeenLastCalledWith({ path: PDF, sessionId: "p1" }))

    await act(async () => ready(openedChat({ id: "p1", cwd: PDF_CWD, kind: "pdf", pdfText: { state: "ready", path: "/docs/Mineru/attention.md" } })))
    expect(await screen.findByText(/session p1/)).toBeTruthy()
    expect(leadShown()).toBe(`File: ${PDF}\nText: /docs/Mineru/attention.md`)
  })

  it("shows a session that already knows the PDF even while its text is extracted again", async () => {
    api.openWorkbenchChat.mockResolvedValue(openedChat({ id: "p2", cwd: PDF_CWD, kind: "pdf", messageCount: 3, pdfText: { state: "pending" } }))
    render(<FileChat path={PDF} width={320} onClose={() => {}} />)
    expect(await screen.findByText(/session p2/)).toBeTruthy()
    expect(leadShown()).toBe("none")
  })
})

describe("Canvas chat", () => {
  const CANVAS = "project/proj/document/board.canvas"

  it("opens the canvas key's chat, names the canvas and takes a screenshot into the draft", async () => {
    api.openWorkbenchChat.mockResolvedValue(openedChat({ id: "c1", cwd: "/work/canvas", kind: "canvas" }))
    const png = new File(["png"], "shot.png", { type: "image/png" })
    render(<FileChat path={CANVAS} width={320} onClose={() => {}} lead={() => canvasLead(CANVAS)} insertion={{ id: 1, text: "", files: [png] }} />)
    expect(await screen.findByText(/session c1/)).toBeTruthy()
    expect(api.openWorkbenchChat).toHaveBeenCalledWith({ path: CANVAS, sessionId: undefined })
    expect(leadShown()).toBe("Canvas: board.canvas")
    expect(screen.getByText("insert - files 1")).toBeTruthy()
  })
})
