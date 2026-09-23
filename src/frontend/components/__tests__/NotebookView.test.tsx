import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { ApiError } from "@/lib/api"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { cellOutputKey } from "@/components/NotebookView"
import type { NotebookCellOutput, NotebookOutputs } from "@/lib/api"

const api = vi.hoisted(() => ({
  ApiError: class ApiError extends Error {
    constructor(message: string, readonly status: number) {
      super(message)
      this.name = "ApiError"
    }
  },
  fetchNotebook: vi.fn(),
  saveNotebook: vi.fn(),
  runNotebook: vi.fn(),
}))

vi.mock("@/lib/api", () => api)
vi.mock("@/components/LaTeXMarkdown", () => ({
  LaTeXMarkdown: ({ content }: { content: string }) => <div data-testid="cell-markdown">{content}</div>,
}))
vi.mock("@/components/PlotElement", () => ({
  PlotElement: ({ figure }: { figure: unknown }) => <div data-testid="cell-plot">{JSON.stringify(figure)}</div>,
}))
vi.mock("@/contexts/SettingsContext", () => ({
  useSettings: () => ({ selectedModel: "test-model" }),
}))

import { NotebookView } from "@/components/NotebookView"

// jsdom ships no WebCrypto; mirror the browser contract so the sidecar lookup works.
async function sha256Hex(text: string): Promise<string> {
  const bytes = new TextEncoder().encode(text)
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes)
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("")
}

const CHAT_ID = "chat-1"

// One markdown + two code cells, percent format.
const NOTEBOOK_SOURCE = "# %% [markdown]\n# # Title\n\n# %%\na = 1\n\n# %%\nb = 2\n"

function snapshot(overrides: { outputs?: NotebookOutputs; revision_id?: string } = {}) {
  return {
    source: NOTEBOOK_SOURCE,
    outputs: {} as NotebookOutputs,
    revision_id: "rev-1",
    ...overrides,
  }
}

function outputsFor(source: string, records: NotebookCellOutput[]): Promise<Record<string, NotebookCellOutput[]>> {
  return sha256Hex(source).then((key) => ({ [key]: records }))
}

beforeEach(() => {
  // Node's webcrypto is the same engine browsers use.
  const { webcrypto } = require("node:crypto")
  vi.stubGlobal("crypto", webcrypto as unknown as Crypto)
  api.fetchNotebook.mockReset()
  api.saveNotebook.mockReset()
  api.runNotebook.mockReset()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe("NotebookView rendering", () => {
  it("renders one card per parsed cell with markdown rendered", async () => {
    api.fetchNotebook.mockResolvedValue(snapshot())
    render(<NotebookView chatId={CHAT_ID} exists />)

    await waitFor(() => expect(screen.getByText("# Title")).toBeInTheDocument())
    // Two code cells' editors plus the rendered markdown body.
    const editors = screen.getAllByRole("textbox")
    expect(editors).toHaveLength(2)
    expect(editors[0]).toHaveValue("a = 1")
    expect(editors[1]).toHaveValue("b = 2")
  })

  it("stays hidden when the GET confirms no notebook", async () => {
    api.fetchNotebook.mockRejectedValue(new ApiError("No notebook for this chat", 404))
    const { container } = render(<NotebookView chatId={CHAT_ID} exists />)

    await waitFor(() => expect(api.fetchNotebook).toHaveBeenCalledTimes(1))
    expect(container).toBeEmptyDOMElement()
  })

  it("renders a cell's sidecar outputs below it, errors as danger text", async () => {
    const outputs = await outputsFor("b = 2", [
      { type: "stdout", text: "hello" },
      { type: "error", text: "ValueError: bad" },
    ])
    api.fetchNotebook.mockResolvedValue(snapshot({ outputs }))
    render(<NotebookView chatId={CHAT_ID} exists />)

    await waitFor(() => expect(screen.getByText("hello")).toBeInTheDocument())
    expect(screen.getByText("ValueError: bad")).toBeInTheDocument()
  })

  it("dims outputs as stale once the cell source no longer matches the sidecar key", async () => {
    const outputs = await outputsFor("b = 2", [{ type: "result", text: "RESULT_OF_B" }])
    api.fetchNotebook.mockResolvedValue(snapshot({ outputs }))
    render(<NotebookView chatId={CHAT_ID} exists />)

    await waitFor(() => expect(screen.getByText("RESULT_OF_B")).toBeInTheDocument())
    // Edit the second cell so its digest no longer matches the sidecar key.
    const second = screen.getAllByRole("textbox")[1]
    fireEvent.change(second, { target: { value: "b = 3" } })

    await waitFor(() => expect(screen.getByText(/stale — edited since this ran/)).toBeInTheDocument())
    expect(screen.getByText("RESULT_OF_B")).toBeInTheDocument()
  })
})

describe("NotebookView modes", () => {
  it("toggles between Notebook cells and the whole-file Edit editor", async () => {
    api.fetchNotebook.mockResolvedValue(snapshot())
    render(<NotebookView chatId={CHAT_ID} exists />)
    await waitFor(() => expect(screen.getByText("# Title")).toBeInTheDocument())

    fireEvent.click(screen.getByRole("button", { name: "Edit" }))

    // Whole-file editor holds the serialized percent format verbatim.
    const fileEditor = screen.getByRole("textbox")
    expect(fileEditor).toHaveValue(NOTEBOOK_SOURCE)

    fireEvent.click(screen.getByRole("button", { name: "Notebook" }))
    expect(screen.getAllByRole("textbox")).toHaveLength(2)
  })
})

describe("NotebookView run", () => {
  it("runs a cell with upto = its 1-based index and adopts the returned revision", async () => {
    const runOutputs = await outputsFor("b = 2", [{ type: "result", text: "RAN_B" }])
    api.fetchNotebook.mockResolvedValue(snapshot())
    api.runNotebook.mockResolvedValue({ revision_id: "rev-2", outputs: runOutputs })
    render(<NotebookView chatId={CHAT_ID} exists />)

    // Cells render before the run button can be found.
    await waitFor(() => expect(screen.getAllByRole("textbox").length).toBeGreaterThan(1))
    fireEvent.click(screen.getByRole("button", { name: "Run cell 2" }))

    await waitFor(() => expect(api.runNotebook).toHaveBeenCalledWith(CHAT_ID, 2))
    await waitFor(() => expect(screen.getByText("RAN_B")).toBeInTheDocument())
  })
})

describe("NotebookView save", () => {
  it("sends the current source with the base revision after the debounce", async () => {
    vi.useFakeTimers()
    api.fetchNotebook.mockResolvedValue(snapshot())
    api.saveNotebook.mockResolvedValue({ revision_id: "rev-2" })
    try {
      render(<NotebookView chatId={CHAT_ID} exists />)
      await vi.waitFor(() => expect(screen.getByText("# Title")).toBeInTheDocument())

      fireEvent.change(screen.getAllByRole("textbox")[0], { target: { value: "a = 10" } })
      expect(api.saveNotebook).not.toHaveBeenCalled()

      await act(async () => { vi.advanceTimersByTime(1000) })
      // Save body: percent-format source with the edited first cell + base revision rev-1.
      expect(api.saveNotebook).toHaveBeenCalledWith(
        CHAT_ID,
        expect.stringContaining("a = 10"),
        expect.anything(),
        "rev-1",
      )
    } finally {
      vi.useRealTimers()
    }
  })

  it("marks the conflict state and refetches when the save rejects with 409", async () => {
    vi.useFakeTimers()
    api.fetchNotebook.mockResolvedValueOnce(snapshot())
    api.saveNotebook.mockRejectedValue(new ApiError("Stale notebook revision", 409))
    api.fetchNotebook.mockResolvedValue(snapshot({ revision_id: "rev-remote" }))
    try {
      render(<NotebookView chatId={CHAT_ID} exists />)
      await vi.waitFor(() => expect(screen.getByText("# Title")).toBeInTheDocument())

      fireEvent.change(screen.getAllByRole("textbox")[0], { target: { value: "a = 10" } })
      await act(async () => { vi.advanceTimersByTime(1000) })

      await vi.waitFor(() => expect(screen.getByText(/changed elsewhere/)).toBeInTheDocument())
      // The refetch adopted the remote revision, so the next save bases on it.
      expect(api.fetchNotebook).toHaveBeenCalledTimes(2)
    } finally {
      vi.useRealTimers()
    }
  })
})

describe("cellOutputKey", () => {
  it("matches the backend sidecar key: sha256 of the raw cell source", async () => {
    const key = await cellOutputKey("b = 2")
    // Known digest of "b = 2" — same formula lib/sandbox_notebook.cell_output_key uses.
    expect(key).toBe(await sha256Hex("b = 2"))
    expect(key).toMatch(/^[0-9a-f]{64}$/)
  })
})