import { act, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { ChatSidebarProvider, useSidebarElements, type ChatSidebarContextValue } from "@/features/chat/context-panel/ChatContextPanel"

function DocumentsBodyHarness() {
  const elements = useSidebarElements({
    slug: "proj",
    allAttachments: [],
    expandedEntries: [],
    onToggleExpand: () => {},
    editingDocPath: null,
    onEditDocument: () => {},
    isElementOpen: () => true,
    onElementOpenChange: () => {},
  })
  return <>{elements.find((element) => element.id === "documents")?.body}</>
}

function DocumentsHarness({ context }: { context: ChatSidebarContextValue }) {
  return <ChatSidebarProvider value={context}><DocumentsBodyHarness /></ChatSidebarProvider>
}

afterEach(() => vi.useRealTimers())

describe("DocumentsBody title interaction", () => {
  const document = { path: "project/proj/document/brief.md", name: "brief.md", mime: "text/markdown" }

  it("attaches after a single click", () => {
    vi.useFakeTimers()
    const onSelectDocument = vi.fn()
    render(<DocumentsHarness context={{ documents: [document], onOpenDocuments: () => {}, onSelectDocument }} />)

    fireEvent.click(screen.getByText("brief.md"))
    expect(onSelectDocument).not.toHaveBeenCalled()
    act(() => vi.advanceTimersByTime(180))

    expect(onSelectDocument).toHaveBeenCalledWith(document.path)
  })

  it("edits and saves the title on double click without attaching", () => {
    const onSelectDocument = vi.fn()
    const onRenameDocument = vi.fn()
    render(<DocumentsHarness context={{ documents: [document], onOpenDocuments: () => {}, onSelectDocument, onRenameDocument }} />)

    fireEvent.doubleClick(screen.getByText("brief.md"))
    const input = screen.getByLabelText("Rename brief.md")
    fireEvent.change(input, { target: { value: "summary" } })
    fireEvent.keyDown(input, { key: "Enter" })

    expect(onRenameDocument).toHaveBeenCalledWith(document.path, "summary")
    expect(onSelectDocument).not.toHaveBeenCalled()
  })
})

describe("DocumentsBody list", () => {
  it("follows the documents in the context when one is deleted", () => {
    const first = { path: "project/proj/document/a.md", name: "a.md", mime: "text/markdown" }
    const second = { path: "project/proj/document/b.md", name: "b.md", mime: "text/markdown" }
    const { rerender } = render(<DocumentsHarness context={{ documents: [first, second], onOpenDocuments: () => {} }} />)
    expect(screen.getByText("b.md")).toBeTruthy()

    rerender(<DocumentsHarness context={{ documents: [first], onOpenDocuments: () => {} }} />)

    expect(screen.queryByText("b.md")).toBeNull()
    expect(screen.getByText("a.md")).toBeTruthy()
  })
})
