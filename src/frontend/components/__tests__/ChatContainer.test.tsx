import { act, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { ChatSidebarProvider, useSidebarElements, type ChatSidebarContextValue } from "@/components/ChatContainer"

function DocumentsBodyHarness() {
  const elements = useSidebarElements({
    chatId: "chat-1",
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
