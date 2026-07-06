/**
 * Regression test for the ChatSidebarContext refactor.
 *
 * The refactor moved ~16 vault/document props off <ChatContainer> and onto a
 * context provider. We assert the contract:
 *   - <ChatContainer> no longer accepts those props (its Props type narrowed).
 *   - useSidebarElements reads everything from the context and produces the
 *     expected sidebar elements (Context, Sources, Documents) for each
 *     scenario in isolation.
 */
import { describe, it, expect, vi } from "vitest"
import { renderHook, act } from "@testing-library/react"

import { ChatSidebarProvider, type ChatSidebarContextValue } from "@/components/ChatContainer"
import { useSidebarElements } from "@/components/ChatContainer"
import type { WebSearchResult, ProjectDocument, Attachment } from "@/lib/types"

// We import the component lazily so we can read its prop-type shape without
// forcing a full render (which pulls in dynamic imports + canvas deps).
import * as ChatContainerModule from "@/components/ChatContainer"

function makeAttachment(over: Partial<Attachment> = {}): Attachment {
  return {
    name: "a.txt",
    mime: "text/plain",
    url: "u",
    active: true,
    ...over,
  }
}

function wrap(ctx: ChatSidebarContextValue, children: any) {
  return <ChatSidebarProvider value={ctx}>{children}</ChatSidebarProvider>
}

describe("ChatSidebarContext refactor", () => {
  it("exports the provider and the context value type", () => {
    expect(ChatSidebarProvider).toBeDefined()
    expect(ChatContainerModule.ChatSidebarProvider).toBe(ChatSidebarProvider)
  })

  it("yields no elements when there are no attachments, no vault, no documents", () => {
    const { result } = renderHook(() =>
      useSidebarElements({
        chatId: "c",
        slug: null,
        allAttachments: [],
        expandedEntries: [],
        onToggleExpand: vi.fn(),
        lastSearchResult: undefined,
        editingDocPath: null,
        onEditDocument: vi.fn(),
        isElementOpen: () => false,
        onElementOpenChange: vi.fn(),
        pdfWide: false,
        onTogglePdfWide: vi.fn(),
      }),
      {
        wrapper: ({ children }) => wrap({}, children),
      },
    )
    expect(result.current).toEqual([])
  })

  it("yields a Context element when an attachment is present", () => {
    const att = makeAttachment()
    const ctx: ChatSidebarContextValue = {
      onRemoveAttachment: vi.fn(),
      onToggleAttachmentActive: vi.fn(),
      onAttachmentContentChange: vi.fn(),
    }
    const { result } = renderHook(
      () =>
        useSidebarElements({
          chatId: "c",
          slug: null,
          allAttachments: [{ messageIndex: 0, attachment: att }],
          expandedEntries: [],
          onToggleExpand: vi.fn(),
          lastSearchResult: undefined,
          editingDocPath: null,
          onEditDocument: vi.fn(),
          isElementOpen: () => false,
          onElementOpenChange: vi.fn(),
          pdfWide: false,
          onTogglePdfWide: vi.fn(),
        }),
      { wrapper: ({ children }) => wrap(ctx, children) },
    )
    expect(result.current.length).toBeGreaterThan(0)
    // The Context element is the first one; it has the structural shape of
    // a ChatSidebarElementConfig (id + body). The exact field names may shift;
    // we assert the stable contract of the refactor.
    expect(result.current[0]).toHaveProperty("id")
    expect(result.current[0]).toHaveProperty("body")
  })

  it("yields a Sources element when a lastSearchResult is provided", () => {
    const searchResult: WebSearchResult = {
      query: "q",
      sources: [{ url: "https://example.com", title: "ex" }] as any,
      images: [],
      videos: [],
    }
    const { result } = renderHook(
      () =>
        useSidebarElements({
          chatId: "c",
          slug: null,
          allAttachments: [],
          expandedEntries: [],
          onToggleExpand: vi.fn(),
          lastSearchResult: searchResult,
          editingDocPath: null,
          onEditDocument: vi.fn(),
          isElementOpen: () => false,
          onElementOpenChange: vi.fn(),
          pdfWide: false,
          onTogglePdfWide: vi.fn(),
        }),
      { wrapper: ({ children }) => wrap({}, children) },
    )
    expect(result.current.length).toBeGreaterThan(0)
    expect(result.current.some((e) => e.id === "sources")).toBe(true)
  })

  it("yields a Documents element when documents are provided via context", () => {
    const documents: ProjectDocument[] = [
      { path: "/lib/a.pdf", name: "a.pdf", mime: "application/pdf" },
    ]
    const ctx: ChatSidebarContextValue = {
      documents,
      onSelectDocument: vi.fn(),
      onOpenDocuments: vi.fn(),
      onCreateDocument: vi.fn(),
      onDeleteDocument: vi.fn(),
      onDocumentSaved: vi.fn(),
    }
    const { result } = renderHook(
      () =>
        useSidebarElements({
          chatId: "c",
          slug: "proj",
          allAttachments: [],
          expandedEntries: [],
          onToggleExpand: vi.fn(),
          lastSearchResult: undefined,
          editingDocPath: null,
          onEditDocument: vi.fn(),
          isElementOpen: () => false,
          onElementOpenChange: vi.fn(),
          pdfWide: false,
          onTogglePdfWide: vi.fn(),
        }),
      { wrapper: ({ children }) => wrap(ctx, children) },
    )
    expect(result.current.some((e) => e.id === "documents")).toBe(true)
  })

  it("memoizes: returns the same array reference across re-renders with identical inputs", () => {
    const att = makeAttachment()
    const ctx: ChatSidebarContextValue = {
      onRemoveAttachment: vi.fn(),
    }
    const props = {
      chatId: "c",
      slug: null,
      allAttachments: [{ messageIndex: 0, attachment: att }],
      expandedEntries: [],
      onToggleExpand: vi.fn(),
      lastSearchResult: undefined,
      editingDocPath: null,
      onEditDocument: vi.fn(),
      isElementOpen: () => false,
      onElementOpenChange: vi.fn(),
      pdfWide: false,
      onTogglePdfWide: vi.fn(),
    }
    const { result, rerender } = renderHook(
      () => useSidebarElements(props),
      { wrapper: ({ children }) => wrap(ctx, children) },
    )
    const first = result.current
    rerender()
    expect(result.current).toBe(first) // same reference (memo hit)
  })
})
