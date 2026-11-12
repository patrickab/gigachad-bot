import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { PromptEditor } from "@/features/settings/PromptEditor"
import { UndoDeleteProvider } from "@/contexts/UndoDeleteContext"
import { promptsResource } from "@/lib/backend/resources"

const api = vi.hoisted(() => ({
  fetchPromptList: vi.fn(async () => [{ slug: "coder", name: "Coder", includes: [] }]),
  fetchPromptBlocks: vi.fn(async () => ({})),
  fetchPromptRaw: vi.fn(async () => ({ content: "old" })),
  savePrompt: vi.fn(async () => {}),
  fetchPrompts: vi.fn(async () => ({ coder: "new" })),
}))
vi.mock("@/lib/backend/api", async (importOriginal) => ({ ...await importOriginal<typeof import("@/lib/backend/api")>(), ...api }))

describe("PromptEditor", () => {
  it("shares the resolved prompts with every tab once a prompt is saved", async () => {
    const tabPrompts = vi.fn()
    const unsubscribe = promptsResource.subscribe(() => tabPrompts(promptsResource.get()))
    render(<UndoDeleteProvider><PromptEditor open onClose={() => {}} /></UndoDeleteProvider>)

    fireEvent.click(await screen.findByText("Coder"))
    fireEvent.change(await screen.findByPlaceholderText("Write your prompt..."), { target: { value: "new" } })
    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => expect(tabPrompts).toHaveBeenCalledWith({ coder: "new" }))
    expect(api.savePrompt).toHaveBeenCalledWith("coder", "new")
    unsubscribe()
  })
})
