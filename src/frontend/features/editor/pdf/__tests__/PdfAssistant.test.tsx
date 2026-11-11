import { afterEach, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react"
import { PdfAssistant } from "../PdfAssistant"
import { TopBarSlotProvider, TopBarSlot } from "@/features/navigation/TopBarSlots"

const attach = vi.hoisted(() => vi.fn())
vi.mock("@/lib/backend/references", () => ({ attachReference: attach }))
vi.mock("@/components/content/AssistantChat", () => ({
  AssistantChat: () => null,
  SendStopButton: () => null,
  useAssistantChat: () => ({}),
  useOneShotStream: () => ({}),
}))

afterEach(cleanup)

function mount(path: string) {
  return render(<TopBarSlotProvider>
    <TopBarSlot name="pdf">{(ref) => <div ref={ref} />}</TopBarSlot>
    <PdfAssistant path={path} model="m" chatOpen={false} onChatOpenChange={() => {}}><div>pages</div></PdfAssistant>
  </TopBarSlotProvider>)
}

it("switches a PDF with an extraction between its pages and the rendered Markdown with its library images", async () => {
  attach.mockResolvedValue({ parsedMd: "# Extracted heading\n\n![](images/Paper-1.jpg)" })
  mount("PDFs/Paper.pdf")
  fireEvent.click(await screen.findByRole("button", { name: "Text" }))
  const text = screen.getByRole("region", { name: "Extracted text" })
  expect(within(text).getByRole("heading", { name: "Extracted heading" })).toBeTruthy()
  expect(text.querySelector("img")?.getAttribute("src")).toMatch(/\/assets\/Mineru\/images\/Paper\/Paper-1\.jpg$/)
  expect(screen.queryByText("pages")).toBeNull()
  fireEvent.click(screen.getByRole("button", { name: "PDF" }))
  expect(screen.getByText("pages")).toBeTruthy()
})

it("reads a stored project PDF's text by its library copy", async () => {
  attach.mockResolvedValue({ parsedMd: "text" })
  mount("project/proj/document/Paper.pdf")
  await screen.findByRole("button", { name: "Text" })
  expect(attach).toHaveBeenCalledWith("PDFs/Paper.pdf")
})
