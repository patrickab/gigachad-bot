import { act, fireEvent, render, screen } from "@testing-library/react"
import { useState } from "react"
import { describe, expect, it, vi } from "vitest"
import { ReasoningSelector } from "@/components/ReasoningSelector"

const fetchReasoningLevels = vi.hoisted(() => vi.fn())
vi.mock("@/lib/api", () => ({ fetchReasoningLevels }))

const LEVELS: Record<string, string[]> = {
  "plain-model": [],
  "thinker-model": ["none", "low", "medium", "high"],
}
fetchReasoningLevels.mockImplementation(async (model: string) => ({ levels: LEVELS[model] ?? [] }))

/** Owns the effort like a tab does, so model switches feed back into the selector. */
function Harness({ initialModel }: { initialModel: string }) {
  const [model, setModel] = useState(initialModel)
  const [effort, setEffort] = useState("medium")
  return <>
    <button onClick={() => setModel("thinker-model")}>use thinker</button>
    <button onClick={() => setModel("plain-model")}>use plain</button>
    <span data-testid="effort">{effort}</span>
    <ReasoningSelector model={model} reasoningEffort={effort} onReasoningChange={setEffort} />
  </>
}

const settle = () => act(async () => {})

describe("ReasoningSelector", () => {
  it("keeps the default effort when the first selected model can reason", async () => {
    render(<Harness initialModel="plain-model" />)
    await settle()
    expect(screen.getByTestId("effort")).toHaveTextContent("none")

    fireEvent.click(screen.getByText("use thinker"))
    await settle()
    expect(screen.getByTestId("effort")).toHaveTextContent("medium")
  })

  it("restores the chosen effort after a detour through a model that cannot reason", async () => {
    render(<Harness initialModel="thinker-model" />)
    await settle()
    fireEvent.click(screen.getByRole("button", { name: /Reasoning/ }))
    fireEvent.click(screen.getByRole("menuitemradio", { name: /high/ }))
    expect(screen.getByTestId("effort")).toHaveTextContent("high")

    fireEvent.click(screen.getByText("use plain"))
    await settle()
    expect(screen.getByTestId("effort")).toHaveTextContent("none")

    fireEvent.click(screen.getByText("use thinker"))
    await settle()
    expect(screen.getByTestId("effort")).toHaveTextContent("high")
  })

  it("keeps an explicit off on a reasoning model", async () => {
    render(<Harness initialModel="thinker-model" />)
    await settle()
    fireEvent.click(screen.getByRole("button", { name: /Reasoning/ }))
    fireEvent.click(screen.getByRole("menuitemradio", { name: /off/ }))
    await settle()
    expect(screen.getByTestId("effort")).toHaveTextContent("none")
  })
})
