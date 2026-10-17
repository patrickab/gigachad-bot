import { act, fireEvent, render, screen } from "@testing-library/react"
import { useState } from "react"
import { describe, expect, it, vi } from "vitest"
import { ReasoningSelector, effectiveReasoningEffort, useReasoningLevels } from "@/components/ReasoningSelector"

const fetchReasoningLevels = vi.hoisted(() => vi.fn())
vi.mock("@/lib/api", () => ({ fetchReasoningLevels }))

const LEVELS: Record<string, string[]> = {
  "plain-model": [],
  "thinker-model": ["none", "low", "medium", "high"],
  "deep-thinker-model": ["none", "low", "medium", "high", "xhigh"],
}
fetchReasoningLevels.mockImplementation(async (model: string) => ({ levels: LEVELS[model] ?? [] }))

/** Owns the effort like a tab does and shows the effective one sent to the backend. */
function Harness({ initialModel }: { initialModel: string }) {
  const [model, setModel] = useState(initialModel)
  const [storedEffort, setStoredEffort] = useState("medium")
  const levels = useReasoningLevels(model)
  const effort = effectiveReasoningEffort(levels, storedEffort)
  return <>
    <button onClick={() => setModel("thinker-model")}>use thinker</button>
    <button onClick={() => setModel("deep-thinker-model")}>use deep thinker</button>
    <button onClick={() => setModel("plain-model")}>use plain</button>
    <span data-testid="effort">{effort}</span>
    <ReasoningSelector levels={levels} reasoningEffort={effort} onReasoningChange={setStoredEffort} />
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

  it("falls back to medium for a level the model lacks and restores it afterwards", async () => {
    render(<Harness initialModel="deep-thinker-model" />)
    await settle()
    fireEvent.click(screen.getByRole("button", { name: /Reasoning/ }))
    fireEvent.click(screen.getByRole("menuitemradio", { name: /xhigh/ }))

    fireEvent.click(screen.getByText("use thinker"))
    await settle()
    expect(screen.getByTestId("effort")).toHaveTextContent("medium")

    fireEvent.click(screen.getByText("use deep thinker"))
    await settle()
    expect(screen.getByTestId("effort")).toHaveTextContent("xhigh")
  })
})
