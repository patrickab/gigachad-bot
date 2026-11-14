import { describe, it, expect, vi, beforeEach } from "vitest"
import { fireEvent, render, waitFor } from "@testing-library/react"
import type { FactoryWorkflow, TaskCreate } from "@/lib/domain/factory"

const api = vi.hoisted(() => ({
  createFactoryTask: vi.fn<(body: TaskCreate) => Promise<unknown>>(),
  fetchFactoryWorkflows: vi.fn<() => Promise<FactoryWorkflow[]>>(),
}))

vi.mock("@/lib/backend/api", () => api)

import { NewTaskDialog } from "@/features/factory/NewTaskDialog"

const WORKFLOW: FactoryWorkflow = { id: "code", name: "Code", phases: ["code"], merge: "review", preview: true }

const png = (name: string, content: string) => new File([content], name, { type: "image/png" })

beforeEach(() => {
  api.createFactoryTask.mockReset().mockResolvedValue({})
  api.fetchFactoryWorkflows.mockReset().mockResolvedValue([WORKFLOW])
})

describe("NewTaskDialog images", () => {
  it("sends picked and pasted images as base64 under distinct filenames, without the removed one", async () => {
    const view = render(<NewTaskDialog repo="gigachad-bot" open onClose={vi.fn()} />)
    await view.findByText("code")
    fireEvent.change(view.getByLabelText("Title"), { target: { value: "Fix the header" } })
    const request = view.getByLabelText("Request")
    fireEvent.change(request, { target: { value: "See the screenshots" } })

    fireEvent.change(view.getByLabelText("Image files"), { target: { files: [png("shot.png", "one"), png("drop.png", "dropped")] } })
    await view.findByAltText("drop.png")
    fireEvent.paste(request, { clipboardData: { files: [png("shot.png", "two")], getData: () => "" } })
    await view.findByAltText("shot-2.png")
    fireEvent.click(view.getByRole("button", { name: "Remove drop.png" }))
    fireEvent.click(view.getByRole("button", { name: "Start" }))

    await waitFor(() => expect(api.createFactoryTask).toHaveBeenCalled())
    expect(api.createFactoryTask.mock.calls[0][0].images).toEqual([
      { filename: "shot.png", data_base64: btoa("one") },
      { filename: "shot-2.png", data_base64: btoa("two") },
    ])
  })

  it("leaves a paste that also carries text to the textarea", async () => {
    const view = render(<NewTaskDialog repo="gigachad-bot" open onClose={vi.fn()} />)
    await view.findByText("code")

    const pasted = fireEvent.paste(view.getByLabelText("Request"), { clipboardData: { files: [png("cells.png", "cells")], getData: () => "a\tb" } })

    expect(pasted).toBe(true)
    expect(view.queryByAltText("cells.png")).toBeNull()
  })
})
