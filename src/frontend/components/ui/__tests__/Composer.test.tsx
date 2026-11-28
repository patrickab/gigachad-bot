import { useRef, useState } from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { afterEach, expect, it, vi } from "vitest"
import { ComposerTextarea } from "@/components/ui/Composer"

afterEach(() => vi.unstubAllGlobals())

function Textarea() {
  const ref = useRef<HTMLTextAreaElement>(null)
  const [value, setValue] = useState("")
  return <ComposerTextarea ref={ref} aria-label="Message" value={value} onChange={(e) => setValue(e.target.value)} />
}

it("keeps one resize observer while the text changes", () => {
  let observers = 0
  vi.stubGlobal("ResizeObserver", class { constructor() { observers++ } observe() {} disconnect() {} })
  render(<Textarea />)
  const textarea = screen.getByLabelText("Message")

  for (const text of ["h", "he", "hel", "hell"]) fireEvent.change(textarea, { target: { value: text } })

  expect(observers).toBe(1)
})
