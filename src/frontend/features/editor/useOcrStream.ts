"use client"

import { useEffect, useState } from "react"
import { createOCRStream } from "@/lib/backend/api"
import { useResetOnChange } from "@/hooks/useResetOnChange"

/**
 * Stream the LaTeX OCR of `image` with `model` into `output`.
 *
 * - A new image or model restarts the stream, the effect cleanup aborts the running one (also on unmount)
 * - `streaming` holds while there is an image and its stream has not ended
 * - Callers may edit `output` with `setOutput`, a later token overwrites it with the streamed text
 */
export function useOcrStream(image: string | null, model: string) {
  const [output, setOutput] = useState("")
  const [ended, setEnded] = useState(false)
  useResetOnChange(image, () => setEnded(false))
  useResetOnChange(model, () => setEnded(false))

  useEffect(() => {
    if (!image) return
    const stream = createOCRStream(image, model)
    let text = ""

    async function read() {
      try {
        for await (const event of stream) {
          if (event.event === "token") {
            text += event.data
            setOutput(text)
          } else if (event.event === "done") {
            break
          } else if (event.event === "error") {
            text += `\n\nError: ${event.data}`
            setOutput(text)
            break
          }
        }
      } catch {
        // aborted or error
      } finally {
        setEnded(true)
      }
    }
    read()

    return () => { stream.abort() }
  }, [image, model])

  return { output, setOutput, streaming: image !== null && !ended }
}
