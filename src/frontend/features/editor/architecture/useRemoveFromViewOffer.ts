import { useCallback, useState } from "react"
import type { ArchitectureOperation, ArchitectureView, ArchitectureWorkspace } from "@/lib/domain/architecture"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { useEventListener } from "@/hooks/useEventListener"
import { useResetOnChange } from "@/hooks/useResetOnChange"

/**
 * Removing from a saved view leaves the element in the model; the user is then
 * asked whether to delete it there too. `offered` are the elements in question,
 * and `answer` settles it (Enter deletes, Escape keeps, whatever holds focus).
 */
export function useRemoveFromViewOffer({ run, view, workspace }: {
  run: (ops: ArchitectureOperation[]) => void
  view: ArchitectureView | null
  workspace: ArchitectureWorkspace | null
}) {
  const [offered, setOffered] = useState<string[]>([])
  const removeFromView = useCallback((elements: string[]) => {
    if (!view || elements.length === 0) return
    run([{ op: "removeFromView", view: view.id, elements }])
    setOffered(elements)
  }, [run, view])
  const offeredElements = workspace ? workspace.model.elements.filter((element) => offered.includes(element.id)) : []
  const answer = useCallback((remove: boolean) => {
    if (remove) run([{ op: "delete", elements: offered, relations: [] }])
    setOffered([])
  }, [run, offered])
  // The dialog has the keyboard: Enter deletes, Escape keeps, whatever holds focus.
  useEventListener("window", "keydown", (event) => {
    if (event.key !== "Enter") return
    event.preventDefault()
    event.stopPropagation()
    answer(true)
  }, { active: offered.length > 0, capture: true })
  useEscapeLayer(() => answer(false), offered.length > 0)
  // Another view or file is another question.
  useResetOnChange(view?.id, () => setOffered([]))
  return { removeFromView, offeredElements, answer }
}
