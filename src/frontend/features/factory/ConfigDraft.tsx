"use client"

import { useState } from "react"
import { ErrorText } from "@/components/ui/ErrorText"
import { useAsyncAction } from "@/hooks/useAsyncAction"
import { callFactory } from "@/lib/backend/api"
import { FIELD_CLASS, PRIMARY_BUTTON_CLASS } from "./styles"

/** The generated `factory.yaml` of a repo that has none, editable. Save writes it uncommitted into the checkout (`config.write`). */
export function ConfigDraft({ repo, draft, onSaved }: { repo: string; draft: string; onSaved: (text: string) => void }) {
  const [text, setText] = useState(draft)
  const { run: save, pending, error } = useAsyncAction(async () => {
    await callFactory("config.write", { repo, text })
    onSaved(text)
  }, "Could not save factory.yaml")
  return (
    <form
      onSubmit={(event) => { event.preventDefault(); if (text.trim() && !pending) void save() }}
      className="mx-auto flex min-h-0 w-full max-w-3xl flex-1 flex-col gap-3 px-8 py-6"
    >
      <label htmlFor="factory-yaml" className="text-xs text-ink-subtle">factory.yaml</label>
      <textarea
        id="factory-yaml"
        spellCheck={false}
        value={text}
        onChange={(event) => setText(event.target.value)}
        className={`${FIELD_CLASS} min-h-0 flex-1 resize-none font-mono text-xs leading-relaxed`}
      />
      <ErrorText>{error}</ErrorText>
      <div className="flex justify-end">
        <button type="submit" disabled={pending || !text.trim()} className={PRIMARY_BUTTON_CLASS}>Save</button>
      </div>
    </form>
  )
}
