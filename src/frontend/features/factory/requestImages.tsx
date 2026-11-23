"use client"

import { useRef, useState, type ClipboardEvent } from "react"
import { ImagePlus, X } from "lucide-react"
import { uploadFactoryImage } from "@/lib/backend/api"

/** `text` followed by the stored images' paths, one per line: the core takes text only, the coder opens the paths. */
export function withImages(text: string, paths: readonly string[]): string {
  return paths.length === 0 ? text : `${text}\n\nImages (open them with the read tool):\n${paths.map((path) => `- ${path}`).join("\n")}`
}

/**
 * The images attached to a request or message, until it is sent.
 *
 * - `add` takes picked, pasted or dropped files and keeps the images among them
 * - `attach(text)` stores them on the host and resolves to `text` with their paths (`withImages`); it rejects when a store fails
 */
export function useRequestImages() {
  const [images, setImages] = useState<File[]>([])
  const add = (files: Iterable<File>) => {
    const added = [...files].filter((file) => file.type.startsWith("image/"))
    if (added.length > 0) setImages((current) => [...current, ...added])
  }
  return {
    images,
    add,
    remove: (index: number) => setImages((current) => current.filter((_, at) => at !== index)),
    clear: () => setImages([]),
    /** Paste handler: images on the clipboard are attached, text pastes as usual. */
    onPaste: (event: ClipboardEvent) => {
      const pasted = [...event.clipboardData.files].filter((file) => file.type.startsWith("image/"))
      if (pasted.length === 0) return
      event.preventDefault()
      add(pasted)
    },
    attach: async (text: string) => withImages(text, await Promise.all(images.map(uploadFactoryImage))),
  }
}

/** The button that picks images to attach. `className` is the shape of the buttons beside it. */
export function AttachImageButton({ onPick, disabled, className }: { onPick: (files: File[]) => void; disabled?: boolean; className?: string }) {
  const input = useRef<HTMLInputElement>(null)
  return (
    <>
      <input
        ref={input}
        type="file"
        accept="image/png,image/jpeg,image/gif,image/webp"
        multiple
        hidden
        aria-label="Image files"
        onChange={(event) => {
          onPick([...(event.target.files ?? [])])
          event.target.value = ""
        }}
      />
      <button type="button" aria-label="Attach image" disabled={disabled} onClick={() => input.current?.click()} className={className}>
        <ImagePlus className="h-4 w-4" aria-hidden="true" />
      </button>
    </>
  )
}

/** The attached images by name, each removable. */
export function ImageChips({ images, onRemove }: { images: readonly File[]; onRemove: (index: number) => void }) {
  if (images.length === 0) return null
  return (
    <ul className="flex flex-wrap gap-1.5 pb-2 text-xs text-ink-muted">
      {images.map((image, index) => (
        <li key={`${image.name}-${index}`} className="flex max-w-full items-center gap-1 rounded-full bg-surface-elevated py-0.5 pl-2.5 pr-1">
          <span className="truncate">{image.name || "image"}</span>
          <button type="button" aria-label={`Remove ${image.name || "image"}`} onClick={() => onRemove(index)} className="rounded-full p-0.5 hover:text-ink focus-visible:outline focus-visible:outline-1">
            <X size={12} aria-hidden="true" />
          </button>
        </li>
      ))}
    </ul>
  )
}
