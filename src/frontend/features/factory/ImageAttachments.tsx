"use client"

import { useCallback, useRef, useState, type ClipboardEvent } from "react"
import { ImagePlus, X } from "lucide-react"
import { blobToDataUrl } from "@/lib/utils"
import { ICON_BUTTON_CLASS } from "./styles"
import type { FactoryImage } from "@/lib/domain/factory"

/** An image staged on a task request or message, `dataUrl` feeds the thumbnail and the upload. */
export interface StagedImage {
  filename: string
  dataUrl: string
}

const isImage = (file: File) => file.type.startsWith("image/")

/** *filename*, numbered when *taken* already holds it: the engine stores images by name. */
function uniqueFilename(filename: string, taken: Set<string>): string {
  if (!taken.has(filename)) return filename
  const dot = filename.lastIndexOf(".")
  const [stem, extension] = dot > 0 ? [filename.slice(0, dot), filename.slice(dot)] : [filename, ""]
  let count = 2
  while (taken.has(`${stem}-${count}${extension}`)) count += 1
  return `${stem}-${count}${extension}`
}

export function toFactoryImages(images: StagedImage[]): FactoryImage[] {
  return images.map(({ filename, dataUrl }) => ({ filename, data_base64: dataUrl.slice(dataUrl.indexOf(",") + 1) }))
}

/** Images staged from the file picker or a paste, *onFailed* gets the message of an unreadable file. */
export function useImageAttachments(onFailed: (message: string) => void) {
  const [images, setImages] = useState<StagedImage[]>([])

  const add = useCallback(async (files: File[]) => {
    try {
      const read = await Promise.all(files.filter(isImage).map(async (file) => ({
        filename: file.name || `image.${file.type.split("/")[1] || "png"}`,
        dataUrl: await blobToDataUrl(file),
      })))
      setImages((current) => {
        const taken = new Set(current.map((image) => image.filename))
        const added = read.map((image) => {
          const filename = uniqueFilename(image.filename, taken)
          taken.add(filename)
          return { ...image, filename }
        })
        return [...current, ...added]
      })
    } catch {
      onFailed("Could not read the image")
    }
  }, [onFailed])

  const remove = useCallback((filename: string) => setImages((current) => current.filter((image) => image.filename !== filename)), [])
  const clear = useCallback(() => setImages([]), [])

  // A clipboard that also carries text (copied spreadsheet cells) pastes as text.
  const onPaste = useCallback((event: ClipboardEvent) => {
    const files = Array.from(event.clipboardData.files).filter(isImage)
    if (files.length === 0 || event.clipboardData.getData("text/plain") !== "") return
    event.preventDefault()
    void add(files)
  }, [add])

  return { images, add, remove, clear, onPaste }
}

/** Icon button opening the image file picker, `className` fits it to the row it sits in. */
export function AttachImagesButton({ disabled = false, onPick, className = ICON_BUTTON_CLASS }: { disabled?: boolean; onPick: (files: File[]) => void; className?: string }) {
  const inputRef = useRef<HTMLInputElement>(null)
  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        multiple
        hidden
        aria-label="Image files"
        onChange={(event) => {
          onPick(Array.from(event.target.files ?? []))
          // Picking the same file again must fire a change.
          event.target.value = ""
        }}
      />
      <button type="button" aria-label="Attach images" disabled={disabled} onClick={() => inputRef.current?.click()} className={className}>
        <ImagePlus className="h-4 w-4" />
      </button>
    </>
  )
}

/** Thumbnails of the staged images, each with a remove button. */
export function StagedImages({ images, onRemove }: { images: StagedImage[]; onRemove: (filename: string) => void }) {
  if (images.length === 0) return null
  return (
    <ul aria-label="Attached images" className="flex flex-wrap gap-2">
      {images.map((image) => (
        <li key={image.filename} className="relative">
          {/* eslint-disable-next-line @next/next/no-img-element -- user-supplied data URL, next/image cannot optimize it */}
          <img src={image.dataUrl} alt={image.filename} title={image.filename} className="h-12 w-12 rounded border border-divider object-cover" />
          <button
            type="button"
            aria-label={`Remove ${image.filename}`}
            onClick={() => onRemove(image.filename)}
            className="absolute -right-1.5 -top-1.5 rounded-full border border-divider bg-surface p-0.5 text-ink-subtle transition-colors hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"
          >
            <X className="h-3 w-3" />
          </button>
        </li>
      ))}
    </ul>
  )
}
