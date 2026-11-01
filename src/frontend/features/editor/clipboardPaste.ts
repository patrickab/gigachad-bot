import { storeWorkbenchImage } from "@/lib/backend/api"
import { announceFileChange } from "@/lib/backend/fileEvents"
import { errorMessage } from "@/lib/utils"

/** Whether a screenshot can be pasted into the file at *path*: an absolute LaTeX or Markdown file. */
function takesImages(path: string): boolean {
  return path.startsWith("/") && /\.(tex|md)$/i.test(path)
}

/** Store the PNG in `imgs/` next to the document at *path*, announce what changed, resolve to the line referencing it. Rejects with a readable message. */
export async function storeClipboardImage(path: string, image: Blob): Promise<string> {
  if (!takesImages(path)) throw new Error("Images paste only into LaTeX or Markdown files on disk")
  try {
    const stored = await storeWorkbenchImage(path, image)
    for (const changed of [stored.path, ...stored.changed]) announceFileChange(changed)
    return stored.reference
  } catch (cause) {
    throw new Error(errorMessage(cause, "Could not store the image"))
  }
}

/** What the browser clipboard pastes into the document at *path*: the plain text, else a PNG as its stored reference and a newline. Rejects with a readable message. */
export async function readClipboardForPaste(path: string): Promise<string> {
  let items: ClipboardItems
  try {
    items = await navigator.clipboard.read()
  } catch {
    throw new Error("Clipboard access denied")
  }
  const text = items.find((item) => item.types.includes("text/plain"))
  if (text) return await (await text.getType("text/plain")).text()
  const image = items.find((item) => item.types.includes("image/png"))
  return image ? `${await storeClipboardImage(path, await image.getType("image/png"))}\n` : ""
}
