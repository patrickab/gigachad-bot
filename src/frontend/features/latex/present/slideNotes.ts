/** The fields of a pdf.js annotation that carry a note. */
interface NoteAnnotation {
  subtype?: string
  contents?: string
  contentsObj?: { str?: string }
}

/** A page's speaker notes: the text of its `/Text` annotations (what pdfpc's `\note` writes), one per line. */
export function slideNotes(annotations: NoteAnnotation[]): string {
  return annotations
    .filter((annotation) => annotation.subtype === "Text")
    .map((annotation) => (annotation.contentsObj?.str ?? annotation.contents ?? "").trim())
    .filter(Boolean)
    .join("\n")
}
