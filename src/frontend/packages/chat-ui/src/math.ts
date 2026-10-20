/** Rewrite `\( \)` and `\[ \]` to `$` and `$$` outside code spans and fences. */
export function normalizeMathDelimiters(input: string): string {
  if (input.indexOf("\\(") === -1 && input.indexOf("\\[") === -1) return input
  const segments = input.split(/(```[\s\S]*?```|`[^`]*`)/g)
  return segments
    .map((seg, i) => {
      if (i % 2 === 1) return seg
      return seg
        .replace(/\\\[/g, () => "$$")
        .replace(/\\\]/g, () => "$$")
        .replace(/\\\(/g, () => "$")
        .replace(/\\\)/g, () => "$")
    })
    .join("")
}
