/** Save `blob` as `fileName` through an object URL: the `download` attribute of a cross-origin link is ignored. */
export function saveBlob(blob: Blob, fileName: string) {
  const url = URL.createObjectURL(blob)
  const anchor = Object.assign(document.createElement("a"), { href: url, download: fileName })
  document.body.append(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(url)
}
