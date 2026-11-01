/** The line that names the file at `path` to a new session in `cwd`: relative to `cwd` when inside it, else absolute. */
export function fileLead(path: string, cwd: string): string {
  const base = cwd.endsWith("/") ? cwd : `${cwd}/`
  return `File: ${path.startsWith(base) ? path.slice(base.length) : path}`
}

/** `text` without the `File:` line `fileLead` put in front of a first message, so session titles show the request. */
export function withoutFileLead(text: string): string {
  return text.startsWith("File: ") ? text.replace(/^File: [^\n]*\n*/, "") : text
}
