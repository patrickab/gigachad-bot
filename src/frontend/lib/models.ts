export function displayName(m: string): string {
  const slash = m.indexOf("/")
  return slash === -1 ? m : m.slice(slash + 1)
}
