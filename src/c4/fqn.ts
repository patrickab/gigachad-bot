// Dotted element FQNs (`shop.api.db`): parent, name and ancestry.

export const parentOf = (fqn: string): string | null => {
  const index = fqn.lastIndexOf('.')
  return index === -1 ? null : fqn.slice(0, index)
}

export const nameOf = (fqn: string) => fqn.slice(fqn.lastIndexOf('.') + 1)

/** The top-level element `fqn` lies under, or `fqn` itself at the top level. */
export const topLevelOf = (fqn: string) => {
  const index = fqn.indexOf('.')
  return index === -1 ? fqn : fqn.slice(0, index)
}

export const isWithin = (fqn: string, ancestor: string) => fqn === ancestor || fqn.startsWith(`${ancestor}.`)

/** Rewrites `fqn` when it is, or lies under, a renamed or moved element. */
export const mapPrefix = (fqn: string, renames: Record<string, string>) => {
  for (const [from, to] of Object.entries(renames)) if (isWithin(fqn, from)) return to + fqn.slice(from.length)
  return fqn
}
