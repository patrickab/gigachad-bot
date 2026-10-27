// Element ids: derived from titles, made unique, validated.

export const ID_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/

/** lowerCamelCase ASCII id from a title: "Payment Service" -> "paymentService". */
export function idFromTitle(title: string): string {
  const words = title.normalize('NFKD').replace(/[\u0300-\u036f]/g, '').split(/[^A-Za-z0-9]+/).filter(Boolean)
  const id = words.map((word, index) => {
    const lower = word.toLowerCase()
    return index === 0 ? lower : lower.charAt(0).toUpperCase() + lower.slice(1)
  }).join('')
  if (!id) return 'element'
  return /^[0-9]/.test(id) ? `element${id.charAt(0).toUpperCase()}${id.slice(1)}` : id
}

/** "payment-service" or "paymentService" -> "Payment Service". */
export function titleFromName(name: string): string {
  const words = name.replace(/([a-z0-9])([A-Z])/g, '$1 $2').split(/[^A-Za-z0-9]+/).filter(Boolean)
  return words.map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join(' ') || 'Architecture'
}

export function uniqueId(base: string, taken: ReadonlySet<string>): string {
  if (!taken.has(base)) return base
  let suffix = 2
  while (taken.has(`${base}${suffix}`)) suffix += 1
  return `${base}${suffix}`
}
