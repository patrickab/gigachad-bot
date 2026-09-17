const DEVICE_ID_STORAGE_KEY = "gigachad-device-id"

export function getDeviceId(): string | undefined {
  if (typeof window === "undefined") return undefined
  try {
    const existing = window.localStorage.getItem(DEVICE_ID_STORAGE_KEY)
    if (existing) return existing
    const deviceId = crypto.randomUUID()
    window.localStorage.setItem(DEVICE_ID_STORAGE_KEY, deviceId)
    return deviceId
  } catch {
    return undefined
  }
}
