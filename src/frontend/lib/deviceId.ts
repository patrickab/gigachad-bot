const DEVICE_ID_STORAGE_KEY = "gigachad-device-id"

/** Per-install id sent as X-Device-Id so the sync feed skips this device's own writes. Undefined on the server or when storage is blocked. */
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
