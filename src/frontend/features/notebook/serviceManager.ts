import { ServerConnection, ServiceManager } from "@jupyterlab/services"
import { getApiBase } from "@/lib/backend/config"
import { deviceHeaders } from "@/lib/backend/deviceId"
import { apiWsUrl } from "@/lib/backend/urls"

/** `@jupyterlab/services` pointed at the backend's Jupyter proxy: same API as a Jupyter server, the real token stays on the backend. */
export function createNotebookServiceManager(): ServiceManager {
  const baseUrl = `${getApiBase()}/jupyter/server/`
  const serverSettings = ServerConnection.makeSettings({
    baseUrl,
    wsUrl: apiWsUrl("/jupyter/server/"),
    // The library insists on a token, the proxy replaces whatever it gets.
    token: "proxied",
    appendToken: false,
    init: { credentials: "same-origin" },
    // The library passes a Request and calls `fetch` detached, so `window.fetch` cannot be handed over as is.
    fetch: (input, init) => {
      const request = new Request(input, init)
      for (const [name, value] of Object.entries(deviceHeaders())) request.headers.set(name, value)
      return fetch(request)
    },
  })
  return new ServiceManager({ serverSettings })
}
