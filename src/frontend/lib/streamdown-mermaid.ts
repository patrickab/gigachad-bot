import mermaid from "mermaid"
import type { DiagramPlugin } from "streamdown"
import type { MermaidConfig } from "mermaid"

const DEFAULTS: MermaidConfig = {
  startOnLoad: false,
  theme: "default",
  securityLevel: "strict",
  fontFamily: "monospace",
  suppressErrorRendering: true,
}

export function createMermaidPlugin(options: { config?: MermaidConfig } = {}): DiagramPlugin {
  let initialized = false
  let config: MermaidConfig = { ...DEFAULTS, ...options.config }

  const instance = {
    initialize(next: MermaidConfig) {
      config = { ...DEFAULTS, ...options.config, ...next }
      mermaid.initialize(config)
      initialized = true
    },
    async render(id: string, source: string) {
      if (!initialized) {
        mermaid.initialize(config)
        initialized = true
      }
      return mermaid.render(id, source)
    },
  }

  return {
    name: "mermaid",
    type: "diagram",
    language: "mermaid",
    getMermaid(next?: MermaidConfig) {
      if (next) instance.initialize(next)
      return instance
    },
  }
}

export const mermaidPlugin = createMermaidPlugin()
