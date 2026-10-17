import { createContext } from "react"

// Effective zoom of the canvas an embedded viewport sits in (1 outside a canvas).
// Renderers deep inside an attachment (e.g. a markmap inside a markdown document)
// read it to magnify with their host the way nested canvases and graphs do.
export const HostScaleContext = createContext(1)
