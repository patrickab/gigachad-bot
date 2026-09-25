import { createContext } from "react"

export interface ViewportPoint {
  x: number
  y: number
}

export interface ViewportSize {
  width: number
  height: number
}

export interface InfiniteViewportCamera {
  scale: number
  centerX: number
  centerY: number
}

export type ZoomAnchor = "cursor" | "viewport-center"

export function cameraToOffset(camera: InfiniteViewportCamera, viewport: ViewportSize): ViewportPoint {
  return {
    x: viewport.width / 2 - camera.centerX * camera.scale,
    y: viewport.height / 2 - camera.centerY * camera.scale,
  }
}

export function offsetToCamera(offset: ViewportPoint, scale: number, viewport: ViewportSize): InfiniteViewportCamera {
  return {
    scale,
    centerX: (viewport.width / 2 - offset.x) / scale,
    centerY: (viewport.height / 2 - offset.y) / scale,
  }
}

export function screenToWorld(point: ViewportPoint, camera: InfiniteViewportCamera, viewport: ViewportSize): ViewportPoint {
  const offset = cameraToOffset(camera, viewport)
  return {
    x: (point.x - offset.x) / camera.scale,
    y: (point.y - offset.y) / camera.scale,
  }
}

export function worldToScreen(point: ViewportPoint, camera: InfiniteViewportCamera, viewport: ViewportSize): ViewportPoint {
  const offset = cameraToOffset(camera, viewport)
  return {
    x: point.x * camera.scale + offset.x,
    y: point.y * camera.scale + offset.y,
  }
}

export function zoomCamera(
  camera: InfiniteViewportCamera,
  factor: number,
  viewport: ViewportSize,
  anchor: ZoomAnchor,
  cursor?: ViewportPoint,
): InfiniteViewportCamera {
  const scale = camera.scale * factor
  if (anchor === "viewport-center" || !cursor) return { ...camera, scale }

  const world = screenToWorld(cursor, camera, viewport)
  return {
    scale,
    centerX: world.x - (cursor.x - viewport.width / 2) / scale,
    centerY: world.y - (cursor.y - viewport.height / 2) / scale,
  }
}

// Carries a view across a change of its frame size and, for an embedded view,
// its host's zoom. The world point at the frame's center stays at the center,
// and the scale is multiplied by `factor`, so an embedded canvas magnifies
// with its host like any other object on it while still showing the same view.
export function reframeCamera(offset: ViewportPoint, scale: number, previous: ViewportSize, next: ViewportSize, factor = 1): { offset: ViewportPoint, scale: number } {
  const camera = offsetToCamera(offset, scale, previous)
  const scaled = { ...camera, scale: camera.scale * factor }
  return { offset: cameraToOffset(scaled, next), scale: scaled.scale }
}

// Effective zoom of the canvas an embedded viewport sits in (1 outside a canvas).
// Renderers deep inside an attachment (e.g. a markmap inside a markdown document)
// read it to magnify with their host the way nested canvases and graphs do.
export const HostScaleContext = createContext(1)
