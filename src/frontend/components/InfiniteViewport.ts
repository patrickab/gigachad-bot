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
export type EmbeddingScale = "inherit" | "screen-stable"

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

// A screen-stable frame counteracts its parent's scale at the frame boundary.
// Its parent-world position still follows the parent camera.
export function embeddedViewportSize(size: ViewportSize, parentScale: number, embeddingScale: EmbeddingScale): ViewportSize {
  if (embeddingScale === "screen-stable") return size
  return { width: size.width * parentScale, height: size.height * parentScale }
}

export function nestedEmbeddingScale(embeddingScale: EmbeddingScale | undefined): EmbeddingScale {
  return embeddingScale ?? "screen-stable"
}
