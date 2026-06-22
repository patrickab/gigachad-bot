import { getStroke } from "perfect-freehand"

export interface StrokeData {
  points: number[][]
  color: string
  width: number
}

const STROKE_OPTIONS = {
  smoothing: 0.5,
  streamline: 0.5,
  simulatePressure: true,
  last: true,
} as const

export function getSvgPathFromStroke(points: number[][], closed = true): string {
  const len = points.length
  if (len < 4) return ""

  const avg = (a: number, b: number) => (a + b) / 2

  let a = points[0]!
  let b = points[1]!
  const c = points[2]!

  let result = `M${a[0]!.toFixed(2)},${a[1]!.toFixed(2)} Q${b[0]!.toFixed(2)},${b[1]!.toFixed(2)} ${avg(b[0], c[0]).toFixed(2)},${avg(b[1], c[1]).toFixed(2)} T`

  for (let i = 2, max = len - 1; i < max; i++) {
    a = points[i]!
    b = points[i + 1]!
    result += `${avg(a[0], b[0]).toFixed(2)},${avg(a[1], b[1]).toFixed(2)} `
  }

  if (closed) result += "Z"
  return result
}

export interface EmbedRect {
  url: string
  x: number
  y: number
  width: number
  aspect: number
}

export function strokeToPathData(stroke: StrokeData): string {
  const outline = getStroke(stroke.points, {
    ...STROKE_OPTIONS,
    size: stroke.width,
  })
  return getSvgPathFromStroke(outline)
}

async function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = "anonymous"
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = url
  })
}

// Render images + strokes onto a 2x canvas, translating by (offsetX, offsetY).
async function drawCanvas(
  strokes: StrokeData[],
  images: EmbedRect[],
  w: number,
  h: number,
  offsetX: number,
  offsetY: number,
  bg: string,
): Promise<HTMLCanvasElement> {
  const dpr = 2
  const canvas = document.createElement("canvas")
  canvas.width = w * dpr
  canvas.height = h * dpr
  const ctx = canvas.getContext("2d")
  if (!ctx) throw new Error("No canvas context")
  ctx.scale(dpr, dpr)
  ctx.fillStyle = bg
  ctx.fillRect(0, 0, w, h)
  // draw images first (below strokes)
  for (const embed of images) {
    try {
      const img = await loadImage(embed.url)
      const dx = embed.x - offsetX
      const dy = embed.y - offsetY
      const dh = embed.width * embed.aspect
      ctx.drawImage(img, dx, dy, embed.width, dh)
    } catch { /* skip broken images */ }
  }
  for (const stroke of strokes) {
    const outline = getStroke(
      stroke.points.map((p) => [p[0]! - offsetX, p[1]! - offsetY, p[2] ?? 0.5]),
      { ...STROKE_OPTIONS, size: stroke.width },
    )
    if (outline.length < 4) continue
    ctx.fillStyle = stroke.color
    ctx.fill(new Path2D(getSvgPathFromStroke(outline)))
  }
  return canvas
}

// ponytail: legacy sync path for callers that don't need images
function drawStrokes(
  strokes: StrokeData[],
  w: number,
  h: number,
  offsetX: number,
  offsetY: number,
  bg: string,
): HTMLCanvasElement {
  const dpr = 2
  const canvas = document.createElement("canvas")
  canvas.width = w * dpr
  canvas.height = h * dpr
  const ctx = canvas.getContext("2d")
  if (!ctx) throw new Error("No canvas context")
  ctx.scale(dpr, dpr)
  ctx.fillStyle = bg
  ctx.fillRect(0, 0, w, h)
  for (const stroke of strokes) {
    const outline = getStroke(
      stroke.points.map((p) => [p[0]! - offsetX, p[1]! - offsetY, p[2] ?? 0.5]),
      { ...STROKE_OPTIONS, size: stroke.width },
    )
    if (outline.length < 4) continue
    ctx.fillStyle = stroke.color
    ctx.fill(new Path2D(getSvgPathFromStroke(outline)))
  }
  return canvas
}

function canvasToBlob(canvas: HTMLCanvasElement, type: string, quality?: number): Promise<Blob> {
  return new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("Failed to create blob"))), type, quality),
  )
}

export async function renderPageToPng(
  strokes: StrokeData[],
  pageX: number,
  pageY: number,
  pageW: number,
  pageH: number,
  images: EmbedRect[] = [],
): Promise<Uint8Array> {
  const canvas = images.length > 0
    ? await drawCanvas(strokes, images, pageW, pageH, pageX, pageY, "#ffffff")
    : drawStrokes(strokes, pageW, pageH, pageX, pageY, "#ffffff")
  const blob = await canvasToBlob(canvas, "image/png")
  return new Uint8Array(await blob.arrayBuffer())
}

export async function renderCanvasToJpeg(strokes: StrokeData[], padding = 20, images: EmbedRect[] = []): Promise<Blob> {
  if (strokes.length === 0 && images.length === 0) return Promise.reject(new Error("Nothing to render"))
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const s of strokes) {
    for (const p of s.points) {
      if (p[0]! < minX) minX = p[0]!
      if (p[1]! < minY) minY = p[1]!
      if (p[0]! > maxX) maxX = p[0]!
      if (p[1]! > maxY) maxY = p[1]!
    }
  }
  for (const img of images) {
    if (img.x < minX) minX = img.x
    if (img.y < minY) minY = img.y
    if (img.x + img.width > maxX) maxX = img.x + img.width
    if (img.y + img.width * img.aspect > maxY) maxY = img.y + img.width * img.aspect
  }
  const w = maxX - minX + padding * 2
  const h = maxY - minY + padding * 2
  const canvas = images.length > 0
    ? await drawCanvas(strokes, images, w, h, minX - padding, minY - padding, "#ffffff")
    : drawStrokes(strokes, w, h, minX - padding, minY - padding, "#ffffff")
  return canvasToBlob(canvas, "image/jpeg", 0.92)
}

export function renderStrokesToJpeg(
  strokes: StrokeData[],
  width: number,
  height: number,
  bgColor: string,
): Promise<Blob> {
  return canvasToBlob(drawStrokes(strokes, width, height, 0, 0, bgColor), "image/jpeg", 0.9)
}