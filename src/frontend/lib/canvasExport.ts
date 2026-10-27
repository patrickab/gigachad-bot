import { inkOutlinePath, pointsBounds, type StrokeData } from "@/lib/drawing"

const HIGHLIGHT_OPACITY = 0.4

/** An image to fetch and draw at canvas position (x, y), `width` wide; its height follows the source image. */
export interface EmbedRect {
  url: string
  x: number
  y: number
  width: number
}

interface TextData {
  x: number
  y: number
  width: number
  height: number
  text: string
  color: string
  size: number
}

function handwritingFontFamily(): string {
  if (typeof document === "undefined") return "cursive"
  const name = getComputedStyle(document.documentElement).getPropertyValue("--font-handwriting").trim()
  return name ? `${name}, cursive` : "cursive"
}

// word-wrap a single line to fit maxWidth, using the context's currently-set font
function wrapLine(ctx: CanvasRenderingContext2D, line: string, maxWidth: number): string[] {
  const words = line.split(" ")
  const lines: string[] = []
  let cur = ""
  for (const word of words) {
    const test = cur ? `${cur} ${word}` : word
    if (cur && ctx.measureText(test).width > maxWidth) {
      lines.push(cur)
      cur = word
    } else {
      cur = test
    }
  }
  lines.push(cur)
  return lines
}

function drawTexts(ctx: CanvasRenderingContext2D, texts: TextData[], offsetX: number, offsetY: number) {
  if (texts.length === 0) return
  const family = handwritingFontFamily()
  for (const t of texts) {
    const bx = t.x - offsetX
    const by = t.y - offsetY
    ctx.save()
    ctx.beginPath()
    ctx.rect(bx, by, t.width, t.height)
    ctx.clip()
    ctx.font = `${t.size}px ${family}`
    ctx.fillStyle = t.color
    // 1.2 line spacing, baseline at 0.85em below the box top
    const lineHeight = t.size * 1.2
    const lines = t.text.split("\n").flatMap((line) => wrapLine(ctx, line, t.width))
    lines.forEach((line, i) => {
      ctx.fillText(line, bx, by + t.size * 0.85 + i * lineHeight)
    })
    ctx.restore()
  }
}

// ponytail: fetch + ImageBitmap, not `new Image()` with crossOrigin. The live canvas
// already loaded these URLs without CORS from another origin (API_BASE), so a crossOrigin
// request reuses that cached entry and errors, silently dropping images from the export.
async function loadImage(url: string): Promise<ImageBitmap> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`image ${res.status}: ${url}`)
  return createImageBitmap(await res.blob())
}

interface LoadedEmbed {
  x: number
  y: number
  width: number
  aspect: number
  image: CanvasImageSource
}

// A rendered PDF page already lives in a same-origin canvas. Reuse that decoded
// bitmap for whiteboard screenshots instead of fetching and rendering its source again.
export interface CanvasEmbedRect {
  image: CanvasImageSource
  x: number
  y: number
  width: number
  aspect: number
}
async function loadEmbeds(images: EmbedRect[]): Promise<LoadedEmbed[]> {
  const loaded: Array<LoadedEmbed | null> = await Promise.all(images.map(async (embed): Promise<LoadedEmbed | null> => {
    try {
      const image = await loadImage(embed.url)
      return { ...embed, image, aspect: image.height / image.width }
    } catch (err) {
      console.warn("canvas export: skipping image", embed.url, err)
      return null
    }
  }))
  return loaded.filter((embed): embed is LoadedEmbed => embed !== null)
}

// Render images + strokes onto a 2x canvas, translating by (offsetX, offsetY).
async function drawCanvas(
  strokes: StrokeData[],
  images: LoadedEmbed[],
  w: number,
  h: number,
  offsetX: number,
  offsetY: number,
  bg: string,
  texts: TextData[] = [],
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
    const dx = embed.x - offsetX
    const dy = embed.y - offsetY
    const dh = embed.width * embed.aspect
    ctx.drawImage(embed.image, dx, dy, embed.width, dh)
  }
  // Highlights share one translucent layer under the pen ink, so overlapping highlights don't stack darker.
  const highlights = strokes.filter((stroke) => stroke.highlight)
  if (highlights.length > 0) {
    const layer = document.createElement("canvas")
    layer.width = canvas.width
    layer.height = canvas.height
    const layerContext = layer.getContext("2d")
    if (layerContext) {
      layerContext.scale(dpr, dpr)
      fillStrokes(layerContext, highlights, offsetX, offsetY)
      ctx.save()
      ctx.setTransform(1, 0, 0, 1, 0, 0)
      ctx.globalAlpha = HIGHLIGHT_OPACITY
      ctx.globalCompositeOperation = "multiply"
      ctx.drawImage(layer, 0, 0)
      ctx.restore()
    }
  }
  fillStrokes(ctx, strokes.filter((stroke) => !stroke.highlight), offsetX, offsetY)
  if (texts.length > 0) await document.fonts.ready
  drawTexts(ctx, texts, offsetX, offsetY)
  return canvas
}

function fillStrokes(ctx: CanvasRenderingContext2D, strokes: StrokeData[], offsetX: number, offsetY: number) {
  for (const stroke of strokes) {
    const d = inkOutlinePath(stroke.points.map((p) => [p[0]! - offsetX, p[1]! - offsetY, p[2] ?? 0.5]), stroke.width, !!stroke.highlight)
    if (!d) continue
    ctx.fillStyle = stroke.color
    ctx.fill(new Path2D(d))
  }
}

function canvasToBlob(canvas: HTMLCanvasElement, type: string, quality?: number): Promise<Blob> {
  return new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("Failed to create blob"))), type, quality),
  )
}

/** Page region as PNG bytes on white. Embeds are fetched images plus already-decoded canvas images. */
export async function renderPageToPng(
  strokes: StrokeData[],
  pageX: number,
  pageY: number,
  pageW: number,
  pageH: number,
  images: EmbedRect[] = [],
  texts: TextData[] = [],
  canvasImages: CanvasEmbedRect[] = [],
): Promise<Uint8Array> {
  const loadedImages = [...await loadEmbeds(images), ...canvasImages]
  const canvas = await drawCanvas(strokes, loadedImages, pageW, pageH, pageX, pageY, "#ffffff", texts)
  const blob = await canvasToBlob(canvas, "image/png")
  return new Uint8Array(await blob.arrayBuffer())
}

/** Corner and sample points of every drawn element. */
function* contentPoints(strokes: StrokeData[], images: LoadedEmbed[], texts: TextData[]): Generator<readonly number[]> {
  for (const stroke of strokes) yield* stroke.points
  for (const img of images) {
    yield [img.x, img.y]
    yield [img.x + img.width, img.y + img.width * img.aspect]
  }
  for (const t of texts) {
    yield [t.x, t.y]
    yield [t.x + t.width, t.y + t.height]
  }
}

/** Crops to the bounding box of all content plus `padding`. Rejects when there is nothing to render. */
export async function renderCanvasToJpeg(strokes: StrokeData[], padding = 20, images: EmbedRect[] = [], texts: TextData[] = []): Promise<Blob> {
  const loadedImages = await loadEmbeds(images)
  const bounds = pointsBounds(contentPoints(strokes, loadedImages, texts))
  if (!bounds) return Promise.reject(new Error("Nothing to render"))
  const w = bounds.maxX - bounds.minX + padding * 2
  const h = bounds.maxY - bounds.minY + padding * 2
  const canvas = await drawCanvas(strokes, loadedImages, w, h, bounds.minX - padding, bounds.minY - padding, "#ffffff", texts)
  return canvasToBlob(canvas, "image/jpeg", 0.92)
}

export async function renderStrokesToJpeg(
  strokes: StrokeData[],
  width: number,
  height: number,
  bgColor: string,
): Promise<Blob> {
  return canvasToBlob(await drawCanvas(strokes, [], width, height, 0, 0, bgColor), "image/jpeg", 0.9)
}
