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

export function strokeToPathData(stroke: StrokeData): string {
  const outline = getStroke(stroke.points, {
    ...STROKE_OPTIONS,
    size: stroke.width,
  })
  return getSvgPathFromStroke(outline)
}

export function renderStrokesToJpeg(
  strokes: StrokeData[],
  width: number,
  height: number,
): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const canvas = document.createElement("canvas")
    const scale = 2
    canvas.width = width * scale
    canvas.height = height * scale
    const ctx = canvas.getContext("2d")
    if (!ctx) {
      reject(new Error("No canvas context"))
      return
    }
    ctx.scale(scale, scale)
    ctx.fillStyle = "#ffffff"
    ctx.fillRect(0, 0, width, height)

    for (const stroke of strokes) {
      const outline = getStroke(stroke.points, {
        ...STROKE_OPTIONS,
        size: stroke.width,
      })
      if (outline.length < 4) continue
      const pathData = getSvgPathFromStroke(outline)
      const path = new Path2D(pathData)
      ctx.fillStyle = stroke.color
      ctx.fill(path)
    }

    canvas.toBlob(
      (blob) => {
        if (blob) resolve(blob)
        else reject(new Error("Failed to create blob"))
      },
      "image/jpeg",
      0.9,
    )
  })
}