/**
 * Video paste: an `@…/pastes/<file>.(mp4|webm|mov|mkv)` mention in a prompt (the GUI composer stores pasted videos in
 * `<cwd>/.pi/pastes/`) reaches the model as one 4x3 frame grid image plus a metadata line.
 *
 * - `input` probes the video (`ffprobe`), renders the grid (`ffmpeg … tile=4x3`, temp file, removed) and transforms the prompt
 * - An idle prompt also gets one `video-preview` custom message (`details.videos`: metadata and grid per video) right after it, injected
 *   from `before_agent_start`, so it joins the prompt's run and starts no extra turn
 * - A prompt sent while a run streams (steer, follow-up) is only transformed: a queued custom message would cost an extra model call
 * - ffmpeg and ffprobe come from PATH, a failing video is reported with `notify` and its mention left as typed
 */
import { mkdtemp, readFile, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join, resolve } from "node:path"
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent"

const MENTION = /@(\S*\/pastes\/\S+\.(?:mp4|webm|mov|mkv))(?=\s|$)/gi
const COLUMNS = 4
const ROWS = 3
const TILE_WIDTH = 320
const TOOL_TIMEOUT_MS = 60_000

interface VideoMetadata {
  durationSeconds: number | null
  width: number | null
  height: number | null
  fps: number | null
}

export interface VideoPreviewDetails extends VideoMetadata {
  path: string
  columns: number
  rows: number
  image: { data: string; mimeType: "image/png" }
}

/** `details` of a `video-preview` custom message: one entry per video of the prompt. */
export interface VideoPreviewMessageDetails {
  videos: VideoPreviewDetails[]
}

interface ProbeOutput {
  format?: { duration?: string }
  streams?: { width?: number; height?: number; avg_frame_rate?: string; r_frame_rate?: string }[]
}

function frameRate(rate: string | undefined): number | null {
  const [numerator, denominator] = (rate ?? "").split("/").map(Number)
  const fps = denominator ? numerator / denominator : numerator
  return Number.isFinite(fps) && fps > 0 ? Math.round(fps * 100) / 100 : null
}

/** One line the model reads next to the grid. */
export function metadataLine({ path, durationSeconds, width, height, fps }: Omit<VideoPreviewDetails, "image">): string {
  const facts = [
    durationSeconds === null ? null : `${durationSeconds.toFixed(1)} s`,
    width && height ? `${width}x${height}` : null,
    fps === null ? null : `${fps} fps`,
  ].filter(Boolean)
  return `[Video ${path}: ${facts.join(", ")}. Attached: ${COLUMNS * ROWS} evenly spaced frames as a ${COLUMNS}x${ROWS} grid, left to right, top to bottom]`
}

/** The video mentions in `text`, each once, in order. */
export function videoMentions(text: string): string[] {
  return [...new Set([...text.matchAll(MENTION)].map((match) => match[1]))]
}

export default function (pi: ExtensionAPI) {
  let pending: VideoPreviewDetails[] = []

  async function probe(file: string): Promise<VideoMetadata> {
    const args = ["-v", "error", "-select_streams", "v:0", "-show_entries", "format=duration:stream=width,height,avg_frame_rate,r_frame_rate", "-of", "json", file]
    const result = await pi.exec("ffprobe", args, { timeout: TOOL_TIMEOUT_MS })
    if (result.code !== 0) throw new Error(result.stderr.trim() || `ffprobe exited with ${result.code}`)
    const output = JSON.parse(result.stdout) as ProbeOutput
    const stream = output.streams?.[0]
    if (!stream) throw new Error("no video stream")
    const duration = Number(output.format?.duration)
    return {
      durationSeconds: Number.isFinite(duration) && duration > 0 ? duration : null,
      width: stream.width ?? null,
      height: stream.height ?? null,
      fps: frameRate(stream.avg_frame_rate) ?? frameRate(stream.r_frame_rate),
    }
  }

  async function grid(file: string, durationSeconds: number | null): Promise<string> {
    const folder = await mkdtemp(join(tmpdir(), "pi-video-paste-"))
    try {
      const output = join(folder, "grid.png")
      // Spread the frames over the whole video; an unknown duration samples one frame per second
      const sampleRate = durationSeconds ? (COLUMNS * ROWS) / durationSeconds : 1
      const filter = `fps=${sampleRate},scale=${TILE_WIDTH}:-2,tile=${COLUMNS}x${ROWS}`
      const args = ["-v", "error", "-y", "-i", file, "-vf", filter, "-frames:v", "1", output]
      const result = await pi.exec("ffmpeg", args, { timeout: TOOL_TIMEOUT_MS })
      if (result.code !== 0) throw new Error(result.stderr.trim() || `ffmpeg exited with ${result.code}`)
      return (await readFile(output)).toString("base64")
    } finally {
      await rm(folder, { recursive: true, force: true })
    }
  }

  async function preview(path: string, cwd: string): Promise<VideoPreviewDetails> {
    const file = resolve(cwd, path)
    const metadata = await probe(file)
    const data = await grid(file, metadata.durationSeconds)
    return { path, ...metadata, columns: COLUMNS, rows: ROWS, image: { data, mimeType: "image/png" } }
  }

  pi.on("input", async (event, ctx) => {
    const mentions = videoMentions(event.text)
    if (mentions.length === 0) return { action: "continue" }
    const previews: VideoPreviewDetails[] = []
    for (const path of mentions) {
      try {
        previews.push(await preview(path, ctx.cwd))
      } catch (error) {
        ctx.ui.notify(`Video preview of ${path} failed: ${error instanceof Error ? error.message : String(error)}`, "warning")
      }
    }
    if (previews.length === 0) return { action: "continue" }
    pending = event.streamingBehavior ? [] : previews
    const lines = previews.map(({ image: _image, ...details }) => metadataLine(details))
    const images = previews.map(({ image }) => ({ type: "image" as const, ...image }))
    return { action: "transform", text: `${event.text}\n\n${lines.join("\n")}`, images: [...(event.images ?? []), ...images] }
  })

  pi.on("before_agent_start", async (event) => {
    // Only the run of the prompt the previews were made for: a prompt another extension handled leaves them behind
    const videos = pending.filter(({ path }) => event.messageType === "user" && event.prompt.includes(path))
    pending = []
    if (videos.length === 0) return
    const details: VideoPreviewMessageDetails = { videos }
    return { message: { customType: "video-preview", content: `Video preview: ${videos.map(({ path }) => path).join(", ")}`, display: true, details } }
  })
}
