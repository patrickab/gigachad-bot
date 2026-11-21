"use client"

import { registerMessageRenderer } from "./renderers"
import type { CustomMessage } from "./types"

type Image = { data: string; mimeType: string }

/** One frame the model saw, `seconds` null for the 4x3 grid older sessions stored as one `image`. */
interface PreviewedFrame {
  seconds: number | null
  image: Image
}

/** One video of a `video-preview` message (`~/.pi/agent/extensions/video-paste.ts`, from `~/.os`): its mention path, metadata and frames. */
export interface PreviewedVideo {
  path: string
  durationSeconds: number | null
  width: number | null
  height: number | null
  fps: number | null
  frames: PreviewedFrame[]
}

type StoredVideo = Omit<PreviewedVideo, "frames"> & { frames?: { seconds: number; image: Image }[]; image?: Image }

const isImage = (image: Partial<Image> | undefined): image is Image => typeof image?.data === "string" && typeof image.mimeType === "string"

function previewedVideo(value: unknown): PreviewedVideo | null {
  const video = value as Partial<StoredVideo> | null
  if (typeof video?.path !== "string") return null
  const { image, frames, ...facts } = video as StoredVideo
  if (Array.isArray(frames) && frames.length > 0 && frames.every((frame) => typeof frame?.seconds === "number" && isImage(frame.image))) {
    return { ...facts, frames }
  }
  return isImage(image) ? { ...facts, frames: [{ seconds: null, image }] } : null
}

/** The videos of `message`'s details, null when they are not a `video-preview`'s. */
export function previewedVideos(message: CustomMessage): PreviewedVideo[] | null {
  const stored = (message.details as { videos?: unknown } | undefined)?.videos
  if (!Array.isArray(stored) || stored.length === 0) return null
  const videos = stored.map(previewedVideo)
  return videos.every((video) => video !== null) ? (videos as PreviewedVideo[]) : null
}

function factsOf({ path, durationSeconds, width, height, fps }: PreviewedVideo): string {
  return [
    path.split("/").at(-1),
    durationSeconds === null ? null : `${durationSeconds.toFixed(1)} s`,
    width && height ? `${width}×${height}` : null,
    fps === null ? null : `${fps} fps`,
  ].filter(Boolean).join(" · ")
}

/** `83.4` as `1:23.4`, the clock the model's metadata line uses. */
function clockTime(seconds: number): string {
  const minutes = Math.floor(seconds / 60)
  return `${minutes}:${(seconds - minutes * 60).toFixed(1).padStart(4, "0")}`
}

/** The frames the model saw, each with its time, the file name and metadata below. */
export function VideoPreviewCard({ video }: { video: PreviewedVideo }) {
  return (
    <figure className="my-2">
      <div className="grid grid-cols-2 gap-2">
        {video.frames.map(({ seconds, image }, index) => (
          <div key={index} className={seconds === null ? "col-span-2" : undefined}>
            {/* eslint-disable-next-line @next/next/no-img-element -- inline data URL from the session, not optimizable by next/image */}
            <img
              src={`data:${image.mimeType};base64,${image.data}`}
              alt={seconds === null ? `Frames of ${video.path}` : `${video.path} at ${clockTime(seconds)}`}
              className="max-w-full rounded-lg"
            />
            {seconds !== null && <div className="mt-0.5 text-xs tabular-nums text-ink-muted">{clockTime(seconds)}</div>}
          </div>
        ))}
      </div>
      <figcaption className="mt-1 text-xs text-ink-muted">{factsOf(video)}</figcaption>
    </figure>
  )
}

registerMessageRenderer("video-preview", (message, next) => {
  const videos = previewedVideos(message)
  return videos ? videos.map((video, index) => <VideoPreviewCard key={`${video.path}-${index}`} video={video} />) : next()
})
