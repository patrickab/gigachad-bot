"use client"

import { registerMessageRenderer } from "./renderers"
import type { CustomMessage } from "./types"

/** One video of a `video-preview` message (`pi-extensions/video-paste.ts`): its mention path, metadata and frame grid. */
export interface PreviewedVideo {
  path: string
  durationSeconds: number | null
  width: number | null
  height: number | null
  fps: number | null
  image: { data: string; mimeType: string }
}

function isPreviewedVideo(value: unknown): value is PreviewedVideo {
  const video = value as Partial<PreviewedVideo> | null
  return typeof video?.path === "string" && typeof video.image?.data === "string" && typeof video.image.mimeType === "string"
}

/** The videos of `message`'s details, null when they are not a `video-preview`'s. */
export function previewedVideos(message: CustomMessage): PreviewedVideo[] | null {
  const videos = (message.details as { videos?: unknown } | undefined)?.videos
  return Array.isArray(videos) && videos.length > 0 && videos.every(isPreviewedVideo) ? videos : null
}

function factsOf({ path, durationSeconds, width, height, fps }: PreviewedVideo): string {
  return [
    path.split("/").at(-1),
    durationSeconds === null ? null : `${durationSeconds.toFixed(1)} s`,
    width && height ? `${width}×${height}` : null,
    fps === null ? null : `${fps} fps`,
  ].filter(Boolean).join(" · ")
}

/** The frame grid the model saw, its file name and metadata below. */
export function VideoPreviewCard({ video }: { video: PreviewedVideo }) {
  return (
    <figure className="my-2">
      {/* eslint-disable-next-line @next/next/no-img-element -- inline data URL from the session, not optimizable by next/image */}
      <img src={`data:${video.image.mimeType};base64,${video.image.data}`} alt={`Frames of ${video.path}`} className="max-w-full rounded-lg" />
      <figcaption className="mt-1 text-xs text-ink-muted">{factsOf(video)}</figcaption>
    </figure>
  )
}

registerMessageRenderer("video-preview", (message, next) => {
  const videos = previewedVideos(message)
  return videos ? videos.map((video, index) => <VideoPreviewCard key={`${video.path}-${index}`} video={video} />) : next()
})
