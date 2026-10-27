"use client";

import type { ComponentProps } from "react";
import { MarkdownBody as UpstreamMarkdownBody } from "../../../components/MarkdownBody";
import { EmbeddedMarkdown } from "../../components/EmbeddedMarkdown";
import { useIsEmbedded } from "../../components/useIsEmbedded";

// Every other importer of `components/MarkdownBody` gets this file, `EmbeddedMarkdown` included: keep upstream's exports.
export * from "../../../components/MarkdownBody";

type Props = ComponentProps<typeof UpstreamMarkdownBody>;

/**
 * Upstream markdown, rendered by gigachad-bot's shared renderer for user, assistant and subagent text when embedded.
 *
 * - Embedded user and assistant text, a subagent's task and result (`markdown-subagent-text`, its transcript dialog):
 *   `EmbeddedMarkdown` (`custom/components/EmbeddedMarkdown.tsx`), same props, same link and image behavior.
 *   Upstream gives every other surface its own class (compaction, custom and hub messages, release notes),
 *   so a renamed or new class stays on upstream's renderer. Tool and thinking output never reach `MarkdownBody`
 * - Standalone and every other surface: upstream unchanged
 */
/** `MarkdownBody` classes of the surfaces the shared renderer draws, `undefined` is assistant text. */
const SHARED_RENDERER_CLASSES = new Set([undefined, "markdown-user-message", "markdown-subagent-text", "gc-markdown-compact"]);

export function MarkdownBody(props: Props) {
  const embedded = useIsEmbedded();
  const isPreview = props.className === "gc-markdown-compact";
  return embedded && SHARED_RENDERER_CLASSES.has(props.className) ? <EmbeddedMarkdown {...props} compact={isPreview} /> : <UpstreamMarkdownBody {...props} />;
}
