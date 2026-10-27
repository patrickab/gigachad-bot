"use client";

import type { ComponentProps } from "react";
import { MarkdownBody as UpstreamMarkdownBody } from "../../../components/MarkdownBody";
import { EmbeddedMarkdown } from "../../components/EmbeddedMarkdown";
import { useIsEmbedded } from "../../components/useIsEmbedded";

// Every other importer of `components/MarkdownBody` gets this file, `EmbeddedMarkdown` included: keep upstream's exports.
export * from "../../../components/MarkdownBody";

type Props = ComponentProps<typeof UpstreamMarkdownBody>;

/**
 * Upstream markdown, rendered by gigachad-bot's shared renderer for user and assistant text when embedded.
 *
 * - Embedded user and assistant text: `EmbeddedMarkdown` (`custom/components/EmbeddedMarkdown.tsx`), same props, same link and image behavior.
 *   Upstream gives every other surface its own class (compaction, custom and hub messages, subagent transcripts, release notes),
 *   so a renamed or new class stays on upstream's renderer. Tool and thinking output never reach `MarkdownBody`
 * - Standalone and every other surface: upstream unchanged
 */
export function MarkdownBody(props: Props) {
  const embedded = useIsEmbedded();
  const isPreview = props.className === "gc-markdown-compact";
  const isConversationText = props.className === undefined || props.className === "markdown-user-message" || isPreview;
  return embedded && isConversationText ? <EmbeddedMarkdown {...props} compact={isPreview} /> : <UpstreamMarkdownBody {...props} />;
}
