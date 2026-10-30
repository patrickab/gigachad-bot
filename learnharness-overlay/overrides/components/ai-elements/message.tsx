"use client";

import { type ComponentProps, type ReactNode, useSyncExternalStore } from "react";
import {
  MessageResponse as UpstreamMessageResponse,
  ToolContentResponse as UpstreamToolContentResponse,
} from "../../../../components/ai-elements/message";
import { isEmbedded } from "../../../embed/mode";

// Every importer of `components/ai-elements/message` gets this file (`reasoning.tsx` too, through `./message`): keep upstream's exports.
export * from "../../../../components/ai-elements/message";

const subscribeNever = () => () => {};

/** `isEmbedded()` for render: false on the server and during hydration, the real value right after, so markup never mismatches. */
function useIsEmbedded(): boolean {
  return useSyncExternalStore(subscribeNever, isEmbedded, () => false);
}

/**
 * The root `@gigachad/chat-ui/markdown.css` styles (`.gc-markdown`, `data-markdown-root`) around upstream's markdown,
 * embedded only. `display: contents` keeps the caller's layout (inline explanations, flex gaps) as upstream draws it.
 * Standalone LearnHarness renders upstream's markup unchanged.
 */
function GigachadMarkdownRoot({ children }: { children: ReactNode }) {
  if (!useIsEmbedded()) return children;
  return (
    <div data-markdown-root="" className="gc-markdown" style={{ display: "contents" }}>
      {children}
    </div>
  );
}

/**
 * Upstream's `MessageResponse` (Streamdown with LearnHarness's plugins, mermaid, callouts and math repair), in
 * gigachad-bot's markdown root while embedded. The look is `custom/theme/prose.css`.
 */
export function MessageResponse(props: ComponentProps<typeof UpstreamMessageResponse>) {
  return (
    <GigachadMarkdownRoot>
      <UpstreamMessageResponse {...props} />
    </GigachadMarkdownRoot>
  );
}

/** Upstream's `ToolContentResponse` in the same root: upstream's own calls `MessageResponse` inside its module, past this file. */
export function ToolContentResponse(props: ComponentProps<typeof UpstreamToolContentResponse>) {
  return (
    <GigachadMarkdownRoot>
      <UpstreamToolContentResponse {...props} />
    </GigachadMarkdownRoot>
  );
}
