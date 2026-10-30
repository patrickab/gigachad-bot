"use client";

import { useRef, type ComponentProps } from "react";
import { TopBar as UpstreamTopBar } from "../../../components/top-bar";
import { useSessionTitleBar } from "../../embed/chrome";

export * from "../../../components/top-bar";

/**
 * Upstream's top bar inside a `data-cf-top-bar` marker (`display: contents`). The session's title bar (title, rename,
 * Plan chip, coverage ring, phase chip) is marked `data-cf-session-title` by its content (`embed/chrome.tsx`
 * `useSessionTitleBar`): hidden in drive mode by `theme/chrome.css`, still mounted. The shell shows its title and chips
 * (`session-meta`) and opens its Plan popup through it (`open-panel`). Other bars (the chain table's search and close)
 * stay visible.
 */
export function TopBar(props: ComponentProps<typeof UpstreamTopBar>) {
  const root = useRef<HTMLDivElement>(null);
  useSessionTitleBar(root);
  return (
    <div ref={root} data-cf-top-bar="">
      <UpstreamTopBar {...props} />
    </div>
  );
}
