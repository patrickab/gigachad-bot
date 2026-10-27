"use client";

import { useEffect, useLayoutEffect, useRef, useState, type ComponentProps } from "react";
import { createPortal } from "react-dom";
import { ComposerPanels as UpstreamComposerPanels } from "../../../components/ComposerPanels";
import { useIsEmbedded } from "../../components/useIsEmbedded";

export * from "../../../components/ComposerPanels";

type Props = ComponentProps<typeof UpstreamComposerPanels>;

/**
 * omp-web's end-of-transcript marker next to `from`: `<div ref={messagesEndRef} />`, last in the chat column of
 * `components/ChatWindow.tsx`. Null when the shape differs, the panels then stay where upstream puts them.
 */
function findTranscriptEnd(from: Element): HTMLElement | null {
  const log = from.closest(":has([role=log])")?.querySelector("[role=log]");
  const end = log?.firstElementChild?.firstElementChild?.lastElementChild;
  const isBareMarker = end instanceof HTMLDivElement && end.childElementCount === 0 && end.attributes.length === 0;
  return isBareMarker ? end : null;
}

/**
 * Upstream's composer panels; embedded, Tasks and Subagents close the chat history instead of riding on the composer.
 *
 * - Rendered into upstream's end marker, so they scroll with the transcript, always last, and upstream's
 *   scroll-to-bottom and follow logic include them
 * - The /btw side question stays above the composer
 * - No native `title` tooltips in the moved panels (subagent chips show their whole task), `aria-label`s stay
 * - Standalone, or no transcript yet (new session): upstream unchanged
 */
export function ComposerPanels(props: Props) {
  const embedded = useIsEmbedded();
  const sentinelRef = useRef<HTMLSpanElement>(null);
  const panelsRef = useRef<HTMLDivElement>(null);
  const [transcriptEnd, setTranscriptEnd] = useState<HTMLElement | null>(null);
  useLayoutEffect(() => {
    if (embedded && sentinelRef.current) setTranscriptEnd(findTranscriptEnd(sentinelRef.current));
  }, [embedded]);
  useEffect(() => {
    const panels = panelsRef.current;
    if (!panels) return;
    // React sets `title` again whenever a label changes (Running → Completed), so watch for it.
    const stripTitles = () => panels.querySelectorAll("[title]").forEach((element) => element.removeAttribute("title"));
    stripTitles();
    const observer = new MutationObserver(stripTitles);
    observer.observe(panels, { subtree: true, childList: true, attributeFilter: ["title"] });
    return () => observer.disconnect();
  }, [transcriptEnd]);

  if (!embedded) return <UpstreamComposerPanels {...props} />;
  return (
    <>
      <span ref={sentinelRef} hidden />
      {transcriptEnd ? (
        <>
          <UpstreamComposerPanels {...props} todoPhases={[]} subagents={[]} liveStatus={null} />
          {createPortal(
            <div ref={panelsRef} className="cf-session-panels">
              <UpstreamComposerPanels {...props} btw={null} />
            </div>,
            transcriptEnd,
          )}
        </>
      ) : (
        <UpstreamComposerPanels {...props} />
      )}
    </>
  );
}
