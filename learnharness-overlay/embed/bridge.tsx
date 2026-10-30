"use client";

import { useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { IS_STATE, LEARN_PROTOCOL_VERSION, type LearnFrameMessage } from "@gigachad/chat-ui/learn-protocol";
import type { LearnChannel } from "./channel";
import { createShellDispatcher, openLearnChannel } from "./channel";
import { setShellDrive } from "./drive";
import { isEmbedded } from "./mode";
import { pathFor, sessionIdOf } from "./routes";
import { guardSettingsDialog } from "./settings";
import { applyTheme } from "./theme";

/** The configured frame ancestors, inlined at build time by `custom/framing.ts`. */
const SHELL_ORIGINS = (process.env.GIGACHAD_LEARNHARNESS_FRAME_ANCESTORS ?? "").split(" ").filter(Boolean);

const dispatcher = createShellDispatcher();
let channel: LearnChannel | null = null;
/** State posted before the channel opened (effects of components below `EmbedBridge` run first), handed to it on open. */
const earlyState: LearnFrameMessage[] = [];

/**
 * Handle one shell message type, returns the unsubscribe. Register in an effect: handlers run while embedded only, and
 * drive messages (`ANY_VERSION_SHELL_MESSAGE` false) reach them only from a shell at this frame's protocol version.
 */
export const onShellMessage = dispatcher.on;

/** Post a frame message to the shell. No-op outside a frame. State messages (`IS_STATE`) are replayed on every `hello`. */
export function postToFrameShell(message: LearnFrameMessage): void {
  if (channel) channel.post(message);
  else if (IS_STATE[message.type] !== false && isEmbedded()) earlyState.push(message);
}

/**
 * The frame's side of the learn embed protocol (`@gigachad/chat-ui/learn-protocol`), mounted once at the app root.
 *
 * - Embedded only: standalone LearnHarness opens no channel
 * - Only a shell at one of the configured frame ancestors is answered (`./channel`)
 * - Every `hello` sets drive mode (`./drive`): on only when the shell speaks `LEARN_PROTOCOL_VERSION`
 * - Theme from `hello`/`set-theme`, navigation from `open-session`/`new-session`/`open-home`, `view` on every page change.
 *   Every other shell message goes to the handlers the bridges below register with `onShellMessage`
 * - The settings dialog stays off the hidden Providers section, embedded or not (`./settings`)
 */
export function EmbedBridge() {
  const router = useRouter();
  const pathname = usePathname();
  const [opened, setOpened] = useState<LearnChannel | null>(null);

  useEffect(() => guardSettingsDialog(), []);

  useEffect(() => {
    if (!isEmbedded()) return;
    const stops = [
      onShellMessage("hello", ({ theme, version }) => {
        applyTheme(theme);
        setShellDrive(version === LEARN_PROTOCOL_VERSION);
      }),
      onShellMessage("set-theme", ({ theme }) => applyTheme(theme)),
      onShellMessage("open-session", (message) => router.push(pathFor(message))),
      onShellMessage("new-session", (message) => router.push(pathFor(message))),
      onShellMessage("open-home", (message) => router.push(pathFor(message))),
    ];
    const next = openLearnChannel(window, SHELL_ORIGINS, dispatcher);
    earlyState.splice(0).forEach((message) => next.post(message));
    channel = next;
    setOpened(next);
    return () => {
      stops.forEach((stop) => stop());
      next.close();
      if (channel === next) channel = null;
      setShellDrive(false);
    };
  }, [router]);

  useEffect(() => {
    opened?.post({ type: "view", sessionId: sessionIdOf(pathname) });
  }, [opened, pathname]);

  return null;
}
