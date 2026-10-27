"use client";

import { useEffect } from "react";
import { useTheme } from "@/hooks/useTheme";
import { OPEN_SESSION_EVENT } from "@/lib/notification-client";
import { onShellMessage, postToShell } from "../embed/channel";
import { applyShellInsets, settleTranscriptAtBottom, trackComposerArea } from "../embed/insets";
import type { FrameMessage, KeyChord } from "@gigachad/chat-ui/protocol";
import { isEmbedded } from "./EmbedMode";

const VIEW_POLL_MS = 400;

/** Upstream hooks the bridge drives: the hidden sidebar exists exactly while full-page Settings is closed. */
const SIDEBAR_SELECTOR = "#workspace-sidebar";
const SETTINGS_ROW_SELECTOR = ".sidebar-settings-row";

function isSettingsOpen(): boolean {
  return document.querySelector(SIDEBAR_SELECTOR) === null;
}

function currentView(): Extract<FrameMessage, { type: "view" }> {
  return { type: "view", sessionId: new URLSearchParams(window.location.search).get("session"), settingsOpen: isSettingsOpen() };
}

function isSameView(left: FrameMessage | null, right: FrameMessage): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function matchesChord(event: KeyboardEvent, chord: KeyChord): boolean {
  return event.key === chord.key && event.altKey === chord.altKey && event.ctrlKey === chord.ctrlKey && event.metaKey === chord.metaKey && event.shiftKey === chord.shiftKey;
}

/** Open full-page Settings through the hidden sidebar row. Its handler toggles, so it only clicks while Settings is closed. */
function openSettings() {
  if (isSettingsOpen()) return;
  document.querySelector<HTMLElement>(SETTINGS_ROW_SELECTOR)?.click();
}

/**
 * Connect the app to gigachad-bot's shell while it runs in its frame (`@gigachad/chat-ui/protocol`).
 *
 * - In: open a session (upstream's notification-click event), open Settings, theme. New session: `custom/overrides/hooks/useKeyboardShortcuts.ts`
 * - Out: the current view, polled from the URL and the sidebar's presence, and the shell's own shortcuts
 */
export function EmbedBridge() {
  const { setTheme } = useTheme();

  useEffect(() => {
    if (!isEmbedded()) return;
    let forwardedChords: KeyChord[] = [];
    let lastView: FrameMessage | null = null;

    const reportViewChange = () => {
      const view = currentView();
      if (isSameView(lastView, view)) return;
      lastView = view;
      postToShell(view);
    };
    const forwardShellShortcut = (event: KeyboardEvent) => {
      const chord = forwardedChords.find((candidate) => matchesChord(event, candidate));
      if (!chord) return;
      event.preventDefault();
      event.stopPropagation();
      postToShell({ type: "key", ...chord });
    };

    const unsubscribers = [
      onShellMessage("hello", (message) => {
        forwardedChords = message.forwardKeys;
        setTheme(message.theme);
      }),
      onShellMessage("open-session", (message) => window.dispatchEvent(new CustomEvent(OPEN_SESSION_EVENT, { detail: message.sessionId }))),
      onShellMessage("open-settings", openSettings),
      onShellMessage("set-theme", (message) => setTheme(message.theme)),
      onShellMessage("insets", applyShellInsets),
      trackComposerArea(),
      settleTranscriptAtBottom(),
    ];
    window.addEventListener("keydown", forwardShellShortcut, true);
    const viewPoll = window.setInterval(reportViewChange, VIEW_POLL_MS);
    reportViewChange();

    return () => {
      unsubscribers.forEach((unsubscribe) => unsubscribe());
      window.removeEventListener("keydown", forwardShellShortcut, true);
      window.clearInterval(viewPoll);
    };
  }, [setTheme]);

  return null;
}
