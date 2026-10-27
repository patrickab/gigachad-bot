"use client";

import { useEffect, useRef } from "react";
import { useGlobalKeyboardShortcuts as upstreamUseGlobalKeyboardShortcuts } from "../../../hooks/useKeyboardShortcuts";
import { isEmbedded } from "../../components/EmbedMode";
import { onShellMessage, startEmbedChannel } from "../../embed/channel";

export { registerAbortHandler } from "../../../hooks/useKeyboardShortcuts";

/** Upstream keeps its options interface private, its parameter is the contract. */
type ShortcutOptions = Parameters<typeof upstreamUseGlobalKeyboardShortcuts>[0];

/**
 * Upstream's global shortcuts, plus the shell's `new-session` while embedded.
 *
 * - AppShell hands its new-session callback (Ctrl+Alt+N) to this hook, the only in-app way to open the composer in a given cwd
 * - Standalone: upstream unchanged
 */
export function useGlobalKeyboardShortcuts(options: ShortcutOptions): void {
  const onNewSessionRef = useRef(options.onNewSession);
  useEffect(() => { onNewSessionRef.current = options.onNewSession; }, [options.onNewSession]);

  useEffect(() => {
    if (!isEmbedded()) return;
    const unsubscribe = onShellMessage("new-session", (message) => onNewSessionRef.current?.(message.cwd));
    startEmbedChannel();
    return unsubscribe;
  }, []);

  upstreamUseGlobalKeyboardShortcuts(options);
}
