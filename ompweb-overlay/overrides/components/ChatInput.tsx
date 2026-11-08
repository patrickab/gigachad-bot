"use client";

import { forwardRef, memo, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState, type ComponentPropsWithoutRef } from "react";
import { ChatInput as UpstreamChatInput, type AttachedImage, type ChatInputHandle } from "../../../components/ChatInput";
import { toast } from "@/components/ui/toast";
import { clearDraft, subscribeDraftRecovery } from "@/lib/draft-store";
import { selectableThinkingLevels } from "@/lib/thinking-levels";
import { isEmbedded } from "../../components/EmbedMode";
import { useIsEmbedded } from "../../components/useIsEmbedded";
import { isShellConnected, onShellMessage, postToShell } from "../../embed/channel";
import { fileCompletions } from "../../embed/completions";
import { submitShellMessage } from "../../embed/submit";
import { useShellCommands } from "../../embed/slash-commands";
import type { Completion, ComposerImage } from "@gigachad/chat-ui/protocol";

export * from "../../../components/ChatInput";

type Props = ComponentPropsWithoutRef<typeof UpstreamChatInput>;

/** Session the composer belongs to: ChatWindow's `draftKey` is the session id, or `new:<cwd>` on the new-session screen. */
function composerSessionId(draftKey: string | undefined): string | null {
  return draftKey && !draftKey.startsWith("new:") ? draftKey : null;
}

function toAttachedImage(image: ComposerImage): AttachedImage {
  return { ...image, previewUrl: `data:${image.mimeType};base64,${image.data}` };
}

/** `@` completions for the shell's draft. No `cwd` (new session without a directory): none, like upstream. */
async function completionsFor(props: Props, text: string, caret: number): Promise<Completion[]> {
  if (!props.cwd) return [];
  return (await fileCompletions(props.cwd, text, caret)) ?? [];
}

/**
 * Upstream composer, driven by gigachad-bot's composer while embedded (`custom/theme/embed.css` hides this one).
 *
 * - Out: `composer` state (session, busy, model, thinking level) on change. `composer-restore` for what omp hands back to the input box
 * - In: `composer-abort`, `composer-set-model`, `composer-set-thinking`, through the same props upstream's own controls call
 * - `composer-send` goes through `submitShellMessage` like upstream's Enter key: a follow-up (or steer, per Settings) while busy.
 *   Text the submit path keeps (refused, `/btw` with images, failed builtin) returns as `composer-restore`, as do rejected prompts
 * - `composer-complete` is answered with `composer-completions` (`@` files), `/` commands go out as `composer-commands` state
 * - Standalone: upstream unchanged
 */
export const ChatInput = memo(forwardRef<ChatInputHandle, Props>(function ChatInput(props, ref) {
  const propsRef = useRef(props);
  useEffect(() => { propsRef.current = props; });
  const sessionId = composerSessionId(props.draftKey);
  useShellCommands(useIsEmbedded(), sessionId, props.cwd ?? undefined, props.slashCommands, props.onLoadSlashCommands);
  const { isStreaming, availableThinkingLevels } = props;
  const provider = props.model?.provider;
  const modelId = props.model?.modelId;
  const thinkingLevel = props.thinkingLevel ?? "auto";
  // Upstream shows the reasoning menu only with a change handler, listing `selectableThinkingLevels`.
  const hasThinkingMenu = Boolean(props.onThinkingLevelChange);
  const thinkingLevels = useMemo(
    () => (hasThinkingMenu ? selectableThinkingLevels(availableThinkingLevels) : []),
    [hasThinkingMenu, availableThinkingLevels],
  );
  // Upstream's context ring reads the same fallback chain, the shell shows the count instead of the percentage.
  const contextTokens = (props.contextUsage ?? props.sessionStats?.contextUsage)?.tokens ?? null;

  useEffect(() => {
    if (!isEmbedded()) return;
    const model = provider && modelId ? { provider, modelId } : null;
    postToShell({ type: "composer", sessionId, streaming: isStreaming, model, thinkingLevel, thinkingLevels, contextTokens });
  }, [sessionId, isStreaming, provider, modelId, thinkingLevel, thinkingLevels, contextTokens]);

  /** One shell message at a time, as upstream's composer holds a second Enter while one is stored. */
  const submittingRef = useRef(false);

  /** Submit a shell message like upstream's Enter key; text it keeps goes back to the shell. */
  const submitFromShell = useCallback(async (sessionId: string | null, text: string, images: ComposerImage[]) => {
    let outcome: "cleared" | "kept" = "kept";
    if (!submittingRef.current) {
      submittingRef.current = true;
      try {
        outcome = await submitShellMessage(propsRef.current, text, images.map(toAttachedImage));
      } catch (error) {
        toast.error(error instanceof Error ? error.message : String(error));
      } finally {
        submittingRef.current = false;
      }
    }
    if (outcome === "cleared" || (!text && images.length === 0)) return;
    postToShell({ type: "composer-restore", sessionId, text, replace: null, images });
  }, []);

  useEffect(() => {
    if (!isEmbedded()) return;
    const isForThisComposer = (target: string | null) => target === composerSessionId(propsRef.current.draftKey);
    const unsubscribers = [
      onShellMessage("composer-send", (message) => {
        if (!isForThisComposer(message.sessionId)) return;
        void submitFromShell(message.sessionId, message.text, message.images);
      }),
      onShellMessage("composer-abort", (message) => {
        if (isForThisComposer(message.sessionId)) propsRef.current.onAbort();
      }),
      onShellMessage("composer-set-model", (message) => {
        if (isForThisComposer(message.sessionId)) propsRef.current.onModelChange?.(message.provider, message.modelId);
      }),
      onShellMessage("composer-set-thinking", (message) => {
        if (isForThisComposer(message.sessionId)) propsRef.current.onThinkingLevelChange?.(message.level);
      }),
      onShellMessage("composer-complete", ({ sessionId: target, requestId, text, caret }) => {
        if (!isForThisComposer(target)) return;
        const answer = (items: Completion[]) => postToShell({ type: "composer-completions", sessionId: target, requestId, items });
        completionsFor(propsRef.current, text, caret).then(answer, () => answer([]));
      }),
    ];
    return () => unsubscribers.forEach((unsubscribe) => unsubscribe());
  }, [submitFromShell]);

  // A recovery (abort withdrew queued messages, queue Edit) goes to the shell. The hidden input box got it too:
  // drop its draft and remount it empty, so it never resurfaces there.
  const [hiddenInputGeneration, setHiddenInputGeneration] = useState(0);
  useEffect(() => {
    if (!isEmbedded()) return;
    return subscribeDraftRecovery((key, recovery) => {
      if (key !== propsRef.current.draftKey || !isShellConnected()) return;
      const { text, replace = null, images = [] } = recovery;
      postToShell({ type: "composer-restore", sessionId: composerSessionId(key), text, replace, images });
      clearDraft(key);
      setHiddenInputGeneration((generation) => generation + 1);
    });
  }, []);

  // Text upstream puts into the input box by handle (Edit's prefill, file mentions) goes to the shell's composer instead:
  // the hidden box would swallow it. The shell merges it before its draft, like a restore.
  const upstreamHandleRef = useRef<ChatInputHandle | null>(null);
  useImperativeHandle(ref, () => {
    const toShell = (fallback: (text: string) => void) => (text: string) => {
      if (!isEmbedded() || !isShellConnected()) return fallback(text);
      if (text) postToShell({ type: "composer-restore", sessionId: composerSessionId(propsRef.current.draftKey), text, replace: null, images: [] });
    };
    return {
      focus: () => upstreamHandleRef.current?.focus(),
      insertText: toShell((text) => upstreamHandleRef.current?.insertText(text)),
      insertIfEmpty: toShell((text) => upstreamHandleRef.current?.insertIfEmpty(text)),
      prependText: toShell((text) => upstreamHandleRef.current?.prependText(text)),
      addFiles: (files) => upstreamHandleRef.current?.addFiles(files),
      openContextPanel: () => upstreamHandleRef.current?.openContextPanel(),
      retargetPendingRecovery: (from, to) => upstreamHandleRef.current?.retargetPendingRecovery(from, to),
    };
  }, []);

  return <UpstreamChatInput key={hiddenInputGeneration} ref={upstreamHandleRef} {...props} />;
}));
