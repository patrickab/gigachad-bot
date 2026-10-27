"use client";

import { forwardRef, memo, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState, type ComponentPropsWithoutRef } from "react";
import { ChatInput as UpstreamChatInput, type AttachedImage, type AttachedTextFile, type ChatInputHandle } from "../../../components/ChatInput";
import { useComposerSubmit, type ComposerSubmit } from "@/components/ChatInput-submit";
import { toast } from "@/components/ui/toast";
import { getSubmitDuringRunBehavior } from "@/lib/composer-prefs";
import { clearDraft, subscribeDraftRecovery } from "@/lib/draft-store";
import { selectableThinkingLevels } from "@/lib/thinking-levels";
import { isEmbedded } from "../../components/EmbedMode";
import { isShellConnected, onShellMessage, postToShell } from "../../embed/channel";
import { commandCompletions, fileCompletions, slashQueryOf } from "../../embed/completions";
import type { Completion, ComposerImage } from "@gigachad/chat-ui/protocol";
import type { SlashCommandInfo } from "@/hooks/useAgentSession";

export * from "../../../components/ChatInput";

type Props = ComponentPropsWithoutRef<typeof UpstreamChatInput>;

/** Session the composer belongs to: ChatWindow's `draftKey` is the session id, or `new:<cwd>` on the new-session screen. */
function composerSessionId(draftKey: string | undefined): string | null {
  return draftKey && !draftKey.startsWith("new:") ? draftKey : null;
}

function toAttachedImage(image: ComposerImage): AttachedImage {
  return { ...image, previewUrl: `data:${image.mimeType};base64,${image.data}` };
}

/** A shell message being submitted: upstream's submit path clears the input once it dispatched it, else the text stays the user's. */
interface ShellSubmission {
  text: string;
  images: ComposerImage[];
  attachedImages: AttachedImage[];
  cleared: boolean;
  /** Builtin command started for it; `/btw` decides after the submit call returned. */
  builtin?: Promise<unknown>;
}

const NO_TEXT_FILES: AttachedTextFile[] = [];
/** The shell owns the draft: submissions always pass their text as `overrideText`. */
const NO_DRAFT_TEXT = "";

function showAttachError(error: string | null) {
  if (error) toast.error(error);
}

/**
 * Completions for the shell's draft: omp commands for a `/word` draft, else files for an `@` token before the caret.
 *
 * - `loadCommands` runs only when the session reported no commands yet, the way upstream's menu asks once per open
 * - No `cwd` (new session without a directory): no file completions, like upstream
 */
async function completionsFor(props: Props, text: string, caret: number, loadCommands: () => Promise<SlashCommandInfo[]>): Promise<Completion[]> {
  const slashQuery = slashQueryOf(text);
  if (slashQuery !== null) {
    const commands = props.slashCommands?.length ? props.slashCommands : await loadCommands();
    return commandCompletions(commands, text, slashQuery);
  }
  if (!props.cwd) return [];
  return (await fileCompletions(props.cwd, text, caret)) ?? [];
}

/**
 * Upstream composer, driven by gigachad-bot's composer while embedded (`custom/theme/embed.css` hides this one).
 *
 * - Out: `composer` state (session, busy, model, thinking level) on change. `composer-restore` for what omp hands back to the input box
 * - In: `composer-abort`, `composer-set-model`, `composer-set-thinking`, through the same props upstream's own controls call
 * - `composer-send` goes through upstream's `useComposerSubmit` like its Enter key: a follow-up (or steer, per Settings) while busy.
 *   Text the submit path keeps (refused, `/btw` with images, failed builtin) returns as `composer-restore`, as do rejected prompts
 * - `composer-complete` is answered with `composer-completions` (commands, `@` files)
 * - Standalone: upstream unchanged
 */
export const ChatInput = memo(forwardRef<ChatInputHandle, Props>(function ChatInput(props, ref) {
  const propsRef = useRef(props);
  useEffect(() => { propsRef.current = props; });
  const sessionId = composerSessionId(props.draftKey);
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

  const valueRef = useRef(NO_DRAFT_TEXT);
  const imagesRef = useRef<AttachedImage[]>([]);
  const textFilesRef = useRef(NO_TEXT_FILES);
  const draftKeyRef = useRef(props.draftKey);
  const submissionBlockedRef = useRef(Boolean(props.submissionBlocked));
  const submissionRef = useRef<ShellSubmission | null>(null);
  useEffect(() => {
    draftKeyRef.current = props.draftKey;
    submissionBlockedRef.current = Boolean(props.submissionBlocked);
  });

  /** Points the hook's shared refs back at `submission`, for a submit path resuming after an await. */
  const resume = useCallback((submission: ShellSubmission) => {
    submissionRef.current = submission;
    imagesRef.current = submission.attachedImages;
  }, []);
  const clearInput = useCallback(() => {
    if (submissionRef.current) submissionRef.current.cleared = true;
  }, []);
  const { onBuiltinCommand } = props;
  const runBuiltinCommand = useMemo(() => onBuiltinCommand && (async (message: string) => {
    const submission = submissionRef.current;
    const pending = onBuiltinCommand(message);
    if (submission) submission.builtin = pending;
    try {
      return await pending;
    } finally {
      if (submission) resume(submission);
    }
  }), [onBuiltinCommand, resume]);

  const submit = useComposerSubmit({
    value: NO_DRAFT_TEXT,
    valueRef,
    imagesRef,
    textFilesRef,
    draftKeyRef,
    submissionBlockedRef,
    isStreaming: props.isStreaming,
    advisorEnabled: props.advisorEnabled,
    clearInput,
    setAttachError: showAttachError,
    onSend: props.onSend,
    onSteer: props.onSteer,
    onFollowUp: props.onFollowUp,
    onPromptWithStreamingBehavior: props.onPromptWithStreamingBehavior,
    onBuiltinCommand: runBuiltinCommand,
    onAudioUnlock: props.onAudioUnlock,
  });
  const submitRef = useRef<ComposerSubmit>(submit);
  useEffect(() => { submitRef.current = submit; });

  /** Submit a shell message through upstream's submit path the way its Enter key does; text it keeps goes back to the shell. */
  const submitFromShell = useCallback(async (sessionId: string | null, text: string, images: ComposerImage[]) => {
    const submission: ShellSubmission = { text, images, attachedImages: images.map(toAttachedImage), cleared: false };
    resume(submission);
    const { isStreaming: busy, onSteer, onFollowUp } = propsRef.current;
    const { handleSend, sendQueued } = submitRef.current;
    try {
      if (busy && (onSteer || onFollowUp)) sendQueued(getSubmitDuringRunBehavior() === "steer" && onSteer ? "steer" : "followup", text);
      else await handleSend(text);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    }
    await submission.builtin?.catch(() => undefined);
    if (submissionRef.current === submission) submissionRef.current = null;
    if (submission.cleared || (!text && images.length === 0)) return;
    postToShell({ type: "composer-restore", sessionId, text, replace: null, images });
  }, [resume]);

  // One command load per composer key: a session without commands must not reload on every keystroke.
  const commandsRequestedForRef = useRef<string | undefined>(undefined);

  useEffect(() => {
    if (!isEmbedded()) return;
    const isForThisComposer = (target: string | null) => target === composerSessionId(propsRef.current.draftKey);
    const loadCommands = async () => {
      const { draftKey, onLoadSlashCommands } = propsRef.current;
      if (!onLoadSlashCommands || commandsRequestedForRef.current === draftKey) return [];
      commandsRequestedForRef.current = draftKey;
      return onLoadSlashCommands();
    };
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
        completionsFor(propsRef.current, text, caret, loadCommands).then(answer, () => answer([]));
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
