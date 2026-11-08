import type { ComponentProps } from "react";
import type { AttachedImage, ChatInput } from "../../components/ChatInput";
import { toast } from "@/components/ui/toast";
import { toastBtwError } from "@/hooks/useBtw";
import { getSubmitDuringRunBehavior } from "@/lib/composer-prefs";
import { translate } from "@/lib/i18n";
import { validateOutgoingPrompt } from "@/lib/image-attachments";
import { expandWebSlashCommand } from "@/lib/web-slash-commands";

type ChatInputProps = ComponentProps<typeof ChatInput>;

/** The upstream composer props a shell submission goes through. */
export type ShellSubmitProps = Pick<
  ChatInputProps,
  "isStreaming" | "submissionBlocked" | "advisorEnabled" | "onSend" | "onSteer" | "onFollowUp" | "onBuiltinCommand" | "onAudioUnlock"
>;

/**
 * cleared -> the message reached omp-web (prompt, queue or builtin), the shell drops it
 * kept -> refused or not delivered, the shell keeps it in its composer
 */
export type ShellSubmitOutcome = "cleared" | "kept";

function refusesOversized(message: string, images: AttachedImage[]): boolean {
  const error = validateOutgoingPrompt(message, images);
  if (error) toast.error(error);
  return error !== null;
}

async function runBuiltin(props: ShellSubmitProps, message: string): Promise<ShellSubmitOutcome | null> {
  const result = await props.onBuiltinCommand!(message);
  if (!result.handled) return null;
  return result.error || result.retainInput ? "kept" : "cleared";
}

/** Queue during a run, as upstream's Enter key does: web slash commands expanded, `/advisor` gated. */
async function queue(props: ShellSubmitProps, message: string, images: AttachedImage[]): Promise<ShellSubmitOutcome> {
  const send = getSubmitDuringRunBehavior() === "steer" && props.onSteer ? props.onSteer : props.onFollowUp ?? props.onSteer;
  if (!send) return "kept";
  let prompt = message;
  if (message.startsWith("/")) {
    if (message.slice(1).split(/\s+/)[0] === "advisor" && !props.advisorEnabled) {
      toast.error(translate("agentSession.advisorDisabled"));
      return "kept";
    }
    const expansion = expandWebSlashCommand(message);
    if (expansion.kind === "usage-error") {
      toast.error(translate("chatInput.commandUsageTitle"), translate("agentSession.commandRequiresArgs", {
        command: expansion.command,
        usage: translate(expansion.argumentHintKey),
      }));
      return "kept";
    }
    if (expansion.kind === "expand") prompt = expansion.prompt;
  }
  if (refusesOversized(prompt, images)) return "kept";
  return (await send(prompt, images.length ? images : undefined)) ? "cleared" : "kept";
}

/**
 * Submit a shell composer message (text and images, no files or uploads) the way upstream's Enter key does.
 *
 * - `/btw`: a side question beside any run, never with images
 * - During a run: a follow-up or steer per Settings. `!cmd` runs at once instead
 * - Idle: builtin slash commands first, then a prompt. Only a rejected prompt is kept, a superseded one was processed
 */
export async function submitShellMessage(props: ShellSubmitProps, text: string, images: AttachedImage[]): Promise<ShellSubmitOutcome> {
  const message = text.trim();
  if (!message && !images.length) return "cleared";
  if (props.submissionBlocked) return "kept";
  props.onAudioUnlock?.();
  if (/^\/btw(\s|$)/.test(message) && props.onBuiltinCommand) {
    if (images.length) {
      toast.error(translate("btw.attachmentsUnsupported"));
      return "kept";
    }
    try {
      return (await runBuiltin(props, message)) ?? "kept";
    } catch (error) {
      toastBtwError(error);
      return "kept";
    }
  }
  const shellCommand = message.startsWith("!") && !images.length;
  if (props.isStreaming && !shellCommand) return queue(props, message, images);
  if (message.startsWith("/") && !images.length && props.onBuiltinCommand) {
    const expansion = expandWebSlashCommand(message);
    if (expansion.kind === "expand" && refusesOversized(expansion.prompt, images)) return "kept";
    const outcome = await runBuiltin(props, message);
    if (outcome) return outcome;
  }
  if (refusesOversized(message, images)) return "kept";
  const result = await props.onSend(message, images.length ? images : undefined);
  return result.status === "rejected" ? "kept" : "cleared";
}
