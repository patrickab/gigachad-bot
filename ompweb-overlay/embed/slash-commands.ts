import { useEffect, useEffectEvent, useMemo, useState } from "react";
import type { SlashCommandInfo } from "@/hooks/useAgentSession";
import type { Skill, SlashCommand } from "@gigachad/chat-ui/protocol";
import { onShellMessage, postToShell } from "./channel";

/** Skills omp resolves for `cwd` (`omp skill list`), without a session process. */
async function fetchSkills(cwd: string): Promise<Skill[]> {
  const response = await fetch(`/api/skills?${new URLSearchParams({ cwd })}`);
  if (!response.ok) throw new Error(`skills failed: ${response.status}`);
  const body = (await response.json()) as { skills?: Skill[] };
  return (body.skills ?? []).map(({ name, description }) => ({ name, description }));
}

/** Skills come as their own list, `/skill:<name>` commands are dropped here. */
function toSlashCommands(commands: SlashCommandInfo[] | undefined): SlashCommand[] {
  return (commands ?? []).flatMap((command) => (command.source === "skill" ? [] : [{ name: command.name, description: command.description, kind: command.source }]));
}

/**
 * Report the composer's skills and slash commands to the shell (`composer-commands`), whose `/` menu filters them itself.
 *
 * - Skills: `/api/skills` for `cwd` on every mount or directory change, so installs show without a reload
 * - Commands: upstream's list once loaded. The shell's first `/` asks for it (`composer-load-commands`), loading resumes an idle session's omp process.
 *   The new-session screen never loads them: upstream's load would create the session
 */
export function useShellCommands(
  embedded: boolean,
  sessionId: string | null,
  cwd: string | undefined,
  commands: SlashCommandInfo[] | undefined,
  loadCommands: (() => unknown) | undefined,
): void {
  const [loadedSkills, setLoadedSkills] = useState<{ cwd: string; skills: Skill[] } | null>(null);
  useEffect(() => {
    if (!embedded || !cwd) return;
    let current = true;
    const settle = (skills: Skill[]) => { if (current) setLoadedSkills({ cwd, skills }); };
    fetchSkills(cwd).then(settle, () => settle([]));
    return () => { current = false; };
  }, [embedded, cwd]);
  const skills = useMemo(() => (loadedSkills && loadedSkills.cwd === cwd ? loadedSkills.skills : []), [loadedSkills, cwd]);
  const slashCommands = useMemo(() => toSlashCommands(commands), [commands]);

  useEffect(() => {
    if (embedded) postToShell({ type: "composer-commands", sessionId, skills, commands: slashCommands });
  }, [embedded, sessionId, skills, slashCommands]);

  const load = useEffectEvent((requestedFor: string) => {
    if (requestedFor === sessionId && !commands?.length) void loadCommands?.();
  });
  useEffect(() => {
    if (!embedded) return;
    return onShellMessage("composer-load-commands", (message) => load(message.sessionId));
  }, [embedded]);
}
