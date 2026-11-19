/**
 * `/files`: pick a file the current branch changed through `edit` or `write`, then open it.
 *
 * - The list comes from the session branch (`ctx.sessionManager.getBranch()`), so it follows resume, fork and `/tree` with no state of its own
 * - RPC (gigachad-bot's Pi tab): `notify` with `Open file: <abs path>` (C7), which the tab opens in its file panel
 * - TUI: hands the terminal to `$VISUAL`/`$EDITOR` (Pi's Ctrl+G order, `nano` last) and resumes
 */
import { spawnSync } from "node:child_process";
import { statSync } from "node:fs";
import { homedir } from "node:os";
import { resolve } from "node:path";
import type { ExtensionAPI, ExtensionCommandContext, SessionEntry } from "@earendil-works/pi-coding-agent";

/** The notice gigachad-bot's Pi tab turns into opening the path that follows (`OPEN_FILE_NOTICE_PREFIX` in `usePiSession.ts`). */
const OPEN_FILE_NOTICE_PREFIX = "Open file: ";

function isFile(path: string): boolean {
  try {
    return statSync(path).isFile();
  } catch {
    return false;
  }
}

/** Absolute paths of existing files that successful `edit`/`write` calls on `branch` touched, most recent last. */
function changedFiles(branch: readonly SessionEntry[], cwd: string): string[] {
  const callPaths = new Map<string, string>();
  const paths = new Set<string>();
  for (const entry of branch) {
    if (entry.type !== "message") continue;
    const message = entry.message;
    if (message.role === "assistant") {
      for (const part of message.content) {
        if (part.type === "toolCall" && (part.name === "edit" || part.name === "write") && typeof part.arguments.path === "string") callPaths.set(part.id, part.arguments.path);
      }
    } else if (message.role === "toolResult" && !message.isError) {
      const path = callPaths.get(message.toolCallId);
      if (!path) continue;
      const absolute = resolve(cwd, path);
      paths.delete(absolute);
      paths.add(absolute);
    }
  }
  return [...paths].filter(isFile);
}

/** `path` relative to `cwd` (`./src/app.ts`) or home (`~/notes.md`) when inside them. */
function displayPath(path: string, cwd: string): string {
  const repo = resolve(cwd);
  const home = homedir();
  if (path.startsWith(`${repo}/`)) return `.${path.slice(repo.length)}`;
  if (path.startsWith(`${home}/`)) return `~${path.slice(home.length)}`;
  return path;
}

/** Run the user's editor on `path` with the terminal handed over, resolving to its exit code. */
function editInTerminal(ctx: ExtensionCommandContext, path: string): Promise<number | null> {
  const editor = process.env.VISUAL || process.env.EDITOR || "nano";
  return ctx.ui.custom<number | null>((tui, _theme, _keybindings, done) => {
    tui.stop();
    // `$1` keeps a path with spaces whole while `editor` may carry its own arguments
    const result = spawnSync("sh", ["-c", `${editor} "$1"`, "files", path], { cwd: ctx.cwd, stdio: "inherit" });
    tui.start();
    tui.requestRender(true);
    done(result.status);
    return { render: () => [], invalidate: () => {} };
  });
}

export default function filesExtension(pi: ExtensionAPI): void {
  pi.registerCommand("files", {
    description: "Open a file this session branch changed",
    handler: async (_args, ctx) => {
      if (!ctx.hasUI) return;
      const files = changedFiles(ctx.sessionManager.getBranch(), ctx.cwd).reverse();
      if (files.length === 0) {
        ctx.ui.notify("No files changed in this session branch", "info");
        return;
      }
      const labels = files.map((path) => displayPath(path, ctx.cwd));
      const selected = await ctx.ui.select("Files changed in this session", labels);
      const path = selected === undefined ? undefined : files[labels.indexOf(selected)];
      if (!path) return;
      if (ctx.mode !== "tui") {
        ctx.ui.notify(`${OPEN_FILE_NOTICE_PREFIX}${path}`, "info");
        return;
      }
      const exitCode = await editInTerminal(ctx, path);
      if (exitCode !== 0) ctx.ui.notify(`Editor exited with code ${exitCode}`, "error");
    },
  });
}
