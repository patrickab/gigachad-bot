/**
 * The merge queue of a repository: who may run the merge gates and move the target next. Shared by every task of the
 * repository, each in its own Pi process, through files in its git directory (`<git common dir>/factory-merge-queue`).
 *
 * - `joinQueue` writes a ticket named `<time>-<pid>-<start time>-<random>`: first come, first served
 * - `queueAhead` is how many live tickets are older than one's own. 0 means it is the ticket's turn. A ticket whose
 *   owner process no longer runs (a crashed session, the pid and its start time no longer match) is removed on the way
 * - `leaveQueue` removes a ticket. A merge leaves when it merged, failed, conflicted or was cancelled
 */
import { randomUUID } from "node:crypto";
import { mkdir, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";

const QUEUE_DIR = "factory-merge-queue";

export interface Ticket {
  path: string;
  name: string;
}

/** `<pid>-<start time>` of a running process, which a reused pid never repeats; null when it does not run. */
async function identity(pid: number): Promise<string | null> {
  try {
    const stat = await readFile(`/proc/${pid}/stat`, "utf8");
    // The fields after the name in parentheses (which may hold spaces), start time is the 20th of them.
    return `${pid}-${stat.slice(stat.lastIndexOf(")") + 2).split(" ")[19]}`;
  } catch {
    return null;
  }
}

/** A ticket of this process in the queue of the repository whose git common dir is `gitDir`. */
export async function joinQueue(gitDir: string): Promise<Ticket> {
  const dir = join(gitDir, QUEUE_DIR);
  await mkdir(dir, { recursive: true });
  const owner = (await identity(process.pid)) ?? `${process.pid}-0`;
  const name = `${String(Date.now()).padStart(15, "0")}-${owner}-${randomUUID().slice(0, 8)}`;
  await writeFile(join(dir, name), "");
  return { path: join(dir, name), name };
}

/** How many live tickets are ahead of `ticket`, removing those whose owner no longer runs. */
export async function queueAhead(ticket: Ticket): Promise<number> {
  const dir = join(ticket.path, "..");
  let ahead = 0;
  for (const name of (await readdir(dir)).sort()) {
    if (name >= ticket.name) continue;
    const [, pid, start] = name.split("-");
    if ((await identity(Number(pid))) !== `${pid}-${start}`) {
      await rm(join(dir, name), { force: true });
      continue;
    }
    ahead += 1;
  }
  return ahead;
}

export async function leaveQueue(ticket: Ticket): Promise<void> {
  await rm(ticket.path, { force: true });
}
