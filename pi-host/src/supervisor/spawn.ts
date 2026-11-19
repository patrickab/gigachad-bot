/**
 * Spawn strategies: how a profile's `pi --mode rpc` child is started (plan D5).
 *
 * - Consumes: the Pi binary, the allowlisted child env (M6), a profile, cwd and Pi arguments
 * - Provides: `PiChild` (stdio pipes, exit, signals) and `spawnStrategyFor(profile)`
 *
 * Batch 1 has only `coding` (Pi defaults, local process, C9). A sandboxed profile later maps to a strategy that runs
 * `agent-sandbox pi-rpc --profile ... -- <pi args>`, which keeps the same stdio, stdin-EOF and SIGTERM contract.
 */
import type { Subprocess } from "bun";

export type Profile = "coding";
export const PROFILES: readonly Profile[] = ["coding"];

export type SpawnRequest = {
  profile: Profile;
  cwd: string;
  /** Arguments after the binary, starting with `--mode rpc`. */
  piArgs: string[];
  env: Record<string, string>;
};

export type PiChild = Subprocess<"pipe", "pipe", "pipe">;

export interface SpawnStrategy {
  spawn(request: SpawnRequest): PiChild;
}

/** Runs the Pi binary directly on this machine. */
export function localSpawn(piBin: string): SpawnStrategy {
  return {
    spawn: ({ cwd, piArgs, env }) =>
      Bun.spawn([piBin, ...piArgs], { cwd, env, stdin: "pipe", stdout: "pipe", stderr: "pipe" }),
  };
}

/** The strategy that starts a profile's children. */
export function spawnStrategyFor(_profile: Profile, piBin: string): SpawnStrategy {
  return localSpawn(piBin);
}
