import { useSyncExternalStore } from "react";

/**
 * Class on <html> while the shell drives the frame: its model selector and chrome stand in for upstream's,
 * which stay mounted but hidden. The hook for every drive-only rule in `custom/theme/`, never set outside a frame.
 */
export const DRIVE_CLASS = "cf-shell-drive";

/** The parts of `document.documentElement` drive mode touches. */
type DriveRoot = { classList: { toggle(token: string, force: boolean): unknown } };

let driven = false;
const listeners = new Set<() => void>();

/**
 * Enter or leave drive mode: `embed/bridge.tsx` calls it on every `hello`, true only when the shell speaks this
 * frame's protocol version. A mismatched shell leaves upstream's UI fully visible.
 */
export function setShellDrive(next: boolean, root: DriveRoot = document.documentElement): void {
  root.classList.toggle(DRIVE_CLASS, next);
  if (next === driven) return;
  driven = next;
  listeners.forEach((listener) => listener());
}

/** Whether the shell drives the frame right now. */
export function isShellDrive(): boolean {
  return driven;
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** True while the shell drives the frame (`html.cf-shell-drive`): hide-and-drive overrides render their hidden form only then. */
export function useShellDrive(): boolean {
  return useSyncExternalStore(subscribe, isShellDrive, () => false);
}
