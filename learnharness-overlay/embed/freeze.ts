import { isShellDrive } from "./drive";
import { sessionIdOf } from "./routes";

/**
 * A learn session keeps the model and thinking level it was stamped with: the shell groups sessions by what they ran on,
 * and only LearnHarness's draft surfaces (home, `/chat/new`) pick the strong tier for the next one.
 *
 * `overrides/components/model-picker.tsx` disables the session's pills. Upstream's keyboard paths reach the control by
 * window events instead (`components/model-picker.tsx`: ⌘K opens the model menu, ⇧⇥ cycles the thinking level), so they
 * are swallowed while a session page is open. Capture listeners on the target run before upstream's bubble listeners.
 *
 * One way out: a session whose model gigachad no longer serves cannot run another turn. Upstream's send guard then opens
 * the model menu with the same event (`setup-cta.tsx` `openModelPicker`). The menu is portaled out of the disabled
 * pills, so once the session's model is confirmed unavailable the event is dispatched again and let through, and the
 * learner picks a served model for the session.
 *
 * While the shell drives the frame (`./drive`) its model selector stands in for the hidden pills, whose menu would still
 * portal out of its hidden wrapper onto the page. So the model menu event is swallowed on every page and goes to the
 * shell as `model-open` instead (`./model.tsx`): at once on a draft surface, after the same unavailable check in a
 * session. The thinking cycle keeps its non-drive path: upstream's hidden control writes the strong tier on a draft
 * surface, a session swallows it.
 */
export const SESSION_PICKER_EVENTS = ["lh:kbd-open-model-picker", "lh:kbd-cycle-thinking"] as const;
const OPEN_MODEL_PICKER = SESSION_PICKER_EVENTS[0];

/** The parts of `window` the freeze uses. */
type FreezeWindow = {
  location: { pathname: string };
  addEventListener(type: string, listener: (event: Event) => void, options: { capture: boolean }): void;
  removeEventListener(type: string, listener: (event: Event) => void, options: { capture: boolean }): void;
  dispatchEvent(event: Event): boolean;
};

/** True when session *sessionId*'s model is known to be unavailable. A failed read answers false: the freeze holds. */
export type ModelUnavailable = (sessionId: string) => Promise<boolean>;

/** Drive mode as the freeze sees it: whether the shell drives, and where upstream's model menu requests go then. */
export type FreezeDrive = { active(): boolean; openModel(sessionId: string | null): void };

let modelOpener: ((sessionId: string | null) => void) | null = null;

/** Where the freeze sends model menu requests in drive mode (`./model.tsx` posts `model-open`), returns the unregister. */
export function onModelOpen(opener: (sessionId: string | null) => void): () => void {
  modelOpener = opener;
  return () => {
    if (modelOpener === opener) modelOpener = null;
  };
}

/** The frame's drive mode. Swallowing does not wait for an opener: the hidden menu must never open over the page. */
const shellDrive: FreezeDrive = {
  active: isShellDrive,
  openModel: (sessionId) => modelOpener?.(sessionId),
};

type ModelRef = { provider?: string; id?: string };

async function readJson<T>(path: string): Promise<T> {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return (await response.json()) as T;
}

/**
 * Upstream's own test (`setup-cta.tsx` `selectedModelUnavailable`) on the model the session runs on: its stamped model,
 * else the strong tier for a legacy unstamped session (`model-picker.tsx`, as `resolveConversationSpec` resolves it).
 */
export const sessionModelUnavailable: ModelUnavailable = async (sessionId) => {
  try {
    const [session, catalog] = await Promise.all([
      readJson<{ settings?: { model?: ModelRef } }>(`/api/chats/${encodeURIComponent(sessionId)}`),
      readJson<{ models?: ModelRef[] }>("/api/models"),
    ]);
    const model =
      session.settings?.model ??
      (await readJson<{ settings?: { models?: { strong?: { model?: ModelRef } } } | null }>("/api/settings")).settings?.models
        ?.strong?.model;
    if (!model?.provider || !model.id || !Array.isArray(catalog.models)) return false;
    return !catalog.models.some((entry) => entry.provider === model.provider && entry.id === model.id);
  } catch {
    return false;
  }
};

/** Swallow the session picker's keyboard events on `/chat/<id>` pages, and the model menu everywhere in drive mode. Returns the stop. */
export function freezeSessionPicker(
  win: FreezeWindow,
  unavailable: ModelUnavailable = sessionModelUnavailable,
  drive: FreezeDrive = shellDrive,
): () => void {
  const released = new WeakSet<Event>();
  let stopped = false;
  const release = async (sessionId: string) => {
    if (!(await unavailable(sessionId)) || stopped || sessionIdOf(win.location.pathname) !== sessionId) return;
    if (drive.active()) {
      drive.openModel(sessionId);
      return;
    }
    const event = new CustomEvent(OPEN_MODEL_PICKER);
    released.add(event);
    win.dispatchEvent(event);
  };
  const swallow = (event: Event) => {
    if (released.has(event)) return;
    const sessionId = sessionIdOf(win.location.pathname);
    const openModel = event.type === OPEN_MODEL_PICKER;
    if (sessionId === null) {
      if (!openModel || !drive.active()) return;
      event.stopImmediatePropagation();
      drive.openModel(null);
      return;
    }
    event.stopImmediatePropagation();
    if (openModel) void release(sessionId);
  };
  for (const type of SESSION_PICKER_EVENTS) win.addEventListener(type, swallow, { capture: true });
  return () => {
    stopped = true;
    for (const type of SESSION_PICKER_EVENTS) win.removeEventListener(type, swallow, { capture: true });
  };
}
