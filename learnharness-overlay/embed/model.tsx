"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ThinkingLevel } from "@earendil-works/pi-agent-core";
import type { LearnFrameMessageOf, LearnModelRef, LearnShellMessageOf } from "@gigachad/chat-ui/learn-protocol";
import type { ChatSettings, ModelCatalogResponse, ModelsSettings } from "@/src/core";
import { useModelCatalog } from "../../components/model-catalog";
import { composeSlotModel } from "../../components/settings/agent-section";
import { notifySettingsSaved, useSettingsSaved } from "../../components/settings-signal";
import { selectedModelUnavailable } from "../../components/setup-cta";
import { defaultOrderLevel, populateLevel } from "../../components/thinking-default";
import { onShellMessage, postToFrameShell } from "./bridge";
import { onModelOpen, sessionModelUnavailable, type ModelUnavailable } from "./freeze";
import { isEmbedded } from "./mode";

/**
 * The shell model selector's side in the frame. Upstream's `ModelControl` stays mounted, hidden in drive mode
 * (`overrides/components/model-picker.tsx`), because its `onModelChange` feeds the send guard. This module repeats its
 * reads and its two writes (`components/model-picker.tsx` `readRecord`, `save`, `saveChat`, `pickModel`) for the shell:
 *
 * - `model`: what the control shows, posted by every mounted control (`useModelState`)
 * - `models`: what `/api/models` serves, the only models the shell offers (`ModelBridge`)
 * - `model-open`: upstream asked for its model menu, the freeze (`./freeze`) hands it here in drive mode
 * - `set-model`, `set-thinking`: a draft writes the strong tier, a session stays read-only unless its
 *   model is no longer served. Every successful write ends in `notifySettingsSaved("models")`, so the hidden control,
 *   the send guard's model and the `model` post re-read. A dropped or failed pick reposts the last `model`, the shell's
 *   selector falls back to the truth.
 */

type ModelMessage = LearnFrameMessageOf<"model">;
type SessionRecord = { settings?: ChatSettings };
type SettingsBody = { settings: { models?: ModelsSettings } | null };
/** What a pick did: written to LearnHarness, dropped by the rules above, or refused/unreachable. */
export type PickResult = "written" | "ignored" | "failed";

/**
 * The control's effective identity, exactly as upstream's `ModelControl` derives it: a session shows its own record,
 * a legacy unstamped one the strong tier; a draft (`sessionId` null) is the strong tier. Null until the session record
 * is read. `unavailable` is upstream's send guard test (`selectedModelUnavailable`), `locked` every session that is not.
 */
export function modelState(
  sessionId: string | null,
  models: ModelsSettings | undefined,
  session: SessionRecord | undefined,
  catalog: ModelCatalogResponse | undefined,
): ModelMessage | null {
  if (sessionId !== null && session === undefined) return null;
  const strong = models?.strong;
  const settings: ChatSettings | undefined =
    sessionId === null
      ? strong
      : { model: session?.settings?.model ?? strong?.model, thinkingLevel: session?.settings?.thinkingLevel ?? strong?.thinkingLevel };
  const model = settings?.model;
  const entry = model && catalog?.models.find((candidate) => candidate.provider === model.provider && candidate.id === model.id);
  const levels = entry?.supportedLevels ?? [];
  const level = settings?.thinkingLevel ?? (entry ? defaultOrderLevel(levels) : undefined);
  const unavailable = selectedModelUnavailable(catalog, model ? `${model.provider}:${model.id}` : undefined);
  return {
    type: "model",
    sessionId,
    model: model ? { provider: model.provider, modelId: model.id } : null,
    thinkingLevel: level ?? null,
    thinkingLevels: [...levels],
    locked: sessionId !== null && !unavailable,
    unavailable,
  };
}

/** The served catalog as the shell's model refs. */
export function servedModels(catalog: Pick<ModelCatalogResponse, "models">): LearnModelRef[] {
  return catalog.models.map((entry) => ({ provider: entry.provider, modelId: entry.id }));
}

let lastModel: ModelMessage | null = null;

function postModel(message: ModelMessage): void {
  lastModel = message;
  postToFrameShell(message);
}

async function readJson<T>(path: string): Promise<T> {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return (await response.json()) as T;
}

async function put(path: string, body: unknown): Promise<PickResult> {
  const response = await fetch(path, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!response.ok) return "failed";
  notifySettingsSaved("models");
  return "written";
}

/** Upstream's draft write (`save`): the strong slot through `composeSlotModel`, cheap and vision preserved. */
const writeStrong = (models: ModelsSettings) => put("/api/settings/models", models);

/** Upstream's session write (`saveChat`). Its signal-free write would leave the hidden control and the send guard stale. */
const writeSession = (sessionId: string, settings: ChatSettings) => put(`/api/chats/${encodeURIComponent(sessionId)}/settings`, settings);

async function readModels(): Promise<ModelsSettings | undefined> {
  return (await readJson<SettingsBody>("/api/settings")).settings?.models ?? undefined;
}

/**
 * `set-model` as upstream's `pickModel`: a served model only, its level populated from the stored one
 * (`populateLevel`). A draft writes the strong tier; a session only while *unavailable* confirms its model is gone.
 */
export async function applyModelPick(
  message: LearnShellMessageOf<"set-model">,
  unavailable: ModelUnavailable = sessionModelUnavailable,
): Promise<PickResult> {
  try {
    const { sessionId } = message;
    if (sessionId !== null && !(await unavailable(sessionId))) return "ignored";
    const catalog = await readJson<ModelCatalogResponse>("/api/models");
    const entry = catalog.models.find((candidate) => candidate.provider === message.provider && candidate.id === message.modelId);
    if (!entry) return "ignored";
    const models = await readModels();
    if (sessionId === null) {
      const level = populateLevel(models?.strong?.thinkingLevel, entry.supportedLevels);
      return await writeStrong(composeSlotModel(models, "strong", entry.provider, entry.id, level));
    }
    const session = await readJson<SessionRecord>(`/api/chats/${encodeURIComponent(sessionId)}`);
    const stored = session.settings?.thinkingLevel ?? models?.strong?.thinkingLevel;
    return await writeSession(sessionId, {
      model: { provider: entry.provider, id: entry.id },
      thinkingLevel: populateLevel(stored, entry.supportedLevels),
    });
  } catch {
    return "failed";
  }
}

/**
 * `set-thinking` as upstream's thinking menu: a draft sets the strong tier's level, one its model supports.
 * A session's level is read-only: a served model is locked, an unserved one has no levels to pick from.
 */
export async function applyThinkingPick(message: LearnShellMessageOf<"set-thinking">): Promise<PickResult> {
  if (message.sessionId !== null) return "ignored";
  try {
    const [catalog, models] = await Promise.all([readJson<ModelCatalogResponse>("/api/models"), readModels()]);
    const model = models?.strong?.model;
    const entry = model && catalog.models.find((candidate) => candidate.provider === model.provider && candidate.id === model.id);
    if (!entry || !entry.supportedLevels.includes(message.level as ThinkingLevel)) return "ignored";
    return await writeStrong(composeSlotModel(models, "strong", entry.provider, entry.id, message.level));
  } catch {
    return "failed";
  }
}

async function settle(pick: Promise<PickResult>): Promise<void> {
  if ((await pick) !== "written" && lastModel) postToFrameShell(lastModel);
}

/**
 * Post what the model control with *chatId* shows as `model`, on every change. Called by the `ModelControl` override,
 * so every surface that mounts the control reports it. Reads like upstream's `readRecord`, on mount and on the models signal.
 */
export function useModelState(chatId: string | undefined): void {
  const sessionId = chatId ?? null;
  const { catalog } = useModelCatalog();
  const [models, setModels] = useState<{ value: ModelsSettings | undefined }>();
  const [session, setSession] = useState<{ id: string; record: SessionRecord }>();
  const current = useRef(sessionId);
  current.current = sessionId;

  const read = useCallback(() => {
    if (sessionId !== null) {
      readJson<SessionRecord>(`/api/chats/${encodeURIComponent(sessionId)}`)
        .then((record) => {
          if (current.current === sessionId) setSession({ id: sessionId, record: { settings: record.settings } });
        })
        .catch(() => {});
    }
    readModels()
      .then((value) => setModels({ value }))
      .catch(() => {});
  }, [sessionId]);

  useEffect(() => {
    if (isEmbedded()) read();
  }, [read]);
  useSettingsSaved((section) => {
    if (section === "models" && isEmbedded()) read();
  });

  const record = sessionId !== null && session?.id === sessionId ? session.record : undefined;
  const message = models ? modelState(sessionId, models.value, record, catalog) : null;
  const posted = useRef<string | null>(null);
  useEffect(() => {
    const key = message ? JSON.stringify(message) : null;
    if (!message || key === posted.current) return;
    posted.current = key;
    postModel(message);
  });
}

/** Posts the served catalog, repeated whenever it is re-read (`useModelCatalog` follows the models and keys signals). */
function ServedModels() {
  const { catalog } = useModelCatalog();
  useEffect(() => {
    if (catalog) postToFrameShell({ type: "models", served: servedModels(catalog) });
  }, [catalog]);
  return null;
}

/** Mounted once by `root.tsx`: the shell's picks, upstream's model menu requests in drive mode, the served catalog while embedded. */
export function ModelBridge() {
  const [embedded, setEmbedded] = useState(false);

  useEffect(() => {
    setEmbedded(isEmbedded());
    const stops = [
      onShellMessage("set-model", (message) => void settle(applyModelPick(message))),
      onShellMessage("set-thinking", (message) => void settle(applyThinkingPick(message))),
      onModelOpen((sessionId) => postToFrameShell({ type: "model-open", sessionId })),
    ];
    return () => stops.forEach((stop) => stop());
  }, []);

  return embedded ? <ServedModels /> : null;
}
