"use client";

import { useSyncExternalStore } from "react";
import { isEmbedded } from "./EmbedMode";

const subscribeNever = () => () => {};

/** `isEmbedded()` for render: false on the server and during hydration, the real value right after, so markup never mismatches. */
export function useIsEmbedded(): boolean {
  return useSyncExternalStore(subscribeNever, isEmbedded, () => false);
}
