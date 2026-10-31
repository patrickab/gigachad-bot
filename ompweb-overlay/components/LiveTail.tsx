"use client";

import { useCallback, useLayoutEffect, useRef, useSyncExternalStore, type ReactNode } from "react";
import { createPortal } from "react-dom";

/**
 * Upstream renders the running turn's live tail (`ChatLiveTail`: streaming reply, pending tool rows, pending bash)
 * after the committed transcript, below the newest Q&A card. Embedded, it portals into that card instead,
 * so a reply sits in its card from its first token.
 *
 * - Host: the newest Q&A card, it offers a slot in its answer region while a tail is mounted
 * - Tail: the live tail, rendered nowhere until the slot exists, inline when no card hosts it (e.g. a branch summary turn)
 * - Everything settles in layout effects, before paint, so the tail never flashes below the card
 */
const listeners = new Set<() => void>();
let hostCount = 0;
let tailCount = 0;
let slot: HTMLElement | null = null;

function notify() {
  listeners.forEach((listener) => listener());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

const hasHost = () => hostCount > 0;
const hasTail = () => tailCount > 0;
const currentSlot = () => slot;
const never = () => false;
const noSlot = () => null;

/** Count a mounted host or tail for as long as `active`. */
function useRegistration(active: boolean, change: (delta: number) => void) {
  useLayoutEffect(() => {
    if (!active) return;
    change(1);
    notify();
    return () => {
      change(-1);
      notify();
    };
  }, [active, change]);
}

const changeHosts = (delta: number) => { hostCount += delta; };
const changeTails = (delta: number) => { tailCount += delta; };

/**
 * For the newest Q&A card: whether a tail wants in, and the ref for the slot it renders into.
 *
 * Render the slot after the card's answer nodes, in the answer region, while `tailActive` or the turn has an answer.
 */
export function useLiveTailHost(newest: boolean): { tailActive: boolean; slotRef: (element: HTMLElement | null) => void } {
  useRegistration(newest, changeHosts);
  const tailActive = useSyncExternalStore(subscribe, hasTail, never) && newest;
  const ownedRef = useRef<HTMLElement | null>(null);
  const slotRef = useCallback((element: HTMLElement | null) => {
    if (element) slot = element;
    else if (slot === ownedRef.current) slot = null;
    ownedRef.current = element;
    notify();
  }, []);
  return { tailActive, slotRef };
}

/** The live tail, inside the newest Q&A card's slot. */
export function LiveTail({ children }: { children: ReactNode }) {
  useRegistration(true, changeTails);
  const host = useSyncExternalStore(subscribe, hasHost, never);
  const target = useSyncExternalStore(subscribe, currentSlot, noSlot);
  if (target) return createPortal(children, target);
  return host ? null : children;
}
