"use client";

import { useEffect } from "react";
import { freezeSessionPicker } from "./freeze";

/** Keeps the keyboard off an open session's model and thinking pills (`./freeze`), embedded or not. */
export function SessionPickerFreeze() {
  useEffect(() => freezeSessionPicker(window), []);
  return null;
}
