import "./theme";
// `@/custom/...` like the rest of the overlay's cross-directory imports. The override aliases never capture these
// (`next.config.ts` `overrideAliases`: no suffixed sibling keys).
import { EmbedBridge } from "@/custom/embed/bridge";
import { ChromeBridge } from "@/custom/embed/chrome";
import { EmbedMode } from "@/custom/embed/mode";
import { ModelBridge } from "@/custom/embed/model";
import { SessionPickerFreeze } from "@/custom/embed/session-picker";

/** Everything the overlay mounts once at the app root, first in <body> of app/layout.tsx (seams.patch). */
export function CustomRoot() {
  return (
    <>
      <EmbedMode />
      <EmbedBridge />
      <ModelBridge />
      <ChromeBridge />
      <SessionPickerFreeze />
    </>
  );
}
