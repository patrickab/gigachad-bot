import "./theme";
import { EmbedBridge } from "./components/EmbedBridge";
import { EmbedMode } from "./components/EmbedMode";

/** Everything the overlay mounts once at the app root, rendered by app/layout.tsx (seams.patch). */
export function CustomRoot() {
  return (
    <>
      <EmbedMode />
      <EmbedBridge />
    </>
  );
}
