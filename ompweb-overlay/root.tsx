import "./theme";
import "./components/EmbedMode";
import { EmbedBridge } from "./components/EmbedBridge";

/** Everything the overlay mounts once at the app root, rendered by app/client.tsx (seams.patch). */
export function CustomRoot() {
  return <EmbedBridge />;
}
