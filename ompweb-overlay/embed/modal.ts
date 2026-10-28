import { postToShell } from "./channel";

/** Base UI's portaled modal backdrop and dialog while open (upstream `components/ui/primitives.tsx` `DialogContent`). */
const OPEN_BACKDROP = "[data-base-ui-portal] > [role=presentation][data-open]";
const OPEN_DIALOG = "[data-base-ui-portal] > :is([role=dialog], [role=alertdialog])[data-open]";

/** Report whether a modal dialog is open (`modal` message), so the shell dims its chrome alike. Returns the stop. */
export function trackModal(): () => void {
  let open: boolean | null = null;
  const report = () => {
    const next = document.querySelector(OPEN_BACKDROP) !== null;
    if (next === open) return;
    open = next;
    postToShell({ type: "modal", open });
  };
  const observer = new MutationObserver(report);
  observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["data-open"] });
  report();
  return () => observer.disconnect();
}

/** Close the open modal dialog the way Escape inside it does (`close-modal` message). */
export function closeModal() {
  document.querySelector(OPEN_DIALOG)?.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true, cancelable: true }));
}
