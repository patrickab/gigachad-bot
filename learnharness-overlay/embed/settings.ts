/**
 * The Providers settings section stays hidden: provider keys live in gigachad-bot and reach LearnHarness through the
 * LiteLLM proxy only. `custom/theme/settings.css` hides its nav item and content, this moves the dialog off it.
 *
 * Upstream's settings dialog opens on Providers by default and some callers deep-link to it (setup flow, dictation
 * error). Its section state is private, so whenever the hidden nav item becomes the current one, the visible
 * `LANDING_SECTION` is clicked instead, before the browser paints.
 */
export const HIDDEN_SECTION_SELECTOR = '[data-testid="settings-nav-keys"]';
export const LANDING_SECTION_SELECTOR = '[data-testid="settings-nav-agent"]';
const HIDDEN_SECTION_CURRENT = `${HIDDEN_SECTION_SELECTOR}[aria-current="true"]`;

/** The parts of `document` the guard queries. */
type SettingsDocument = { querySelector(selector: string): { click(): void } | null };

/** Switch an open settings dialog from the hidden section to the landing section, true when it did. */
export function leaveHiddenSection(doc: SettingsDocument): boolean {
  if (!doc.querySelector(HIDDEN_SECTION_CURRENT)) return false;
  const landing = doc.querySelector(LANDING_SECTION_SELECTOR);
  landing?.click();
  return landing !== null;
}

/** Keep every settings dialog off the hidden section, returns the stop. */
export function guardSettingsDialog(doc: Document = document): () => void {
  const observer = new MutationObserver(() => leaveHiddenSection(doc));
  observer.observe(doc.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["aria-current"] });
  leaveHiddenSection(doc);
  return () => observer.disconnect();
}
