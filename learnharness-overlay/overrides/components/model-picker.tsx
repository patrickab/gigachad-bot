"use client";

import type { ComponentProps } from "react";
import { ModelControl as UpstreamModelControl } from "../../../components/model-picker";
import { useShellDrive } from "../../embed/drive";
import { useModelState } from "../../embed/model";

export * from "../../../components/model-picker";

/**
 * Upstream's model control, read-only inside a learn session (`custom/embed/freeze.ts` has the why and the keyboard half).
 *
 * - With a `chatId` (the session's stats bar) a disabled fieldset turns its pills and menus inert, upstream still renders them
 * - Without one (home, `/chat/new`) it is upstream's control: the pick sets the strong tier, which stamps the next session
 * - While the shell drives the frame its model selector stands in: the control stays mounted (its `onModelChange` feeds
 *   the send guard) inside a hidden, inert `data-cf-model-host`, and `custom/embed/model.tsx` posts what it shows
 */
export function ModelControl(props: ComponentProps<typeof UpstreamModelControl>) {
  const drive = useShellDrive();
  useModelState(props.chatId);
  const control = props.chatId ? (
    <fieldset disabled data-testid="session-model-frozen" style={{ display: "contents" }}>
      <UpstreamModelControl {...props} />
    </fieldset>
  ) : (
    <UpstreamModelControl {...props} />
  );
  return (
    <div data-cf-model-host hidden={drive} inert={drive} style={{ display: drive ? "none" : "contents" }}>
      {control}
    </div>
  );
}
