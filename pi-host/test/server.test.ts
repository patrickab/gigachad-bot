import { expect, test } from "bun:test";
import { watchEventLoop } from "../src/server.ts";

test("the event-loop watch logs a tick that came late by more than a second, and nothing for ticks on time", async () => {
  const logged: string[] = [];
  // A clock that runs with real time, plus a jump that plays a synchronous stall without blocking the test process.
  let stalled = 0;
  const stop = watchEventLoop((message) => logged.push(message), () => performance.now() + stalled);
  try {
    await Bun.sleep(700);
    expect(logged).toEqual([]);

    stalled = 1_800;
    await Bun.sleep(600);
    expect(logged).toHaveLength(1);
    expect(logged[0]).toMatch(/^pi-host: event loop blocked for 1\.\d s$/);
  } finally {
    stop();
  }
});
