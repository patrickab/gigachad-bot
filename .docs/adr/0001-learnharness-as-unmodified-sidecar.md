# LearnHarness runs as an unmodified sidecar, not a fork or a port

gigachad-bot gets learn mode (learn sessions, practice sets, Knowledge) by serving unmodified upstream LearnHarness (`amosblomqvist/learnharness`) at a pinned ref as a loopback sidecar, shown in one frame. It works like omp tabs: a launcher, a private overlay, and a seam patch of about four lines. The goal is to receive every upstream change without adapting gigachad. That rules out a fork and a native port, both of which turn upstream progress into our maintenance.

## Considered options

- **Fork or vendor LearnHarness into gigachad**: rejected. Upstream redesigns weekly under an explicit no-backward-compatibility policy, so a fork diverges within days.
- **Native port to Python and Postgres**: rejected. It is about 27k lines of engine, and we would lose upstream progress entirely.
- **Compile LearnHarness components into gigachad's Next app**: rejected. Upstream churn would break gigachad's own build.
- **Rebuild LearnHarness's UI in gigachad against its HTTP API**: rejected. Upstream UI changes would never reach us.

## Consequences

- **Restyle only the shell.**
  - The overlay styles LearnHarness only through its theme tokens and global CSS, scoped to `html.cf-embed`.
  - It overrides a LearnHarness component only where function requires it. Today that is only freezing the model selector inside a learn session.
  - LearnHarness keeps its own sidebar, composer and question dock, so upstream UI changes leak through.
- **Seam policy.**
  - Whatever gigachad can do with app-internal seams, without hacking that risks conflicts, stays in gigachad.
  - A seam is proposed upstream only when it is the only maintainable path.
- **Models.**
  - LearnHarness keeps its three model tiers: strong (the conversation, picked per learn session and frozen once started), cheap (background jobs) and vision. They all come from gigachad's models through LearnHarness's custom OpenAI-compatible endpoint, served by a LiteLLM proxy sidecar.
  - The OMP gateway is the fallback.
  - gigachad generates the proxy config and LearnHarness's model catalog from its own catalog.
- **Knowledge is LearnHarness-owned.**
  - Knowledge is global, stored in LearnHarness's files under `~/.local/share/gigachad-bot/learnharness/`, outside Postgres, sync and per-user scoping.
  - Chat sessions do not read it, and gigachad does not feed it.
- **Host access only.** It serves one learner. LearnHarness has no auth, so it binds to loopback behind a host-gated proxy.
- **One adapter on each side.**
  - Backend: `src/lib/learnharness.py`, with pydantic contracts.
  - Frontend: `features/learn/`.
  - Both own all knowledge of LearnHarness, so an upstream change fails loudly in one place. gigachad-side features use this boundary only: project grouping and the `start_learn_session` chat tool.
