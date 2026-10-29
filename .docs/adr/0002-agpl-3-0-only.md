# gigachad-bot is licensed AGPL-3.0-only

gigachad-bot is public and was unlicensed. It is licensed `AGPL-3.0-only`, the same license as LearnHarness. The private LearnHarness overlay combines with AGPL code and is served over the network, and later upstream contributions (UI, seams) must flow both ways without friction. A permissive license was the alternative. It was rejected because the overlay would have to live outside the repo or under split licensing.

## Consequences

- Network users must be able to get the source. The public repository, including `learnharness-overlay/`, plus the pinned upstream ref satisfies this.
- Before relicensing, run a dependency license scan. Every dependency must be AGPL-compatible.
