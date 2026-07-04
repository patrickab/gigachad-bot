## Never start the dev server autonomously

The backend (`uvicorn`, port 8001) and frontend (`next dev`, port 2999) are
started via `run.sh` or manually by the user, in a terminal they control.

- Never run `run.sh`, `uvicorn`, `npm run dev`, or equivalent yourself —
  even in the background, even to verify a change.
- If you need to see the app running (UI check, manual test, screenshot),
  ask the user to start it and confirm it's up, then use their already-running
  instance. Do not launch your own copy alongside theirs.
