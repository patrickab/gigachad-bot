# Zen quick fix

## Recover a new tab without changing Zen settings

When a new Zen tab cannot load Gigachad:

1. In that tab, open:

   ```text
   https://gigachad-backend.tail8cc40f.ts.net/api/config
   ```

2. Wait until it displays JSON.
3. In the same tab, open:

   ```text
   https://gigachad-bot-two.vercel.app
   ```

The API visit establishes the private Tailnet connection before the Vercel app
starts its bootstrap requests. This requires no Zen profile, extension, script,
or `about:config` change.

## If the API link fails

The backend or Tailscale Serve path is unavailable. From a Tailnet machine,
check:

```bash
curl --fail --show-error https://gigachad-backend.tail8cc40f.ts.net/healthz
```

Restart the production backend only if that health check fails:

```bash
systemctl --user restart gigachad-bot.service
```
