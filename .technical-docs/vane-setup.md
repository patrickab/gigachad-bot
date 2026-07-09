### Routing web search to DeepSeek

Cloud providers other than Gemini are **not** auto-registered in Vane. Without a
DeepSeek provider, a `deepseek/*` web search falls through to Ollama's
subscription-gated `deepseek-*:cloud` copy and fails with a `403`. DeepSeek also
rejects the `json_schema` response format Vane's planner sends, so requests go
through the `deepseek-shim` service (in `docker-compose.vane.yml`, auto-started
with the stack), which rewrites it to DeepSeek's `json_object` mode. Register the
provider pointing at that shim. Run once — it persists in the `vane_data` volume:

```bash
# 1. Create the provider (type "openai", baseURL = the shim, not the API directly).
#    Inline chatModels are ignored here; models are added in step 2.
PID=$(curl -s -X POST http://localhost:3001/api/providers \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "DeepSeek",
    "type": "openai",
    "config": {"apiKey": "'"$DEEPSEEK_API_KEY"'", "baseURL": "http://deepseek-shim:8000/v1"}
  }' | python3 -c "import sys,json; print(json.load(sys.stdin)['provider']['id'])")

# 2. Add the chat models (one POST each).
for K in deepseek-v4-pro deepseek-v4-flash; do
  curl -s -X POST "http://localhost:3001/api/providers/$PID/models" \
    -H 'Content-Type: application/json' \
    -d "{\"type\":\"chat\",\"key\":\"$K\",\"name\":\"$K\"}"
done
```

**The name and keys are significant**: the backend maps the `deepseek/` model
prefix to a Vane provider named exactly `DeepSeek` and matches the model key
exactly (`deepseek/deepseek-v4-pro` → key `deepseek-v4-pro`). You can also do
this in the Vane UI at `http://localhost:3001` (Settings → Providers, type
"OpenAI", base URL `http://deepseek-shim:8000/v1`) — keep the same name and keys.

Verify the models are attached, then restart the backend (clears the resolution
cache):

```bash
curl -s http://localhost:3001/api/providers \
  | python3 -c "import sys,json; print([m['key'] for p in json.load(sys.stdin)['providers'] if p['name']=='DeepSeek' for m in p['chatModels']])"
# -> ['deepseek-v4-pro', 'deepseek-v4-flash']
```
