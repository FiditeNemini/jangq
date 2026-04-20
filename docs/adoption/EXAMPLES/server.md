# Serving a JANG model as an OpenAI-compatible HTTP server

JANG models can be served via [Osaurus](https://github.com/dinoki-ai/osaurus) — an
OpenAI-compatible HTTP server with native support for JANG and JANGTQ.

## Quick start

```bash
pip install osaurus
osaurus serve --model /path/to/your-JANG-model --port 8080
```

Osaurus auto-detects the JANG format by reading `jang_config.json` and dispatches
to the correct loader (`load_jang_model` or `load_jangtq_model`).

## Call it like OpenAI

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jang",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello"}
    ],
    "stream": true
  }'
```

## Tool calling

If the source model has a `tool_call_parser` set in its `config.json`, JANG preserves
it through conversion. Osaurus honors it automatically:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jang",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }]
  }'
```

## Reasoning / thinking

For models with `reasoning_parser` or `enable_thinking` set in `config.json`, the
reasoning trace will be returned in the `reasoning` field alongside the final answer:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Paris.",
      "reasoning": "The question asks for the capital of France..."
    }
  }]
}
```

## Vision/Language and video

VL models are loaded via `mlx_vlm` automatically when `preprocessor_config.json` is
present. Pass images as base64 in standard OpenAI format:

```json
{
  "model": "jang",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe this image"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]
  }]
}
```

## Health check and model list

```bash
curl http://localhost:8080/health
# {"status": "ok"}

curl http://localhost:8080/v1/models
# {"data": [{"id": "jang", "object": "model", ...}]}
```

## Running as a macOS LaunchAgent

Create `~/Library/LaunchAgents/ai.jangq.osaurus.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>       <string>ai.jangq.osaurus</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/osaurus</string>
        <string>serve</string>
        <string>--model</string>
        <string>/path/to/your-JANG-model</string>
        <string>--port</string>
        <string>8080</string>
    </array>
    <key>RunAtLoad</key>   <true/>
    <key>KeepAlive</key>   <true/>
</dict>
</plist>
```

Then load it:

```bash
launchctl load ~/Library/LaunchAgents/ai.jangq.osaurus.plist
```

## Alternative runtimes

At time of writing, LM Studio, Ollama, and Inferencer do not yet support JANG.
If you are integrating JANG into one of those runtimes, see
[PORTING.md](../PORTING.md) for the format spec and open an issue on their repo
linking to it.
