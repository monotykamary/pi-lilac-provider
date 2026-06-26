<div align="center">

# 💜 pi-lilac-provider

**Kimi K2.6, GLM 5.1, Gemma 4 & more on idle GPUs via [Lilac](https://getlilac.com/)**

_A [pi](https://github.com/earendil-works/pi-coding-agent) provider extension for cost-efficient GPU inference._

[![pi extension](https://img.shields.io/badge/pi-extension-blueviolet)](https://github.com/earendil-works/pi-coding-agent)
[![license](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)

</div>

---

Access Kimi K2.6, GLM 5.1, MiniMax M2.7, and Gemma 4 models through Lilac's OpenAI-compatible API on idle GPUs.

## Features

- **3 AI Models** — Kimi K2.6, GLM 5.1, and Gemma 4
- **OpenAI-Compatible API** — Just change the base URL and API key
- **Cost Tracking** — Per-model pricing with cache read discounts
- **Reasoning Models** — Chain-of-thought via `chat_template_kwargs` (all models)
- **Vision Support** — Image input on Kimi K2.6 and Gemma 4
- **Context Caching** — Cache read pricing on Kimi K2.6 and GLM 5.1
- **Idle GPU Scheduling** — Lilac leverages idle GPU capacity for cost-efficient inference

## Installation

### Option 1: Using `pi install` (Recommended)

Install directly from GitHub:

```bash
pi install https://github.com/monotykamary/pi-lilac-provider
```

Then set your API key and run pi:
```bash
# Recommended: add to auth.json
# See Authentication section below

# Or set as environment variable
export LILAC_API_KEY=your-api-key-here

pi
```

### Option 2: Manual Clone

1. Clone this repository:
   ```bash
   git clone https://github.com/monotykamary/pi-lilac-provider.git
   cd pi-lilac-provider
   ```

2. Set your Lilac API key:
   ```bash
   # Recommended: add to auth.json
   # See Authentication section below

   # Or set as environment variable
   export LILAC_API_KEY=your-api-key-here
   ```

3. Run pi with the extension:
   ```bash
   pi -e /path/to/pi-lilac-provider
   ```

## Available Models

| Model | Context | Vision | Reasoning | Input $/M | Cache Read $/M | Output $/M |
|-------|---------|--------|-----------|-----------|-----------------|------------|
| Gemma 4 | 262K | ✅ | ✅ | $0.11 | — | $0.35 |
| GLM 5.1 | 203K | ❌ | ✅ | $0.90 | $0.27 | $3.00 |
| GLM 5.2 | 524K | ❌ | ✅ | $0.90 | $0.27 | $3.00 |
| Kimi K2.6 | 262K | ✅ | ✅ | $0.70 | $0.20 | $3.50 |
| MiniMax M2.7 | 205K | ❌ | ✅ | $0.30 | $0.06 | $1.20 |
| MiniMax M3 | 1.0M | ✅ | ✅ | $0.28 | $0.05 | $1.10 |

*Costs are per million tokens. Prices subject to change — check [getlilac.com](https://getlilac.com/) for current pricing.*

**Notes:**
- **Gemma 4** has reasoning **off by default** — pi enables it when you set a thinking level (Shift+Tab)
- **Kimi K2.6** and **GLM 5.1** have reasoning **on by default**
- **Cache read** pricing applies to repeated input tokens served from cache on supported models
- **Gemma 4** does not support cache read pricing

## Usage

After loading the extension, use the `/model` command in pi to select your preferred model:

```
/model lilac moonshotai/kimi-k2.6
```

Or start pi directly with a Lilac model:

```bash
pi --provider lilac --model moonshotai/kimi-k2.6
```

### Thinking Mode

All Lilac models toggle reasoning via `chat_template_kwargs`, but the key each
model's chat template honors differs per family. The provider uses pi's
`chat-template` thinkingFormat with per-model `chatTemplateKwargs` (configured in
`patch.json`) so the right key reaches each template:

| Model | Reasoning key | Default |
|-------|---------------|---------|
| Kimi K2.6 | `thinking` (bool) | on |
| GLM 5.1 | `enable_thinking` (bool) | on |
| GLM 5.2 | `enable_thinking` (bool) + `reasoning_effort` (`max`\|`high`) | on (`max`) |
| Gemma 4 | `enable_thinking` (bool) | off |
| MiniMax M2.7 | `thinking` + `enable_thinking` (bool) | on |
| MiniMax M3 | `thinking_mode` (`disabled`\|`adaptive`\|`enabled`) | adaptive (server) |

Kimi K2.6, GLM 5.1, Gemma 4, and MiniMax M2.7 use the forward-compatible form
that sends **both** `thinking` and `enable_thinking`, so whichever key the
template honors is set. GLM 5.2 additionally maps pi's thinking levels to
`reasoning_effort` (`high` = lower-latency, `xhigh` = `max`). MiniMax M3 uses
the `thinking_mode` enum, exposed as three pi thinking levels: `off` →
`disabled` (never think), `minimal` → `adaptive` (the model decides), `high` →
`enabled` (always think). Pi starts at `off` (`disabled`); cycle to `minimal`
for M3's adaptive "model decides" mode. (The selector/footer show pi's level
names — `minimal`/`high` — not the `thinking_mode` values; pi has no per-model
level-relabel hook.)

**Preserved thinking (full-history reasoning).** By default these templates
trim older assistant reasoning between turns (each vendor's default), which
degrades multi-turn recall. Three models opt into full-history preservation via
a template flag sent alongside the reasoning key:

| Model | Flag | Effect |
|-------|------|--------|
| Kimi K2.6 | `preserve_thinking: true` | keeps every assistant turn's reasoning (default: only the last) |
| GLM 5.1 | `clear_thinking: false` | keeps reasoning for all turns (default: clears before the last user message) |
| GLM 5.2 | `clear_thinking: false` | keeps reasoning for all turns (default: clears before the last user message) |

Kimi K2.6 and GLM 5.2 are E2E-verified on the sibling neuralwatt provider via a
3-turn, two-20-digit-number recall test (Kimi 0/6 → 6/6, GLM 5.2 1/4 → 4/4);
GLM 5.1 uses the same `clear_thinking` mechanism (confirmed in its HuggingFace
chat template). Gemma 4 and MiniMax M2.7/M3 expose no family-wide preserve flag,
so their older assistant reasoning is trimmed per the template default.

In pi, reasoning models automatically use the appropriate thinking format. Use
Shift+Tab to control thinking level.

### Vision

Kimi K2.6 and Gemma 4 support image inputs. Pass images in messages and pi will handle the formatting automatically.

Gemma 4 also supports video by accepting a sequence of frames as images.

## Authentication

The Lilac API key can be configured in multiple ways (resolved in this order):

1. **`auth.json`** (recommended) — Add to `~/.pi/agent/auth.json`:
   ```json
   { "lilac": { "type": "api_key", "key": "your-api-key" } }
   ```
   The `key` field supports literal values, env var names, and shell commands (prefix with `!`). See [pi's auth file docs](https://github.com/badlogic/pi-mono) for details.
2. **Runtime override** — Use the `--api-key` CLI flag
3. **Environment variable** — Set `LILAC_API_KEY`

Get your API key at [getlilac.com](https://getlilac.com/).

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LILAC_API_KEY` | No | Your Lilac API key (fallback if not in auth.json) |

## Configuration

Add to your pi configuration for automatic loading:

```json
{
  "extensions": [
    "/path/to/pi-lilac-provider"
  ]
}
```

### Compat Settings

Lilac's API is OpenAI-compatible with these specifics:

- **`thinkingFormat: "chat-template"`** — All reasoning models. Lilac's vLLM backend toggles reasoning via `chat_template_kwargs`, but the honored key differs per model family. Per-model `chatTemplateKwargs` in `patch.json` send the right key(s): `thinking`+`enable_thinking` (bool) for Kimi K2.6, GLM 5.1, Gemma 4, and MiniMax M2.7; `enable_thinking` + `reasoning_effort` for GLM 5.2; `thinking_mode` (adaptive|enabled|disabled) for MiniMax M3. Kimi K2.6, GLM 5.1, and GLM 5.2 additionally send a preservation flag (`preserve_thinking: true` / `clear_thinking: false`) to retain full reasoning history across turns — see [Preserved thinking](#thinking-mode) above. Override these per-model via [Model Overrides](#model-overrides).
- **`maxTokensField: "max_completion_tokens"`** — All models. Lilac supports `max_completion_tokens` (preferred for reasoning models as it includes reasoning tokens).
- **`supportsDeveloperRole: true`** — All models. Lilac's vLLM backend maps the developer role to system.
- **`supportsStore: false`** — All models. Lilac doesn't support the `store` parameter.

### Known Caveats

- **GLM 5.1 intermittent tool call loss**: vLLM's streaming parser intermittently emits `finish_reason: "tool_calls"` without any `delta.tool_calls` chunks — even with `tool_stream: true` (set via `zaiToolStream` in compat). Pi maps this to `stopReason: "toolUse"` with zero toolCall blocks, causing an "abrupt stop". The extension's `message_end` handler converts this to a retryable error that triggers pi's built-in auto-retry mechanism, so the agent automatically re-prompts and typically succeeds on the next attempt.
- **GLM 5.1 chain-of-thought leakage**: On the current vLLM build, disabling reasoning on GLM 5.1 may still leak chain-of-thought into `content` terminated by a `</think>` marker. Post-process the response to discard text up to and including the first `</think>` when reasoning is disabled. See [vllm-project/vllm#31319](https://github.com/vllm-project/vllm/issues/31319).
- **Gemma 4 reasoning parser**: vLLM's reasoning parser can fail to populate the `reasoning` field when special tokens are stripped before the parser runs. Clients that require a clean split should post-process `<|channel|>thought ... <|channel|>` markers. See [vllm-project/vllm#38855](https://github.com/vllm-project/vllm/issues/38855).
- **Gemma 4 structured output**: Combining `enable_thinking: false` with `response_format: json_schema` can silently disable xgrammar-backed structured output. If you rely on structured output with Gemma 4, leave thinking enabled or validate output client-side. See [vllm-project/vllm#39130](https://github.com/vllm-project/vllm/issues/39130).

### Patch Overrides

The `patch.json` file contains overrides that are applied on top of `models.json` data. This is useful for:
- Correcting API-derived values (e.g., GLM 5.1's `maxTokens` — API returns context length, actual max output is 131K)
- Marking models as reasoning-capable when the API features list doesn't include it
- Adding compat settings that the API doesn't provide
- Overriding pricing when official rates change

### Model Overrides

`modelOverrides` lets you override compat flags and other model properties per model id, **on top of** `patch.json` + `custom-models.json`, without editing the extension. Keyed by model id; `compat` (including nested `chatTemplateKwargs`), `thinkingLevelMap`, and `cost` are deep-merged **recursively** (toggle one flag without redeclaring the rest), scalars and arrays are replaced. Applied at session start, so edits take effect on the next `pi` session.

Create `~/.pi/agent/extensions/lilac.json` (auto-populated with defaults on first run):

```jsonc
{
  "modelOverrides": {
    // Disable full-history reasoning for kimi-k2.6 (e.g. to save tokens):
    "moonshotai/kimi-k2.6": { "compat": { "chatTemplateKwargs": { "preserve_thinking": false } } },
    // Toggle the GLM 5.2 clear_thinking flag without redeclaring the rest of compat:
    "zai-org/glm-5.2": { "compat": { "chatTemplateKwargs": { "clear_thinking": true } } },
    // Override a single thinking level without redeclaring the whole map:
    "zai-org/glm-5.1": { "thinkingLevelMap": { "high": "max" } }
  }
}
```

The full set of overridable fields matches the model schema (`compat`, `thinkingLevelMap`, `cost`, `contextWindow`, `maxTokens`, `reasoning`, `input`). See [Compat Settings](#compat-settings) for the catalog of compat flags and what `chatTemplateKwargs` values mean per family. An invalid JSON file is left untouched (defaults are used) so a typo isn't silently wiped — fix the file and restart pi.

## Updating Models

Run the update script to fetch the latest models from Lilac's API:

```bash
export LILAC_API_KEY=your-api-key
node scripts/update-models.js
```

This will:
1. Fetch models from `https://api.getlilac.com/v1/models`
2. Convert per-token pricing to per-million-tokens
3. Preserve existing curated data (pricing, compat) for known models
4. Apply overrides from `patch.json`
5. Update `models.json` and the README model table

A GitHub Actions workflow runs this daily and creates a PR if models have changed.

## License

MIT
