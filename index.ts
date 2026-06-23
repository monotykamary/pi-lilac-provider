/**
 * Lilac Provider Extension
 *
 * Registers Lilac (getlilac.com) as a custom provider using the openai-completions API.
 * Base URL: https://api.getlilac.com/v1
 *
 * Lilac serves models via a customized fork of vLLM tuned for idle-GPU scheduling
 * and shared warm endpoints. All models use chat_template_kwargs to toggle reasoning:
 *
 *   - Kimi K2.6: reasoning ON by default, honors `thinking` key
 *   - GLM 5.1:   reasoning ON by default, honors `enable_thinking` key
 *   - Gemma 4:   reasoning OFF by default, honors `enable_thinking` key
 *
 * The forward-compatible approach is to send both `thinking` and `enable_thinking`
 * in chat_template_kwargs — pi's `qwen-chat-template` thinkingFormat does this.
 *
 * Key API notes:
 *   - Uses `max_completion_tokens` (preferred for reasoning models)
 *   - All reasoning models return chain-of-thought in `reasoning` field
 *   - Developer role is NOT supported by GLM, Kimi, or MiniMax chat templates;
 *     prompts with role: "developer" are silently dropped. Only Gemma 4 handles it.
 *     supportsDeveloperRole is set to false for affected models via patch.json.
 *   - Context caching supported on Kimi K2.6 and GLM 5.1 (cacheRead pricing)
 *   - Gemma 4 does NOT support cache read pricing
 *   - `store` parameter is NOT supported
 *
 * GLM 5.1 caveats:
 *   - vLLM's streaming parser intermittently omits `delta.tool_calls` when the
 *     model decides to call tools, finishing with `finish_reason: "tool_calls"` but
 *     an empty delta. Even with `tool_stream: true` set via `zaiToolStream`, this
 *     can still occur intermittently. The `message_end` handler converts the
 *     resulting `stopReason: "toolUse"` with zero toolCall blocks into a retryable
 *     error (matching pi's auto-retry pattern) so the agent re-prompts automatically.
 *   - GLM's chat template does not handle the `developer` role — prompts sent
 *     with `role: "developer"` are silently dropped. `supportsDeveloperRole: false`
 *     in models.json forces pi to use `role: "system"` instead.
 *   - On current vLLM builds, disabling reasoning may still leak chain-of-thought
 *     into `content` terminated by a ``` marker. Clients that require
 *     hard-suppressed output should post-process accordingly.
 *     See: https://github.com/vllm-project/vllm/issues/31319
 *
 * Kimi K2.6 / MiniMax M2.7 caveat: Their chat templates also do not handle the
 * `developer` role — prompts are silently dropped. `supportsDeveloperRole: false`
 * is set for these models as well.
 *
 * Gemma 4 caveat: vLLM's reasoning parser can fail to populate the `reasoning`
 * field when special tokens are stripped. Combining `enable_thinking: false`
 * with `response_format: json_schema` can silently disable structured output.
 * See: https://github.com/vllm-project/vllm/issues/38855
 * See: https://github.com/vllm-project/vllm/issues/39130
 *
 * Model resolution strategy: Stale-While-Revalidate
 *   1. Serve stale immediately: disk cache → embedded models.json (zero-latency)
 *   2. Revalidate in background: live API /models → merge with embedded → cache → hot-swap
 *   3. patch.json + custom-models.json applied on top of whichever source won
 *
 * Merge order: [live|cache|embedded] → apply patch.json → merge custom-models.json
 *
 * Usage:
 *   # Option 1: Store in auth.json (recommended)
 *   # Add to ~/.pi/agent/auth.json:
 *   #   "lilac": { "type": "api_key", "key": "your-api-key" }
 *
 *   # Option 2: Set as environment variable
 *   export LILAC_API_KEY=your-api-key
 *
 *   # Run pi with the extension
 *   pi -e /path/to/pi-lilac-provider
 *
 * Then use /model to select from available models
 */

import type { ExtensionAPI, ModelRegistry } from "@earendil-works/pi-coding-agent";
import modelsData from "./models.json" with { type: "json" };
import customModelsData from "./custom-models.json" with { type: "json" };
import patchData from "./patch.json" with { type: "json" };
import fs from "fs";
import os from "os";
import path from "path";

// ─── Types ────────────────────────────────────────────────────────────────────

interface JsonDiscount {
  supplyState: string;
  discountPercent: number;
  creditMultiplier: number;
}

// Maps pi's thinking levels (off, minimal, low, medium, high, xhigh) to the
// provider-specific effort string sent on the wire. A `null` value marks a
// level as unsupported — clampThinkingLevel skips it when resolving the
// user's selection. Mirrors pi-ai's ThinkingLevelMap shape.
type ThinkingLevelMap = {
  off?: string | null;
  minimal?: string | null;
  low?: string | null;
  medium?: string | null;
  high?: string | null;
  xhigh?: string | null;
};

interface JsonModel {
  id: string;
  name: string;
  reasoning: boolean;
  input: string[];
  cost: {
    input: number;
    output: number;
    cacheRead: number;
    cacheWrite: number;
  };
  contextWindow: number;
  maxTokens: number;
  thinkingLevelMap?: ThinkingLevelMap;
  compat?: {
    supportsDeveloperRole?: boolean;
    supportsStore?: boolean;
    maxTokensField?: "max_completion_tokens" | "max_tokens";
    thinkingFormat?: "openai" | "zai" | "qwen" | "qwen-chat-template";
    supportsReasoningEffort?: boolean;
  };
  discount?: JsonDiscount;
}

interface PatchEntry {
  name?: string;
  reasoning?: boolean;
  input?: string[];
  cost?: {
    input?: number;
    output?: number;
    cacheRead?: number;
    cacheWrite?: number;
  };
  contextWindow?: number;
  maxTokens?: number;
  thinkingLevelMap?: ThinkingLevelMap;
  compat?: Record<string, unknown>;
}

type PatchData = Record<string, PatchEntry>;

// ─── Patch Application ────────────────────────────────────────────────────────

function applyPatch(model: JsonModel, patch: PatchEntry): JsonModel {
  const result = { ...model };

  if (patch.name !== undefined) result.name = patch.name;
  if (patch.reasoning !== undefined) result.reasoning = patch.reasoning;
  if (patch.input !== undefined) result.input = patch.input;
  if (patch.contextWindow !== undefined) result.contextWindow = patch.contextWindow;
  if (patch.maxTokens !== undefined) result.maxTokens = patch.maxTokens;

  if (patch.cost) {
    result.cost = {
      input: patch.cost.input ?? result.cost.input,
      output: patch.cost.output ?? result.cost.output,
      cacheRead: patch.cost.cacheRead ?? result.cost.cacheRead,
      cacheWrite: patch.cost.cacheWrite ?? result.cost.cacheWrite,
    };
  }
  if (patch.compat) {
    result.compat = { ...(result.compat || {}), ...patch.compat };
  }
  if (patch.thinkingLevelMap !== undefined) {
    result.thinkingLevelMap = patch.thinkingLevelMap;
  }

  if (!result.reasoning && result.compat?.thinkingFormat) {
    delete result.compat.thinkingFormat;
  }
  if (!result.reasoning && result.thinkingLevelMap) {
    delete result.thinkingLevelMap;
  }
  if (result.compat && Object.keys(result.compat).length === 0) {
    delete result.compat;
  }

  return result;
}

/** Full pipeline: base models → patch → custom → result */
function buildModels(base: JsonModel[], custom: JsonModel[], patch: PatchData): JsonModel[] {
  const modelMap = new Map<string, JsonModel>();

  for (const model of base) {
    modelMap.set(model.id, model);
  }

  for (const [id, patchEntry] of Object.entries(patch)) {
    const existing = modelMap.get(id);
    if (existing) {
      modelMap.set(id, applyPatch(existing, patchEntry));
    }
  }

  for (const model of custom) {
    const existing = modelMap.get(model.id);
    const patchEntry = patch[model.id];
    if (existing && patchEntry) {
      modelMap.set(model.id, applyPatch(model, patchEntry));
    } else if (existing) {
      modelMap.set(model.id, model);
    } else if (patchEntry) {
      modelMap.set(model.id, applyPatch(model, patchEntry));
    } else {
      modelMap.set(model.id, model);
    }
  }

  return Array.from(modelMap.values());
}

// ─── Stale-While-Revalidate Model Sync ────────────────────────────────────────

const PROVIDER_ID = "lilac";
const BASE_URL = "https://api.getlilac.com/v1";
const STATUS_URL = "https://api.getlilac.com/status";
const MODELS_URL = `${BASE_URL}/models`;
const CACHE_DIR = path.join(os.homedir(), ".pi", "agent", "cache");
const CACHE_PATH = path.join(CACHE_DIR, `${PROVIDER_ID}-models.json`);
const DISCOUNT_CACHE_PATH = path.join(CACHE_DIR, `${PROVIDER_ID}-discounts.json`);
const LIVE_FETCH_TIMEOUT_MS = 8000;

/** Transform a model from the Lilac /v1/models API. Lilac returns rich metadata. */
function transformApiModel(apiModel: any): JsonModel | null {
  const features: string[] = apiModel.supported_features || [];
  const modalities = apiModel.architecture?.input_modalities || [];
  const hasImage = modalities.includes("image");
  const pricing = apiModel.pricing || {};

  // Lilac API returns per-token pricing (e.g. "0.0000007" = $0.70/M tokens)
  const toPerM = (v: any) => Math.round((typeof v === "string" ? parseFloat(v) : (v || 0)) * 1_000_000 * 100) / 100;

  const inputTypes: string[] = ["text"];
  if (hasImage) inputTypes.push("image");
  // Video is sent as image frames, so we don't add a separate "video" input type

  const model: JsonModel = {
    id: apiModel.id,
    name: apiModel.name || apiModel.id,
    reasoning: features.includes("reasoning"),
    input: inputTypes,
    cost: {
      input: toPerM(pricing.prompt),
      output: toPerM(pricing.completion),
      cacheRead: toPerM(pricing.input_cache_read),
      cacheWrite: 0,
    },
    contextWindow: apiModel.context_length || 131072,
    maxTokens: apiModel.top_provider?.max_completion_tokens || apiModel.context_length || 131072,
  };

  // All Lilac models use chat_template_kwargs for reasoning toggle
  if (features.includes("reasoning")) {
    model.compat = {
      supportsDeveloperRole: true,
      supportsStore: false,
      maxTokensField: "max_completion_tokens",
      thinkingFormat: "qwen-chat-template",
      supportsReasoningEffort: true,
    };
  } else {
    model.compat = {
      supportsDeveloperRole: true,
      supportsStore: false,
      maxTokensField: "max_completion_tokens",
    };
  }

  return model;
}

async function fetchLiveModels(apiKey: string, signal?: AbortSignal): Promise<JsonModel[] | null> {
  try {
    const response = await fetch(MODELS_URL, {
      headers: { Authorization: `Bearer ${apiKey}` },
      signal: signal ? AbortSignal.any([AbortSignal.timeout(LIVE_FETCH_TIMEOUT_MS), signal]) : AbortSignal.timeout(LIVE_FETCH_TIMEOUT_MS),
    });
    if (!response.ok) return null;
    const data = await response.json();
    const apiModels = Array.isArray(data) ? data : (data.data || []);
    if (!Array.isArray(apiModels) || apiModels.length === 0) return null;
    return apiModels.map(transformApiModel).filter((m): m is JsonModel => m !== null);
  } catch {
    return null;
  }
}

function loadCachedModels(): JsonModel[] | null {
  try {
    const data = JSON.parse(fs.readFileSync(CACHE_PATH, "utf8"));
    return Array.isArray(data) ? data : null;
  } catch {
    return null;
  }
}

function cacheModels(models: JsonModel[]): void {
  try {
    fs.mkdirSync(CACHE_DIR, { recursive: true });
    fs.writeFileSync(CACHE_PATH, JSON.stringify(models, null, 2) + "\n");
  } catch {
    // Cache write failure is non-fatal
  }
}

function mergeWithEmbedded(liveModels: JsonModel[], embeddedModels: JsonModel[]): JsonModel[] {
  const embeddedMap = new Map(embeddedModels.map(m => [m.id, m]));
  const seen = new Set<string>();
  const result: JsonModel[] = [];
  for (const liveModel of liveModels) {
    const embedded = embeddedMap.get(liveModel.id);
    seen.add(liveModel.id);
    if (embedded) {
      result.push({
        ...liveModel,
        ...embedded,
        contextWindow: liveModel.contextWindow || embedded.contextWindow,
      });
    } else {
      result.push(liveModel);
    }
  }
  // Append any embedded models that the live API didn't return
  for (const em of embeddedModels) {
    if (!seen.has(em.id)) {
      result.push(em);
    }
  }
  return result;
}

function loadStaleModels(embeddedModels: JsonModel[]): JsonModel[] {
  const cached = loadCachedModels();
  if (!cached || cached.length === 0) return embeddedModels;

  // Merge embedded models that are missing from cache (newly added models)
  const cachedMap = new Map(cached.map(m => [m.id, m]));
  for (const em of embeddedModels) {
    if (!cachedMap.has(em.id)) {
      cached.push(em);
    }
  }
  return cached;
}

async function fetchStatusDiscounts(apiKey: string, signal?: AbortSignal): Promise<Map<string, JsonDiscount> | null> {
  try {
    const response = await fetch(STATUS_URL, {
      headers: { Authorization: `Bearer ${apiKey}` },
      signal: signal ? AbortSignal.any([AbortSignal.timeout(LIVE_FETCH_TIMEOUT_MS), signal]) : AbortSignal.timeout(LIVE_FETCH_TIMEOUT_MS),
    });
    if (!response.ok) return null;
    const data = await response.json() as Record<string, unknown>;
    const discounts = new Map<string, JsonDiscount>();
    // The /status endpoint returns per-model discount data in a "models" array.
    // Each model object has: id, current_subscription_supply_state,
    // current_subscription_discount_percent, current_subscription_credit_multiplier.
    const models = data.models;
    if (Array.isArray(models)) {
      for (const m of models) {
        if (!m || typeof m !== "object" || !m.id) continue;
        discounts.set(m.id, {
          supplyState: String(m.current_subscription_supply_state || "unknown"),
          discountPercent: Number(m.current_subscription_discount_percent ?? 0),
          creditMultiplier: parseFloat(String(m.current_subscription_credit_multiplier ?? "1")),
        });
      }
    }
    return discounts;
  } catch {
    return null;
  }
}

function applyDiscounts(models: JsonModel[], discounts: Map<string, JsonDiscount> | null): JsonModel[] {
  if (!discounts || discounts.size === 0) return models;
  return models.map(model => {
    const discount = discounts.get(model.id);
    if (!discount) return model;
    // credit_multiplier from /status is the effective price factor.
    // E.g. "0.75" means pay 75% of list price. For MiniMax with "1.00" there's no discount.
    // discountPercent is informational (it equals (1 - creditMultiplier) * 100).
    const factor = discount.creditMultiplier;
    const applyFactor = (n: number) => n > 0 ? Math.round(n * factor * 10000) / 10000 : n;
    return {
      ...model,
      cost: {
        input: applyFactor(model.cost.input),
        output: applyFactor(model.cost.output),
        cacheRead: applyFactor(model.cost.cacheRead),
        cacheWrite: model.cost.cacheWrite,
      },
      discount,
    };
  });
}

function cacheDiscounts(discounts: Map<string, JsonDiscount>): void {
  try {
    fs.mkdirSync(CACHE_DIR, { recursive: true });
    fs.writeFileSync(DISCOUNT_CACHE_PATH, JSON.stringify(Object.fromEntries(discounts), null, 2) + "\n");
  } catch {
    // non-fatal
  }
}

function loadCachedDiscounts(): Map<string, JsonDiscount> | null {
  try {
    const data = JSON.parse(fs.readFileSync(DISCOUNT_CACHE_PATH, "utf8")) as Record<string, JsonDiscount>;
    const map = new Map<string, JsonDiscount>();
    for (const [key, value] of Object.entries(data)) {
      if (value && typeof value === "object") {
        map.set(key, {
          supplyState: String(value.supplyState || "unknown"),
          discountPercent: Number(value.discountPercent ?? 0),
          creditMultiplier: Number(value.creditMultiplier ?? 1),
        });
      }
    }
    return map;
  } catch {
    return null;
  }
}

function formatDiscountStatus(modelId?: string): string {
  if (!modelId) return "supply: —";
  if (!latestDiscounts) return "supply: checking…";
  const discount = latestDiscounts.get(modelId);
  if (!discount) return "supply: —";
  return `supply: ${discount.supplyState} · sub-discount: ${discount.discountPercent}%`;
}

function dimStatus(ctx: any, text: string): string {
  try {
    return ctx.ui.theme.fg("dim", text);
  } catch {
    return text;
  }
}

function discountsChanged(
  a: Map<string, JsonDiscount> | null,
  b: Map<string, JsonDiscount> | null,
): boolean {
  if (!a || !b) return true;
  if (a.size !== b.size) return true;
  for (const [key, valA] of a) {
    const valB = b.get(key);
    if (!valB) return true;
    if (valA.supplyState !== valB.supplyState) return true;
    if (valA.discountPercent !== valB.discountPercent) return true;
    if (valA.creditMultiplier !== valB.creditMultiplier) return true;
  }
  return false;
}



// ─── API Key Resolution (via ModelRegistry) ────────────────────────────────────

let cachedApiKey: string | undefined;
let revalidateAbort: AbortController | null = null;
let latestDiscounts: Map<string, JsonDiscount> | null = null;
let lastDiscountFetchTime = 0;
const STATUS_CACHE_TTL_MS = 30000;

async function resolveApiKey(modelRegistry: ModelRegistry): Promise<void> {
  cachedApiKey = await modelRegistry.getApiKeyForProvider("lilac") ?? undefined;
}

// ─── Extension Entry Point ────────────────────────────────────────────────────

export default function (pi: ExtensionAPI) {
  const embeddedModels = modelsData as JsonModel[];
  const customModels = customModelsData as JsonModel[];
  const patches = patchData as PatchData;

  const staleBase = loadStaleModels(embeddedModels);
  latestDiscounts = loadCachedDiscounts();
  const staleModels = applyDiscounts(buildModels(staleBase, customModels, patches), latestDiscounts);

  pi.registerProvider("lilac", {
    baseUrl: BASE_URL,
    apiKey: "$LILAC_API_KEY",
    api: "openai-completions",
    models: staleModels,
  });

  const DISCOUNT_ENTRY_TYPE = "lilac-discount";

  interface DiscountEntry {
    modelId: string;
    supplyState: string;
    discountPercent: number;
    creditMultiplier: number;
  }

  function replayDiscountEvents(ctx: any): void {
    latestDiscounts = loadCachedDiscounts() ?? new Map();
    for (const entry of ctx.sessionManager.getBranch()) {
      if (entry.type === "custom" && entry.customType === DISCOUNT_ENTRY_TYPE && entry.data) {
        const d = entry.data as DiscountEntry;
        latestDiscounts.set(d.modelId, {
          supplyState: d.supplyState,
          discountPercent: d.discountPercent,
          creditMultiplier: d.creditMultiplier,
        });
      }
    }
  }

  pi.on("session_start", async (_event, ctx) => {
    revalidateAbort?.abort();
    revalidateAbort = new AbortController();
    const signal = revalidateAbort.signal;

    // Replay persisted discount state from session JSONL (synchronous, zero-latency)
    replayDiscountEvents(ctx);

    // Show status immediately with replayed/cached data — don't block pi startup
    const model = ctx.model;
    if (model?.provider === "lilac") {
      ctx.ui.setStatus("lilac", dimStatus(ctx, formatDiscountStatus(model.id)));
    }

    // Fire-and-forget: resolve API key, then fetch live data in background.
    // Provider and status are hot-swapped when results arrive.
    resolveApiKey(ctx.modelRegistry).then(() => {
      if (!cachedApiKey || signal.aborted) return;

      Promise.all([
        fetchLiveModels(cachedApiKey, signal),
        fetchStatusDiscounts(cachedApiKey, signal),
      ]).then(([liveModels, discounts]) => {
        if (signal.aborted) return;

        if (discounts) {
          lastDiscountFetchTime = Date.now();
          cacheDiscounts(discounts);
          latestDiscounts = discounts;
        }

        if (liveModels && liveModels.length > 0) {
          const merged = mergeWithEmbedded(liveModels, embeddedModels);
          cacheModels(merged);
          pi.registerProvider("lilac", {
            baseUrl: BASE_URL,
            apiKey: "$LILAC_API_KEY",
            api: "openai-completions",
            models: applyDiscounts(buildModels(merged, customModels, patches), latestDiscounts),
          });
        } else if (discounts) {
          pi.registerProvider("lilac", {
            baseUrl: BASE_URL,
            apiKey: "$LILAC_API_KEY",
            api: "openai-completions",
            models: applyDiscounts(buildModels(staleBase, customModels, patches), latestDiscounts),
          });
        }

        if (model?.provider === "lilac") {
          ctx.ui.setStatus("lilac", dimStatus(ctx, formatDiscountStatus(model.id)));
        }
      }).catch(() => { /* network errors are non-fatal */ });
    });
  });

  pi.on("turn_end", async (_event, ctx) => {
    if (!ctx.model || ctx.model.provider !== "lilac" || !latestDiscounts) return;
    const discount = latestDiscounts.get(ctx.model.id);
    if (!discount) return;
    pi.appendEntry(DISCOUNT_ENTRY_TYPE, {
      modelId: ctx.model.id,
      supplyState: discount.supplyState,
      discountPercent: discount.discountPercent,
      creditMultiplier: discount.creditMultiplier,
    } as DiscountEntry);
  });

  pi.on("before_provider_request", async (_event, ctx) => {
    if (ctx.model?.provider !== "lilac") return;

    // Always show status for active lilac model
    ctx.ui.setStatus("lilac", dimStatus(ctx, formatDiscountStatus(ctx.model.id)));

    if (!cachedApiKey) return;

    const now = Date.now();
    if (latestDiscounts && now - lastDiscountFetchTime < STATUS_CACHE_TTL_MS) {
      return;
    }

    const discounts = await fetchStatusDiscounts(cachedApiKey);
    if (!discounts) return;
    if (!discountsChanged(latestDiscounts, discounts)) {
      lastDiscountFetchTime = now;
      ctx.ui.setStatus("lilac", dimStatus(ctx, formatDiscountStatus(ctx.model.id)));
      return;
    }

    lastDiscountFetchTime = now;
    cacheDiscounts(discounts);
    latestDiscounts = discounts;

    const base = loadStaleModels(embeddedModels);
    pi.registerProvider("lilac", {
      baseUrl: BASE_URL,
      apiKey: "$LILAC_API_KEY",
      api: "openai-completions",
      models: applyDiscounts(buildModels(base, customModels, patches), discounts),
    });
    ctx.ui.setStatus("lilac", dimStatus(ctx, formatDiscountStatus(ctx.model.id)));
  });

  pi.on("model_select", async (event, ctx) => {
    if (event.model.provider === "lilac") {
      ctx.ui.setStatus("lilac", dimStatus(ctx, formatDiscountStatus(event.model.id)));
    } else {
      ctx.ui.setStatus("lilac", undefined);
    }
  });

  pi.on("session_tree", async (_event, ctx) => {
    replayDiscountEvents(ctx);
    const model = ctx.model;
    if (model?.provider === "lilac") {
      ctx.ui.setStatus("lilac", dimStatus(ctx, formatDiscountStatus(model.id)));
    }
  });

  // vLLM's streaming parser intermittently emits finish_reason: "tool_calls" without
  // any delta.tool_calls chunks — even with tool_stream: true (set via zaiToolStream
  // in compat). Pi maps that to stopReason: "toolUse" but there are zero toolCall
  // blocks to execute, so the agent loop ends with nothing to do ("abrupt stop").
  // The message_end handler converts this to a retryable error so pi's auto-retry
  // mechanism re-prompts the agent.
  pi.on("message_end", async (event, mctx) => {
    const message = event.message;
    if (message.role !== "assistant") return;
    if (message.provider !== "lilac" && mctx.model?.provider !== "lilac") return;
    if (message.stopReason !== "toolUse") return;

    const content = message.content;
    const hasToolCalls = Array.isArray(content) &&
      content.some((block: any) => block.type === "toolCall");

    if (hasToolCalls) return;

    // vLLM emitted finish_reason: "tool_calls" without any delta.tool_calls chunks.
    // Convert to a retryable error so pi's auto-retry mechanism re-prompts the
    // agent. The error message matches the "stream ended before" pattern in
    // _isRetryableError, which triggers automatic backoff-and-retry.
    return {
      message: {
        ...message,
        stopReason: "error",
        errorMessage: "stream ended before tool_calls were received (vLLM phantom tool_use)",
      },
    };
  });

  pi.on("session_shutdown", () => {
    revalidateAbort?.abort();
  });
}

export { fetchStatusDiscounts, applyDiscounts, loadCachedDiscounts, cacheDiscounts };
export type { JsonDiscount, JsonModel, PatchEntry, PatchData };
