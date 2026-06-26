/**
 * Lilac Provider Extension
 *
 * Registers Lilac (getlilac.com) as a custom provider using the openai-completions API.
 * Base URL: https://api.getlilac.com/v1
 *
 * Lilac serves models via a customized fork of vLLM tuned for idle-GPU scheduling
 * and shared warm endpoints. Reasoning is toggled via chat_template_kwargs, but
 * the key each model's chat template honors differs per family, so per-model
 * chatTemplateKwargs are configured in patch.json:
 *
 *   - Kimi K2.6:    honors `thinking` (bool); `enable_thinking` ignored. ON by default.
 *   - GLM 5.1:      honors `enable_thinking` (bool). ON by default.
 *   - GLM 5.2:      honors `enable_thinking` (bool) + `reasoning_effort` (max|high). ON by default.
 *   - Gemma 4:      honors `enable_thinking` (bool). OFF by default.
 *   - MiniMax M2.7: forward-compatible `thinking` + `enable_thinking` (bool).
 *   - MiniMax M3:   honors `thinking_mode` (disabled|adaptive|enabled); bool keys ignored.
 *
 * We use pi's `chat-template` thinkingFormat (NOT `qwen-chat-template`, which
 * sends only `enable_thinking` + `preserve_thinking` and is ignored by Kimi and
 * MiniMax M3). The forward-compatible form sends BOTH `thinking` and
 * `enable_thinking` so whichever key the template honors is set:
 *   { chat_template_kwargs: { thinking: <bool>, enable_thinking: <bool> } }
 * GLM 5.2 adds `reasoning_effort` (high = lower-latency, xhigh = max) via a
 * thinkingLevelMap. MiniMax M3 maps to the `thinking_mode` enum as three pi
 * thinking levels — off→disabled, minimal→adaptive (model decides), high→enabled
 * — so adaptive is selectable via pi's Shift+Tab cycle (off→minimal→high). Pi
 * shows the pi level names (minimal/high) in the selector/footer, not the
 * thinking_mode values; there's no per-model level-relabel hook.
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

// A chat_template_kwargs value, mirroring pi-ai's ChatTemplateKwargSchema. Scalar
// values are passed through verbatim; { $var } values are resolved by pi-ai from
// the turn's thinking state ("thinking.enabled" → bool, "thinking.effort" → the
// mapped effort string). omitWhenOff drops the key entirely when thinking is off.
type ChatTemplateKwargValue =
  | string
  | number
  | boolean
  | null
  | { $var: "thinking.enabled" | "thinking.effort"; omitWhenOff?: boolean };

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
    thinkingFormat?:
      | "openai"
      | "openrouter"
      | "together"
      | "deepseek"
      | "zai"
      | "qwen"
      | "chat-template"
      | "qwen-chat-template"
      | "string-thinking"
      | "ant-ling";
    chatTemplateKwargs?: Record<string, ChatTemplateKwargValue>;
    zaiToolStream?: boolean;
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

// User Configuration: ~/.pi/agent/extensions/lilac.json lets a user override model
// properties per id ON TOP of patch.json + custom-models.json (so they win).
// Recursively deep-merges `compat` (incl. nested `chatTemplateKwargs`),
// `thinkingLevelMap`, and `cost` (toggle one flag without redeclaring the rest),
// replaces scalars and arrays. Lets a user toggle chat_template_kwargs (e.g.
// preserve_thinking / clear_thinking) or a single thinking level without editing
// the extension. See README "Model Overrides".
interface ModelOverride {
  thinkingLevelMap?: ThinkingLevelMap;
  compat?: Record<string, unknown>;
}

interface LilacConfig {
  modelOverrides?: Record<string, ModelOverride>;
}

const CONFIG_PATH = path.join(os.homedir(), ".pi", "agent", "extensions", "lilac.json");
const DEFAULT_CONFIG: LilacConfig = { modelOverrides: {} };

// Validate user-supplied modelOverrides from the config file. Non-object ids and
// non-object overrides are dropped silently so a malformed file doesn't crash
// model registration.
function parseModelOverrides(raw: unknown): Record<string, ModelOverride> | undefined {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return undefined;
  const result: Record<string, ModelOverride> = {};
  for (const [id, override] of Object.entries(raw as Record<string, unknown>)) {
    if (typeof id !== "string" || !override || typeof override !== "object" || Array.isArray(override)) continue;
    const o = override as Record<string, unknown>;
    const parsed: ModelOverride = {};
    if (o.thinkingLevelMap && typeof o.thinkingLevelMap === "object") {
      const m: Record<string, string | null> = {};
      for (const [k, v] of Object.entries(o.thinkingLevelMap as Record<string, unknown>)) {
        if (v === null || typeof v === "string") m[k] = v;
      }
      if (Object.keys(m).length > 0) parsed.thinkingLevelMap = m as ThinkingLevelMap;
    }
    if (o.compat && typeof o.compat === "object") parsed.compat = o.compat as Record<string, unknown>;
    if (Object.keys(parsed).length > 0) result[id] = parsed;
  }
  return Object.keys(result).length > 0 ? result : undefined;
}

// Reads ~/.pi/agent/extensions/lilac.json. Missing file → populate with defaults
// so the user can discover it, then return defaults. An existing-but-invalid file
// is left untouched (defaults returned) so a user's typo isn't silently wiped —
// they fix the file and restart pi. Loaded lazily on first use (not at import) so
// importing the module has no filesystem side effects and unit tests can import
// the pure helpers safely.
function loadConfig(): LilacConfig {
  let rawText: string;
  try {
    rawText = fs.readFileSync(CONFIG_PATH, "utf8");
  } catch {
    try {
      fs.mkdirSync(path.dirname(CONFIG_PATH), { recursive: true });
      fs.writeFileSync(CONFIG_PATH, JSON.stringify(DEFAULT_CONFIG, null, 2) + "\n");
    } catch {
      // Write failure is non-fatal — defaults still work in memory
    }
    return { ...DEFAULT_CONFIG };
  }
  try {
    const raw = JSON.parse(rawText);
    return { modelOverrides: parseModelOverrides(raw.modelOverrides) };
  } catch {
    // File exists but is invalid JSON — return defaults WITHOUT overwriting.
    return { ...DEFAULT_CONFIG };
  }
}

// Recursively deep-merge `override` into `base`. Plain objects are merged
// key-by-key (so a user can toggle a single chatTemplateKwargs flag without
// redeclaring the rest, and a single compat flag without redeclaring
// chatTemplateKwargs); arrays and non-plain-object values replace the base value
// (so an overridden { $var } schema object, or an `input` array, replaces wholesale).
function isPlainObject(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function deepMerge<T>(base: T, override: unknown): T {
  if (!isPlainObject(override) || !isPlainObject(base)) return override as T;
  const result: Record<string, unknown> = { ...(base as Record<string, unknown>) };
  for (const [k, v] of Object.entries(override)) {
    result[k] = isPlainObject(v) && isPlainObject(result[k]) ? deepMerge(result[k], v) : v;
  }
  return result as T;
}

// Apply a user-supplied modelOverride (from lilac.json) on top of a built model.
// Recursively deep-merges compat (incl. nested chatTemplateKwargs) /
// thinkingLevelMap / cost so a user can toggle a single flag (e.g.
// chatTemplateKwargs.preserve_thinking) without redeclaring the rest; replaces
// scalars and arrays. No reasoning-cleanup (unlike applyPatch) — the override is
// authoritative.
function applyModelOverride(model: JsonModel, override: ModelOverride): JsonModel {
  const result = { ...model };
  for (const [key, value] of Object.entries(override)) {
    (result as any)[key] = isPlainObject(value) && isPlainObject((result as any)[key])
      ? deepMerge((result as any)[key], value)
      : value;
  }
  return result;
}

let config: LilacConfig | undefined;
function getConfig(): LilacConfig {
  if (!config) config = loadConfig();
  return config;
}

function activeOverrides(): Record<string, ModelOverride> {
  return getConfig().modelOverrides ?? {};
}

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

/** Full pipeline: base models → patch → custom → user modelOverrides → result */
function buildModels(
  base: JsonModel[],
  custom: JsonModel[],
  patch: PatchData,
  overrides: Record<string, ModelOverride> = {},
): JsonModel[] {
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

  // User-supplied modelOverrides (from ~/.pi/agent/extensions/lilac.json) applied
  // LAST so they win over patch.json + custom-models.json. Recursively deep-merges
  // compat (incl. chatTemplateKwargs) / thinkingLevelMap / cost so a user can
  // toggle a single flag (e.g. chatTemplateKwargs.preserve_thinking) without
  // redeclaring the rest.
  for (const [id, override] of Object.entries(overrides)) {
    const existing = modelMap.get(id);
    if (existing) {
      modelMap.set(id, applyModelOverride(existing, override));
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

  // All Lilac models toggle reasoning via chat_template_kwargs, but the key each
  // model's chat template honors differs per family. Default newly discovered
  // models to the forward-compatible both-keys form (works across all current
  // Lilac templates); per-model overrides in patch.json refine this — e.g. GLM
  // 5.2 adds reasoning_effort, MiniMax M3 uses the thinking_mode enum.
  if (features.includes("reasoning")) {
    model.compat = {
      supportsDeveloperRole: true,
      supportsStore: false,
      maxTokensField: "max_completion_tokens",
      thinkingFormat: "chat-template",
      chatTemplateKwargs: {
        thinking: { $var: "thinking.enabled" },
        enable_thinking: { $var: "thinking.enabled" },
      },
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
  // Base model set changed; force getListModels() to rebuild list prices.
  listModelsCache = null;
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

/**
 * Apply the current discount to a single model's cost IN PLACE.
 *
 * applyDiscounts() returns discounted COPIES for the registry; those copies
 * cannot reach the model pi already bound for the current turn. This mutates
 * that bound object directly so the current turn's calculateCost() (in pi-ai)
 * reads the discounted price. Cost is always recomputed from list price — the
 * patch-applied, pre-discount value from buildModels() — so re-applying a
 * changed discount never compounds a factor already present on the object.
 * cacheWrite is left untouched (Lilac does not discount it, matching
 * applyDiscounts).
 *
 * Targets the object pi bound for the turn, captured before any await or
 * registerProvider() call. registerProvider() refreshes the session model for
 * SUBSEQUENT turns only (via prepareNextTurn); it cannot affect the current
 * turn's cost calc, which is why the in-place mutation is required.
 */
function applyDiscountInPlace(
  model: { id: string; cost: { input: number; output: number; cacheRead: number; cacheWrite: number } } | undefined,
  listModels: JsonModel[],
  discounts: Map<string, JsonDiscount> | null,
): void {
  if (!model?.cost) return;
  const list = listModels.find(m => m.id === model.id);
  if (!list) return;
  // A missing discount entry means list price (factor 1) — e.g. a model whose
  // discount was removed since it was last priced.
  const rawFactor = discounts?.get(model.id)?.creditMultiplier;
  const factor = Number.isFinite(rawFactor) && rawFactor !== undefined ? rawFactor : 1;
  const applyFactor = (n: number) => n > 0 ? Math.round(n * factor * 10000) / 10000 : n;
  model.cost.input = applyFactor(list.cost.input);
  model.cost.output = applyFactor(list.cost.output);
  model.cost.cacheRead = applyFactor(list.cost.cacheRead);
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

/**
 * Paint the footer status from the LIVE session model — the single source of
 * truth for the display. ctx.model is a lazy getter (not a snapshot), so this
 * reflects the currently-selected model even when called from a deferred
 * post-await callback that captured a different (stale) model at hook start.
 *
 * Set for lilac models, cleared for everything else. Every status paint site
 * calls this, so whichever handler runs last wins — including after a switch
 * to a non-lilac model, fixing a race where deferred post-await setStatus()
 * calls re-painted a stale captured lilac model's discount over the clear
 * that model_select had just issued.
 *
 * Only the DISPLAY follows the live model. Cost mutation (applyDiscountInPlace)
 * and registerProvider() in before_provider_request still target the turn's
 * captured in-flight model — that's the object pi bound and whose .cost
 * calculateCost() reads, and it may legitimately differ from the live model
 * after a mid-turn /model switch.
 *
 * Wrapped in try/catch: ctx.model / ctx.ui assert the extension runner is
 * still active and throw if the session ended mid-fetch, so a late deferred
 * callback after session_shutdown no-ops instead of throwing.
 */
function syncStatus(ctx: any): void {
  try {
    const model = ctx.model;
    if (model?.provider === "lilac") {
      ctx.ui.setStatus("lilac", dimStatus(ctx, formatDiscountStatus(model.id)));
    } else {
      ctx.ui.setStatus("lilac", undefined);
    }
  } catch {
    // Runner stale (session ended mid-fetch) — nothing to paint.
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
// Turn-initiated /status fetches are throttled to once per TTL window so a burst
// of messages doesn't hammer the endpoint. Background polling (see
// STATUS_POLL_INTERVAL_MS) and turn fetches both stamp lastDiscountFetchTime, so
// they cooperate: a poll that just ran lets the next turn skip its own fetch
// within the TTL.
const STATUS_CACHE_TTL_MS = 60000;
// Lilac refreshes discounts ~every 10 minutes (per their docs: "Discounts refresh
// approximately every 10 minutes and are locked in when a request starts"). Poll
// every 5 min during idle — half the refresh window — so a long-idle session
// catches supply/sub changes within ~5 min instead of waiting up to a full
// window. Turn fetches alone only refresh on a user message and are
// TTL-throttled to 1/min.
const STATUS_POLL_INTERVAL_MS = 5 * 60 * 1000;
let pollInterval: ReturnType<typeof setInterval> | null = null;
// List-price (patch-applied, pre-discount) models, cached until the base set
// changes. Reset in cacheModels() so the next getListModels() rebuilds from the
// refreshed disk cache / embedded set.
let listModelsCache: JsonModel[] | null = null;

async function resolveApiKey(modelRegistry: ModelRegistry): Promise<void> {
  cachedApiKey = await modelRegistry.getApiKeyForProvider("lilac") ?? undefined;
}

// ─── Extension Entry Point ────────────────────────────────────────────────────

export default function (pi: ExtensionAPI) {
  const embeddedModels = modelsData as JsonModel[];
  const customModels = customModelsData as JsonModel[];
  const patches = patchData as PatchData;

  // List-price models (patch applied, pre-discount), cached at module scope and
  // rebuilt only when the base set changes (see cacheModels). Used to recompute
  // the in-flight model's cost without compounding an already-applied discount.
  function getListModels(): JsonModel[] {
    if (!listModelsCache) {
      listModelsCache = buildModels(loadStaleModels(embeddedModels), customModels, patches, activeOverrides());
    }
    return listModelsCache;
  }

  const staleBase = loadStaleModels(embeddedModels);
  latestDiscounts = loadCachedDiscounts();
  const staleModels = applyDiscounts(buildModels(staleBase, customModels, patches, activeOverrides()), latestDiscounts);

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

  /**
   * Background /status poll, fired every STATUS_POLL_INTERVAL_MS (5 min) from
   * session_start to cover idle sessions. Mirrors the discount half of
   * before_provider_request, but without an in-flight turn model to mutate: it
   * only refreshes latestDiscounts, re-registers (so the next turn's models carry
   * the new price), and re-paints the footer from the LIVE model. Passes the
   * session AbortSignal so the fetch dies on session_shutdown or a subsequent
   * session_start; bails on a missing API key or an aborted signal.
   */
  function pollStatusDiscounts(ctx: any, signal: AbortSignal): void {
    if (signal.aborted || !cachedApiKey) return;
    fetchStatusDiscounts(cachedApiKey, signal).then(discounts => {
      if (signal.aborted || !discounts) return;
      lastDiscountFetchTime = Date.now();
      if (!discountsChanged(latestDiscounts, discounts)) {
        syncStatus(ctx);
        return;
      }
      cacheDiscounts(discounts);
      latestDiscounts = discounts;
      const freshList = getListModels();
      pi.registerProvider("lilac", {
        baseUrl: BASE_URL,
        apiKey: "$LILAC_API_KEY",
        api: "openai-completions",
        models: applyDiscounts(freshList, discounts),
      });
      syncStatus(ctx);
    }).catch(() => { /* network errors are non-fatal */ });
  }

  pi.on("session_start", async (_event, ctx) => {
    revalidateAbort?.abort();
    revalidateAbort = new AbortController();
    const signal = revalidateAbort.signal;

    // Tear down any poll interval left from a prior session before starting a
    // fresh one (defensive; session_shutdown normally handles this).
    if (pollInterval) {
      clearInterval(pollInterval);
      pollInterval = null;
    }

    // Replay persisted discount state from session JSONL (synchronous, zero-latency)
    replayDiscountEvents(ctx);

    // Show status immediately with replayed/cached data — don't block pi startup.
    // syncStatus reads the LIVE ctx.model so a switch away from lilac before the
    // background fetch resolves never leaves a stale discount painted.
    syncStatus(ctx);

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
            models: applyDiscounts(buildModels(merged, customModels, patches, activeOverrides()), latestDiscounts),
          });
        } else if (discounts) {
          pi.registerProvider("lilac", {
            baseUrl: BASE_URL,
            apiKey: "$LILAC_API_KEY",
            api: "openai-completions",
            models: applyDiscounts(buildModels(staleBase, customModels, patches, activeOverrides()), latestDiscounts),
          });
        }

        // Re-paint from the LIVE model: if the user switched to a non-lilac
        // model (or a different lilac model) during the fetch, this reflects
        // the new selection instead of re-showing the stale captured model.
        syncStatus(ctx);
      }).catch(() => { /* network errors are non-fatal */ });
    });

    // Background poll for idle sessions: Lilac refreshes discounts ~every 10
    // minutes, so poll every 5 min (half the refresh window) to catch supply/sub
    // changes while the user is idle (turn fetches only run when a message is. The callback
    // bails on a missing API key or an aborted/shut-down session. Cleared in
    // session_shutdown and at the top of the next session_start.
    pollInterval = setInterval(() => pollStatusDiscounts(ctx, signal), STATUS_POLL_INTERVAL_MS);
    // Don't keep the process alive solely for discount polling.
    pollInterval.unref?.();
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
    // Capture the in-flight model BEFORE any await or registerProvider() call.
    // ctx.model is a lazy getter to the live session model; at hook start (before
    // any refresh) it is the same object pi's agent loop bound as the model for
    // THIS turn — the exact object whose .cost calculateCost() reads when the
    // response arrives. registerProvider() can only refresh the session model
    // for SUBSEQUENT turns (prepareNextTurn), so to affect the current turn's
    // cost we mutate this object in place. Capturing before the await also guards
    // against a concurrent session_start /status sync swapping the reference
    // mid-hook.
    const inFlightModel = ctx.model;
    if (!inFlightModel || inFlightModel.provider !== "lilac") return;

    const listModels = getListModels();

    // Align the in-flight model with the best-known discount right away. This
    // covers turns that don't trigger a fresh fetch (within TTL, or the fetch
    // returned unchanged) and recomputes from list price so it never compounds
    // a previously-applied factor.
    applyDiscountInPlace(inFlightModel, listModels, latestDiscounts);
    // Display follows the LIVE model (clears if the user switched away during
    // this turn); the cost mutation above still targets the captured in-flight
    // model pi bound for this turn.
    syncStatus(ctx);

    if (!cachedApiKey) return;

    const now = Date.now();
    if (latestDiscounts && now - lastDiscountFetchTime < STATUS_CACHE_TTL_MS) {
      return;
    }

    const discounts = await fetchStatusDiscounts(cachedApiKey);
    if (!discounts) return;

    lastDiscountFetchTime = now;

    if (!discountsChanged(latestDiscounts, discounts)) {
      syncStatus(ctx);
      return;
    }

    cacheDiscounts(discounts);
    latestDiscounts = discounts;

    // Re-read list prices in case a concurrent /models sync (session_start)
    // invalidated the list-price cache during the await above. Then re-apply the
    // freshly-fetched discount so THIS turn (not just later ones) is costed at
    // the new price, and re-register so other lilac models pick it up on their
    // next request.
    const freshList = getListModels();
    applyDiscountInPlace(inFlightModel, freshList, discounts);
    pi.registerProvider("lilac", {
      baseUrl: BASE_URL,
      apiKey: "$LILAC_API_KEY",
      api: "openai-completions",
      models: applyDiscounts(freshList, discounts),
    });
    // Display reflects the LIVE model: post-await the user may have switched to
    // a non-lilac model, so syncStatus clears instead of re-painting the stale
    // captured in-flight lilac model's discount.
    syncStatus(ctx);
  });

  pi.on("model_select", async (_event, ctx) => {
    // ctx.model is the live session model (pi sets state.model before emitting
    // this event), so syncStatus paints/clears consistently with every other
    // handler — one source of truth for the footer.
    syncStatus(ctx);
  });

  pi.on("session_tree", async (_event, ctx) => {
    replayDiscountEvents(ctx);
    syncStatus(ctx);
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
    if (pollInterval) {
      clearInterval(pollInterval);
      pollInterval = null;
    }
  });
}

export { fetchStatusDiscounts, applyDiscounts, applyDiscountInPlace, loadCachedDiscounts, cacheDiscounts, buildModels, applyModelOverride, parseModelOverrides, loadConfig, getConfig };
export type { JsonDiscount, JsonModel, PatchEntry, PatchData, ModelOverride, LilacConfig };
