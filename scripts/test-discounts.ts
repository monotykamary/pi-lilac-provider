#!/usr/bin/env node
/**
 * E2E test for Lilac discount metadata enumeration.
 *
 * Verifies:
 * 1. fetchStatusDiscounts parses the /status endpoint (real response format).
 * 2. applyDiscounts attaches metadata to matching models and mutates costs.
 * 3. cacheDiscounts / loadCachedDiscounts round-trip correctly.
 * 4. The provider extension registers models with discount metadata on init
 *    and refreshes them after session_start.
 * 5. session_start replays persisted discount events and sets footer status.
 * 6. model_select sets/clears footer status for lilac/non-lilac models.
 * 7. turn_end appends discount entry to session JSONL.
 * 8. before_provider_request refreshes discounts with a 30s cache.
 * 9. formatDiscountStatus returns fallbacks when data is missing.
 */

import type { ExtensionAPI, ModelRegistry } from "@earendil-works/pi-coding-agent";
import fs from "fs";

// Isolate cache to a temp directory so the test is deterministic.
const tmpHome = `/tmp/pi-lilac-test-${Date.now()}`;
fs.mkdirSync(tmpHome, { recursive: true });
process.env.HOME = tmpHome;

const originalFetch = globalThis.fetch;

function mockFetch(responses: Record<string, { status?: number; body?: unknown }>) {
  return async (url: string, _init?: RequestInit) => {
    for (const [pattern, response] of Object.entries(responses)) {
      if (url.includes(pattern)) {
        return new Response(JSON.stringify(response.body ?? {}), {
          status: response.status ?? 200,
          headers: { "content-type": "application/json" },
        });
      }
    }
    return new Response(JSON.stringify({ error: "not found" }), { status: 404 });
  };
}

const {
  default: registerLilac,
  fetchStatusDiscounts,
  applyDiscounts,
  loadCachedDiscounts,
  cacheDiscounts,
} = await import("../index.ts");

function assert(condition: boolean, message: string) {
  if (condition) {
    console.log(`  ✓ ${message}`);
  } else {
    console.error(`  ✗ ${message}`);
    throw new Error(`Assertion failed: ${message}`);
  }
}

// ─── Test 1: fetchStatusDiscounts (real API format) ────────────────────────────

console.log("\n--- Test 1: fetchStatusDiscounts ---");
globalThis.fetch = mockFetch({
  "/status": {
    body: {
      updated_at: "2026-06-06T09:18:14Z",
      current_subscription_supply_updated_at: "2026-06-06T09:11:49Z",
      window: "24h",
      window_secs: 86400,
      stale: false,
      models: [
        {
          id: "google/gemma-4-31b-it",
          name: "Gemma 4",
          tps: 62.67,
          ttfb_seconds: 0.96,
          uptime_pct: 100.0,
          current_subscription_supply_state: "medium",
          current_subscription_discount_percent: 25,
          current_subscription_credit_multiplier: "0.75",
        },
        {
          id: "zai-org/glm-5.1",
          name: "GLM 5.1",
          tps: 163.32,
          ttfb_seconds: 0.92,
          uptime_pct: 100.0,
          current_subscription_supply_state: "high",
          current_subscription_discount_percent: 50,
          current_subscription_credit_multiplier: "0.50",
        },
        {
          id: "minimaxai/minimax-m2.7",
          name: "MiniMax M2.7",
          tps: null,
          ttfb_seconds: null,
          uptime_pct: null,
          current_subscription_supply_state: "low",
          current_subscription_discount_percent: 0,
          current_subscription_credit_multiplier: "1.00",
        },
      ],
    },
  },
}) as any;

const discounts = await fetchStatusDiscounts("test-key");
assert(discounts !== null, "returns a non-null result");
assert(discounts!.has("google/gemma-4-31b-it"), "includes gemma");
assert(discounts!.get("google/gemma-4-31b-it")!.discountPercent === 25, "gemma discount is 25%");
assert(discounts!.get("google/gemma-4-31b-it")!.supplyState === "medium", "gemma state is medium");
assert(discounts!.get("google/gemma-4-31b-it")!.creditMultiplier === 0.75, "gemma credit multiplier is 0.75");
assert(discounts!.get("zai-org/glm-5.1")!.creditMultiplier === 0.50, "glm credit multiplier is 0.50");
assert(discounts!.get("minimaxai/minimax-m2.7")!.creditMultiplier === 1.00, "minimax credit multiplier is 1.00 (no discount)");

// ─── Test 2: applyDiscounts ───────────────────────────────────────────────────

console.log("\n--- Test 2: applyDiscounts ---");
const models = [
  { id: "google/gemma-4-31b-it", name: "Gemma 4", cost: { input: 0.11, output: 0.35, cacheRead: 0, cacheWrite: 0 } },
  { id: "moonshotai/kimi-k2.6", name: "Kimi K2.6", cost: { input: 0.70, output: 3.50, cacheRead: 0.20, cacheWrite: 0 } },
] as any[];
const applied = applyDiscounts(models, discounts);
assert(applied[0].discount != null, "gemma has discount field");
assert(applied[0].discount.discountPercent === 25, "gemma discount attached correctly");
// credit_multiplier 0.75 means pay 75% of list price
assert(applied[0].cost.input === 0.0825, "gemma input cost = 0.11 * 0.75 = 0.0825");
assert(applied[0].cost.output === 0.2625, "gemma output cost = 0.35 * 0.75 = 0.2625");
assert(applied[1].discount == null, "kimi has no discount (not in status)");
assert(applied[1].cost.input === 0.70, "kimi cost unchanged when no discount");

// ─── Test 3: cache round-trip ─────────────────────────────────────────────────

console.log("\n--- Test 3: cache round-trip ---");
const freshDiscounts = new Map([
  [
    "moonshotai/kimi-k2.6",
    { supplyState: "medium", discountPercent: 25, creditMultiplier: 0.75 },
  ],
]);
cacheDiscounts(freshDiscounts);
const loaded = loadCachedDiscounts();
assert(loaded !== null, "loaded discounts are non-null");
assert(loaded!.has("moonshotai/kimi-k2.6"), "loaded discounts include kimi");
assert(loaded!.get("moonshotai/kimi-k2.6")!.discountPercent === 25, "cached discount percent preserved");
assert(loaded!.get("moonshotai/kimi-k2.6")!.creditMultiplier === 0.75, "cached credit multiplier preserved");

// ─── Test 4: provider e2e registration ─────────────────────────────────────────

console.log("\n--- Test 4: provider e2e registration ---");
globalThis.fetch = mockFetch({
  "/models": {
    body: {
      data: [
        {
          id: "moonshotai/kimi-k2.6",
          name: "Kimi K2.6",
          supported_features: ["reasoning"],
          architecture: { input_modalities: ["text", "image"] },
          pricing: { prompt: "0.0000007", completion: "0.0000035", input_cache_read: "0.0000002" },
          context_length: 262144,
          top_provider: { max_completion_tokens: 262144 },
        },
      ],
    },
  },
  "/status": {
    body: {
      updated_at: "2026-06-06T12:00:00Z",
      current_subscription_supply_updated_at: "2026-06-06T12:00:00Z",
      window: "24h",
      window_secs: 86400,
      stale: false,
      models: [
        {
          id: "moonshotai/kimi-k2.6",
          name: "Kimi K2.6",
          tps: 75.2,
          ttfb_seconds: 0.21,
          uptime_pct: 99.99,
          current_subscription_supply_state: "healthy",
          current_subscription_discount_percent: 25,
          current_subscription_credit_multiplier: "0.75",
        },
      ],
    },
  },
}) as any;

const providers: any[] = [];
const handlers = new Map<string, ((...args: any[]) => void | Promise<void>)[]>();
const statuses = new Map<string, string | undefined>();
const appendedEntries: { customType: string; data: any }[] = [];

const mockTheme = {
  fg: (_color: string, text: string) => text, // strip theme for easy assertions
};

const mockUi = {
  setStatus: (key: string, text: string | undefined) => {
    statuses.set(key, text);
  },
  theme: mockTheme,
};

const mockApi: ExtensionAPI = {
  registerProvider: (name: string, config: any) => {
    providers.push({ name, config });
  },
  on: (event: string, handler: (...args: any[]) => void | Promise<void>) => {
    if (!handlers.has(event)) handlers.set(event, []);
    handlers.get(event)!.push(handler);
  },
  setStatus: mockUi.setStatus,
  registerCommand: () => {},
  setHiddenThinkingLabel: () => {},
  setLabel: () => {},
  appendEntry: (customType: string, data?: any) => {
    appendedEntries.push({ customType, data });
  },
  exec: async () => ({ exitCode: 0, stdout: "", stderr: "" }),
} as any;

registerLilac(mockApi);

const initialProvider = providers.find((p: any) => p.name === "lilac");
assert(initialProvider != null, "provider registered on init");
assert(initialProvider.config.models.length > 0, "models registered on init");

// Trigger session_start to fetch live data
const mockRegistry: ModelRegistry = {
  getApiKeyForProvider: async () => "test-key",
} as any;

for (const handler of handlers.get("session_start") || []) {
  await handler(
    {},
    {
      modelRegistry: mockRegistry,
      ui: mockUi,
      model: { id: "moonshotai/kimi-k2.6", provider: "lilac" },
      sessionManager: { getBranch: () => [] },
    }
  );
}

// Allow micro-tasks to flush
await new Promise((r) => setTimeout(r, 100));

const updatedProvider = providers[providers.length - 1];
assert(updatedProvider != null, "provider re-registered after session_start");

const kimi = updatedProvider.config.models.find((m: any) => m.id === "moonshotai/kimi-k2.6");
assert(kimi != null, "kimi model present after live fetch");
assert(kimi.discount != null, "kimi has discount metadata");
assert(kimi.discount.discountPercent === 25, "kimi discount is 25%");
assert(kimi.discount.creditMultiplier === 0.75, "kimi credit multiplier is 0.75");
// List costs after transformApiModel: input=0.70, output=3.50, cacheRead=0.20
// credit_multiplier 0.75 → effective cost = list * 0.75
assert(kimi.cost.input === 0.525, "kimi input cost = 0.70 * 0.75 = 0.525");
assert(kimi.cost.output === 2.625, "kimi output cost = 3.50 * 0.75 = 2.625");
assert(kimi.cost.cacheRead === 0.15, "kimi cacheRead cost = 0.20 * 0.75 = 0.15");

const gemma = updatedProvider.config.models.find((m: any) => m.id === "google/gemma-4-31b-it");
assert(gemma != null, "gemma model still present (from embedded fallback)");
assert(gemma.discount == null, "gemma has no discount (not in status response)");
assert(gemma.cost.input === 0.11, "gemma cost unchanged (no discount in status)");

// ─── Test 5: session_start sets footer status ─────────────────────────────────

console.log("\n--- Test 5: session_start sets footer status ---");
assert(statuses.get("lilac") === "supply: healthy · sub-discount: 25%", "status set after session_start in correct format");

// ─── Test 6: model_select for lilac model updates status ──────────────────────

console.log("\n--- Test 6: model_select for lilac model ---");
for (const handler of handlers.get("model_select") || []) {
  await handler(
    {
      type: "model_select",
      model: { id: "moonshotai/kimi-k2.6", provider: "lilac" },
      previousModel: undefined,
      source: "set",
    },
    { ui: mockUi }
  );
}
assert(statuses.get("lilac") === "supply: healthy · sub-discount: 25%", "model_select keeps status for lilac model");

// ─── Test 7: model_select for non-lilac model clears status ───────────────────

console.log("\n--- Test 7: model_select for non-lilac model ---");
for (const handler of handlers.get("model_select") || []) {
  await handler(
    {
      type: "model_select",
      model: { id: "claude-sonnet-4", provider: "anthropic" },
      previousModel: undefined,
      source: "set",
    },
    { ui: mockUi }
  );
}
assert(statuses.get("lilac") === undefined, "model_select clears status for non-lilac model");

// ─── Test 8: before_provider_request with fresh cache sets status ─────────────

console.log("\n--- Test 8: before_provider_request with fresh cache ---");
for (const handler of handlers.get("before_provider_request") || []) {
  await handler(
    { type: "before_provider_request", payload: {} },
    {
      ui: mockUi,
      model: { id: "moonshotai/kimi-k2.6", provider: "lilac" },
    }
  );
}
assert(statuses.get("lilac") === "supply: healthy · sub-discount: 25%", "before_provider_request sets status when cache is fresh");

// ─── Test 9: turn_end appends discount entry ──────────────────────────────────

console.log("\n--- Test 9: turn_end appends discount entry ---");
for (const handler of handlers.get("turn_end") || []) {
  await handler(
    { type: "turn_end" },
    {
      ui: mockUi,
      model: { id: "moonshotai/kimi-k2.6", provider: "lilac" },
    }
  );
}
assert(appendedEntries.length > 0, "at least one entry was appended");
const discountEntry = appendedEntries.find(e => e.customType === "lilac-discount");
assert(discountEntry != null, "lilac-discount entry was appended");
assert(discountEntry.data.modelId === "moonshotai/kimi-k2.6", "entry has correct modelId");
assert(discountEntry.data.discountPercent === 25, "entry has correct discountPercent");
assert(discountEntry.data.creditMultiplier === 0.75, "entry has correct creditMultiplier");
assert(discountEntry.data.supplyState === "healthy", "entry has correct supplyState");

// ─── Test 10: formatDiscountStatus fallbacks ───────────────────────────────────

console.log("\n--- Test 10: formatDiscountStatus fallbacks ---");
assert(
  statuses.get("lilac") === "supply: healthy · sub-discount: 25%",
  "known model shows full discount status",
);

// Unknown model (not in latestDiscounts) shows fallback dash
statuses.delete("lilac");
for (const handler of handlers.get("model_select") || []) {
  await handler(
    {
      type: "model_select",
      model: { id: "some/unknown-model", provider: "lilac" },
      previousModel: undefined,
      source: "set",
    },
    { ui: mockUi },
  );
}
assert(statuses.get("lilac") === "supply: —", "unknown model shows fallback dash");

// Known model restores full status
for (const handler of handlers.get("model_select") || []) {
  await handler(
    {
      type: "model_select",
      model: { id: "moonshotai/kimi-k2.6", provider: "lilac" },
      previousModel: undefined,
      source: "set",
    },
    { ui: mockUi },
  );
}
assert(
  statuses.get("lilac") === "supply: healthy · sub-discount: 25%",
  "known model restores full discount status",
);

// ─── Cleanup ──────────────────────────────────────────────────────────────────

globalThis.fetch = originalFetch;
fs.rmSync(tmpHome, { recursive: true, force: true });

console.log("\n--- All tests passed ---\n");
