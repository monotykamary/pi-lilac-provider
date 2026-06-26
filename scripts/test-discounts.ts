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
 * 8. before_provider_request refreshes discounts with a 60s cache.
 * 9. formatDiscountStatus returns fallbacks when data is missing.
 * 10. applyDiscountInPlace mutates the in-flight model's cost in place,
 *     recomputed from list price so re-applied discounts never compound.
 * 11. before_provider_request mutates the bound (in-flight) model object so the
 *     current turn's cost calc sees the discount in real time.
 * 12. session_start schedules a 5-minute background /status poll to cover idle
 *     sessions; session_shutdown clears it.
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
  applyDiscountInPlace,
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
    { ui: mockUi, model: { id: "moonshotai/kimi-k2.6", provider: "lilac" } }
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
    { ui: mockUi, model: { id: "claude-sonnet-4", provider: "anthropic" } }
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
    { ui: mockUi, model: { id: "some/unknown-model", provider: "lilac" } },
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
    { ui: mockUi, model: { id: "moonshotai/kimi-k2.6", provider: "lilac" } },
  );
}
assert(
  statuses.get("lilac") === "supply: healthy · sub-discount: 25%",
  "known model restores full discount status",
);

// ─── Test 11: re-register stability ───────────────────────────────────────────

console.log("\n--- Test 11: re-register stability ---");

// Simulate multiple rapid re-registers (e.g. before_provider_request fires
// right after session_start's background fetch completes). Each re-register
// should produce a clean model list — no duplicates, no missing models.
//
// In pi's real runtime:
//   1. pi captures state.model (old object) for the in-flight stream
//   2. before_provider_request re-registers (creates new model objects in registry)
//   3. _refreshCurrentModelFromRegistry updates state.model to the new object
//   4. the in-flight stream still uses the old model reference for cost calc
//   5. the next turn picks up the new model with updated costs
//
// So re-registering never corrupts an in-flight request — the stream holds
// its own model reference. We verify the provider produces a stable model list
// across successive re-registers.

const modelIdsAfterSignup = updatedProvider.config.models.map((m: any) => m.id).sort();

// Build fresh model list with known LIST prices for cost verification
const freshModels = updatedProvider.config.models.map((m: any) => ({
  ...m,
  cost: { input: 0.70, output: 3.50, cacheRead: 0.20, cacheWrite: 0 },
  discount: undefined,
}));

// Re-register with changed discounts
const changedDiscounts = new Map([
  ["moonshotai/kimi-k2.6", { supplyState: "low", discountPercent: 10, creditMultiplier: 0.90 }],
]);

const reRegisterModels = applyDiscounts(freshModels, changedDiscounts);
mockApi.registerProvider("lilac", {
  baseUrl: "https://api.getlilac.com/v1",
  apiKey: "$LILAC_API_KEY",
  api: "openai-completions",
  models: reRegisterModels,
});
const afterFirst = providers[providers.length - 1];
const modelIdsAfterFirst = afterFirst.config.models.map((m: any) => m.id).sort();
assert(modelIdsAfterFirst.length === modelIdsAfterSignup.length, "model count preserved after re-register");
assert(JSON.stringify(modelIdsAfterFirst) === JSON.stringify(modelIdsAfterSignup), "model IDs stable after re-register");

// Re-register again with original discounts (simulates rapid successive updates)
// Use the same discount data that session_start fetched (which includes kimi at 25%)
const sessionDiscounts = new Map([
  ["moonshotai/kimi-k2.6", { supplyState: "healthy", discountPercent: 25, creditMultiplier: 0.75 }],
]);
const reRegisterModels2 = applyDiscounts(
  freshModels.map((m: any) => ({ ...m, discount: undefined })),
  sessionDiscounts,
);
mockApi.registerProvider("lilac", {
  baseUrl: "https://api.getlilac.com/v1",
  apiKey: "$LILAC_API_KEY",
  api: "openai-completions",
  models: reRegisterModels2,
});
const afterSecond = providers[providers.length - 1];
const modelIdsAfterSecond = afterSecond.config.models.map((m: any) => m.id).sort();
assert(modelIdsAfterSecond.length === modelIdsAfterSignup.length, "model count preserved after second re-register");
assert(JSON.stringify(modelIdsAfterSecond) === JSON.stringify(modelIdsAfterSignup), "model IDs stable after second re-register");

// Verify the costs reflect the LATEST registration (not stale from an earlier one)
const kimiFinal = afterSecond.config.models.find((m: any) => m.id === "moonshotai/kimi-k2.6");
assert(kimiFinal.discount.discountPercent === 25, "final registration uses correct discount (not stale from first re-register)");
assert(kimiFinal.cost.input === 0.525, "final registration uses correct cost (0.70 * 0.75)");

// ─── Test 12: applyDiscountInPlace mutates the in-flight model ───────────────

console.log("\n--- Test 12: applyDiscountInPlace (in-flight mutation) ---");

// applyDiscountInPlace closes the one-turn gap: registerProvider() refreshes
// the registry for LATER turns, but the model pi already bound for the current
// turn is a separate object whose .cost calculateCost() reads. We mutate THAT
// object in place, recomputed from list price so a changed discount never
// compounds on top of an already-applied factor.

const listModels = [
  { id: "moonshotai/kimi-k2.6", name: "Kimi K2.6", cost: { input: 0.70, output: 3.50, cacheRead: 0.20, cacheWrite: 0 } },
] as any[];

// A model object that is "already bound" for the turn — the same reference pi's
// agent loop holds. It currently carries an OLD discount (factor 0.75).
const inFlight = {
  id: "moonshotai/kimi-k2.6",
  cost: { input: 0.525, output: 2.625, cacheRead: 0.15, cacheWrite: 0 },
};
const inFlightRef = inFlight;

// New discount: factor 0.50 (changed). Must recompute from LIST price (0.70),
// NOT compound on top of the already-applied 0.75 (which would give 0.2625).
const newDiscounts = new Map([
  ["moonshotai/kimi-k2.6", { supplyState: "high", discountPercent: 50, creditMultiplier: 0.50 }],
]);
applyDiscountInPlace(inFlight, listModels, newDiscounts);
assert(inFlight === inFlightRef, "mutates the same object in place (no new object created)");
assert(inFlight.cost.input === 0.35, "recomputed from list price: 0.70 * 0.50 = 0.35 (not compounded 0.2625)");
assert(inFlight.cost.output === 1.75, "output = 3.50 * 0.50 = 1.75");
assert(inFlight.cost.cacheRead === 0.1, "cacheRead = 0.20 * 0.50 = 0.1");
assert(inFlight.cost.cacheWrite === 0, "cacheWrite left untouched (Lilac does not discount it)");

// Idempotency + no-compounding: applying the same factor again yields the same
// values (always recomputed from list price, never compounding).
applyDiscountInPlace(inFlight, listModels, newDiscounts);
assert(inFlight.cost.input === 0.35, "re-applying same factor is idempotent (no compounding)");

// Discount removed (model no longer in /status) → reverts to list price.
const emptyDiscounts = new Map();
applyDiscountInPlace(inFlight, listModels, emptyDiscounts);
assert(inFlight.cost.input === 0.70, "removed discount reverts to list price (factor 1)");
assert(inFlight.cost.output === 3.50, "removed discount: output reverts to list price");

// null discounts → list price, no throw.
applyDiscountInPlace(inFlight, listModels, null);
assert(inFlight.cost.input === 0.70, "null discounts → list price");

// Unknown model id (not in listModels) → no-op, no throw.
const unknown = { id: "some/unknown-model", cost: { input: 9.99, output: 9.99, cacheRead: 9.99, cacheWrite: 0 } };
applyDiscountInPlace(unknown, listModels, newDiscounts);
assert(unknown.cost.input === 9.99, "unknown model id → no mutation (no list-price match)");

// undefined model → no throw.
applyDiscountInPlace(undefined, listModels, newDiscounts);
assert(true, "undefined model does not throw");

// ─── Test 13: before_provider_request mutates the bound (in-flight) model ────

console.log("\n--- Test 13: before_provider_request mutates the bound model ---");

// Simulate the object pi's agent loop bound for the turn. It holds an OUTDATED
// cost (list price, as if no discount had been applied yet). This same reference
// is what calculateCost() reads for the current turn — so the handler MUST mutate
// it in place, not replace it. latestDiscounts currently holds kimi at 0.75 (from
// Test 4), and the TTL is fresh, so the handler takes the within-TTL path (which
// previously did NOT touch the in-flight model at all).
const boundModel = {
  id: "moonshotai/kimi-k2.6",
  provider: "lilac",
  cost: { input: 0.70, output: 3.50, cacheRead: 0.20, cacheWrite: 0 }, // list price (stale)
};
const boundRef = boundModel;

for (const handler of handlers.get("before_provider_request") || []) {
  await handler(
    { type: "before_provider_request", payload: {} },
    {
      ui: mockUi,
      model: boundModel,
    }
  );
}

assert(boundModel === boundRef, "handler did not replace the bound model object (same reference)");
assert(boundModel.cost.input === 0.525, "bound model cost mutated in place to 0.70 * 0.75 = 0.525");
assert(boundModel.cost.output === 2.625, "bound model output mutated to 3.50 * 0.75 = 2.625");
assert(boundModel.cost.cacheRead === 0.15, "bound model cacheRead mutated to 0.20 * 0.75 = 0.15");

// ─── Test 14: session_start clears status after switch during fetch ─────────

console.log("\n--- Test 14: session_start clears status after switch ---");

// Regression: the session_start background fetch resolves up to ~8s after hook
// start. If the user switches to a non-lilac model during that window, the
// deferred callback must NOT re-paint the stale captured lilac model's discount
// over the clear that model_select issued. syncStatus() reads the LIVE ctx.model
// (a lazy getter in production), so mutating startCtx.model mid-flight mimics
// the session model changing and the deferred paint clears instead.
globalThis.fetch = mockFetch({
  "/models": { body: { data: [] } }, // no live models → liveModels null
  "/status": {
    body: {
      models: [{
        id: "moonshotai/kimi-k2.6",
        current_subscription_supply_state: "healthy",
        current_subscription_discount_percent: 25,
        current_subscription_credit_multiplier: "0.75",
      }],
    },
  },
}) as any;

statuses.clear();
const startCtx: any = {
  modelRegistry: mockRegistry,
  ui: mockUi,
  model: { id: "moonshotai/kimi-k2.6", provider: "lilac" }, // start on lilac
  sessionManager: { getBranch: () => [] },
};
for (const handler of handlers.get("session_start") || []) {
  await handler({}, startCtx);
}
// Immediate paint reflects the lilac model (still selected at hook start).
assert(
  statuses.get("lilac") === "supply: healthy · sub-discount: 25%",
  "immediate session_start paint shows lilac discount",
);

// While the background fetch is in flight, the user switches to a non-lilac model.
startCtx.model = { id: "claude-sonnet-4", provider: "anthropic" };

// Flush the deferred .then() chain (resolveApiKey → fetch → syncStatus).
await new Promise((r) => setTimeout(r, 100));

assert(
  statuses.get("lilac") === undefined,
  "deferred session_start clears status after switch to non-lilac (no stale re-paint)",
);

// ─── Test 15: before_provider_request clears status after switch ─────────────

console.log("\n--- Test 15: before_provider_request clears status after switch ---");

// Regression: every 60s (when the discount TTL expires) before_provider_request
// awaits a fresh /status fetch. If the user switches to a non-lilac model during
// that await, the post-await paint must clear (live model) instead of re-painting
// the stale captured in-flight lilac model. Cost mutation still targets the
// captured in-flight model (Tests 12-13); only the DISPLAY follows the live model.

// Force the 60s TTL to look expired so the fetch path executes. Offset 120s for
// a clear 2x margin over the 60s TTL (the original used 60s over a 30s TTL).
const realDateNow = Date.now;
Date.now = () => realDateNow.call(Date) + 120000;
try {
  globalThis.fetch = mockFetch({
    "/status": {
      body: {
        // CHANGED discount (low/10%) vs the cached healthy/25% → exercises the
        // registerProvider + syncStatus path, not just the unchanged early-return.
        models: [{
          id: "moonshotai/kimi-k2.6",
          current_subscription_supply_state: "low",
          current_subscription_discount_percent: 10,
          current_subscription_credit_multiplier: "0.90",
        }],
      },
    },
  }) as any;

  statuses.clear();
  // Live model holder; reassigning .model mid-await mimics ctx.model being a
  // lazy getter to the session model.
  const bprCtx: any = {
    ui: mockUi,
    model: { id: "moonshotai/kimi-k2.6", provider: "lilac" },
  };

  // Kick off before_provider_request; it captures inFlightModel=lilac, mutates
  // cost, paints, then awaits the fetch (TTL expired).
  const bprPromise = (async () => {
    for (const handler of handlers.get("before_provider_request") || []) {
      await handler({ type: "before_provider_request", payload: {} }, bprCtx);
    }
  })();

  // Switch to non-lilac while the fetch is in flight (before the await resumes).
  bprCtx.model = { id: "claude-sonnet-4", provider: "anthropic" };

  await bprPromise;

  assert(
    statuses.get("lilac") === undefined,
    "post-await syncStatus clears after switch to non-lilac (no stale re-paint)",
  );
} finally {
  Date.now = realDateNow;
}

// ─── Test 16: session_start shows new lilac model after switch ───────────────

console.log("\n--- Test 16: session_start shows new lilac model after switch ---");

// Companion to Test 14: switching to a DIFFERENT lilac model during the fetch
// must paint the NEW model's discount, proving syncStatus reads the live model
// (not a stale capture that would show the old model's discount).
globalThis.fetch = mockFetch({
  "/models": { body: { data: [] } },
  "/status": {
    body: {
      models: [
        {
          id: "moonshotai/kimi-k2.6",
          current_subscription_supply_state: "healthy",
          current_subscription_discount_percent: 25,
          current_subscription_credit_multiplier: "0.75",
        },
        {
          id: "zai-org/glm-5.1",
          current_subscription_supply_state: "high",
          current_subscription_discount_percent: 50,
          current_subscription_credit_multiplier: "0.50",
        },
      ],
    },
  },
}) as any;

statuses.clear();
const startCtx2: any = {
  modelRegistry: mockRegistry,
  ui: mockUi,
  model: { id: "moonshotai/kimi-k2.6", provider: "lilac" }, // start on kimi
  sessionManager: { getBranch: () => [] },
};
for (const handler of handlers.get("session_start") || []) {
  await handler({}, startCtx2);
}
// Switch to a different lilac model (glm) during the in-flight fetch.
startCtx2.model = { id: "zai-org/glm-5.1", provider: "lilac" };
await new Promise((r) => setTimeout(r, 100));
assert(
  statuses.get("lilac") === "supply: high · sub-discount: 50%",
  "deferred session_start paints the NEW lilac model's discount (glm), not the stale kimi capture",
);

// ─── Test 17: session_start schedules a 5-min idle poll; shutdown clears it ─

console.log("\n--- Test 17: session_start schedules idle poll; session_shutdown clears it ---");

// Lilac refreshes discounts ~every 10 minutes, so session_start polls /status
// every 5 min (half the refresh window) to cover idle sessions (turn fetches
// only run when the user sends a message), and session_shutdown must clear it so
// it neither leaks nor keeps the process alive. Wrap the global timer APIs to
// capture the scheduled delay + handle, then confirm shutdown clears it.

const realSetInterval = globalThis.setInterval.bind(globalThis);
const realClearInterval = globalThis.clearInterval.bind(globalThis);
let scheduledDelay: number | null = null;
let scheduledHandle: ReturnType<typeof setInterval> | null = null;
const clearedHandles = new Set<ReturnType<typeof setInterval>>();

globalThis.setInterval = ((fn: (...args: any[]) => void, delay?: number, ...rest: any[]) => {
  scheduledDelay = delay ?? null;
  const h = realSetInterval(fn, delay as any, ...rest);
  scheduledHandle = h;
  return h;
}) as any;
globalThis.clearInterval = ((handle: ReturnType<typeof setInterval>) => {
  clearedHandles.add(handle);
  return realClearInterval(handle);
}) as any;

try {
  // Benign fetch mock so the session_start fire-and-forget /models + /status
  // fetch doesn't hit the network. (The poll itself never fires — 5 min — so
  // only the startup fetch needs mocking here.)
  globalThis.fetch = mockFetch({
    "/models": { body: { data: [] } },
    "/status": {
      body: {
        models: [
          {
            id: "moonshotai/kimi-k2.6",
            current_subscription_supply_state: "healthy",
            current_subscription_discount_percent: 25,
            current_subscription_credit_multiplier: "0.75",
          },
        ],
      },
    },
  }) as any;

  const pollCtx: any = {
    modelRegistry: mockRegistry,
    ui: mockUi,
    model: { id: "moonshotai/kimi-k2.6", provider: "lilac" },
    sessionManager: { getBranch: () => [] },
  };

  for (const handler of handlers.get("session_start") || []) {
    await handler({}, pollCtx);
  }

  assert(
    scheduledDelay === 5 * 60 * 1000,
    "session_start schedules a 5-minute (300000ms) /status poll for idle sessions",
  );
  assert(scheduledHandle !== null, "poll interval handle was captured");
  const capturedHandle = scheduledHandle;

  for (const handler of handlers.get("session_shutdown") || []) {
    await handler({}, pollCtx);
  }

  assert(
    capturedHandle !== null && clearedHandles.has(capturedHandle),
    "session_shutdown clears the scheduled poll interval (no leak / process-hang)",
  );
} finally {
  // Restore globals; clear any interval this test scheduled before it can fire.
  if (scheduledHandle) realClearInterval(scheduledHandle);
  globalThis.setInterval = realSetInterval as any;
  globalThis.clearInterval = realClearInterval as any;
}

// ─── Cleanup ──────────────────────────────────────────────────────────────────

globalThis.fetch = originalFetch;
fs.rmSync(tmpHome, { recursive: true, force: true });

console.log("\n--- All tests passed ---\n");
