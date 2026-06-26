#!/usr/bin/env node
/**
 * Wire-level test for preserved-thinking (full-history reasoning) flags.
 *
 * Verifies, against the REAL pi-ai streamSimple + a stubbed fetch, that the
 * `chat_template_kwargs` each Lilac reasoning model puts on the wire include the
 * preservation flags configured in patch.json — and that those static flags
 * coexist with the { $var }-resolved thinking keys at every thinking level.
 *
 * Background: reasoning models trim older assistant reasoning across turns by
 * default (each vendor's template default). Two template-level flags opt into
 * full-history preservation (confirmed in each model's HuggingFace chat
 * template; Kimi K2.6 and GLM 5.2 additionally E2E-verified on the sibling
 * neuralwatt provider's 3-turn / two-20-digit-number recall test):
 *
 *   Kimi K2.6    → preserve_thinking: true   (template: "if preserve_thinking, keep -1 ... retain reasoning")
 *   GLM 5.1      → clear_thinking: false      (same clear_thinking mechanism as GLM 5.2)
 *   GLM 5.2      → clear_thinking: false      (template: keep reasoning when "clear_thinking is defined and not clear_thinking")
 *
 * Lilac uses pi-ai's `chat-template` thinkingFormat, which calls
 * buildChatTemplateKwargs → resolveChatTemplateKwargValue. Static primitive
 * kwargs pass through verbatim; only { $var } objects are resolved against the
 * turn's thinking state. So the preserve flags ride onto the wire as plain
 * booleans next to the { $var } thinking/enable_thinking keys — no onPayload
 * hook needed (unlike neuralwatt, which drives the openai reasoning_effort path
 * and must inject via onPayload because the two paths are mutually exclusive).
 *
 * Gemma 4 and MiniMax M2.7/M3 expose NO family-wide preserve flag (their HF
 * templates read only enable_thinking / thinking_mode + reasoning_content), so
 * no flag is added for them here — asserted as a regression guard.
 *
 * Run: node scripts/test-preserved-thinking.ts
 */
import fs from "fs";
import path from "path";
import { pathToFileURL } from "url";

const HERE = path.dirname(new URL(import.meta.url).pathname);

// ─── Faithful replica of index.ts applyPatch / buildModels ────────────────────
// Mirrors the provider's model pipeline so the test exercises the same objects
// the runtime registers. (Kept inline so the test has no TS-import dependency
// on index.ts, which pulls in pi-coding-agent types.)

function applyPatch(model, patch) {
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
  if (patch.compat) result.compat = { ...(result.compat || {}), ...patch.compat };
  if (patch.thinkingLevelMap !== undefined) result.thinkingLevelMap = patch.thinkingLevelMap;
  if (!result.reasoning && result.compat?.thinkingFormat) delete result.compat.thinkingFormat;
  if (!result.reasoning && result.thinkingLevelMap) delete result.thinkingLevelMap;
  if (result.compat && Object.keys(result.compat).length === 0) delete result.compat;
  return result;
}

function buildModels(base, custom, patch) {
  const map = new Map();
  for (const m of base) map.set(m.id, m);
  for (const [id, p] of Object.entries(patch)) {
    const ex = map.get(id);
    if (ex) map.set(id, applyPatch(ex, p));
  }
  for (const m of custom) {
    const ex = map.get(m.id);
    const p = patch[m.id];
    if (ex && p) map.set(m.id, applyPatch(m, p));
    else if (ex) map.set(m.id, m);
    else if (p) map.set(m.id, applyPatch(m, p));
    else map.set(m.id, m);
  }
  return Array.from(map.values());
}

// ─── Load REAL pi-ai streamSimple from the global pi install ──────────────────
// Imported by absolute path so its relative deps (openai, ../models.js, ...) and
// the `openai` SDK resolve from the global pi node_modules tree.
const PI_AI_API = path.join(
  os_home_pi_ai(),
  "dist/api/openai-completions.js",
);
function os_home_pi_ai() {
  // Resolve the pi-ai package shipped inside the globally-installed pi agent.
  const candidates = [
    "/Users/monotykamary/.npm-global/lib/node_modules/@earendil-works/pi-coding-agent/node_modules/@earendil-works/pi-ai",
  ];
  for (const c of candidates) if (fs.existsSync(c)) return c;
  throw new Error("Could not locate pi-ai package in the global pi install.");
}

const { streamSimple } = await import(pathToFileURL(PI_AI_API).href);

// ─── Build the exact model objects the provider registers ────────────────────
const embedded = JSON.parse(fs.readFileSync(path.join(HERE, "..", "models.json"), "utf8"));
const custom = JSON.parse(fs.readFileSync(path.join(HERE, "..", "custom-models.json"), "utf8"));
const patch = JSON.parse(fs.readFileSync(path.join(HERE, "..", "patch.json"), "utf8"));
const models = new Map(buildModels(embedded, custom, patch).map((m) => [m.id, m]));

// ─── Stub fetch to capture the request body (fires before response parsing) ──
const captured = [];
const originalFetch = globalThis.fetch;
globalThis.fetch = async (url, init) => {
  captured.push({ url: String(url), body: init?.body ?? null });
  // Minimal SSE response that ends immediately — enough for the OpenAI SDK to
  // construct the stream; we only need the request body, captured above.
  return new Response(new ReadableStream({ start(c) { c.close(); } }), {
    headers: { "content-type": "text/event-stream" },
  });
};

async function wire(modelId, reasoning) {
  const model = models.get(modelId);
  if (!model) throw new Error(`unknown model: ${modelId}`);
  captured.length = 0;
  const ctx = { messages: [{ role: "user", content: [{ type: "text", text: "hi" }] }] };
  const s = streamSimple(
    { ...model, provider: "lilac", api: "openai-completions", baseUrl: "https://api.getlilac.com/v1" },
    ctx,
    { apiKey: "sk-test", reasoning },
  );
  // The fetch fires inside stream()'s async IIFE; poll for it.
  const deadline = Date.now() + 2000;
  while (captured.length === 0 && Date.now() < deadline) await new Promise((r) => setTimeout(r, 20));
  try { s.end?.(); } catch {}
  const hit = captured.find((c) => c.url.includes("/chat/completions"));
  if (!hit) throw new Error(`no /chat/completions request captured for ${modelId} (${reasoning})`);
  return JSON.parse(hit.body);
}

// ─── Assertions ───────────────────────────────────────────────────────────────
let failures = 0;
function eq(actual, expected, msg) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  const ok = a === e;
  console.log(`${ok ? "✓" : "✗"} ${msg}`);
  if (!ok) {
    failures++;
    console.log(`    expected ${e}`);
    console.log(`    actual   ${a}`);
  }
}
function truthy(actual, msg) {
  const ok = !!actual;
  console.log(`${ok ? "✓" : "✗"} ${msg}`);
  if (!ok) {
    failures++;
    console.log(`    expected truthy, got ${JSON.stringify(actual)}`);
  }
}
function falsy(actual, msg) {
  const ok = !actual;
  console.log(`${ok ? "✓" : "✗"} ${msg}`);
  if (!ok) {
    failures++;
    console.log(`    expected falsy/undefined, got ${JSON.stringify(actual)}`);
  }
}

console.log("\n=== patch.json data ===");
const kimiPatch = patch["moonshotai/kimi-k2.6"]?.compat?.chatTemplateKwargs;
eq(kimiPatch?.preserve_thinking, true, "kimi-k2.6 patch sets preserve_thinking: true");
const glm51Patch = patch["zai-org/glm-5.1"]?.compat?.chatTemplateKwargs;
eq(glm51Patch?.clear_thinking, false, "glm-5.1 patch sets clear_thinking: false");
const glm52Patch = patch["zai-org/glm-5.2"]?.compat?.chatTemplateKwargs;
eq(glm52Patch?.clear_thinking, false, "glm-5.2 patch sets clear_thinking: false");

console.log("\n=== Kimi K2.6 on the wire (real pi-ai) ===");
{
  const high = await wire("moonshotai/kimi-k2.6", "high");
  eq(high.chat_template_kwargs, { thinking: true, enable_thinking: true, preserve_thinking: true },
    "kimi @ high → thinking+enable_thinking true AND preserve_thinking true");
  const off = await wire("moonshotai/kimi-k2.6", "off");
  eq(off.chat_template_kwargs, { thinking: false, enable_thinking: false, preserve_thinking: true },
    "kimi @ off → thinking false but preserve_thinking still true (level-independent)");
  const minimal = await wire("moonshotai/kimi-k2.6", "minimal");
  eq(minimal.chat_template_kwargs, { thinking: true, enable_thinking: true, preserve_thinking: true },
    "kimi @ minimal → preserve_thinking present at every level");
}

console.log("\n=== GLM 5.2 on the wire (real pi-ai) ===");
{
  const high = await wire("zai-org/glm-5.2", "high");
  eq(high.chat_template_kwargs, { enable_thinking: true, reasoning_effort: "high", clear_thinking: false },
    "glm-5.2 @ high → enable_thinking+reasoning_effort AND clear_thinking false");
  const xhigh = await wire("zai-org/glm-5.2", "xhigh");
  eq(xhigh.chat_template_kwargs, { enable_thinking: true, reasoning_effort: "max", clear_thinking: false },
    "glm-5.2 @ xhigh → reasoning_effort max, clear_thinking false");
  const off = await wire("zai-org/glm-5.2", "off");
  eq(off.chat_template_kwargs, { enable_thinking: false, clear_thinking: false },
    "glm-5.2 @ off → reasoning_effort omitted (omitWhenOff), clear_thinking false persists");
}

console.log("\n=== GLM 5.1 on the wire (real pi-ai) ===");
{
  const high = await wire("zai-org/glm-5.1", "high");
  eq(high.chat_template_kwargs, { thinking: true, enable_thinking: true, clear_thinking: false },
    "glm-5.1 @ high → thinking+enable_thinking true AND clear_thinking false");
  const off = await wire("zai-org/glm-5.1", "off");
  eq(off.chat_template_kwargs, { thinking: false, enable_thinking: false, clear_thinking: false },
    "glm-5.1 @ off → thinking false but clear_thinking false persists");
}

console.log("\n=== Gemma 4 / MiniMax (no family-wide preserve flag — regression guard) ===");
{
  const gemma = await wire("google/gemma-4-31b-it", "high");
  eq(gemma.chat_template_kwargs, { thinking: true, enable_thinking: true },
    "gemma-4 @ high → only thinking/enable_thinking (no preserve/clear flag)");
  falsy(gemma.chat_template_kwargs?.preserve_thinking, "gemma-4 has no preserve_thinking");
  falsy(gemma.chat_template_kwargs?.clear_thinking, "gemma-4 has no clear_thinking");

  const m3 = await wire("minimaxai/minimax-m3", "high");
  eq(m3.chat_template_kwargs, { thinking_mode: "enabled" },
    "minimax-m3 @ high → only thinking_mode (no preserve/clear flag)");
  falsy(m3.chat_template_kwargs?.preserve_thinking, "minimax-m3 has no preserve_thinking");

  const m27 = await wire("minimaxai/minimax-m2.7", "high");
  eq(m27.chat_template_kwargs, { thinking: true, enable_thinking: true },
    "minimax-m2.7 @ high → only thinking/enable_thinking (no preserve/clear flag)");
  falsy(m27.chat_template_kwargs?.clear_thinking, "minimax-m2.7 has no clear_thinking");
}

globalThis.fetch = originalFetch;

console.log(`\n${failures === 0 ? "ALL PASS" : `${failures} FAILURE(S)`}`);
if (failures > 0) process.exit(1);
