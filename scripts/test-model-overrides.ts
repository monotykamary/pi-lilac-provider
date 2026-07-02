#!/usr/bin/env node
/**
 * Test for user model overrides (~/.pi/agent/extensions/lilac.json).
 *
 * Verifies, against the REAL exported helpers from index.ts (not a re-implementation):
 *   - applyModelOverride: override wins over the base value; deep-merge compat /
 *     cost / thinkingLevelMap preserves non-overridden fields; scalars replaced;
 *     input model not mutated.
 *   - parseModelOverrides: drops invalid entries (non-object override, non-string
 *     thinkingLevelMap values), keeps valid ones.
 *   - loadConfig: auto-populates a scaffold on a missing file; parses an existing
 *     file's modelOverrides; returns defaults on invalid JSON WITHOUT overwriting
 *     the user's file; returns undefined overrides for a file lacking the key.
 *   - buildModels end-to-end: a user override on a real model id wins over
 *     patch.json (e.g. disable preserve_thinking on kimi-k2.6) while the rest of
 *     compat survives; with no overrides the built models are unchanged; an
 *     override for an unknown id adds no models.
 *
 * Config FS is isolated to a temp HOME so nothing touches the real ~/.pi.
 */

import fs from "fs";
import path from "path";
import { getAgentDir } from "@earendil-works/pi-coding-agent";

const root = path.resolve(import.meta.dirname, "..");
const modelsData = JSON.parse(fs.readFileSync(path.join(root, "models.json"), "utf8"));
const customModelsData = JSON.parse(fs.readFileSync(path.join(root, "custom-models.json"), "utf8"));
const patchData = JSON.parse(fs.readFileSync(path.join(root, "patch.json"), "utf8"));

// Isolate config + cache to a temp agent dir so loadConfig never touches the real ~/.pi.
// Must be set before importing index.ts, which computes CONFIG_PATH at module scope.
const tmpHome = `/tmp/pi-lilac-override-test-${Date.now()}`;
fs.mkdirSync(tmpHome, { recursive: true });
process.env.HOME = tmpHome;
process.env.PI_CODING_AGENT_DIR = path.join(tmpHome, ".pi", "agent");

const {
  buildModels,
  applyModelOverride,
  parseModelOverrides,
  loadConfig,
} = await import("../index.ts");

let passed = 0;
let failed = 0;
function assert(condition: boolean, message: string) {
  if (condition) {
    console.log(`  ✓ ${message}`);
    passed++;
  } else {
    console.error(`  ✗ ${message}`);
    failed++;
  }
}
function eq<T>(actual: T, expected: T, message: string) {
  const ok = JSON.stringify(actual) === JSON.stringify(expected);
  if (ok) {
    console.log(`  ✓ ${message}`);
    passed++;
  } else {
    console.error(`  ✗ ${message}\n     expected: ${JSON.stringify(expected)}\n     actual:   ${JSON.stringify(actual)}`);
    failed++;
  }
}

const KIMI = "moonshotai/kimi-k2.6";
const GLM52 = "zai-org/glm-5.2";
const GLM51 = "zai-org/glm-5.1";

// ─── applyModelOverride ────────────────────────────────────────────────────────

console.log("\n--- applyModelOverride ---");

{
  const base = {
    id: KIMI,
    reasoning: true,
    compat: {
      thinkingFormat: "chat-template",
      supportsDeveloperRole: false,
      chatTemplateKwargs: { thinking: { $var: "thinking.enabled" }, preserve_thinking: true },
    },
    thinkingLevelMap: { low: "low", high: "high" },
  } as any;

  const out = applyModelOverride(base, { compat: { chatTemplateKwargs: { preserve_thinking: false } } } as any);
  assert(out.compat.chatTemplateKwargs.preserve_thinking === false, "override wins over base for a compat flag it sets");
  assert(out.compat.thinkingFormat === "chat-template", "deep-merge compat preserves non-overridden thinkingFormat");
  assert(out.compat.supportsDeveloperRole === false, "deep-merge compat preserves non-overridden supportsDeveloperRole");
  assert((out.compat.chatTemplateKwargs as any).thinking?.$var === "thinking.enabled", "deep-merge chatTemplateKwargs preserves non-overridden thinking key");
  assert((base.compat.chatTemplateKwargs as any).preserve_thinking === true, "does not mutate the input model");
}

{
  const base = { id: GLM52, thinkingLevelMap: { low: "low", high: "high" } } as any;
  const out = applyModelOverride(base, { thinkingLevelMap: { high: "max" } } as any);
  eq(out.thinkingLevelMap, { low: "low", high: "max" }, "deep-merge thinkingLevelMap overrides a single level, keeps the rest");
}

{
  const base = { id: KIMI, reasoning: true, contextWindow: 131072 } as any;
  const out = applyModelOverride(base, { reasoning: false, contextWindow: 65536 } as any);
  assert(out.reasoning === false, "scalar override replaces reasoning");
  assert(out.contextWindow === 65536, "scalar override replaces contextWindow");
}

{
  const base = { id: GLM52, cost: { input: 1, output: 2, cacheRead: 0.2, cacheWrite: 0 } } as any;
  // cost is a plain object -> recursively deep-merged (not on the ModelOverride
  // type, so exercise it via a loose cast).
  const out = applyModelOverride(base, { cost: { input: 5 } } as any);
  eq(out.cost, { input: 5, output: 2, cacheRead: 0.2, cacheWrite: 0 }, "deep-merge cost overrides one field, keeps the rest");
}

{
  // Overriding a { $var } schema object wholesale REPLACES it (no deep-merge into $var)
  const base = { id: KIMI, compat: { chatTemplateKwargs: { thinking: { $var: "thinking.enabled" }, preserve_thinking: true } } } as any;
  const out = applyModelOverride(base, { compat: { chatTemplateKwargs: { thinking: { $var: "thinking.effort" } } } } as any);
  eq(out.compat.chatTemplateKwargs.thinking, { $var: "thinking.effort" }, "overriding a { $var } object replaces it (no merge into $var)");
  assert(out.compat.chatTemplateKwargs.preserve_thinking === true, "sibling chatTemplateKwargs key survives a $var replacement");
}

{
  // Overriding an array (input) replaces it wholesale (no index-wise merge)
  const base = { id: KIMI, input: ["text", "image"] } as any;
  const out = applyModelOverride(base, { input: ["text"] } as any);
  eq(out.input, ["text"], "array override replaces wholesale (no index-wise merge)");
}

// ─── parseModelOverrides ───────────────────────────────────────────────────────

console.log("\n--- parseModelOverrides ---");

eq(parseModelOverrides(undefined), undefined, "undefined input -> undefined");
eq(parseModelOverrides("nope"), undefined, "non-object input -> undefined");
eq(parseModelOverrides({}), undefined, "empty object -> undefined (no valid entries)");
eq(parseModelOverrides({ badId: "not-an-object" } as any), undefined, "non-object override value dropped -> undefined");
eq(parseModelOverrides({ id: 123 } as any), undefined, "non-object override (number) dropped -> undefined");
{
  const r = parseModelOverrides({
    [KIMI]: { compat: { chatTemplateKwargs: { preserve_thinking: false } } },
    bad: 123,
    alsobad: "string",
  });
  assert(r !== undefined && Object.keys(r!).length === 1 && r![KIMI] !== undefined, "keeps valid override, drops invalid ids");
  assert((r![KIMI] as any).compat.chatTemplateKwargs.preserve_thinking === false, "valid override compat preserved");
}
{
  // thinkingLevelMap: only string/null values kept; non-string values dropped
  const r = parseModelOverrides({ [GLM52]: { thinkingLevelMap: { high: "max", bad: 42, off: null } } });
  eq(r, { [GLM52]: { thinkingLevelMap: { high: "max", off: null } } }, "thinkingLevelMap keeps string/null values, drops others");
}
{
  // an override whose fields all parse to nothing is dropped
  const r = parseModelOverrides({ [KIMI]: { thinkingLevelMap: { bad: 42 } } });
  eq(r, undefined, "override with no usable fields is dropped -> undefined");
}

// ─── loadConfig ────────────────────────────────────────────────────────────────

console.log("\n--- loadConfig ---");

const cfgPath = path.join(getAgentDir(), "extensions", "lilac.json");

{
  // Fresh tmpHome: no config file -> loadConfig auto-populates the scaffold and returns defaults
  assert(!fs.existsSync(cfgPath), "scaffold not present before first loadConfig");
  const cfg = loadConfig();
  eq(cfg, { modelOverrides: {}, flexThreshold: null }, "missing file -> defaults (empty modelOverrides, flex off)");
  assert(fs.existsSync(cfgPath), "loadConfig auto-populates the scaffold file on missing file");
  eq(JSON.parse(fs.readFileSync(cfgPath, "utf8")), { modelOverrides: {}, flexThreshold: null }, "scaffold file contains the default shape");
}

{
  // Existing file with valid modelOverrides -> parsed
  fs.writeFileSync(cfgPath, JSON.stringify({
    modelOverrides: { [KIMI]: { compat: { chatTemplateKwargs: { preserve_thinking: false } } } },
  }));
  const cfg = loadConfig();
  assert((cfg.modelOverrides as any)?.[KIMI]?.compat?.chatTemplateKwargs?.preserve_thinking === false, "existing file's modelOverrides parsed");
}

{
  // Existing file WITHOUT a modelOverrides key -> undefined overrides (not an error)
  fs.writeFileSync(cfgPath, JSON.stringify({ unrelatedKey: true }));
  const cfg = loadConfig();
  assert(cfg.modelOverrides === undefined, "file without modelOverrides key -> undefined overrides");
  assert(JSON.parse(fs.readFileSync(cfgPath, "utf8")).unrelatedKey === true, "file without modelOverrides key is not rewritten");
}

{
  // Existing file with invalid JSON -> defaults returned, file left UNTOUCHED (typo not wiped)
  fs.writeFileSync(cfgPath, "not json {{{");
  const cfg = loadConfig();
  eq(cfg, { modelOverrides: {}, flexThreshold: null }, "invalid JSON -> defaults");
  assert(fs.readFileSync(cfgPath, "utf8") === "not json {{{", "invalid file is not overwritten (typo preserved)");
}

// ─── buildModels end-to-end (real models.json + patch.json) ────────────────────

console.log("\n--- buildModels end-to-end ---");

function find(models: any[], id: string): any {
  const m = models.find((x) => x.id === id);
  if (!m) throw new Error(`model ${id} not built`);
  return m;
}

{
  // No overrides -> identical to today: kimi preserve_thinking: true survives patch
  const models = buildModels(modelsData, customModelsData, patchData, {});
  const kimi = find(models, KIMI);
  assert(kimi.compat.chatTemplateKwargs.preserve_thinking === true, "no overrides -> kimi preserve_thinking stays true (patch wins)");
  assert(kimi.compat.thinkingFormat === "chat-template", "no overrides -> kimi thinkingFormat intact");
  const glm52 = find(models, GLM52);
  assert(glm52.compat.chatTemplateKwargs.clear_thinking === false, "no overrides -> glm-5.2 clear_thinking stays false (patch wins)");
}

{
  // User override wins over patch.json: disable preserve_thinking on kimi
  const overrides = { [KIMI]: { compat: { chatTemplateKwargs: { preserve_thinking: false } } } } as any;
  const models = buildModels(modelsData, customModelsData, patchData, overrides);
  const kimi = find(models, KIMI);
  assert(kimi.compat.chatTemplateKwargs.preserve_thinking === false, "override wins over patch: kimi preserve_thinking -> false");
  // deep-merge: the $var thinking keys + thinkingFormat survive
  assert((kimi.compat.chatTemplateKwargs as any).thinking?.$var === "thinking.enabled", "override deep-merges: thinking $var key survives");
  assert((kimi.compat.chatTemplateKwargs as any).enable_thinking?.$var === "thinking.enabled", "override deep-merges: enable_thinking $var key survives");
  assert(kimi.compat.thinkingFormat === "chat-template", "override deep-merges: thinkingFormat survives");
  assert(kimi.compat.supportsDeveloperRole === false, "override deep-merges: supportsDeveloperRole survives");
}

{
  // Override a single thinking level on glm-5.2 without redeclaring the map;
  // the patch-applied clear_thinking flag survives a thinkingLevelMap-only override
  const overrides = { [GLM52]: { thinkingLevelMap: { high: "max" } } } as any;
  const models = buildModels(modelsData, customModelsData, patchData, overrides);
  const glm = find(models, GLM52);
  assert((glm.thinkingLevelMap as any)?.high === "max", "override thinkingLevelMap.high wins over patch");
  assert(glm.compat.chatTemplateKwargs.clear_thinking === false, "non-overridden clear_thinking survives a thinkingLevelMap-only override");
}

{
  // Override on glm-5.1 toggles clear_thinking (patch sets false); other compat survives
  const overrides = { [GLM51]: { compat: { chatTemplateKwargs: { clear_thinking: true } } } } as any;
  const models = buildModels(modelsData, customModelsData, patchData, overrides);
  const glm = find(models, GLM51);
  assert(glm.compat.chatTemplateKwargs.clear_thinking === true, "override wins over patch: glm-5.1 clear_thinking -> true");
  assert((glm.compat.chatTemplateKwargs as any).thinking?.$var === "thinking.enabled", "override deep-merges: glm-5.1 thinking $var key survives");
  assert(glm.compat.zaiToolStream === true, "override deep-merges: glm-5.1 zaiToolStream survives");
}

{
  // Override for an unknown id is a no-op (adds no models)
  const before = buildModels(modelsData, customModelsData, patchData, {});
  const after = buildModels(modelsData, customModelsData, patchData, { "no/such-model": { reasoning: false } } as any);
  eq(after.length, before.length, "override for an unknown id adds no models");
}

// ─── Summary ───────────────────────────────────────────────────────────────────

console.log(`\n${failed === 0 ? "ALL PASS" : `${failed} FAILED`}`);
process.exit(failed === 0 ? 0 : 1);
