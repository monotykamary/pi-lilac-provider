#!/usr/bin/env node
/**
 * Tests for the lilac-flex feature: discount-threshold gating.
 *
 * Verifies, against the REAL exported helpers and registered handlers/commands
 * from index.ts (not a re-implementation):
 *   - parseFlexThreshold: number in [0,100] passes; null/"off"/"" -> null;
 *     out-of-range / non-numeric / object -> undefined.
 *   - loadConfig: parses flexThreshold from an existing file; missing key ->
 *     undefined (off); invalid value -> undefined while modelOverrides survive.
 *   - updateConfig: sets flexThreshold, persists to disk, and PRESERVES the
 *     user's modelOverrides (doesn't clobber them); normalizes the file shape.
 *   - input gate (integration via the registered handler): blocks (handled) when
 *     discount < threshold; allows (continue) when >= threshold, when flex is
 *     off, for non-lilac models, for non-interactive sources, and for models
 *     with no discount entry (treated as 0% -> blocked when flex on).
 *   - /lilac-flex command (integration via the registered handler): direct arg
 *     ("/lilac-flex 75", "off", "50%"), picker presets, custom input, cancel,
 *     invalid arg, and non-interactive guard.
 *
 * Config FS + discount cache are isolated to a temp HOME so nothing touches the
 * real ~/.pi.
 */

import fs from "fs";
import os from "os";
import path from "path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// Isolate config + cache to a temp HOME so loadConfig/cacheDiscounts never touch
// the real ~/.pi. Must be set before importing index.ts, which computes
// CONFIG_PATH / CACHE_PATH at module scope.
const tmpHome = `/tmp/pi-lilac-flex-test-${Date.now()}`;
fs.mkdirSync(tmpHome, { recursive: true });
process.env.HOME = tmpHome;

const {
  default: registerLilac,
  parseFlexThreshold,
  loadConfig,
  getConfig,
  updateConfig,
  cacheDiscounts,
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

const cfgPath = path.join(os.homedir(), ".pi", "agent", "extensions", "lilac.json");
const KIMI = "moonshotai/kimi-k2.6";

// ─── parseFlexThreshold ───────────────────────────────────────────────────────

console.log("\n--- parseFlexThreshold ---");
eq(parseFlexThreshold(null), null, "null -> null (off)");
eq(parseFlexThreshold(0), 0, "0 -> 0");
eq(parseFlexThreshold(100), 100, "100 -> 100");
eq(parseFlexThreshold(75), 75, "75 -> 75");
eq(parseFlexThreshold(50.5), 50.5, "fractional number preserved");
eq(parseFlexThreshold(-1), undefined, "negative -> undefined");
eq(parseFlexThreshold(101), undefined, ">100 -> undefined");
eq(parseFlexThreshold(NaN), undefined, "NaN -> undefined");
eq(parseFlexThreshold("75"), 75, "numeric string -> number");
// Note: parseFlexThreshold is strict for the JSON file (number or off/none/null).
// Trailing-% stripping lives in parseFlexArg for typed command/input args.
eq(parseFlexThreshold("50%"), undefined, "string with trailing % -> undefined (strict; parseFlexArg handles %)");
eq(parseFlexThreshold("off"), null, "'off' -> null");
eq(parseFlexThreshold("OFF"), null, "'OFF' (case-insensitive) -> null");
eq(parseFlexThreshold("none"), null, "'none' -> null");
eq(parseFlexThreshold(""), null, "empty string -> null");
eq(parseFlexThreshold("abc"), undefined, "non-numeric string -> undefined");
eq(parseFlexThreshold(true), undefined, "boolean -> undefined");
eq(parseFlexThreshold({ a: 1 }), undefined, "object -> undefined");
eq(parseFlexThreshold([75]), undefined, "array -> undefined");

// ─── loadConfig: flexThreshold parsing ────────────────────────────────────────

console.log("\n--- loadConfig: flexThreshold ---");

// The config dir doesn't exist yet in the temp HOME; direct writes below bypass
// loadConfig's missing-file scaffold, so create it first.
fs.mkdirSync(path.dirname(cfgPath), { recursive: true });

{
  fs.writeFileSync(cfgPath, JSON.stringify({ flexThreshold: 75 }));
  const cfg = loadConfig();
  assert(cfg.flexThreshold === 75, "file with flexThreshold:75 -> 75");
}

{
  fs.writeFileSync(cfgPath, JSON.stringify({ flexThreshold: null }));
  const cfg = loadConfig();
  assert(cfg.flexThreshold === null, "file with flexThreshold:null -> null");
}

{
  fs.writeFileSync(cfgPath, JSON.stringify({ unrelated: true }));
  const cfg = loadConfig();
  assert(cfg.flexThreshold === undefined, "file without flexThreshold key -> undefined (off)");
}

{
  // Invalid flexThreshold falls back to undefined, but modelOverrides still parse.
  fs.writeFileSync(cfgPath, JSON.stringify({
    flexThreshold: "not-a-number",
    modelOverrides: { [KIMI]: { compat: { chatTemplateKwargs: { preserve_thinking: false } } } },
  }));
  const cfg = loadConfig();
  assert(cfg.flexThreshold === undefined, "invalid flexThreshold -> undefined (off)");
  assert((cfg.modelOverrides as any)?.[KIMI]?.compat?.chatTemplateKwargs?.preserve_thinking === false, "modelOverrides still parsed alongside invalid flexThreshold");
}

// ─── updateConfig: round-trip + modelOverrides preservation ───────────────────

console.log("\n--- updateConfig ---");

{
  // Start from a file with real modelOverrides, then set flex via updateConfig:
  // the overrides must survive (read-merge-write, not clobber).
  fs.writeFileSync(cfgPath, JSON.stringify({
    modelOverrides: { [KIMI]: { compat: { chatTemplateKwargs: { preserve_thinking: false } } } },
  }));
  updateConfig((c) => ({ ...c, flexThreshold: 75 }));
  const cfg = getConfig();
  assert(cfg.flexThreshold === 75, "updateConfig sets flexThreshold to 75");
  assert((cfg.modelOverrides as any)?.[KIMI]?.compat?.chatTemplateKwargs?.preserve_thinking === false, "updateConfig preserves existing modelOverrides");
  const onDisk = JSON.parse(fs.readFileSync(cfgPath, "utf8"));
  assert(onDisk.flexThreshold === 75, "file on disk has flexThreshold 75");
  assert(onDisk.modelOverrides[KIMI].compat.chatTemplateKwargs.preserve_thinking === false, "file on disk preserves modelOverrides");
}

{
  // Setting to null persists null and keeps the modelOverrides scaffold shape.
  updateConfig((c) => ({ ...c, flexThreshold: null }));
  const onDisk = JSON.parse(fs.readFileSync(cfgPath, "utf8"));
  assert(onDisk.flexThreshold === null, "updateConfig null -> file has flexThreshold null");
  assert(onDisk.modelOverrides[KIMI].compat.chatTemplateKwargs.preserve_thinking === false, "modelOverrides still preserved after setting flex off");
}

// ─── Setup: register extension + seed latestDiscounts from cache ──────────────

console.log("\n--- input gate ---");

// Seed the discount cache BEFORE registering so the extension's init picks it up
// via loadCachedDiscounts() (latestDiscounts). Kimi at 25%, glm-5.1 uncached.
cacheDiscounts(new Map([
  [KIMI, { supplyState: "medium", discountPercent: 25, creditMultiplier: 0.75 }],
]));

const handlers = new Map<string, ((...args: any[]) => any)[]>();
const commands = new Map<string, { handler: (...args: any[]) => any }>();

const mockApi: ExtensionAPI = {
  registerProvider: () => {},
  on: (event: string, handler: (...args: any[]) => any) => {
    if (!handlers.has(event)) handlers.set(event, []);
    handlers.get(event)!.push(handler);
  },
  registerCommand: (name: string, opts: any) => {
    commands.set(name, opts);
  },
  appendEntry: () => {},
  exec: async () => ({ exitCode: 0, stdout: "", stderr: "" }),
} as any;

registerLilac(mockApi);

const inputHandlers = handlers.get("input") ?? [];
assert(inputHandlers.length === 1, "exactly one input handler registered");

function runInput(source: string, model: any) {
  const notifications: { msg: string; level: string }[] = [];
  const ctx = {
    model,
    ui: {
      notify: (msg: string, level: string) => notifications.push({ msg, level }),
      setStatus: () => {},
      theme: { fg: (_c: string, t: string) => t },
    },
  };
  return Promise.all(inputHandlers.map((h) => h({ source }, ctx))).then((results) => ({
    result: results[0],
    notifications,
  }));
}

// Reset flex to a known state for the gate suite.
updateConfig((c) => ({ ...c, flexThreshold: null }));

// flex off -> always allow
{
  const { result, notifications } = await runInput("interactive", { id: KIMI, provider: "lilac" });
  eq(result, { action: "continue" }, "flex off -> continue");
  assert(notifications.length === 0, "flex off -> no notify");
}

// flex 75, kimi at 25% -> blocked
updateConfig((c) => ({ ...c, flexThreshold: 75 }));
{
  const { result, notifications } = await runInput("interactive", { id: KIMI, provider: "lilac" });
  eq(result, { action: "handled" }, "25% < 75% threshold -> handled (blocked)");
  assert(notifications.length === 1, "blocked -> one notify");
  assert(notifications[0].level === "warning", "blocked notify is a warning");
  assert(notifications[0].msg.includes("25%") && notifications[0].msg.includes("75%"), "blocked notify mentions current and threshold");
}

// flex 20, kimi at 25% -> allowed (>= threshold)
updateConfig((c) => ({ ...c, flexThreshold: 20 }));
{
  const { result, notifications } = await runInput("interactive", { id: KIMI, provider: "lilac" });
  eq(result, { action: "continue" }, "25% >= 20% threshold -> continue");
  assert(notifications.length === 0, "allowed -> no notify");
}

// flex 75, non-lilac model -> allowed (gate only applies to lilac)
updateConfig((c) => ({ ...c, flexThreshold: 75 }));
{
  const { result, notifications } = await runInput("interactive", { id: "anthropic/claude", provider: "anthropic" });
  eq(result, { action: "continue" }, "non-lilac model -> continue");
  assert(notifications.length === 0, "non-lilac model -> no notify");
}

// flex 75, rpc source -> allowed (only interactive is gated)
{
  const { result } = await runInput("rpc", { id: KIMI, provider: "lilac" });
  eq(result, { action: "continue" }, "rpc source -> continue (not gated)");
}

// flex 75, extension source -> allowed (no loop)
{
  const { result } = await runInput("extension", { id: KIMI, provider: "lilac" });
  eq(result, { action: "continue" }, "extension source -> continue (no loop)");
}

// flex 75, lilac model with no discount entry -> 0% -> blocked, mentions list price
{
  const { result, notifications } = await runInput("interactive", { id: "zai-org/glm-5.1", provider: "lilac" });
  eq(result, { action: "handled" }, "uncached lilac model -> handled (0% < 75%)");
  assert(notifications.length === 1, "uncached model blocked -> one notify");
  assert(notifications[0].msg.includes("no discount on this model"), "uncached model notify mentions list price");
}

// flex exactly equal to discount -> allowed (>= is inclusive)
updateConfig((c) => ({ ...c, flexThreshold: 25 }));
{
  const { result } = await runInput("interactive", { id: KIMI, provider: "lilac" });
  eq(result, { action: "continue" }, "25% >= 25% threshold (inclusive) -> continue");
}

// ─── /lilac-flex command ──────────────────────────────────────────────────────

console.log("\n--- /lilac-flex command ---");

assert(commands.has("lilac-flex"), "/lilac-flex command registered");
const cmd = commands.get("lilac-flex")!;

function runCmd(args: string, opts: { select?: string; input?: string; hasUI?: boolean; model?: any } = {}) {
  const notifications: { msg: string; level: string }[] = [];
  const ctx = {
    hasUI: opts.hasUI ?? true,
    model: opts.model ?? { id: KIMI, provider: "lilac" },
    ui: {
      notify: (msg: string, level: string) => notifications.push({ msg, level }),
      setStatus: () => {},
      theme: { fg: (_c: string, t: string) => t },
      select: async (_title: string, _labels: string[]) => opts.select,
      input: async (_title: string, _placeholder: string) => opts.input,
    },
  };
  return cmd.handler(args, ctx).then(() => ({ notifications, flexThreshold: getConfig().flexThreshold ?? null }));
}

// direct numeric arg
{
  const { flexThreshold } = await runCmd("75");
  assert(flexThreshold === 75, "/lilac-flex 75 -> flexThreshold 75");
  assert(JSON.parse(fs.readFileSync(cfgPath, "utf8")).flexThreshold === 75, "/lilac-flex 75 persisted to disk");
}

// trailing % accepted
{
  const { flexThreshold } = await runCmd("50%");
  assert(flexThreshold === 50, "/lilac-flex 50% -> flexThreshold 50");
}

// off keyword
{
  const { flexThreshold } = await runCmd("off");
  assert(flexThreshold === null, "/lilac-flex off -> flexThreshold null");
}

// invalid arg -> no change, usage notify
{
  const { flexThreshold, notifications } = await runCmd("bogus");
  assert(flexThreshold === null, "/lilac-flex bogus -> no change (still off)");
  assert(notifications.some((n) => n.msg.includes("Usage")), "/lilac-flex bogus -> usage notify");
}

// picker: "≥ 75% discount" preset
{
  const { flexThreshold } = await runCmd("", { select: "≥ 75% discount" });
  assert(flexThreshold === 75, "picker '≥ 75% discount' -> 75");
}

// picker: "Off" preset
{
  const { flexThreshold } = await runCmd("", { select: "Off — allow all discounts" });
  assert(flexThreshold === null, "picker 'Off' -> null");
}

// picker: "Custom…" then input "60"
{
  const { flexThreshold } = await runCmd("", { select: "Custom…", input: "60" });
  assert(flexThreshold === 60, "picker Custom + input 60 -> 60");
}

// picker: custom input invalid -> no change, warning
{
  const { flexThreshold, notifications } = await runCmd("", { select: "Custom…", input: "abc" });
  assert(flexThreshold === 60, "invalid custom input -> no change (still 60)");
  assert(notifications.some((n) => n.level === "warning" && n.msg.includes("Invalid")), "invalid custom input -> warning notify");
}

// picker cancelled (select returns undefined) -> no change, no notify
{
  const { flexThreshold, notifications } = await runCmd("", { select: undefined });
  assert(flexThreshold === 60, "picker cancelled -> no change (still 60)");
  assert(notifications.length === 0, "picker cancelled -> no notify");
}

// non-interactive guard -> error notify, no change
{
  const { flexThreshold, notifications } = await runCmd("75", { hasUI: false });
  assert(flexThreshold === 60, "non-interactive -> no change (still 60)");
  assert(notifications.some((n) => n.level === "error" && n.msg.includes("interactive")), "non-interactive -> error notify");
}

// setting flex via command is visible to the gate (config cache in sync)
{
  await runCmd("75");
  const { result } = await runInput("interactive", { id: KIMI, provider: "lilac" });
  eq(result, { action: "handled" }, "after /lilac-flex 75, gate blocks kimi@25%");
  await runCmd("off");
  const { result: afterOff } = await runInput("interactive", { id: KIMI, provider: "lilac" });
  eq(afterOff, { action: "continue" }, "after /lilac-flex off, gate allows kimi@25%");
}

// ─── Summary ───────────────────────────────────────────────────────────────────

console.log(`\n${failed === 0 ? "ALL PASS" : `${failed} FAILED`}`);
process.exit(failed === 0 ? 0 : 1);
