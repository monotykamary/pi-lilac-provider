// Stub for @earendil-works/pi-coding-agent peer dependency.
//
// Tests import index.ts, which calls getAgentDir() at module top-level to build
// CACHE_DIR + FIREWORKS_CONFIG_PATH. We point getAgentDir at a per-run temp dir
// (PI_CODING_AGENT_DIR, set in vitest.setup.ts) so config/cache reads+writes
// never touch the user's real ~/.pi/agent. Also re-exports the type surface the
// extension imports, so TypeScript resolves without the real package installed.

import os from "os";
import path from "path";

/** Mirrors the real getAgentDir(): respects PI_CODING_AGENT_DIR, defaults to ~/.pi/agent */
export function getAgentDir(): string {
  return process.env.PI_CODING_AGENT_DIR || path.join(os.homedir(), ".pi", "agent");
}

export interface ExtensionAPI {
  registerProvider(_name: string, _provider: any): void;
  on(_event: string, _handler: any): void;
  registerShortcut(_binding: string, _opts: any): void;
  registerCommand(_name: string, _opts: any): void;
  appendEntry(_type: string, _data: any): void;
  events: { emit(_event: string, _data: any): void };
}

export interface ModelRegistry {
  getApiKeyForProvider(_provider: string): Promise<string | null>;
}

export interface ExtensionContext {
  hasUI: boolean;
  ui: any;
  sessionManager: any;
  modelRegistry: ModelRegistry;
  model: any;
  mode: string;
}

// Minimal re-exports the settings panel dynamically imports. Real shapes are
// provided by pi at runtime; these stubs exist for tsc/editor support only.
export function getSettingsListTheme(): any {
  return {};
}
export class DynamicBorder {
  constructor(_fn: (s: string) => string) {}
}
