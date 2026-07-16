// Minimal type stub for @earendil-works/pi-tui (a peer/ambient dependency provided
// by pi at runtime). Declares the symbols index.ts statically type-imports
// (Input, matchesKey, Key, truncateToWidth, visibleWidth, wrapTextWithAnsi,
// fuzzyFilter, SettingsListTheme) plus the ones the settings panel dynamically
// imports (SettingsList, Container). Used for editor/tsc support; the real
// package is loaded by pi when the extension runs.

export class Container {
  addChild(_child: any): void {}
  invalidate(): void {}
  render(_width: number): any {}
}

export interface SettingsListItem {
  id: string;
  label: string;
  description?: string;
  currentValue: string;
  values: string[];
}

export interface SettingsListTheme {
  label: (text: string, selected: boolean) => string;
  value: (text: string, selected: boolean) => string;
  description: (text: string) => string;
  cursor: string;
  hint: (text: string) => string;
}

export class SettingsList {
  constructor(
    _items: SettingsListItem[],
    _height: number,
    _theme: any,
    _onChange: (id: string, newValue: string) => void,
    _onDone: () => void,
    _opts?: { enableSearch?: boolean },
  ) {}
  handleInput(_data: string): void {}
  invalidate(): void {}
  render(_width: number): any {}
}

export class Input {
  focused = false;
  onSubmit?: (value: string) => void;
  onEscape?: () => void;
  setValue(_value: string): void {}
  getValue(): string { return ""; }
  handleInput(_data: string): void {}
  invalidate(): void {}
  render(_width: number): string[] { return []; }
}

export const Key: Record<string, string> = {};
export function matchesKey(_data: string, _key: any): boolean { return false; }
export function truncateToWidth(s: string, _width: number, _ellipsis?: string): string { return s; }
export function visibleWidth(_s: string): number { return 0; }
export function wrapTextWithAnsi(_s: string, _width: number): string[] { return []; }
export function fuzzyFilter<T>(_items: T[], _query: string, _keyFn: (item: T) => string): T[] { return _items; }
