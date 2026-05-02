#!/usr/bin/env node
/**
 * Update Lilac models from API
 *
 * Fetches models from https://api.getlilac.com/v1/models and updates:
 * - models.json: Provider model definitions (enriched with pricing & compat)
 * - README.md: Model table in the Available Models section
 *
 * The Lilac /v1/models API returns model info including:
 *   - Per-token pricing (prompt, completion, input_cache_read)
 *   - Context length, max completion tokens
 *   - Architecture (input modalities, output modalities)
 *   - Supported features (tools, reasoning)
 *   - Supported parameters
 *
 * Pricing is converted from per-token to per-million-tokens for pi.
 *
 * models.json is the source of truth for curated specs — the script preserves
 * existing data and only adds new models with API-derived defaults.
 * Curate models.json manually after new model discovery.
 *
 * patch.json is applied at runtime by the provider — not baked into models.json.
 *
 * Requires LILAC_API_KEY environment variable.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const MODELS_API_URL = 'https://api.getlilac.com/v1/models';
const MODELS_JSON_PATH = path.join(__dirname, '..', 'models.json');
const README_PATH = path.join(__dirname, '..', 'README.md');

// ─── Helpers ─────────────────────────────────────────────────────────────────

function loadJson(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch {
    return {};
  }
}

function saveJson(filePath, data) {
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2) + '\n');
  console.log(`✓ Saved ${path.basename(filePath)}`);
}

// Convert per-token pricing from API to per-million-tokens
function toPerMillion(val) {
  if (val === '' || val === null || val === undefined) return null;
  return Math.round(parseFloat(val) * 1_000_000 * 100) / 100;
}

// ─── API fetch ───────────────────────────────────────────────────────────────

async function fetchModels() {
  const apiKey = process.env.LILAC_API_KEY;
  if (!apiKey) {
    throw new Error('LILAC_API_KEY environment variable is required');
  }

  console.log(`Fetching models from ${MODELS_API_URL}...`);
  const response = await fetch(MODELS_API_URL, {
    headers: { 'Authorization': `Bearer ${apiKey}` },
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }

  const data = await response.json();
  const models = data.data || [];
  console.log(`✓ Fetched ${models.length} models from API`);
  return models;
}

// ─── Transform API model → models.json entry ────────────────────────────────

function transformApiModel(apiModel, existingModelsMap) {
  const id = apiModel.id;

  // Preserve existing curated data (pricing, reasoning, compat, etc.)
  if (existingModelsMap[id]) {
    const existing = { ...existingModelsMap[id] };
    // Update context window from API if changed
    if (apiModel.context_length) {
      existing.contextWindow = apiModel.context_length;
    }
    // Update max output tokens from API (but trust curated value for GLM 5.1)
    if (apiModel.top_provider?.max_completion_tokens && id !== 'zai-org/glm-5.1') {
      existing.maxTokens = apiModel.top_provider.max_completion_tokens;
    }
    // Update features from API
    const features = apiModel.supported_features || [];
    if (features.includes('reasoning')) {
      existing.reasoning = true;
    }
    // Update modalities from API
    const modalities = apiModel.architecture?.input_modalities || [];
    if (modalities.includes('image') && !existing.input.includes('image')) {
      existing.input = ['text', 'image'];
    }
    // Update pricing from API
    const pricing = apiModel.pricing || {};
    const inputCost = toPerMillion(pricing.prompt);
    const outputCost = toPerMillion(pricing.completion);
    const cacheReadCost = toPerMillion(pricing.input_cache_read);
    if (inputCost !== null && inputCost > 0) existing.cost.input = inputCost;
    if (outputCost !== null && outputCost > 0) existing.cost.output = outputCost;
    if (cacheReadCost !== null) existing.cost.cacheRead = cacheReadCost;
    return existing;
  }

  // New model — build from API data + sensible defaults
  const features = apiModel.supported_features || [];
  const modalities = apiModel.architecture?.input_modalities || [];
  const pricing = apiModel.pricing || {};
  const hasReasoning = features.includes('reasoning');
  const hasImage = modalities.includes('image');

  const inputTypes = ['text'];
  if (hasImage) inputTypes.push('image');

  const inputCost = toPerMillion(pricing.prompt) || 0;
  const outputCost = toPerMillion(pricing.completion) || 0;
  const cacheReadCost = toPerMillion(pricing.input_cache_read) || 0;

  const model = {
    id,
    name: apiModel.name || generateDisplayName(id),
    reasoning: hasReasoning,
    input: inputTypes,
    cost: {
      input: inputCost,
      output: outputCost,
      cacheRead: cacheReadCost,
      cacheWrite: 0,
    },
    contextWindow: apiModel.context_length || 131072,
    maxTokens: apiModel.top_provider?.max_completion_tokens || apiModel.context_length || 131072,
  };

  // Add compat — all Lilac models use chat_template_kwargs for reasoning toggle
  const compat = {
    supportsDeveloperRole: true,
    supportsStore: false,
    maxTokensField: 'max_completion_tokens',
  };

  if (hasReasoning) {
    compat.thinkingFormat = 'qwen-chat-template';
  }

  model.compat = compat;

  return model;
}

function generateDisplayName(id) {
  // Handle known naming patterns
  const KNOWN_NAMES = {
    'moonshotai/kimi-k2.6': 'Kimi K2.6',
    'zai-org/glm-5.1': 'GLM 5.1',
    'google/gemma-4-31b-it': 'Gemma 4',
  };
  if (KNOWN_NAMES[id]) return KNOWN_NAMES[id];

  // Fallback: prettify the ID
  return id
    .split('/')
    .pop()
    .replace(/[-_]/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase());
}

// ─── README generation ──────────────────────────────────────────────────────

function formatContext(n) {
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
  if (n >= 1000) return `${Math.round(n / 1000)}K`;
  return n.toString();
}

function formatCost(cost) {
  if (cost === 0) return '—';
  if (cost === null || cost === undefined) return '—';
  return `$${cost.toFixed(2)}`;
}

function generateReadmeTable(models) {
  const lines = [
    '| Model | Context | Vision | Reasoning | Input $/M | Cache Read $/M | Output $/M |',
    '|-------|---------|--------|-----------|-----------|-----------------|------------|',
  ];

  for (const model of models) {
    const context = formatContext(model.contextWindow);
    const vision = model.input.includes('image') ? '✅' : '❌';
    const reasoning = model.reasoning ? '✅' : '❌';
    const inputCost = formatCost(model.cost.input);
    const cacheReadCost = formatCost(model.cost.cacheRead);
    const outputCost = formatCost(model.cost.output);

    lines.push(`| ${model.name} | ${context} | ${vision} | ${reasoning} | ${inputCost} | ${cacheReadCost} | ${outputCost} |`);
  }

  return lines.join('\n');
}

function updateReadme(models) {
  let readme = fs.readFileSync(README_PATH, 'utf8');
  const newTable = generateReadmeTable(models);

  const tableRegex = /(## Available Models\n\n)\| Model \|[^\n]+\|\n\|[-| ]+\|(\n\|[^\n]+\|)*\n*/;

  if (tableRegex.test(readme)) {
    readme = readme.replace(tableRegex, (match, header) => `${header}${newTable}\n\n`);
    fs.writeFileSync(README_PATH, readme);
    console.log('✓ Updated README.md');
  } else {
    console.warn('⚠ Could not find model table in "## Available Models" section');
  }
}

// ─── Main ────────────────────────────────────────────────────────────────────

async function main() {
  try {
    const apiModels = await fetchModels();

    // Load existing models.json — source of truth for curated specs
    const existingModels = loadJson(MODELS_JSON_PATH);
    const existingModelsMap = {};
    for (const m of (Array.isArray(existingModels) ? existingModels : [])) {
      existingModelsMap[m.id] = m;
    }

    // Transform API models, preserving existing data where available
    let models = apiModels.map(m =>
      transformApiModel(m, existingModelsMap)
    );

    // Keep models from models.json that are NOT in the API response
    const apiIds = new Set(apiModels.map(m => m.id));
    for (const existing of Object.values(existingModelsMap)) {
      if (!apiIds.has(existing.id)) {
        models.push(existing);
      }
    }

    // Sort by model name
    models.sort((a, b) => a.name.localeCompare(b.name));

    // Save models.json
    saveJson(MODELS_JSON_PATH, models);

    // Update README
    updateReadme(models);

    // Summary
    const newIds = new Set(models.map(m => m.id));
    const oldIds = new Set(Object.keys(existingModelsMap));
    const added = [...newIds].filter(id => !oldIds.has(id));
    const removed = [...oldIds].filter(id => !newIds.has(id));

    console.log('\n--- Summary ---');
    console.log(`Total models: ${models.length}`);
    console.log(`Reasoning models: ${models.filter(m => m.reasoning).length}`);
    console.log(`Vision models: ${models.filter(m => m.input.includes('image')).length}`);
    console.log(`Cache-enabled models: ${models.filter(m => m.cost.cacheRead > 0).length}`);
    if (added.length > 0) console.log(`New models: ${added.join(', ')} — curate models.json manually`);
    if (removed.length > 0) console.log(`Removed models: ${removed.join(', ')}`);

  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
}

main();
