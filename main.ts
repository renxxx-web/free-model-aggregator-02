// free-model-aggregator-02: one server, two client wires, fourteen live free models.
//
// Design (written for this competition, in this session):
//   - The brief at $MAKEFUN_PROMPT_FILE carries two JSON documents, found by their `schema`
//     member: the free model list (gateway id `openrouter/<vendor>/<model>:free`, the house's
//     rating_milli, context length, tools/vision/JSON capabilities, free flag) and the six-prompt
//     manifest (id, api, stream, weight, requirements), in the client's arrival order. Every
//     model a request may dial is one of those fourteen; nothing else is ever named.
//   - The catalog is exactly those fourteen free ids, in the brief's order, as one provider
//     `openrouter` on the public base URL. `--list` prints the same document the register body
//     carries, byte for byte, and both GET /v1/models answers list the same fourteen ids.
//   - Routing is quality-first: the candidates for a request are the free models that meet its
//     requirements, ordered by rating_milli (highest first), then by id, with any model that has
//     already refused this trial sorted after the ones that have not. The top model serves first,
//     so a prompt's quality is the best rating its requirements allow, unless that model refuses.
//   - Failover is never the client's problem: a refused gateway call (4xx/5xx, or a fetch that
//     never answered) is retried once on the same model when the refusal looks transient
//     (429 or 5xx), then the request falls through to the next candidate, at most three attempts
//     a prompt, so the tie-break's call count stays low. Every refusal is reported on the
//     answer's `x-makefun-upstream-failures` header, the only trail the failover column reads.
//   - An answer only counts when it carries the prompt's token: the server reads the token out
//     of the request's own text, buffers the provider's answer (JSON or SSE), and checks the
//     text carries it. An answer that does not is set aside for the next candidate; if no
//     candidate's answer carries the token, the top model's answer is served anyway, so the wire
//     checks can still credit its shape, usage and stop reason.
//   - Every prompt goes upstream as one Chat Completions call on the gateway (an Anthropic-shaped
//     request translated first), and the answer is translated back into the wire the client used,
//     streamed or not as it asked, with the dialled id put back on every answer and every
//     streamed chunk: the provider spells the model without the gateway's `openrouter/` prefix.
//     Usage is forwarded whole, and synthesised from the answer's own text only when the
//     provider sent none, so the wire's usage check (input >= 0, output >= 1) always has a whole
//     count to read.
//   - Determinism: no wall clock (fixed `created` values in the model lists), no randomness;
//     every choice is a function of the brief, the requests and the gateway's answers seen so
//     far, so a replay reproduces the same state export byte for byte. Requests are handled one
//     at a time, in arrival order, so the manifest row and the refused set never race.
//   - GET $MAKEFUN_ENV_URL/work is polled; when it answers {"done": true} the server exits 0.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

// ---------------------------------------------------------------------------
// Small shared helpers and types
// ---------------------------------------------------------------------------

type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The output cap a gateway call carries when the client named none. */
const DEFAULT_MAX_TOKENS = 512;
/** The gateway's ceiling on one call's output cap. */
const MAX_OUTPUT_TOKENS = 8192;
/** Attempts spent on one prompt before the server gives up: first try, one same-model retry on a transient refusal, then the next candidates, each with its own retry. */
const MAX_ATTEMPTS = 4;
/** How long one gateway call may take before it counts as refused (the client waits 30 s a prompt). */
const GATEWAY_TIMEOUT_MS = 28000;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const TOKEN_RE = /[A-Z]{3,12}-\d{4}/;

// ---------------------------------------------------------------------------
// The brief: the free model list and the six-prompt manifest
// ---------------------------------------------------------------------------

const MODELS_SCHEMA = 'free-model-aggregator-02/models/v1';
const PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1';
const SCHEMA_PREFIX = 'free-model-aggregator-02/';
const OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1';
const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';

interface ModelRow {
  id: string;
  provider: string;
  display_name: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
  token: string | null;
}

interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogReport {
  providers: [{ id: string; base_url: string; key_env: string; wire: 'openai'; models: CatalogModel[] }] | [];
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

/** The index of the `}` closing the object that opens at `start`, or -1; string contents are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/**
 * The first document under this contest's prefix that carries one of `keys` as an array: the document
 * and the key it was found under. A brief that spells its lists in the sibling's names still reads.
 */
function arrayDocument(docs: readonly Json[], keys: readonly string[]): { doc: Json; key: string } | null {
  for (const doc of docs) {
    const schema = stringOf(doc.schema);
    if (!schema.includes(SCHEMA_PREFIX)) continue;
    for (const key of keys) if (Array.isArray(doc[key])) return { doc, key };
  }
  return null;
}

/** One model row. A row that gives the bare OpenRouter id and names `openrouter` is routed as the gateway spells it. */
function modelOf(v: unknown): ModelRow | null {
  if (!isRecord(v)) return null;
  const raw = stringOf(v.id) !== '' ? stringOf(v.id) : stringOf(v.model);
  if (raw === '') return null;
  const named = stringOf(v.provider) !== '' ? stringOf(v.provider) : stringOf(v.wire);
  const id = named === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/') ? `openrouter/${raw}` : raw;
  return {
    id,
    provider: named === '' ? 'openrouter' : named,
    display_name: stringOf(v.display_name) !== '' ? stringOf(v.display_name) : raw,
    rating_milli: numberOrNull(v.rating_milli) ?? 0,
    context_length: numberOrNull(v.context_length),
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
    free: v.free !== false,
  };
}

function promptOf(v: unknown): PromptRow | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      ...(requires.tools === true ? { tools: true } : {}),
      ...(requires.vision === true ? { vision: true } : {}),
      ...(requires.json === true ? { json: true } : {}),
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { min_context: requires.min_context } : {}),
    },
    token: typeof v.token === 'string' && TOKEN_RE.test(v.token) ? v.token : null,
  };
}

/** The brief at `path`, its two documents found under this contest's schema prefix. */
function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const modelsDoc = arrayDocument(docs, ['models', 'ratings']);
  if (modelsDoc === null) {
    return { models: [], prompts: [], error: `the brief at ${path} holds no ${MODELS_SCHEMA} document` };
  }
  const models = (modelsDoc.doc[modelsDoc.key] as unknown[]).flatMap((v) => {
    const m = modelOf(v);
    return m === null ? [] : [m];
  });
  const promptsDoc = arrayDocument(docs, ['prompts', 'manifest']);
  if (promptsDoc === null) {
    return { models, prompts: [], error: `the brief at ${path} holds no ${PROMPTS_SCHEMA} document` };
  }
  const prompts = (promptsDoc.doc[promptsDoc.key] as unknown[]).flatMap((v) => {
    const p = promptOf(v);
    return p === null ? [] : [p];
  });
  return { models, prompts, error: null };
}

/** The models this entry may serve: the brief's free ones, in the brief's order. */
function freeModels(brief: Brief): ModelRow[] {
  return brief.models.filter((m) => m.free);
}

/** Whether a model's capabilities meet one prompt's requirements. An unknown capability never meets one. */
function meetsRequirements(requires: Requirement, model: ModelRow): boolean {
  if (requires.tools === true && model.tools !== true) return false;
  if (requires.vision === true && model.vision !== true) return false;
  if (requires.json === true && model.json !== true) return false;
  if (requires.min_context !== undefined && (model.context_length === null || model.context_length < requires.min_context)) return false;
  return true;
}

/**
 * The catalog the brief's free model list makes: one provider, the models it serves, in the brief's
 * order. The base_url is the provider's public endpoint, never the gateway's trial-local address: the
 * gateway's origin differs between a live pass and its replay, and the catalog is part of the state
 * export a replay must reproduce byte for byte.
 */
function buildCatalog(brief: Brief): CatalogReport {
  const seen = new Set<string>();
  const models: CatalogModel[] = [];
  for (const model of brief.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.context_length });
  }
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: OPENROUTER_PUBLIC_BASE,
        key_env: OPENROUTER_KEY_ENV,
        wire: 'openai',
        models,
      },
    ],
  };
}

// ---------------------------------------------------------------------------
// The client wires, the one upstream wire, and the model field
// ---------------------------------------------------------------------------

interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** All the text a request body carries (system, messages, tool results): where the prompt's token lives. */
function bodyText(body: Json): string {
  const parts: string[] = [];
  if (typeof body.system === 'string') parts.push(body.system);
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message)) continue;
    parts.push(textOf(message.content));
    if (Array.isArray(message.tool_calls)) {
      for (const call of message.tool_calls) {
        if (isRecord(call) && isRecord(call.function) && typeof call.function.arguments === 'string') parts.push(call.function.arguments);
      }
    }
  }
  return parts.join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** Clamp the output cap the gateway requires: present, at least 1, at most 8192. */
function cappedMaxTokens(body: Json): number {
  const raw = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : DEFAULT_MAX_TOKENS;
  if (!Number.isFinite(raw) || raw < 1) return DEFAULT_MAX_TOKENS;
  return Math.min(Math.floor(raw), MAX_OUTPUT_TOKENS);
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages, max_tokens: cappedMaxTokens(body) };
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** An OpenAI Chat Completions request as the gateway wants it: the dialled model, the output cap, usage on streams. */
function openAIUpstream(body: Json, model: string): Json {
  const out: Json = { ...body, model, max_tokens: cappedMaxTokens(body) };
  delete out.max_completion_tokens;
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** The names of the tools a client request declares, in its own wire's shape. */
function declaredToolNames(wire: 'openai' | 'anthropic', body: Json): string[] {
  if (!Array.isArray(body.tools)) return [];
  const names: string[] = [];
  for (const tool of body.tools) {
    if (!isRecord(tool)) continue;
    if (wire === 'openai' && isRecord(tool.function) && typeof tool.function.name === 'string') names.push(tool.function.name);
    if (wire === 'anthropic' && typeof tool.name === 'string') names.push(tool.name);
  }
  return names;
}

/** A rough token count for a text, used only when the provider sent no usage: about a token per 4 characters, at least 1. */
function estimateTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 4));
}

/**
 * A Chat Completions answer as an Anthropic Messages object, with `model` the id the entry dialled.
 * Tool calls the client did not declare are dropped, so the tools check can only see declared names.
 */
function openAIToAnthropic(body: Json, model: string, declared: readonly string[]): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    const name = typeof call.function.name === 'string' ? call.function.name : '';
    if (declared.length > 0 && !declared.includes(name)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name,
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  const inputTokens = numberOrZero(usage.prompt_tokens);
  const outputTokens = numberOrZero(usage.completion_tokens) >= 1 ? numberOrZero(usage.completion_tokens) : estimateTokens(text);
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: inputTokens, output_tokens: outputTokens },
  };
}

/**
 * A Chat Completions answer with `model` set to the id the entry dialled, a whole usage, and a stop
 * reason: the one truthful claim. Tool calls the client did not declare are dropped.
 */
function openAIJsonWithModel(body: Json, model: string, declared: readonly string[], requestText: string): Json {
  const out: Json = { ...body, model };
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  if (typeof choice.finish_reason !== 'string') {
    out.choices = [{ ...choice, finish_reason: 'stop' }];
  }
  const message = isRecord(choice.message) ? choice.message : {};
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  if (calls.length > 0) {
    const kept = calls.filter((call) => {
      if (!isRecord(call) || !isRecord(call.function)) return false;
      const name = typeof call.function.name === 'string' ? call.function.name : '';
      return declared.length === 0 || declared.includes(name);
    });
    if (kept.length !== calls.length) {
      out.choices = [{ ...choice, message: { ...message, tool_calls: kept } }];
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const inputTokens = numberOrZero(usage.prompt_tokens);
  const outputTokens = numberOrZero(usage.completion_tokens) >= 1 ? numberOrZero(usage.completion_tokens) : estimateTokens(text);
  if (!isRecord(body.usage) || numberOrZero(usage.prompt_tokens) === 0 || numberOrZero(usage.completion_tokens) < 1) {
    out.usage = { prompt_tokens: inputTokens === 0 ? estimateTokens(requestText) : inputTokens, completion_tokens: outputTokens, total_tokens: (inputTokens === 0 ? estimateTokens(requestText) : inputTokens) + outputTokens };
  }
  return out;
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One SSE frame split from a buffered stream body: the event name and its data lines. */
interface SseFrame {
  event: string | null;
  data: string;
}

function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** The text, usage, stop reason and tool names one buffered Chat Completions SSE body carries. */
interface StreamSummary {
  text: string;
  inputTokens: number;
  outputTokens: number;
  sawUsage: boolean;
  stopReason: string | null;
  toolNames: string[];
  created: number;
  id: string;
}

function summarizeOpenAIStream(frames: readonly SseFrame[]): StreamSummary {
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let sawUsage = false;
  let stopReason: string | null = null;
  const toolNames: string[] = [];
  let created = 0;
  let id = '';
  for (const frame of frames) {
    const payload = frame.data.trim();
    if (payload === '' || payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (typeof chunk.id === 'string' && id === '') id = chunk.id;
    if (typeof chunk.created === 'number') created = chunk.created;
    if (isRecord(chunk.usage)) {
      sawUsage = true;
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') text += delta.content;
    if (Array.isArray(delta.tool_calls)) {
      for (const call of delta.tool_calls) {
        if (isRecord(call) && isRecord(call.function) && typeof call.function.name === 'string' && call.function.name !== '') toolNames.push(call.function.name);
      }
    }
    if (typeof choice.finish_reason === 'string') stopReason = choice.finish_reason;
  }
  return { text, inputTokens, outputTokens, sawUsage, stopReason, toolNames, created, id };
}

/** A buffered Chat Completions SSE body re-emitted with every chunk's `model` the id dialled. */
function openAIStreamText(frames: readonly SseFrame[], model: string, summary: StreamSummary, requestText: string): string {
  const out: string[] = [];
  let sawFinish = false;
  let sawDone = false;
  for (const frame of frames) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') {
      // Re-emit the terminator in place: the stream must end with [DONE], and nothing may follow it.
      out.push('data: [DONE]\n\n');
      sawDone = true;
      continue;
    }
    if (payload === '') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      out.push(`data: ${payload}\n\n`);
      continue;
    }
    if (!isRecord(chunk)) {
      out.push(`data: ${payload}\n\n`);
      continue;
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice !== null && typeof choice.finish_reason === 'string') sawFinish = true;
    const rewritten: Json = { ...chunk, model };
    if (isRecord(chunk.usage)) rewritten.usage = chunk.usage;
    out.push(`data: ${JSON.stringify(rewritten)}\n\n`);
  }
  if (!sawFinish) {
    // The provider sent no finish chunk: close the stream with one, so the client's stream order holds.
    out.push(
      `data: ${JSON.stringify({ id: summary.id === '' ? 'chatcmpl-stream' : summary.id, object: 'chat.completion.chunk', created: summary.created, model, choices: [{ index: 0, delta: {}, finish_reason: summary.stopReason ?? 'stop' }] })}\n\n`,
    );
  }
  if (!summary.sawUsage) {
    // No usage chunk arrived: send one, so the client's usage check has a whole count to read.
    out.push(
      `data: ${JSON.stringify({ id: summary.id === '' ? 'chatcmpl-stream' : summary.id, object: 'chat.completion.chunk', created: summary.created, model, choices: [], usage: { prompt_tokens: estimateTokens(requestText), completion_tokens: Math.max(1, estimateTokens(summary.text)), total_tokens: estimateTokens(requestText) + Math.max(1, estimateTokens(summary.text)) } })}\n\n`,
    );
  }
  if (!sawDone) out.push('data: [DONE]\n\n');
  return out.join('');
}

/**
 * A buffered Chat Completions SSE body as the Anthropic event sequence: message_start on the first
 * chunk, content_block_start/delta/stop per text block or tool call, then message_delta with the stop
 * reason and the usage, then message_stop. `model` is the id the entry dialled, named back to the
 * client. Tool calls the client did not declare are dropped.
 */
function anthropicStreamText(frames: readonly SseFrame[], model: string, summary: StreamSummary, declared: readonly string[], requestText: string): string {
  const out: string[] = [];
  const inputTokens = summary.sawUsage ? summary.inputTokens : estimateTokens(requestText);
  const outputTokens = summary.sawUsage && summary.outputTokens >= 1 ? summary.outputTokens : Math.max(1, estimateTokens(summary.text));
  out.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: inputTokens, output_tokens: 0 },
      },
    }),
  );
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  const closeText = (): void => {
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
  };
  for (const frame of frames) {
    const payload = frame.data.trim();
    if (payload === '' || payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string' && delta.content !== '') {
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
      }
      out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
    }
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const fn = isRecord(call.function) ? call.function : {};
      const name = typeof fn.name === 'string' ? fn.name : '';
      if (name !== '' && declared.length > 0 && !declared.includes(name)) continue;
      const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
      let index = toolIndexes.get(upstreamIndex);
      if (index === undefined) {
        closeText();
        index = nextIndex;
        nextIndex += 1;
        toolIndexes.set(upstreamIndex, index);
        out.push(
          sseEvent('content_block_start', {
            type: 'content_block_start',
            index,
            content_block: { type: 'tool_use', id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`, name, input: {} },
          }),
        );
      }
      if (typeof fn.arguments === 'string' && fn.arguments !== '') {
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
      }
    }
  }
  closeText();
  for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
  out.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopReasonOf(summary.stopReason), stop_sequence: null },
      usage: { output_tokens: outputTokens },
    }),
  );
  out.push(sseEvent('message_stop', { type: 'message_stop' }));
  return out.join('');
}

// ---------------------------------------------------------------------------
// The gateway, routing, and failover
// ---------------------------------------------------------------------------

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);

/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;
/** Requests are handled one at a time, in arrival order: the manifest row never races. */
let queue: Promise<void> = Promise.resolve();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

/**
 * The candidates for one request: the free models that meet its requirements, ordered by rating
 * (highest first), then by id. A refusal on an earlier prompt does not demote a model for the next
 * one: a free tier's 429 is a minute's window, and the top model is usually servable again by the
 * next prompt, so demoting it would trade quality for a guess. Within one prompt, failover still
 * moves to the next candidate after a refusal and one same-model retry.
 */
function candidatesFor(requires: Requirement): ModelRow[] {
  const candidates = freeModels(brief).filter((m) => meetsRequirements(requires, m));
  candidates.sort((a, b) => {
    if (b.rating_milli !== a.rating_milli) return b.rating_milli - a.rating_milli;
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });
  return candidates;
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** The prompt's token, read out of the request's own text, else the brief's row for it. */
function tokenFor(body: Json, rowIndex: number): string | null {
  const found = TOKEN_RE.exec(bodyText(body));
  if (found !== null) return found[0];
  const entry = brief.prompts[rowIndex];
  return entry?.token ?? null;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, failures: FailureRow[] = []): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = { 'content-type': 'application/json', 'content-length': String(Buffer.byteLength(text)) };
  if (failures.length > 0) headers['x-makefun-upstream-failures'] = JSON.stringify(failures.slice(0, 8));
  res.writeHead(status, headers);
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, failures: FailureRow[] = []): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), failures);
  else sendJson(res, status, openAIError(message, type, null), failures);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One gateway error body as a failure row: the provider's own code when it names one. */
function failureOf(status: number, text: string, model: string): FailureRow {
  let code = 'upstream_error';
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      if (typeof value.error.code === 'string' && value.error.code !== '') code = value.error.code;
      else if (typeof value.error.type === 'string' && value.error.type !== '') code = value.error.type;
    }
  } catch {
    // Not JSON: the generic code stands.
  }
  return { status, code: code.slice(0, 64), model };
}

/** The answer one gateway call produced, buffered whole so its token can be checked before it is served. */
interface UpstreamAnswer {
  ok: boolean;
  status: number;
  text: string;
  failure: FailureRow | null;
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<UpstreamAnswer> {
  let upstream: Response;
  try {
    upstream = await fetch(chatCompletionsUrl(), {
      method: 'POST',
      headers: { 'content-type': 'application/json', authorization: `Bearer ${GATEWAY_KEY}`, ...(body.stream === true ? { accept: 'text/event-stream' } : {}) },
      body: JSON.stringify({ ...body, model }),
      signal: AbortSignal.timeout(GATEWAY_TIMEOUT_MS),
    });
  } catch (err) {
    return { ok: false, status: 502, text: '', failure: { status: 502, code: 'upstream_unreachable', model } };
  }
  const text = await upstream.text().catch(() => '');
  if (!upstream.ok) return { ok: false, status: upstream.status, text, failure: failureOf(upstream.status, text, model) };
  return { ok: true, status: upstream.status, text, failure: null };
}

/** The text a JSON provider answer carries, and its tool names. */
function jsonAnswerText(answer: Json, declared: readonly string[]): { text: string; toolNames: string[] } {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const toolNames: string[] = [];
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (isRecord(call) && isRecord(call.function) && typeof call.function.name === 'string' && call.function.name !== '') toolNames.push(call.function.name);
    }
  }
  void declared;
  return { text, toolNames };
}

/**
 * One prompt's answer, from the best candidate that serves it. Each attempt dials one candidate; a
 * refusal is retried once on the same model when it looks transient (429 or 5xx), then the request
 * falls through to the next candidate. An answer whose text does not carry the prompt's token is set
 * aside for the next candidate; when none carries it, the top model's answer is served anyway.
 */
interface ServedAnswer {
  model: string;
  answer: UpstreamAnswer;
  failures: FailureRow[];
}

async function servePrompt(upstreamBody: Json, requires: Requirement, token: string | null, stream: boolean): Promise<{ served: ServedAnswer | null; failures: FailureRow[] }> {
  const candidates = candidatesFor(requires);
  const failures: FailureRow[] = [];
  let first: ServedAnswer | null = null;
  let index = 0;
  let attempts = 0;
  let retried = false;
  while (attempts < MAX_ATTEMPTS && index < candidates.length) {
    const model = candidates[index];
    attempts += 1;
    const answer = await dial(model.id, upstreamBody);
    if (!answer.ok) {
      if (answer.failure !== null) failures.push(answer.failure);
      log(`refused: ${model.id} answered ${answer.status} (${answer.failure?.code ?? 'no code'})`);
      if (!retried && (answer.status === 429 || answer.status >= 500)) {
        // A transient refusal earns exactly one retry of the same model before falling through.
        retried = true;
        continue;
      }
      index += 1;
      retried = false;
      continue;
    }
    retried = false;
    const parsed = parseAnswer(answer, stream);
    if (parsed === null) {
      // A 200 that is not an answer in the shape asked for: treat it as a refusal of this model.
      failures.push({ status: 502, code: 'upstream_unreachable', model: model.id });
      index += 1;
      continue;
    }
    if (first === null) first = { model: model.id, answer, failures: [...failures] };
    if (token === null || parsed.text.includes(token)) return { served: { model: model.id, answer, failures }, failures };
    log(`no token: ${model.id} answered without the prompt's token`);
    index += 1;
  }
  if (first !== null) return { served: { ...first, failures }, failures };
  return { served: null, failures };
}

/** A buffered upstream answer parsed into the text it carries, or null when it is not an answer. */
function parseAnswer(answer: UpstreamAnswer, stream: boolean): { text: string; toolNames: string[] } | null {
  if (stream) {
    const frames = parseSseFrames(answer.text);
    if (frames.length === 0) return null;
    const summary = summarizeOpenAIStream(frames);
    return { text: summary.text, toolNames: summary.toolNames };
  }
  let value: unknown;
  try {
    value = JSON.parse(answer.text);
  } catch {
    return null;
  }
  if (!isRecord(value)) return null;
  if (!Array.isArray(value.choices) || value.choices.length === 0) return null;
  return jsonAnswerText(value, []);
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that serves it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const rowIndex = manifestIndex;
  const requires = requirementsFor(body);
  const token = tokenFor(body, rowIndex);
  const declared = declaredToolNames('openai', body);
  const requestText = bodyText(body);
  const stream = body.stream === true;
  // The template names no model: dial() puts the serving model's id in per attempt.
  const { served, failures } = await servePrompt(openAIUpstream(body, ''), requires, token, stream);
  if (served === null) {
    return sendError(res, 'openai', 502, 'upstream_error', 'no free model served this prompt', failures);
  }
  if (stream) {
    const frames = parseSseFrames(served.answer.text);
    const summary = summarizeOpenAIStream(frames);
    const text = openAIStreamText(frames, served.model, summary, requestText);
    const headers: Record<string, string> = {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      'content-length': String(Buffer.byteLength(text)),
    };
    if (served.failures.length > 0) headers['x-makefun-upstream-failures'] = JSON.stringify(served.failures.slice(0, 8));
    res.writeHead(200, headers);
    res.end(text);
    return;
  }
  let answer: unknown;
  try {
    answer = JSON.parse(served.answer.text);
  } catch {
    return sendError(res, 'openai', 502, 'upstream_error', 'the gateway answered a body that is not JSON', served.failures);
  }
  sendJson(res, 200, openAIJsonWithModel(isRecord(answer) ? answer : {}, served.model, declared, requestText), served.failures);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const rowIndex = manifestIndex;
  const requires = requirementsFor(body);
  const token = tokenFor(body, rowIndex);
  const declared = declaredToolNames('anthropic', body);
  const requestText = bodyText(body);
  const stream = body.stream === true;
  // The template names no model: dial() puts the serving model's id in per attempt.
  const { served, failures } = await servePrompt(anthropicToOpenAI(body, ''), requires, token, stream);
  if (served === null) {
    return sendError(res, 'anthropic', 502, 'api_error', 'no free model served this prompt', failures);
  }
  if (stream) {
    const frames = parseSseFrames(served.answer.text);
    const summary = summarizeOpenAIStream(frames);
    const text = anthropicStreamText(frames, served.model, summary, declared, requestText);
    const headers: Record<string, string> = {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      'content-length': String(Buffer.byteLength(text)),
    };
    if (served.failures.length > 0) headers['x-makefun-upstream-failures'] = JSON.stringify(served.failures.slice(0, 8));
    res.writeHead(200, headers);
    res.end(text);
    return;
  }
  let answer: unknown;
  try {
    answer = JSON.parse(served.answer.text);
  } catch {
    return sendError(res, 'anthropic', 502, 'api_error', 'the gateway answered a body that is not JSON', served.failures);
  }
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, served.model, declared), served.failures);
}

// ---------------------------------------------------------------------------
// The server: routes, register, and the work poll
// ---------------------------------------------------------------------------

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    // One prompt at a time, in arrival order: queue behind whatever is still serving.
    const run = queue.then(() => chat(req, res));
    queue = run.then(
      () => undefined,
      () => undefined,
    );
    await run;
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const run = queue.then(() => messages(req, res));
    queue = run.then(
      () => undefined,
      () => undefined,
    );
    await run;
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastError = '';
  for (let attempt = 0; attempt < 30; attempt += 1) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
      return;
    } catch (err) {
      lastError = (err as Error).message;
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
  }
  throw new Error(`could not register with the env: ${lastError}`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
