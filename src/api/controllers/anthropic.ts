import { PassThrough } from "stream";
import _ from "lodash";

import chat from "./chat.ts";
import util from "@/lib/util.ts";
import logger from "@/lib/logger.ts";

// ============================================================
//  Anthropic Messages API (/v1/messages) for this proxy.
//  Lets Claude Code (which speaks the Anthropic Messages API)
//  drive the same chat.qwen.ai backend + <tool_call> emulation.
//  Text generation is delegated to the OpenAI-style completion,
//  and results are re-encoded into Anthropic message format.
// ============================================================

const genId = () => util.uuid(false).slice(0, 24);

/** Generate assistant text for a flattened prompt via the base completion. */
async function getText(model: string, prompt: string, token: string): Promise<string> {
  const r: any = await chat.createCompletion(model, [{ role: "user", content: prompt }], token);
  return r?.choices?.[0]?.message?.content || "";
}

// ---- tool-call emulation (self-contained) ----

function buildToolSystemPrompt(tools: any[]): string {
  // Anthropic tools: { name, description, input_schema }
  return [
    "# Tool Calling",
    "",
    "You have the tools below. To call a tool, reply with one or more blocks EXACTLY:",
    "",
    "<tool_call>",
    '{"name": "<tool_name>", "arguments": {<json-arguments matching input_schema>}}',
    "</tool_call>",
    "",
    "Rules:",
    "- Output ONLY the <tool_call> block(s) when calling a tool — no surrounding prose.",
    "- Call a tool ONLY when the task actually needs it (reading/writing files, running",
    "  commands, searching the code). Do NOT call a tool just to talk.",
    "- For greetings, explanations, or anything you can answer from context, reply in PLAIN",
    "  TEXT with NO tool call. Never use a shell/Bash tool merely to print or echo a message.",
    "- Tool results come back inside <tool_response> blocks. When finished, reply in plain text.",
    "",
    "Available tools (JSON):",
    "<tools>",
    JSON.stringify(tools, null, 2),
    "</tools>",
  ].join("\n");
}

function safeParseToolJson(raw: string): any {
  if (!raw) return null;
  let s = raw.trim();
  const fence = s.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/);
  if (fence) s = fence[1].trim();
  try { return JSON.parse(s); } catch { return null; }
}

/** parse <tool_call> blocks -> [{ id, name, input }] and cleaned text */
function parseToolCalls(text: string): { content: string; toolCalls: any[] } {
  // Some models (e.g. deepseek-reasoner) emit <tool_call> without a closing
  // </tool_call>. Balance the tags so those still parse instead of leaking.
  let t = text;
  const opens = (t.match(/<tool_call>/g) || []).length;
  const closes = (t.match(/<\/tool_call>/g) || []).length;
  if (opens > closes) t += "</tool_call>".repeat(opens - closes);

  const toolCalls: any[] = [];
  const regex = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g;
  let m: RegExpExecArray | null;
  while ((m = regex.exec(t)) !== null) {
    const p = safeParseToolJson(m[1]);
    if (p && p.name) {
      const input = _.isString(p.arguments) ? safeParseToolJson(p.arguments) || {} : (p.arguments ?? {});
      toolCalls.push({ id: `toolu_${genId()}`, name: p.name, input });
    }
  }
  let content = t.replace(regex, "").trim();
  if (!toolCalls.length) {
    const bare = safeParseToolJson(t);
    if (bare && bare.name && bare.arguments !== undefined) {
      toolCalls.push({ id: `toolu_${genId()}`, name: bare.name, input: bare.arguments ?? {} });
      content = "";
    }
  }
  return { content, toolCalls };
}

function extractText(content: any): string {
  if (_.isString(content)) return content;
  if (_.isArray(content)) {
    return content
      .filter((b) => b && b.type === "text" && _.isString(b.text))
      .map((b) => b.text)
      .join("\n");
  }
  return "";
}

const cap = (r: string) => (r === "assistant" ? "Assistant" : "User");

/** Anthropic request (system + messages + tools) -> single flattened prompt */
function anthropicPrepare(system: any, messages: any[], tools: any[]): string {
  const parts: string[] = [];
  if (_.isArray(tools) && tools.length) parts.push(buildToolSystemPrompt(tools));

  const sys = _.isString(system) ? system : _.isArray(system) ? extractText(system) : "";
  if (sys) parts.push(`System: ${sys}`);

  for (const msg of messages || []) {
    if (!msg) continue;
    if (_.isString(msg.content)) {
      parts.push(`${cap(msg.role)}: ${msg.content}`);
      continue;
    }
    if (!_.isArray(msg.content)) continue;
    // render blocks
    const textPieces: string[] = [];
    for (const b of msg.content) {
      if (!b) continue;
      if (b.type === "text" && b.text) textPieces.push(b.text);
      else if (b.type === "tool_use")
        textPieces.push(`<tool_call>\n${JSON.stringify({ name: b.name, arguments: b.input ?? {} })}\n</tool_call>`);
      else if (b.type === "tool_result") {
        const c = _.isString(b.content) ? b.content : extractText(b.content);
        textPieces.push(`<tool_response>\n${c}\n</tool_response>`);
      }
    }
    const joined = textPieces.join("\n");
    if (joined) parts.push(`${cap(msg.role)}: ${joined}`);
  }
  parts.push("Assistant:");
  return parts.join("\n\n");
}

/** Build Anthropic content blocks from text + tool calls */
function buildContent(text: string, toolCalls: any[]): any[] {
  const content: any[] = [];
  if (text) content.push({ type: "text", text });
  for (const tc of toolCalls)
    content.push({ type: "tool_use", id: tc.id, name: tc.name, input: tc.input ?? {} });
  return content;
}

/** Non-streaming /v1/messages */
async function createMessages(model: string, body: any, token: string) {
  const { system, messages, tools } = body;
  const useTools = _.isArray(tools) && tools.length > 0;
  const prompt = anthropicPrepare(system, messages, tools);
  const raw = await getText(model, prompt, token);
  let text = raw, toolCalls: any[] = [];
  if (useTools) { const p = parseToolCalls(raw); text = p.content; toolCalls = p.toolCalls; }
  const content = buildContent(text, toolCalls);
  return {
    id: `msg_${genId()}`,
    type: "message",
    role: "assistant",
    model,
    content,
    stop_reason: toolCalls.length ? "tool_use" : "end_turn",
    stop_sequence: null,
    usage: { input_tokens: 1, output_tokens: 1 },
  };
}

/** Streaming /v1/messages (Anthropic SSE). Buffers, then replays events. */
function createMessagesStream(model: string, body: any, token: string) {
  const { system, messages, tools } = body;
  const useTools = _.isArray(tools) && tools.length > 0;
  const prompt = anthropicPrepare(system, messages, tools);
  const msgId = `msg_${genId()}`;

  const ts = new PassThrough();
  const send = (event: string, data: any) => {
    try { ts.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`); } catch {}
  };

  // immediate start so the client gets bytes
  send("message_start", {
    type: "message_start",
    message: {
      id: msgId, type: "message", role: "assistant", model,
      content: [], stop_reason: null, stop_sequence: null,
      usage: { input_tokens: 1, output_tokens: 0 },
    },
  });
  const heartbeat = setInterval(() => { try { ts.write(`event: ping\ndata: {"type":"ping"}\n\n`); } catch {} }, 5000);

  (async () => {
    try {
      const raw = await getText(model, prompt, token);
      let text = raw, toolCalls: any[] = [];
      if (useTools) { const p = parseToolCalls(raw); text = p.content; toolCalls = p.toolCalls; }
      const content = buildContent(text, toolCalls);

      let idx = 0;
      for (const block of content) {
        if (block.type === "text") {
          send("content_block_start", { type: "content_block_start", index: idx, content_block: { type: "text", text: "" } });
          send("content_block_delta", { type: "content_block_delta", index: idx, delta: { type: "text_delta", text: block.text } });
          send("content_block_stop", { type: "content_block_stop", index: idx });
        } else if (block.type === "tool_use") {
          send("content_block_start", { type: "content_block_start", index: idx, content_block: { type: "tool_use", id: block.id, name: block.name, input: {} } });
          send("content_block_delta", { type: "content_block_delta", index: idx, delta: { type: "input_json_delta", partial_json: JSON.stringify(block.input ?? {}) } });
          send("content_block_stop", { type: "content_block_stop", index: idx });
        }
        idx++;
      }
      send("message_delta", { type: "message_delta", delta: { stop_reason: toolCalls.length ? "tool_use" : "end_turn", stop_sequence: null }, usage: { output_tokens: 1 } });
      send("message_stop", { type: "message_stop" });
    } catch (err: any) {
      logger.error("messages stream error:", err?.message || err);
      send("error", { type: "error", error: { type: "api_error", message: String(err?.message || err) } });
    } finally {
      clearInterval(heartbeat);
      ts.end();
    }
  })();

  return ts;
}

export default { createMessages, createMessagesStream, tokenSplit: chat.tokenSplit };
