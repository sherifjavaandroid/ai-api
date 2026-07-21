import { PassThrough } from "stream";
import _ from "lodash";

import deepseekChat from "./chat.ts";
import util from "@/lib/util.ts";
import logger from "@/lib/logger.ts";

// ============================================================
//  Agent layer for kimi-free-api
//  Adds OpenAI tool/function calling (emulated via prompt) and
//  the OpenAI Responses API (/v1/responses) on top of Kimi's
//  chat.qwen.ai-style completion, so agents like Codex / Qwen
//  Code can write files and run commands through Kimi.
// ============================================================

const genId = () => util.uuid(false).slice(0, 24);

/** 生成 Kimi 文本（把提示词作为单条 user 消息发给 Kimi，关闭联网搜索） */
async function getText(
  model: string,
  promptContent: string,
  token: string
): Promise<{ responseContent: string; responseId: string }> {
  const completion: any = await deepseekChat.createCompletion(
    model,
    [{ role: "user", content: promptContent }],
    token,
    false
  );
  const responseContent = completion?.choices?.[0]?.message?.content || "";
  return { responseContent, responseId: completion?.id || genId() };
}

// ---------- tool-call emulation (shared) ----------

function buildToolSystemPrompt(tools: any[]): string {
  const toolDefs = tools
    .map((t) => (t && t.type === "function" && t.function ? t.function : t))
    .filter(Boolean);
  return [
    "# Tool Calling",
    "",
    "You are an agent with access to the tools listed below. When you need to use a tool, you MUST reply with one or more tool-call blocks in EXACTLY this format:",
    "",
    "<tool_call>",
    '{"name": "<tool_name>", "arguments": {<json-arguments>}}',
    "</tool_call>",
    "",
    "Strict rules:",
    "- When calling tools, output ONLY the <tool_call> block(s). No prose before or after.",
    '- "arguments" MUST be a valid JSON object matching the tool schema.',
    "- Emit several <tool_call> blocks to call multiple tools at once.",
    "- Tool results come back inside <tool_response> blocks; use them to continue.",
    "- When the task is done and you are giving the final answer, reply in plain text with NO <tool_call> block.",
    "",
    "Available tools (JSON Schema):",
    "<tools>",
    JSON.stringify(toolDefs, null, 2),
    "</tools>",
  ].join("\n");
}

function safeParseToolJson(raw: string): any {
  if (!raw) return null;
  let s = raw.trim();
  const fence = s.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/);
  if (fence) s = fence[1].trim();
  try {
    return JSON.parse(s);
  } catch {
    return null;
  }
}

function toOpenAIToolCall(parsed: any): any {
  const args = parsed.arguments;
  return {
    id: `call_${genId()}`,
    type: "function",
    function: {
      name: parsed.name,
      arguments: _.isString(args) ? args : JSON.stringify(args ?? {}),
    },
  };
}

function parseToolCalls(text: string): { content: string; toolCalls: any[] } {
  const toolCalls: any[] = [];
  const regex = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g;
  let match: RegExpExecArray | null;
  while ((match = regex.exec(text)) !== null) {
    const parsed = safeParseToolJson(match[1]);
    if (parsed && parsed.name) toolCalls.push(toOpenAIToolCall(parsed));
  }
  let content = text.replace(regex, "").trim();
  if (!toolCalls.length) {
    const bare = safeParseToolJson(text);
    if (bare && bare.name && bare.arguments !== undefined) {
      toolCalls.push(toOpenAIToolCall(bare));
      content = "";
    }
  }
  return { content, toolCalls };
}

function extractTextContent(content: any): string {
  if (_.isString(content)) return content;
  if (_.isArray(content)) {
    return content
      .filter(
        (p) =>
          p &&
          (p.type === "text" ||
            p.type === "input_text" ||
            p.type === "output_text") &&
          p.text
      )
      .map((p) => p.text)
      .join("\n");
  }
  return "";
}

const cap = (r: string) =>
  r === "system" ? "System" : r === "assistant" ? "Assistant" : "User";

/** OpenAI chat messages + tools -> single flattened prompt string */
function messagesPrepareWithTools(messages: any[], tools: any[]): string {
  const parts: string[] = [];
  if (_.isArray(tools) && tools.length) parts.push(buildToolSystemPrompt(tools));
  for (const msg of messages || []) {
    if (!msg) continue;
    if (msg.role === "tool") {
      const name = msg.name || msg.tool_call_id || "";
      parts.push(
        `Tool result${name ? ` for ${name}` : ""}:\n<tool_response>\n${extractTextContent(
          msg.content
        )}\n</tool_response>`
      );
      continue;
    }
    let c = extractTextContent(msg.content);
    if (msg.role === "assistant" && _.isArray(msg.tool_calls) && msg.tool_calls.length) {
      const calls = msg.tool_calls
        .map((tc: any) => {
          let args: any = tc.function?.arguments;
          try {
            args = JSON.parse(args);
          } catch {}
          return `<tool_call>\n${JSON.stringify({ name: tc.function?.name, arguments: args ?? {} })}\n</tool_call>`;
        })
        .join("\n");
      c = c ? `${c}\n${calls}` : calls;
    }
    if (c) parts.push(`${cap(msg.role || "user")}: ${c}`);
  }
  parts.push("Assistant:");
  return parts.join("\n\n");
}

/** Responses API input items + tools -> single flattened prompt string */
function prepareResponsesPrompt(instructions: any, input: any, tools: any[]): string {
  const parts: string[] = [];
  if (_.isArray(tools) && tools.length) parts.push(buildToolSystemPrompt(tools));
  if (instructions && _.isString(instructions)) parts.push(`System: ${instructions}`);
  const items = _.isString(input)
    ? [{ type: "message", role: "user", content: input }]
    : _.isArray(input)
    ? input
    : [];
  for (const item of items) {
    if (!item) continue;
    const type = item.type || "message";
    if (type === "message") {
      const text = extractTextContent(item.content);
      if (text) parts.push(`${cap(item.role || "user")}: ${text}`);
    } else if (type === "function_call") {
      let args: any = item.arguments;
      try {
        args = JSON.parse(args);
      } catch {}
      parts.push(
        `Assistant: <tool_call>\n${JSON.stringify({ name: item.name, arguments: args ?? {} })}\n</tool_call>`
      );
    } else if (type === "function_call_output") {
      const out = _.isString(item.output) ? item.output : JSON.stringify(item.output);
      parts.push(`Tool result:\n<tool_response>\n${out}\n</tool_response>`);
    }
  }
  parts.push("Assistant:");
  return parts.join("\n\n");
}

// ---------- chat/completions with tools ----------

async function createChatCompletion(
  model: string,
  messages: any[],
  token: string,
  tools?: any[],
  toolChoice?: any
) {
  const useTools = _.isArray(tools) && tools.length > 0 && toolChoice !== "none";
  if (!useTools) return deepseekChat.createCompletion(model, messages, token, false);

  const prompt = messagesPrepareWithTools(messages, tools);
  const { responseContent, responseId } = await getText(model, prompt, token);
  const { content, toolCalls } = parseToolCalls(responseContent);
  return {
    id: responseId,
    model,
    object: "chat.completion",
    choices: [
      {
        index: 0,
        message: toolCalls.length
          ? { role: "assistant", content: content || null, tool_calls: toolCalls }
          : { role: "assistant", content },
        finish_reason: toolCalls.length ? "tool_calls" : "stop",
      },
    ],
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
    created: util.unixTimestamp(),
  };
}

async function createChatCompletionStream(
  model: string,
  messages: any[],
  token: string,
  tools?: any[],
  toolChoice?: any
) {
  const useTools = _.isArray(tools) && tools.length > 0 && toolChoice !== "none";
  if (!useTools) return deepseekChat.createCompletionStream(model, messages, token, false);

  const completion: any = await createChatCompletion(model, messages, token, tools, toolChoice);
  const choice = completion.choices[0];
  const ts = new PassThrough();
  const base = { id: completion.id, model, object: "chat.completion.chunk", created: completion.created };
  const send = (delta: any, finish: any = null) =>
    ts.write(`data: ${JSON.stringify({ ...base, choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`);
  send({ role: "assistant", content: "" });
  if (choice.message.tool_calls?.length)
    send({
      tool_calls: choice.message.tool_calls.map((tc: any, i: number) => ({ index: i, ...tc })),
    });
  else if (choice.message.content) send({ content: choice.message.content });
  send({}, choice.finish_reason || "stop");
  ts.write("data: [DONE]\n\n");
  ts.end();
  return ts;
}

// ---------- Responses API ----------

function buildResponsesOutput(textContent: string, toolCalls: any[]): any[] {
  const output: any[] = [];
  if (textContent)
    output.push({
      type: "message",
      id: `msg_${genId()}`,
      status: "completed",
      role: "assistant",
      content: [{ type: "output_text", text: textContent, annotations: [] }],
    });
  for (const tc of toolCalls)
    output.push({
      type: "function_call",
      id: `fc_${genId()}`,
      call_id: tc.id,
      name: tc.function.name,
      arguments: tc.function.arguments,
      status: "completed",
    });
  return output;
}

async function createResponses(model: string, body: any, token: string) {
  const { instructions, input, tools, tool_choice } = body;
  const prompt = prepareResponsesPrompt(instructions, input, tools);
  const { responseContent } = await getText(model, prompt, token);
  const useTools = _.isArray(tools) && tools.length > 0 && tool_choice !== "none";
  let textContent = responseContent;
  let toolCalls: any[] = [];
  if (useTools) {
    const parsed = parseToolCalls(responseContent);
    textContent = parsed.content;
    toolCalls = parsed.toolCalls;
  }
  return {
    id: `resp_${genId()}`,
    object: "response",
    created_at: util.unixTimestamp(),
    status: "completed",
    model,
    output: buildResponsesOutput(textContent, toolCalls),
    usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
  };
}

function createResponsesStream(model: string, body: any, token: string) {
  const { instructions, input, tools, tool_choice } = body;
  const prompt = prepareResponsesPrompt(instructions, input, tools);
  const useTools = _.isArray(tools) && tools.length > 0 && tool_choice !== "none";
  const respId = `resp_${genId()}`;
  const ts = new PassThrough();
  let seq = 0;
  const emit = (type: string, obj: any) => {
    try {
      ts.write(`event: ${type}\ndata: ${JSON.stringify({ type, sequence_number: seq++, ...obj })}\n\n`);
    } catch {}
  };
  const resp = (status: string, out: any[], error?: any) => ({
    id: respId,
    object: "response",
    created_at: util.unixTimestamp(),
    status,
    model,
    output: out,
    error: error || null,
    usage: status === "completed" ? { input_tokens: 1, output_tokens: 1, total_tokens: 2 } : null,
  });

  emit("response.created", { response: resp("in_progress", []) });
  emit("response.in_progress", { response: resp("in_progress", []) });
  const heartbeat = setInterval(() => {
    try {
      ts.write(`: keepalive\n\n`);
    } catch {}
  }, 5000);

  (async () => {
    try {
      const { responseContent } = await getText(model, prompt, token);
      let textContent = responseContent;
      let toolCalls: any[] = [];
      if (useTools) {
        const parsed = parseToolCalls(responseContent);
        textContent = parsed.content;
        toolCalls = parsed.toolCalls;
      }
      const output = buildResponsesOutput(textContent, toolCalls);
      let idx = 0;
      for (const item of output) {
        if (item.type === "message") {
          const text = item.content[0].text;
          emit("response.output_item.added", { output_index: idx, item: { ...item, status: "in_progress", content: [] } });
          emit("response.content_part.added", { item_id: item.id, output_index: idx, content_index: 0, part: { type: "output_text", text: "", annotations: [] } });
          emit("response.output_text.delta", { item_id: item.id, output_index: idx, content_index: 0, delta: text });
          emit("response.output_text.done", { item_id: item.id, output_index: idx, content_index: 0, text });
          emit("response.content_part.done", { item_id: item.id, output_index: idx, content_index: 0, part: item.content[0] });
          emit("response.output_item.done", { output_index: idx, item });
        } else if (item.type === "function_call") {
          emit("response.output_item.added", { output_index: idx, item: { ...item, status: "in_progress", arguments: "" } });
          emit("response.function_call_arguments.delta", { item_id: item.id, output_index: idx, delta: item.arguments });
          emit("response.function_call_arguments.done", { item_id: item.id, output_index: idx, arguments: item.arguments });
          emit("response.output_item.done", { output_index: idx, item });
        }
        idx++;
      }
      emit("response.completed", { response: resp("completed", output) });
      ts.write("data: [DONE]\n\n");
    } catch (err: any) {
      logger.error("responses stream error:", err?.message || err);
      emit("response.failed", { response: resp("failed", [], { code: "upstream_error", message: String(err?.message || err) }) });
      ts.write("data: [DONE]\n\n");
    } finally {
      clearInterval(heartbeat);
      ts.end();
    }
  })();

  return ts;
}

export default {
  tokenSplit: deepseekChat.tokenSplit,
  createChatCompletion,
  createChatCompletionStream,
  createResponses,
  createResponsesStream,
};
