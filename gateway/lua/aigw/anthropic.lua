-- Translation between the OpenAI chat format and the Anthropic Messages API.
--
-- A tier with AIGW_<TIER>_FORMAT=anthropic gets the client's OpenAI request
-- as an Anthropic request, and the client gets the Anthropic answer back in
-- the OpenAI format: as one JSON object, or as an SSE stream of
-- chat.completion.chunk events.
--
-- What maps:
--   system and developer messages  ->  the top-level "system" string
--   user and assistant messages    ->  "messages", text only
--   max_completion_tokens or max_tokens -> "max_tokens" (Anthropic requires
--                                      it, so the tier has a default)
--   temperature, top_p             ->  the same fields
--   stop (string or array)         ->  "stop_sequences"
--   stop_reason end_turn, stop_sequence -> finish_reason "stop"
--   stop_reason max_tokens         ->  "length", tool_use -> "tool_calls"
--   usage input_tokens, output_tokens -> prompt_tokens, completion_tokens
--
-- Tools, images, n > 1 and the tool role do not map. The gateway answers
-- such a request with HTTP 400 before it calls the upstream.

local cjson = require("cjson.safe")

local _M = {}

_M.API_VERSION = "2023-06-01"

local concat, find, sub = table.concat, string.find, string.sub

local FINISH = {
  end_turn = "stop",
  stop_sequence = "stop",
  max_tokens = "length",
  tool_use = "tool_calls",
}

-- Text of an OpenAI message content: a string, or an array of text parts.
local function text_of(content)
  if type(content) == "string" then return content end
  if type(content) ~= "table" then return nil end
  local out = {}
  for _, part in ipairs(content) do
    if type(part) ~= "table" or part.type ~= "text" or type(part.text) ~= "string" then
      return nil
    end
    out[#out + 1] = part.text
  end
  return concat(out, "\n")
end

-- OpenAI request body -> Anthropic request body, or nil and the reason.
function _M.request(body, model, default_max_tokens)
  if body.tools ~= nil or body.functions ~= nil or body.tool_choice ~= nil then
    return nil, "tools are not supported for this model"
  end
  if body.n ~= nil and body.n ~= 1 then
    return nil, "'n' must be 1 for this model"
  end

  local system, messages = {}, {}
  for i, m in ipairs(body.messages) do
    local text = text_of(m.content)
    if not text then
      return nil, ("'messages[%d]': only text content is supported for this model"):format(i - 1)
    end
    if m.role == "system" or m.role == "developer" then
      system[#system + 1] = text
    elseif m.role == "user" or m.role == "assistant" then
      messages[#messages + 1] = { role = m.role, content = text }
    else
      return nil, ("'messages[%d]': the role '%s' is not supported for this model"):format(i - 1, tostring(m.role))
    end
  end
  if #messages == 0 then
    return nil, "the request needs a user message"
  end

  local out = {
    model = model,
    messages = messages,
    max_tokens = tonumber(body.max_completion_tokens) or tonumber(body.max_tokens) or default_max_tokens,
    temperature = tonumber(body.temperature),
    top_p = tonumber(body.top_p),
  }
  if #system > 0 then out.system = concat(system, "\n\n") end
  if body.stream == true then out.stream = true end
  if type(body.stop) == "string" then
    out.stop_sequences = { body.stop }
  elseif type(body.stop) == "table" and #body.stop > 0 then
    out.stop_sequences = body.stop
  end
  return out
end

local function openai_usage(input, output)
  if not input then return nil end
  output = output or 0
  return { prompt_tokens = input, completion_tokens = output, total_tokens = input + output }
end

-- Anthropic message -> OpenAI chat.completion.
function _M.response(resp, created)
  local parts = {}
  for _, block in ipairs(type(resp.content) == "table" and resp.content or {}) do
    if type(block) == "table" and block.type == "text" and type(block.text) == "string" then
      parts[#parts + 1] = block.text
    end
  end
  local u = type(resp.usage) == "table" and resp.usage or {}
  return {
    id = resp.id,
    object = "chat.completion",
    created = created,
    model = resp.model,
    choices = { {
      index = 0,
      message = { role = "assistant", content = concat(parts) },
      finish_reason = FINISH[resp.stop_reason] or "stop",
    } },
    usage = openai_usage(tonumber(u.input_tokens), tonumber(u.output_tokens)),
  }
end

-- Anthropic error body -> OpenAI error body.
function _M.error(resp)
  local e = type(resp) == "table" and type(resp.error) == "table" and resp.error or {}
  return { error = {
    message = type(e.message) == "string" and e.message or "upstream error",
    type = type(e.type) == "string" and e.type or "api_error",
    code = cjson.null,
  } }
end

-- Incremental translator of an Anthropic SSE stream into OpenAI SSE text.
-- feed() takes a chunk of upstream bytes and returns the OpenAI bytes for
-- every complete event in it. An event can span chunks.
local Stream = {}
Stream.__index = Stream

-- with_usage: also send the usage chunk (choices = []) before [DONE].
function _M.stream(created, with_usage)
  return setmetatable({ pending = "", created = created, with_usage = with_usage, out = {} }, Stream)
end

function Stream:_chunk(choices, extra)
  local event = { id = self.id, object = "chat.completion.chunk", created = self.created,
                  model = self.model, choices = choices }
  for k, v in pairs(extra or {}) do event[k] = v end
  self.out[#self.out + 1] = "data: " .. cjson.encode(event) .. "\n\n"
end

function Stream:_event(data)
  local kind = data.type
  if kind == "message_start" and type(data.message) == "table" then
    local m = data.message
    self.id, self.model = m.id, m.model
    self.input = type(m.usage) == "table" and tonumber(m.usage.input_tokens) or nil
    self:_chunk({ { index = 0, delta = { role = "assistant", content = "" }, finish_reason = cjson.null } })
  elseif kind == "content_block_delta" and type(data.delta) == "table" and data.delta.type == "text_delta" then
    self:_chunk({ { index = 0, delta = { content = data.delta.text }, finish_reason = cjson.null } })
  elseif kind == "message_delta" then
    if type(data.delta) == "table" then self.stop_reason = data.delta.stop_reason end
    if type(data.usage) == "table" then self.output = tonumber(data.usage.output_tokens) end
  elseif kind == "message_stop" then
    self:_chunk({ { index = 0, delta = {}, finish_reason = FINISH[self.stop_reason] or "stop" } })
    local usage = openai_usage(self.input, self.output)
    if self.with_usage and usage then self:_chunk(cjson.empty_array, { usage = usage }) end
    self.out[#self.out + 1] = "data: [DONE]\n\n"
  elseif kind == "error" then
    self.out[#self.out + 1] = "data: " .. cjson.encode(_M.error(data)) .. "\n\n"
    self.out[#self.out + 1] = "data: [DONE]\n\n"
  end
  -- ping, content_block_start and content_block_stop carry nothing to map.
end

function Stream:feed(chunk)
  local buf = self.pending .. chunk:gsub("\r\n", "\n")
  local start = 1
  while true do
    local i = find(buf, "\n\n", start, true)
    if not i then break end
    local event = sub(buf, start, i - 1)
    start = i + 2
    for line in event:gmatch("[^\n]+") do
      if sub(line, 1, 5) == "data:" then
        local data = cjson.decode((sub(line, 6):gsub("^ ", "")))
        if type(data) == "table" then self:_event(data) end
      end
    end
  end
  self.pending = sub(buf, start)
  local out = concat(self.out)
  self.out = {}
  return out
end

return _M
