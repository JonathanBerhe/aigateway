-- Fake OpenAI-compatible chat completions upstream, and a fake Anthropic
-- Messages API (messages() below).
--
-- It returns a fixed reply, as JSON or as an SSE stream, and echoes what it
-- received in X-Mock-* response headers. The smoke test reads those headers
-- to make sure that the gateway rewrote the request correctly.
--
-- Test-only request fields (the gateway passes them through unchanged):
--   mock_omit_usage = true    send no usage, so the gateway must estimate tokens
--   mock_split_events = true  split each SSE event over two network writes
--   mock_status = 503         answer with this error status

local cjson = require("cjson.safe")

local _M = {}

-- Fixed so that the tests can tell upstream usage from a local estimate.
local PROMPT_TOKENS = 42

local function words(text)
  local out = {}
  for w in text:gmatch("%S+%s*") do out[#out + 1] = w end
  return out
end

local function sse_writer(split)
  return function(payload)
    local event = "data: " .. payload .. "\n\n"
    if split then
      local half = math.floor(#event / 2)
      ngx.print(event:sub(1, half))
      ngx.flush(true)
      ngx.sleep(0.005)
      ngx.print(event:sub(half + 1))
    else
      ngx.print(event)
    end
    ngx.flush(true)
  end
end

function _M.chat(tier)
  ngx.req.read_body()
  local body = cjson.decode(ngx.req.get_body_data() or "")
  if type(body) ~= "table" then
    ngx.status = 400
    ngx.say('{"error":{"message":"mock: body is not JSON","type":"invalid_request_error"}}')
    return
  end

  local h = ngx.header
  h["X-Mock-Tier"] = tier
  h["X-Mock-Model"] = tostring(body.model)
  h["X-Mock-Authorization"] = ngx.var.http_authorization or "none"
  h["X-Mock-Include-Usage"] = tostring(type(body.stream_options) == "table"
    and body.stream_options.include_usage == true)

  if type(body.mock_status) == "number" then
    ngx.status = body.mock_status
    h.content_type = "application/json"
    ngx.say('{"error":{"message":"mock: forced error","type":"api_error"}}')
    return
  end

  local reply = ("This is a mock reply from the %s tier."):format(tier)
  local parts = words(reply)
  local usage = not body.mock_omit_usage and {
    prompt_tokens = PROMPT_TOKENS,
    completion_tokens = #parts,
    total_tokens = PROMPT_TOKENS + #parts,
  } or nil
  local id, created = "chatcmpl-mock-" .. ngx.var.request_id, ngx.time()

  if body.stream ~= true then
    h.content_type = "application/json"
    ngx.say(cjson.encode({
      id = id, object = "chat.completion", created = created, model = body.model,
      choices = { { index = 0, finish_reason = "stop",
                    message = { role = "assistant", content = reply } } },
      usage = usage,
    }))
    return
  end

  h.content_type = "text/event-stream"
  h.cache_control = "no-cache"
  local send = sse_writer(body.mock_split_events == true)
  local function chunk(choices, extra)
    local event = { id = id, object = "chat.completion.chunk", created = created,
                    model = body.model, choices = choices }
    for k, v in pairs(extra or {}) do event[k] = v end
    send(cjson.encode(event))
  end

  chunk({ { index = 0, delta = { role = "assistant", content = "" } } })
  for _, w in ipairs(parts) do
    chunk({ { index = 0, delta = { content = w } } })
  end
  chunk({ { index = 0, delta = {}, finish_reason = "stop" } })
  if usage and type(body.stream_options) == "table" and body.stream_options.include_usage then
    chunk(cjson.empty_array, { usage = usage })
  end
  ngx.print("data: [DONE]\n\n")
end

-- Fake Anthropic Messages API. It echoes the translated request in X-Mock-*
-- headers and answers in the Anthropic format, as JSON or as an SSE stream.
-- The gateway builds a new body for this API and drops the mock_* fields, so
-- request headers control the test cases: X-Mock-Status forces an error, and
-- X-Mock-Split: true splits each SSE event over two writes.
local function anthropic_event(send_raw, kind, data)
  data.type = kind
  send_raw("event: " .. kind .. "\ndata: " .. cjson.encode(data) .. "\n\n")
end

function _M.messages(tier)
  ngx.req.read_body()
  local body = cjson.decode(ngx.req.get_body_data() or "")
  local h = ngx.header
  h["X-Mock-Tier"] = tier
  h["X-Mock-Format"] = "anthropic"
  if type(body) ~= "table" then
    ngx.status = 400
    h.content_type = "application/json"
    ngx.say('{"type":"error","error":{"type":"invalid_request_error","message":"mock: body is not JSON"}}')
    return
  end
  h["X-Mock-Model"] = tostring(body.model)
  h["X-Mock-Api-Key"] = ngx.var.http_x_api_key or "none"
  h["X-Mock-Authorization"] = ngx.var.http_authorization or "none"
  h["X-Mock-Anthropic-Version"] = ngx.var.http_anthropic_version or "none"
  h["X-Mock-System"] = type(body.system) == "string" and body.system or "none"
  h["X-Mock-Max-Tokens"] = tostring(body.max_tokens)
  local roles = {}
  for _, m in ipairs(type(body.messages) == "table" and body.messages or {}) do roles[#roles + 1] = m.role end
  h["X-Mock-Roles"] = table.concat(roles, ",")
  h["X-Mock-Stop"] = type(body.stop_sequences) == "table" and table.concat(body.stop_sequences, ",") or "none"

  local forced = tonumber(ngx.var.http_x_mock_status)
  if forced then
    ngx.status = forced
    h.content_type = "application/json"
    ngx.say('{"type":"error","error":{"type":"overloaded_error","message":"mock: forced error"}}')
    return
  end

  local reply = ("This is a mock reply from the %s tier."):format(tier)
  local parts = words(reply)
  local id = "msg_mock_" .. ngx.var.request_id
  if body.stream ~= true then
    h.content_type = "application/json"
    ngx.say(cjson.encode({
      id = id, type = "message", role = "assistant", model = body.model,
      content = { { type = "text", text = reply } },
      stop_reason = "end_turn", stop_sequence = cjson.null,
      usage = { input_tokens = PROMPT_TOKENS, output_tokens = #parts },
    }))
    return
  end

  h.content_type = "text/event-stream"
  h.cache_control = "no-cache"
  -- Split each event over two writes when the client asks, to test that the
  -- translator keeps a partial event.
  local split = ngx.var.http_x_mock_split == "true"
  local function send_raw(text)
    if split then
      local half = math.floor(#text / 2)
      ngx.print(text:sub(1, half))
      ngx.flush(true)
      ngx.sleep(0.005)
      ngx.print(text:sub(half + 1))
    else
      ngx.print(text)
    end
    ngx.flush(true)
  end
  anthropic_event(send_raw, "message_start", { message = {
    id = id, type = "message", role = "assistant", model = body.model, content = cjson.empty_array,
    stop_reason = cjson.null, usage = { input_tokens = PROMPT_TOKENS, output_tokens = 1 },
  } })
  anthropic_event(send_raw, "content_block_start", { index = 0, content_block = { type = "text", text = "" } })
  anthropic_event(send_raw, "ping", {})
  for _, w in ipairs(parts) do
    anthropic_event(send_raw, "content_block_delta", { index = 0, delta = { type = "text_delta", text = w } })
  end
  anthropic_event(send_raw, "content_block_stop", { index = 0 })
  anthropic_event(send_raw, "message_delta", { delta = { stop_reason = "end_turn", stop_sequence = cjson.null },
                                               usage = { output_tokens = #parts } })
  anthropic_event(send_raw, "message_stop", {})
end

return _M
