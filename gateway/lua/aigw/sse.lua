-- Incremental parser for OpenAI-style server-sent event (SSE) streams.
--
-- body_filter_by_lua gives the response in chunks of any size: a chunk can
-- hold several events or end in the middle of one. feed() keeps the partial
-- last line and handles every complete line. The parser only observes the
-- stream. It never changes the bytes that go to the client.
--
-- It records:
--   usage    the last `usage` object seen. OpenAI-compatible servers send it in
--            the final chunk when the request sets stream_options.include_usage.
--   content  the generated text, kept up to max_content bytes, for a token
--            estimate when the upstream sends no usage, and for the cache.
--   content_truncated  true when the text was longer than max_content.
--   finish_reason      the last finish_reason seen ("stop", "length", ...).
--   done     true after the `data: [DONE]` line.

local _M = {}
local mt = { __index = _M }

local find, sub, byte = string.find, string.sub, string.byte
local concat = table.concat

local MAX_LINE = 1024 * 1024

-- decode: a JSON decoder that returns nil on bad input (cjson.safe.decode).
function _M.new(decode, max_content)
  return setmetatable({
    decode = decode,
    max_content = max_content or 256 * 1024,
    pending = "",
    parts = {},
    content_bytes = 0,
    usage = nil,
    events = 0,
    done = false,
    dropped_lines = 0,
  }, mt)
end

function _M:_add_content(text)
  local room = self.max_content - self.content_bytes
  if #text > room then
    self.content_truncated = true
    if room <= 0 then return end
    text = sub(text, 1, room)
  end
  self.parts[#self.parts + 1] = text
  self.content_bytes = self.content_bytes + #text
end

function _M:_line(line)
  -- Field lines look like "data: payload". Comments start with ":".
  if sub(line, 1, 5) ~= "data:" then return end
  local payload = sub(line, 6)
  if byte(payload, 1) == 32 then payload = sub(payload, 2) end -- optional space
  if payload == "[DONE]" then
    self.done = true
    return
  end

  local event = self.decode(payload)
  if type(event) ~= "table" then return end
  self.events = self.events + 1

  if type(event.usage) == "table" then
    self.usage = event.usage
  end
  local choices = event.choices
  if type(choices) == "table" then
    for _, choice in ipairs(choices) do
      if type(choice) == "table" then
        local delta = choice.delta
        if type(delta) == "table" and type(delta.content) == "string" then
          self:_add_content(delta.content)
        end
        if type(choice.finish_reason) == "string" then
          self.finish_reason = choice.finish_reason
        end
      end
    end
  end
end

function _M:feed(chunk)
  local data = self.pending .. chunk
  local pos = 1
  while true do
    local nl = find(data, "\n", pos, true)
    if not nl then break end
    local stop = nl - 1
    if byte(data, stop) == 13 then stop = stop - 1 end -- strip \r
    self:_line(sub(data, pos, stop))
    pos = nl + 1
  end
  self.pending = sub(data, pos)
  if #self.pending > MAX_LINE then
    -- A line this long is not an OpenAI event. Drop it to bound memory.
    self.pending = ""
    self.dropped_lines = self.dropped_lines + 1
  end
end

function _M:content()
  return concat(self.parts)
end

return _M
