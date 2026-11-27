-- Client of the semantic cache sidecar (the Rust crate in semcache/).
--
-- The gateway asks the sidecar for a cached answer in the access phase
-- (lookup) and gives it the new answer after a miss (store). Both calls use
-- ngx.socket.tcp, a cosocket: while it waits for the sidecar, the NGINX worker
-- serves other requests. The log phase cannot use cosockets, so the store
-- runs in a zero-delay timer.
--
-- The cache fails open. An error or a timeout means "no cached answer", and
-- the request goes upstream as usual.

local cjson = require("cjson.safe")

local _M = {}

local concat, sort = table.concat, table.sort

-- Request fields that change the answer. They are part of the namespace, so a
-- cached answer is reused only for the same values.
local PARAMS = {
  "temperature", "top_p", "max_tokens", "max_completion_tokens", "seed", "stop",
  "presence_penalty", "frequency_penalty", "logit_bias", "response_format",
  "reasoning_effort",
}

-- Text of a value with object keys sorted, so equal values give equal text
-- in every worker.
local function canonical(v)
  local t = type(v)
  if v == nil or v == cjson.null then
    return "null"
  elseif t == "string" then
    return ("%q"):format(v)
  elseif t ~= "table" then
    return tostring(v)
  end
  local parts = {}
  if #v > 0 then
    for i = 1, #v do parts[i] = canonical(v[i]) end
    return "[" .. concat(parts, ",") .. "]"
  end
  local keys = {}
  for k in pairs(v) do keys[#keys + 1] = tostring(k) end
  sort(keys)
  for i, k in ipairs(keys) do parts[i] = ("%q:%s"):format(k, canonical(v[k])) end
  return "{" .. concat(parts, ",") .. "}"
end
_M.canonical = canonical

-- Text of a message content, or nil when it holds anything but text parts.
local function text_only(content)
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

-- Decide whether a request can use the cache.
-- Returns namespace, text when it can, or nil, reason when it cannot.
--
-- Only single-turn requests qualify: system or developer messages plus one
-- user message, all plain text, no tools, one choice. The answer to a
-- multi-turn chat depends on its history, and the embedding sees only one text.
-- The namespace is an exact hash of the tier, the model, the system prompt and
-- the PARAMS fields. Only the user message is compared by meaning.
function _M.cacheable(body, target)
  if body.n ~= nil and body.n ~= 1 then return nil, "n" end
  if body.tools ~= nil or body.functions ~= nil or body.tool_choice ~= nil then
    return nil, "tools"
  end

  local system, user = {}, nil
  for _, m in ipairs(body.messages) do
    local text = text_only(m.content)
    if not text then return nil, "non_text" end
    if m.role == "system" or m.role == "developer" then
      system[#system + 1] = text
    elseif m.role == "user" and not user then
      user = text
    else
      return nil, "multi_turn"
    end
  end
  if not user or not user:find("%S") then return nil, "no_user_text" end

  local params = {}
  for _, k in ipairs(PARAMS) do params[k] = body[k] end
  local key = concat({ target.name, target.model, concat(system, "\n"), canonical(params) }, "\0")
  return ngx.md5(key), user
end

-- One request and one reply line over a pooled connection.
local function call(cfg, payload)
  local sock = ngx.socket.tcp()
  sock:settimeouts(cfg.timeout, cfg.timeout, cfg.timeout)

  local ok, err
  if cfg.unix then
    ok, err = sock:connect(cfg.unix)
  else
    ok, err = sock:connect(cfg.host, cfg.port)
  end
  if not ok then return nil, "connect: " .. tostring(err) end

  local line, encode_err = cjson.encode(payload)
  if not line then
    sock:close()
    return nil, "encode: " .. tostring(encode_err)
  end
  local _, send_err = sock:send(line .. "\n")
  if send_err then
    sock:close()
    return nil, "send: " .. send_err
  end
  local reply, receive_err = sock:receive("*l")
  if not reply then
    sock:close()
    return nil, "receive: " .. receive_err
  end
  sock:setkeepalive(60000, 64)

  local resp = cjson.decode(reply)
  if type(resp) ~= "table" then return nil, "invalid reply" end
  if resp.ok ~= true then return nil, "sidecar: " .. tostring(resp.error) end
  return resp
end

-- Score a prompt with the trained router of the sidecar.
-- Returns the reply: {score, strong, route_ms, embed_ms, version}, or nil and an error.
-- id may be nil: then the sidecar does not keep the embedding for a lookup.
function _M.route(cfg, id, text)
  return call(cfg, { op = "route", id = id, text = text })
end

-- Score a prompt with the guardrail classifier of the sidecar.
-- Returns the reply: {score, injection, guard_ms, version}, or nil and an error.
function _M.guard(cfg, text)
  return call(cfg, { op = "guard", text = text })
end

-- Returns the sidecar reply: {hit, score, rerank, response, skip, embed_ms}, or nil and an error.
function _M.lookup(cfg, id, ns, text)
  return call(cfg, { op = "lookup", id = id, ns = ns, text = text, threshold = cfg.threshold })
end

-- Store an answer after the response is sent. Safe to call from the log phase.
function _M.store_async(cfg, id, ns, text, response)
  local ok, err = ngx.timer.at(0, function(premature)
    if premature then return end
    local _, store_err = call(cfg, {
      op = "store", id = id, ns = ns, text = text, ttl = cfg.ttl, response = response,
    })
    if store_err then ngx.log(ngx.WARN, "aigw cache: store failed: ", store_err) end
  end)
  if not ok then ngx.log(ngx.WARN, "aigw cache: cannot schedule the store: ", err) end
end

return _M
