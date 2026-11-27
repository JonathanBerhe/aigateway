-- Request handler for POST /v1/chat/completions, one function per NGINX phase.
--
--   access         authenticate, parse the body, check the prompt guard, pick a
--                  tier, rewrite the request
--   (proxy_pass)   NGINX sends the request upstream and streams the reply back
--   header_filter  add X-AIGW-* response headers, choose how to read the body
--   body_filter    observe the reply (SSE or JSON) to find token usage
--   log            finish token accounting and update the metrics
--
-- State for one request lives in ngx.ctx.aigw, which survives across the
-- phases of that request.

local cjson = require("cjson.safe").new()
local config = require("aigw.config")
local core = require("aigw.core")
local request = require("aigw.request")
local sse = require("aigw.sse")
local metrics = require("aigw.metrics")
local semcache = require("aigw.semcache")
local anthropic = require("aigw.anthropic")

-- Keep empty JSON arrays as arrays when a body is decoded and encoded again.
-- Without this, "tools": [] reaches the upstream as "tools": {}.
cjson.decode_array_with_array_mt(true)
-- OpenResty's lua-cjson fork has these two options. Older builds do not.
pcall(cjson.encode_escape_forward_slash, false)
pcall(cjson.encode_number_precision, 16)

local find, concat = string.find, table.concat

local _M = {}

local MODEL_IDS = { config.AUTO_MODEL, "aigw-cheap", "aigw-strong" }

-- Send an OpenAI-style error body and end the request.
local function json_error(status, err_type, message, code)
  ngx.status = status
  ngx.header.content_type = "application/json"
  ngx.say(cjson.encode({
    error = { message = message, type = err_type, code = code or cjson.null },
  }))
  -- ngx.exit(ngx.HTTP_OK) after a response body ends the whole request,
  -- including the proxy_pass content phase. ngx.status keeps the real status.
  return ngx.exit(ngx.HTTP_OK)
end

-- The consumer key comes from "Authorization: Bearer <key>" (OpenAI SDKs)
-- or from "X-Api-Key: <key>".
local function client_key()
  local h = ngx.var.http_authorization
  if h then
    local key = h:match("^%s*[Bb]earer%s+(%S+)%s*$")
    if key then return key end
  end
  return ngx.var.http_x_api_key
end

local function authenticated(cfg)
  if not cfg.api_keys then return true end
  local key = client_key()
  return key ~= nil and cfg.api_keys[key] == true
end

-- Text and finish_reason of a non-streaming chat completion.
local function completion_text(resp)
  local out, finish = {}, nil
  if type(resp.choices) == "table" then
    for _, choice in ipairs(resp.choices) do
      local msg = type(choice) == "table" and choice.message
      if type(msg) == "table" and type(msg.content) == "string" then
        out[#out + 1] = msg.content
      end
      if type(choice) == "table" and type(choice.finish_reason) == "string" then
        finish = choice.finish_reason
      end
    end
  end
  return concat(out, "\n"), finish
end

-- Decide the token counts of a finished (or aborted) request. Upstream usage
-- wins. Without it, fall back to the local estimate from the Rust core.
-- Runs at most once: at the end of the body, or in the log phase if the
-- client went away before the end.
local function finalize(a)
  if a.finalized then return end
  a.finalized = true

  if a.cached then
    -- Served from the semantic cache: these tokens were not spent upstream.
    local u = a.cached.usage
    a.prompt_tokens = type(u) == "table" and tonumber(u.prompt_tokens) or a.prompt_tokens_est
    a.completion_tokens = type(u) == "table" and tonumber(u.completion_tokens)
      or core.estimate_tokens(a.cached.content or "")
    a.token_source = "cache"
  elseif not a.upstream_status or a.upstream_status >= 400 or a.body_truncated then
    a.token_source = "none"
  else
    local usage, completion
    if a.sse then
      usage, completion = a.sse.usage, a.sse:content()
      a.finish_reason, a.content_truncated = a.sse.finish_reason, a.sse.content_truncated
    elseif a.body_parts then
      local resp = cjson.decode(concat(a.body_parts))
      a.body_parts = nil
      if type(resp) == "table" then
        usage = resp.usage
        completion, a.finish_reason = completion_text(resp)
      end
    end
    a.completion_text = completion

    local prompt = type(usage) == "table" and tonumber(usage.prompt_tokens)
    if prompt then
      a.prompt_tokens = prompt
      a.completion_tokens = tonumber(usage.completion_tokens) or 0
      a.token_source = "upstream"
    else
      a.prompt_tokens = a.prompt_tokens_est
      a.completion_tokens = completion and core.estimate_tokens(completion) or 0
      a.token_source = "estimate"
    end
  end

  local var = ngx.var
  var.aigw_prompt_tokens = a.prompt_tokens or 0
  var.aigw_completion_tokens = a.completion_tokens or 0
  var.aigw_token_source = a.token_source
end

-- Answer from the cache, as JSON or as an SSE stream like an upstream would.
-- The whole text goes in one SSE chunk, so no UTF-8 character is split.
local function serve_cached(body, cached, wants_usage)
  local id = "chatcmpl-cache-" .. ngx.var.request_id
  local created = ngx.time()
  local u = cached.usage
  local usage = type(u) == "table" and {
    prompt_tokens = u.prompt_tokens,
    completion_tokens = u.completion_tokens,
    total_tokens = (tonumber(u.prompt_tokens) or 0) + (tonumber(u.completion_tokens) or 0),
  } or nil

  if body.stream ~= true then
    ngx.header.content_type = "application/json"
    ngx.say(cjson.encode({
      id = id, object = "chat.completion", created = created, model = cached.model,
      choices = { { index = 0, finish_reason = cached.finish_reason,
                    message = { role = "assistant", content = cached.content } } },
      usage = usage,
    }))
    return ngx.exit(ngx.HTTP_OK)
  end

  ngx.header.content_type = "text/event-stream"
  ngx.header.cache_control = "no-cache"
  local function event(choices, extra_usage)
    ngx.print("data: ", cjson.encode({
      id = id, object = "chat.completion.chunk", created = created, model = cached.model,
      choices = choices, usage = extra_usage,
    }), "\n\n")
  end
  event({ { index = 0, delta = { role = "assistant", content = cached.content } } })
  event({ { index = 0, delta = {}, finish_reason = cached.finish_reason } })
  if wants_usage and usage then event(cjson.empty_array, usage) end
  ngx.print("data: [DONE]\n\n")
  return ngx.exit(ngx.HTTP_OK)
end

-- Look the request up in the semantic cache.
-- Returns status ("hit", "miss", "skip", "bypass" or "error"), the sidecar
-- reply, and on a miss the {ns, text} needed to store the answer later.
local function cache_lookup(cache_cfg, body, target)
  if ngx.var.http_x_aigw_cache == "bypass" then return "bypass" end
  local ns, text = semcache.cacheable(body, target)
  if not ns then return "skip" end

  local reply, err = semcache.lookup(cache_cfg, ngx.var.request_id, ns, text)
  if not reply then
    ngx.log(ngx.WARN, "aigw cache: lookup failed, the request goes upstream: ", err)
    return "error"
  end
  if reply.hit and type(reply.response) == "table" then return "hit", reply end
  if reply.skip then return "skip", reply end
  return "miss", reply, { ns = ns, text = text }
end

-- Canary of the model router: with AIGW_ROUTER_MODEL_SHARE below 1, only
-- that share of consumers gets the model router. A hash of the consumer key
-- picks them, so one consumer always gets the same router. Without a key,
-- the request id decides.
local function model_router_serves(r)
  if r.share >= 1 then return true end
  if r.share <= 0 then return false end
  local key = client_key() or ngx.var.request_id
  return ngx.crc32_short(key) % 10000 < r.share * 10000
end

-- Count one shadow decision next to the active tier.
local function count_shadow(active_tier, shadow_tier, shadow_router)
  metrics.incr("aigw_router_shadow_total", { active = active_tier, shadow = shadow_tier, router = shadow_router })
end

-- A model shadow calls the sidecar in a timer after the response, so it
-- adds no latency. Only the metrics see its decision.
local function shadow_model_async(s, text, active_tier)
  local ok, err = ngx.timer.at(0, function(premature)
    if premature then return end
    local reply = semcache.route(s, nil, text)
    local shadow = "error"
    if reply and type(reply.strong) == "boolean" then
      shadow = reply.strong and "strong" or "cheap"
    end
    count_shadow(active_tier, shadow, "model")
  end)
  if not ok then ngx.log(ngx.WARN, "aigw: cannot schedule the shadow router: ", err) end
end

-- Prompt guard. The Rust deny list runs first, in microseconds. With
-- AIGW_GUARD=model, a prompt that it lets through goes to the classifier of
-- the sidecar. If the sidecar fails, the prompt passes (fail open), and the
-- result is "error". Returns {result, source, rule or score}, where result
-- is pass, block, flag or error.
local function check_guard(g, text)
  local rule, err = core.guard(text)
  if rule == nil then
    ngx.log(ngx.ERR, "aigw: ", err)
  elseif rule then
    return { result = g.action, source = "regex", rule = rule }
  end
  if g.mode ~= "model" then return { result = "pass", source = "regex" } end

  local reply, sidecar_err = semcache.guard(g, text)
  if reply and type(reply.injection) == "boolean" then
    local score = tonumber(reply.score)
    local version = type(reply.version) == "string" and reply.version or nil
    return { result = reply.injection and g.action or "pass", source = "model", score = score, version = version }
  end
  ngx.log(ngx.WARN, "aigw: guard sidecar: ", sidecar_err or "no decision", "; the prompt passes")
  return { result = "error", source = "model" }
end

-- Pick the tier: the router for "auto" (or no model), the alias otherwise.
-- With AIGW_ROUTER=model, the trained router of the sidecar decides. If the
-- sidecar fails or times out, the heuristic decides instead.
-- Returns tier, score, reason, route version, or nil for an unknown model.
local function pick_tier(cfg, requested, last_user)
  if requested ~= nil and requested ~= config.AUTO_MODEL then
    local tier = cfg.aliases[requested]
    if not tier then return nil end
    return tier, nil, "pinned"
  end
  local asked = cfg.router ~= nil and model_router_serves(cfg.router)
  if asked then
    local reply, err = semcache.route(cfg.router, ngx.var.request_id, last_user)
    if reply and type(reply.strong) == "boolean" then
      local version = type(reply.version) == "string" and reply.version or nil
      return reply.strong and config.TIER_STRONG or config.TIER_CHEAP, tonumber(reply.score), "router_model", version
    end
    ngx.log(ngx.WARN, "aigw: router sidecar: ", err or "no decision", "; using the heuristic router")
  end
  local t, s = core.route(last_user, cfg.router_threshold)
  if not t then
    ngx.log(ngx.ERR, "aigw: ", s, "; sending the request to the cheap tier")
    return config.TIER_CHEAP, nil, "router_error"
  end
  -- router_fallback only when the model router was asked and failed.
  return t, s, asked and "router_fallback" or "router"
end

-- Shadow router: measured next to the active decision, never used.
local function run_shadow(cfg, last_user, active_tier)
  local shadow = cfg.router_shadow
  if shadow.mode == "heuristic" then
    local t = core.route(last_user, cfg.router_threshold)
    local shadow_tier = t and cfg.tiers[t].name or "error"
    count_shadow(active_tier, shadow_tier, "heuristic")
    ngx.var.aigw_shadow_tier = shadow_tier
  else
    shadow_model_async(shadow, last_user, active_tier)
  end
end

-- Rewrite the request for the target: body, path and keys. The consumer key
-- never goes upstream. It is cleared before the upstream key is set, because
-- Anthropic reads its key from the same X-Api-Key header.
-- Returns nil, or status, error type, message and code for an error reply.
local function rewrite_upstream(a, body, target)
  local var = ngx.var
  local wants_usage = type(body.stream_options) == "table" and body.stream_options.include_usage == true
  local upstream_body = body
  if target.format == "anthropic" then
    local translated, reason = anthropic.request(body, target.model, target.max_tokens)
    if not translated then
      return 400, "invalid_request_error", reason, "unsupported_for_model"
    end
    upstream_body = translated
    -- The answer comes back in the Anthropic format. The response filters
    -- translate it, as a stream or as one JSON object.
    a.translate = { stream = body.stream == true, with_usage = wants_usage or target.stream_usage }
  else
    body.model = target.model
    if body.stream == true and target.stream_usage then
      if type(body.stream_options) ~= "table" then body.stream_options = {} end
      body.stream_options.include_usage = true
    end
  end
  local encoded, err = cjson.encode(upstream_body)
  if not encoded then
    ngx.log(ngx.ERR, "aigw: cannot encode the upstream body: ", err)
    return 500, "api_error", "cannot encode the upstream request"
  end
  ngx.req.set_body_data(encoded)

  var.aigw_target_scheme = target.scheme
  var.aigw_target_authority = target.authority
  var.aigw_target_sni = target.host
  ngx.req.clear_header("Authorization")
  ngx.req.clear_header("X-Api-Key")
  ngx.req.clear_header("X-AIGW-Cache")
  if target.format == "anthropic" then
    var.aigw_target_path = target.messages_path
    ngx.req.set_header("X-Api-Key", target.api_key)
    ngx.req.set_header("anthropic-version", anthropic.API_VERSION)
  else
    var.aigw_target_path = target.chat_path
    if target.api_key then
      ngx.req.set_header("Authorization", "Bearer " .. target.api_key)
    end
  end
end

function _M.access()
  local cfg = config.get()
  if not authenticated(cfg) then
    return json_error(401, "authentication_error", "missing or invalid API key", "invalid_api_key")
  end

  -- client_body_buffer_size equals client_max_body_size in nginx.conf, so a
  -- valid body is always in memory. A body in a temp file means a config error.
  ngx.req.read_body()
  local raw = ngx.req.get_body_data()
  if not raw then
    if ngx.req.get_body_file() then
      return json_error(413, "invalid_request_error", "request body is too large")
    end
    return json_error(400, "invalid_request_error", "request body is empty")
  end

  local body = cjson.decode(raw)
  if type(body) ~= "table" then
    return json_error(400, "invalid_request_error", "request body is not a valid JSON object")
  end

  local all_text, last_user = request.prompt_text(body)
  if not all_text then
    return json_error(400, "invalid_request_error", last_user)
  end

  -- The guard runs before the router and the cache, so a blocked prompt
  -- costs no model call. Like the router, it reads the last user message.
  local guard
  if cfg.guard then
    guard = check_guard(cfg.guard, last_user)
    ngx.var.aigw_guard = guard.result
    if guard.result == "block" then
      ngx.ctx.aigw = { guard = guard }
      return json_error(400, "invalid_request_error", "the prompt guard blocked this request", "prompt_blocked")
    end
  end

  local tier, score, reason, route_version = pick_tier(cfg, body.model, last_user)
  if not tier then
    return json_error(404, "invalid_request_error",
      ("model '%s' does not exist. Use one of: %s"):format(tostring(body.model), concat(MODEL_IDS, ", ")),
      "model_not_found")
  end
  local target = cfg.tiers[tier]
  if cfg.router_shadow and reason ~= "pinned" then
    run_shadow(cfg, last_user, target.name)
  end

  local a = {
    tier_name = target.name,
    model = target.model,
    reason = reason,
    score = score,
    prompt_tokens_est = core.estimate_tokens(all_text),
    guard = guard,
    route_version = route_version,
    max_buffered = cfg.max_buffered_response,
  }
  local var = ngx.var
  var.aigw_tier = target.name
  var.aigw_model = target.model
  var.aigw_reason = reason
  if score then var.aigw_score = ("%.3f"):format(score) end
  if route_version then var.aigw_route_version = route_version end
  if guard and guard.version then var.aigw_guard_version = guard.version end

  -- Semantic cache, after the tier choice: the tier is part of the namespace.
  if cfg.cache then
    ngx.update_time()
    local t0 = ngx.now()
    local status, reply, store = cache_lookup(cfg.cache, body, target)
    ngx.update_time()
    a.cache, a.cache_seconds, a.cache_store = status, ngx.now() - t0, store
    a.cache_score = reply and tonumber(reply.score)
    a.cache_rerank = reply and tonumber(reply.rerank)
    var.aigw_cache = status
    if a.cache_score then var.aigw_cache_score = ("%.3f"):format(a.cache_score) end
    if a.cache_rerank then var.aigw_cache_rerank = ("%.3f"):format(a.cache_rerank) end

    if status == "hit" then
      a.cached = reply.response
      ngx.ctx.aigw = a
      local wants_usage = type(body.stream_options) == "table" and body.stream_options.include_usage == true
      return serve_cached(body, reply.response, wants_usage)
    end
  end

  local status, err_type, message, code = rewrite_upstream(a, body, target)
  if status then
    return json_error(status, err_type, message, code)
  end
  ngx.ctx.aigw = a
end

function _M.header_filter()
  local a = ngx.ctx.aigw
  local h = ngx.header
  h["X-Request-Id"] = ngx.var.request_id
  if not a then return end

  if a.guard then
    h["X-AIGW-Guard"] = a.guard.result
    h["X-AIGW-Guard-Rule"] = a.guard.rule
    if a.guard.score then h["X-AIGW-Guard-Score"] = ("%.3f"):format(a.guard.score) end
    h["X-AIGW-Guard-Version"] = a.guard.version
  end
  h["X-AIGW-Route-Version"] = a.route_version
  h["X-AIGW-Tier"] = a.tier_name
  h["X-AIGW-Model"] = a.model
  h["X-AIGW-Route-Reason"] = a.reason
  if a.score then h["X-AIGW-Route-Score"] = ("%.3f"):format(a.score) end
  if a.cache then h["X-AIGW-Cache"] = a.cache end
  if a.cache_score then h["X-AIGW-Cache-Score"] = ("%.3f"):format(a.cache_score) end
  if a.cache_rerank then h["X-AIGW-Cache-Rerank"] = ("%.3f"):format(a.cache_rerank) end

  a.upstream_status = ngx.status
  local ct = h["Content-Type"]
  if type(ct) == "table" then ct = ct[1] end
  local is_sse = ct and find(ct, "text/event-stream", 1, true)

  -- An Anthropic answer changes size when it is translated.
  if a.translate and not a.cached then
    h["Content-Length"] = nil
    if is_sse and a.upstream_status < 400 then
      a.translate.sse = anthropic.stream(ngx.time(), a.translate.with_usage)
    else
      a.translate.parts = {}
      h["Content-Type"] = "application/json"
    end
  end

  if a.upstream_status >= 400 or a.cached then return end

  if is_sse then
    a.sse = sse.new(cjson.decode)
  else
    a.body_parts, a.body_bytes = {}, 0
  end
end

-- Translate an Anthropic answer into the OpenAI format. A stream is
-- translated event by event. A JSON body is held until its end, because only
-- the whole object can be translated.
local function translate_chunk(t, chunk, eof, status)
  if t.sse then
    local out = t.sse:feed(chunk)
    return out
  end
  t.parts[#t.parts + 1] = chunk
  if not eof then return "" end
  local resp = cjson.decode(concat(t.parts))
  t.parts = {}
  local out
  if type(resp) ~= "table" then
    out = { error = { message = "the upstream answer is not JSON", type = "api_error" } }
  elseif status >= 400 then
    out = anthropic.error(resp)
  else
    out = anthropic.response(resp, ngx.time())
  end
  return cjson.encode(out) .. "\n"
end

function _M.body_filter()
  local a = ngx.ctx.aigw
  if not a then return end
  local chunk, eof = ngx.arg[1], ngx.arg[2]
  if a.translate and not a.cached and (a.translate.sse or a.translate.parts) then
    chunk = translate_chunk(a.translate, chunk, eof, a.upstream_status)
    ngx.arg[1] = chunk
  end

  if chunk ~= "" then
    if a.sse then
      a.sse:feed(chunk)
    elseif a.body_parts then
      a.body_bytes = a.body_bytes + #chunk
      if a.body_bytes > a.max_buffered then
        -- Too large to parse cheaply. Stop buffering and give up on usage.
        a.body_parts, a.body_truncated = nil, true
      else
        a.body_parts[#a.body_parts + 1] = chunk
      end
    end
  end

  if eof then finalize(a) end
end

-- Cache only complete, plain answers: finish_reason "stop", nothing cut.
local function should_store(a)
  return a.cache_store and a.token_source ~= "none" and a.finish_reason == "stop"
    and not a.content_truncated and type(a.completion_text) == "string" and a.completion_text ~= ""
end

function _M.log()
  local a = ngx.ctx.aigw
  if a then finalize(a) end
  metrics.record(a or {})

  if a and should_store(a) then
    local cache_cfg = config.get().cache
    local u = a.token_source == "upstream" and {
      prompt_tokens = a.prompt_tokens, completion_tokens = a.completion_tokens,
    } or nil
    semcache.store_async(cache_cfg, ngx.var.request_id, a.cache_store.ns, a.cache_store.text, {
      content = a.completion_text, model = a.model, finish_reason = a.finish_reason, usage = u,
    })
  end
end

-- GET /v1/models: the model names that clients can send.
function _M.models()
  if not authenticated(config.get()) then
    return json_error(401, "authentication_error", "missing or invalid API key", "invalid_api_key")
  end
  local data = {}
  for i, id in ipairs(MODEL_IDS) do
    data[i] = { id = id, object = "model", created = 0, owned_by = "aigw" }
  end
  ngx.header.content_type = "application/json"
  ngx.say(cjson.encode({ object = "list", data = data }))
end

function _M.not_found()
  return json_error(404, "invalid_request_error",
    ("no route for %s %s"):format(ngx.req.get_method(), ngx.var.uri), "not_found")
end

-- GET /healthz on the status port.
function _M.health()
  ngx.header.content_type = "application/json"
  ngx.say(cjson.encode({ status = "ok", core = core.version() }))
end

return _M
