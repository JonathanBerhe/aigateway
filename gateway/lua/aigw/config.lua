-- Gateway configuration.
--
-- load() reads environment variables once, in init_by_lua (the NGINX master
-- process). Workers inherit the result when NGINX forks them, so a bad value
-- stops the gateway at startup instead of failing requests later.
--
-- NGINX removes environment variables from its processes unless nginx.conf
-- lists them with the `env` directive. Keep that list in sync with VARS below.

local _M = {}

-- Every variable this module reads, with its default. false means "no default".
-- tests/unit/config_spec.lua makes sure that nginx.conf lists each one.
local VARS = {
  AIGW_CORE_LIB              = false, -- path to libaigw_core.{so,dylib}; required
  AIGW_API_KEYS              = false, -- comma-separated consumer keys; unset = no auth
  AIGW_ROUTER                = "heuristic", -- heuristic, or model: the trained router of the sidecar
  AIGW_ROUTER_THRESHOLD      = "0.5", -- the heuristic router (also the fallback of model)
  AIGW_ROUTER_ADDR           = false, -- sidecar of the model router; unset = AIGW_CACHE_ADDR
  AIGW_ROUTER_TIMEOUT_MS     = "200",
  AIGW_ROUTER_MODEL_SHARE    = "1", -- with model: the share of consumers that the model router serves (canary)
  AIGW_ROUTER_SHADOW         = "off", -- off, heuristic or model: a second router that is only measured
  AIGW_GUARD                 = "off", -- off, regex (Rust deny list), or model (deny list, then the sidecar classifier)
  AIGW_GUARD_ACTION          = "block", -- block (HTTP 400) or flag (pass with a header)
  AIGW_GUARD_ADDR            = false, -- sidecar of the model guard; unset = AIGW_CACHE_ADDR
  AIGW_GUARD_TIMEOUT_MS      = "200",
  AIGW_MAX_BUFFERED_RESPONSE = "1048576",
  AIGW_CHEAP_BASE_URL        = "http://127.0.0.1:11434/v1",
  AIGW_CHEAP_MODEL           = "qwen2.5:0.5b",
  AIGW_CHEAP_API_KEY         = false,
  AIGW_CHEAP_STREAM_USAGE    = "true",
  AIGW_CHEAP_FORMAT          = "openai", -- openai, or anthropic (the Messages API)
  AIGW_CHEAP_MAX_TOKENS      = "1024", -- anthropic only: max_tokens when the client sends none
  AIGW_STRONG_BASE_URL       = "http://127.0.0.1:11434/v1",
  AIGW_STRONG_MODEL          = "gpt-oss:20b",
  AIGW_STRONG_API_KEY        = false,
  AIGW_STRONG_STREAM_USAGE   = "true",
  AIGW_STRONG_FORMAT         = "openai",
  AIGW_STRONG_MAX_TOKENS     = "1024",
  AIGW_CACHE_ADDR            = false, -- semcache sidecar: unix:/path or host:port; unset = no cache
  AIGW_CACHE_THRESHOLD       = false, -- override; unset = the threshold of the sidecar's model
  AIGW_CACHE_TTL             = "3600",
  AIGW_CACHE_TIMEOUT_MS      = "500",
}
_M.VARS = VARS

-- Tier ids match AIGW_TIER_* in core/include/aigw_core.h.
local TIER_CHEAP, TIER_STRONG = 0, 1
_M.TIER_CHEAP, _M.TIER_STRONG = TIER_CHEAP, TIER_STRONG

-- Model names that clients send. "auto" asks the router to choose.
_M.AUTO_MODEL = "auto"
local ALIASES = { ["aigw-cheap"] = TIER_CHEAP, ["aigw-strong"] = TIER_STRONG }

local current

local function env(name, getenv)
  assert(VARS[name] ~= nil, "unknown configuration variable " .. name)
  local v = getenv(name)
  if v == nil or v == "" then return VARS[name] or nil end
  return v
end

local function parse_bool(name, v)
  if v == "true" or v == "1" then return true end
  if v == "false" or v == "0" then return false end
  error(("%s must be true or false, got %q"):format(name, v))
end

local function parse_number(name, v, min, max)
  local n = tonumber(v)
  if not n or n < min or n > max then
    error(("%s must be a number from %s to %s, got %q"):format(name, min, max, tostring(v)))
  end
  return n
end

-- Split "http://host:port/base" into the parts that the proxy needs.
-- The base path follows the OpenAI SDK convention: it ends with /v1.
local function parse_base_url(name, url)
  local scheme, host, port, path = url:match("^(https?)://([^/:]+):?(%d*)(.*)$")
  if not scheme then
    error(("%s must look like http(s)://host[:port]/path, got %q"):format(name, url))
  end
  path = path:gsub("/+$", "")
  local default_port = scheme == "https" and 443 or 80
  port = tonumber(port) or default_port
  return {
    scheme = scheme,
    host = host,
    port = port,
    -- Value for the Host header and for proxy_pass. Leave out a default port.
    authority = port == default_port and host or (host .. ":" .. port),
    chat_path = path .. "/chat/completions",
    messages_path = path .. "/messages",
  }
end

local function build_target(tier_name, getenv)
  local prefix = "AIGW_" .. tier_name:upper() .. "_"
  local url = env(prefix .. "BASE_URL", getenv)
  local t = parse_base_url(prefix .. "BASE_URL", url)
  t.name = tier_name
  t.base_url = url
  t.model = env(prefix .. "MODEL", getenv)
  t.api_key = env(prefix .. "API_KEY", getenv)
  t.stream_usage = parse_bool(prefix .. "STREAM_USAGE", env(prefix .. "STREAM_USAGE", getenv))
  t.format = env(prefix .. "FORMAT", getenv)
  if t.format ~= "openai" and t.format ~= "anthropic" then
    error(("%sFORMAT must be openai or anthropic, got %q"):format(prefix, t.format))
  end
  t.max_tokens = parse_number(prefix .. "MAX_TOKENS", env(prefix .. "MAX_TOKENS", getenv), 1, 1000000)
  return t
end

-- Set t.unix, or t.host and t.port, from a sidecar address.
local function parse_addr(name, addr, t)
  if addr:sub(1, 5) == "unix:" and #addr > 5 then
    t.unix = addr
  else
    local host, port = addr:match("^([^:]+):(%d+)$")
    if not host then
      error(("%s must be unix:/path or host:port, got %q"):format(name, addr))
    end
    t.host, t.port = host, tonumber(port)
  end
  return t
end

-- Sidecar address and timeout of a model feature: <prefix>_ADDR, or the
-- cache sidecar, and <prefix>_TIMEOUT_MS. `var` names the setting that asked.
local function sidecar(var, prefix, getenv)
  local name, addr = prefix .. "_ADDR", env(prefix .. "_ADDR", getenv)
  if not addr then name, addr = "AIGW_CACHE_ADDR", env("AIGW_CACHE_ADDR", getenv) end
  if not addr then
    error(("%s=model needs the sidecar address: set %s_ADDR or AIGW_CACHE_ADDR"):format(var, prefix))
  end
  return parse_addr(name, addr, {
    addr = addr,
    timeout = parse_number(prefix .. "_TIMEOUT_MS", env(prefix .. "_TIMEOUT_MS", getenv), 1, 60000),
  })
end

-- Settings of the trained router in the sidecar, or nil for the heuristic.
local function build_router(getenv)
  local mode = env("AIGW_ROUTER", getenv)
  if mode == "heuristic" then return nil end
  if mode ~= "model" then
    error(("AIGW_ROUTER must be heuristic or model, got %q"):format(mode))
  end
  local r = sidecar("AIGW_ROUTER", "AIGW_ROUTER", getenv)
  r.share = parse_number("AIGW_ROUTER_MODEL_SHARE", env("AIGW_ROUTER_MODEL_SHARE", getenv), 0, 1)
  return r
end

-- The shadow router, or nil. It runs on every auto request, and only the
-- metrics see its decision. It must differ from the active router.
local function build_shadow(getenv)
  local mode = env("AIGW_ROUTER_SHADOW", getenv)
  if mode == "off" then return nil end
  if mode ~= "heuristic" and mode ~= "model" then
    error(("AIGW_ROUTER_SHADOW must be off, heuristic or model, got %q"):format(mode))
  end
  if mode == env("AIGW_ROUTER", getenv) then
    error("AIGW_ROUTER_SHADOW must differ from AIGW_ROUTER, both are " .. mode)
  end
  if mode == "heuristic" then return { mode = mode } end
  local s = sidecar("AIGW_ROUTER_SHADOW", "AIGW_ROUTER", getenv)
  s.mode = mode
  return s
end

-- Prompt guard settings, or nil when AIGW_GUARD is off.
local function build_guard(getenv)
  local mode = env("AIGW_GUARD", getenv)
  if mode == "off" then return nil end
  if mode ~= "regex" and mode ~= "model" then
    error(("AIGW_GUARD must be off, regex or model, got %q"):format(mode))
  end
  local action = env("AIGW_GUARD_ACTION", getenv)
  if action ~= "block" and action ~= "flag" then
    error(("AIGW_GUARD_ACTION must be block or flag, got %q"):format(action))
  end
  if mode == "regex" then return { mode = mode, action = action } end
  local g = sidecar("AIGW_GUARD", "AIGW_GUARD", getenv)
  g.mode, g.action = mode, action
  return g
end

-- Semantic cache settings, or nil when AIGW_CACHE_ADDR is not set.
local function build_cache(getenv)
  local addr = env("AIGW_CACHE_ADDR", getenv)
  if not addr then return nil end
  -- A threshold is only valid for the model it was measured on. The sidecar
  -- knows its model, so by default the gateway sends no threshold.
  local threshold = env("AIGW_CACHE_THRESHOLD", getenv)
  local c = {
    addr = addr,
    threshold = threshold and parse_number("AIGW_CACHE_THRESHOLD", threshold, 0, 1) or nil,
    ttl = parse_number("AIGW_CACHE_TTL", env("AIGW_CACHE_TTL", getenv), 1, 30 * 86400),
    timeout = parse_number("AIGW_CACHE_TIMEOUT_MS", env("AIGW_CACHE_TIMEOUT_MS", getenv), 1, 60000),
  }
  return parse_addr("AIGW_CACHE_ADDR", addr, c)
end

local function parse_keys(v)
  if not v then return nil end
  local keys, n = {}, 0
  for key in v:gmatch("[^,%s]+") do
    keys[key] = true
    n = n + 1
  end
  return n > 0 and keys or nil
end

-- Read the configuration. getenv defaults to os.getenv; tests pass a table lookup.
function _M.load(getenv)
  getenv = getenv or os.getenv
  local core_lib = env("AIGW_CORE_LIB", getenv)
  if not core_lib then
    error("AIGW_CORE_LIB is not set: point it to libaigw_core.so or libaigw_core.dylib")
  end

  current = {
    core_lib = core_lib,
    api_keys = parse_keys(env("AIGW_API_KEYS", getenv)),
    router = build_router(getenv),
    router_shadow = build_shadow(getenv),
    guard = build_guard(getenv),
    router_threshold = parse_number("AIGW_ROUTER_THRESHOLD",
      env("AIGW_ROUTER_THRESHOLD", getenv), 0, 1),
    max_buffered_response = parse_number("AIGW_MAX_BUFFERED_RESPONSE",
      env("AIGW_MAX_BUFFERED_RESPONSE", getenv), 0, 64 * 1024 * 1024),
    tiers = {
      [TIER_CHEAP] = build_target("cheap", getenv),
      [TIER_STRONG] = build_target("strong", getenv),
    },
    aliases = ALIASES,
    cache = build_cache(getenv),
  }
  return current
end

function _M.get()
  if not current then error("aigw.config.load() was not called") end
  return current
end

return _M
