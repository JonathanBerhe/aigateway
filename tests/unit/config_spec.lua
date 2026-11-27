local t = require("t")
local config = require("aigw.config")

local function env(tbl)
  return function(name) return tbl[name] end
end

local function load(extra)
  local vars = { AIGW_CORE_LIB = "/tmp/libaigw_core.so" }
  for k, v in pairs(extra or {}) do vars[k] = v end
  return config.load(env(vars))
end

return {
  cases = {
    { "AIGW_CORE_LIB is required", function()
      t.errors(function() config.load(env({})) end, "AIGW_CORE_LIB")
    end },

    { "defaults point both tiers at a local Ollama", function()
      local c = load()
      local cheap, strong = c.tiers[config.TIER_CHEAP], c.tiers[config.TIER_STRONG]
      t.eq(cheap.name, "cheap")
      t.eq(strong.name, "strong")
      t.eq(cheap.scheme, "http")
      t.eq(cheap.authority, "127.0.0.1:11434")
      t.eq(cheap.chat_path, "/v1/chat/completions")
      t.eq(cheap.api_key, nil)
      t.eq(cheap.stream_usage, true)
      t.eq(c.api_keys, nil, "auth is off without AIGW_API_KEYS")
      t.eq(c.router_threshold, 0.5)
    end },

    { "an https base URL leaves out the default port", function()
      local s = load({ AIGW_STRONG_BASE_URL = "https://api.example.com/v1/" }).tiers[config.TIER_STRONG]
      t.eq(s.scheme, "https")
      t.eq(s.port, 443)
      t.eq(s.host, "api.example.com")
      t.eq(s.authority, "api.example.com")
      t.eq(s.chat_path, "/v1/chat/completions")
    end },

    { "an empty variable means the default", function()
      t.eq(load({ AIGW_CHEAP_MODEL = "" }).tiers[config.TIER_CHEAP].model, config.VARS.AIGW_CHEAP_MODEL)
    end },

    { "consumer keys are split on commas and spaces", function()
      local keys = load({ AIGW_API_KEYS = "alpha, beta,,gamma " }).api_keys
      t.truthy(keys.alpha and keys.beta and keys.gamma, "all three keys are present")
      t.eq(keys[""], nil)
    end },

    { "a tier speaks the OpenAI format unless it is set to anthropic", function()
      local c = load({ AIGW_STRONG_FORMAT = "anthropic", AIGW_STRONG_BASE_URL = "https://api.anthropic.com/v1" })
      t.eq(c.tiers[config.TIER_CHEAP].format, "openai")
      local s = c.tiers[config.TIER_STRONG]
      t.eq(s.format, "anthropic")
      t.eq(s.messages_path, "/v1/messages")
      t.eq(s.max_tokens, 1024)
      t.errors(function() load({ AIGW_CHEAP_FORMAT = "gemini" }) end, "AIGW_CHEAP_FORMAT")
    end },

    { "bad values stop the load", function()
      t.errors(function() load({ AIGW_ROUTER_THRESHOLD = "2" }) end, "AIGW_ROUTER_THRESHOLD")
      t.errors(function() load({ AIGW_CHEAP_BASE_URL = "ftp://x/v1" }) end, "AIGW_CHEAP_BASE_URL")
      t.errors(function() load({ AIGW_STRONG_STREAM_USAGE = "maybe" }) end, "AIGW_STRONG_STREAM_USAGE")
    end },

    { "the router is the heuristic by default", function()
      t.eq(load().router, nil)
      t.eq(load({ AIGW_ROUTER = "heuristic", AIGW_CACHE_ADDR = "unix:/tmp/s.sock" }).router, nil)
    end },

    { "the model router uses the cache sidecar unless it has its own address", function()
      local r = load({ AIGW_ROUTER = "model", AIGW_CACHE_ADDR = "unix:/tmp/s.sock" }).router
      t.eq(r.unix, "unix:/tmp/s.sock")
      t.eq(r.timeout, 200)
      r = load({ AIGW_ROUTER = "model", AIGW_CACHE_ADDR = "unix:/tmp/s.sock", AIGW_ROUTER_ADDR = "127.0.0.1:7700" }).router
      t.eq(r.host, "127.0.0.1")
      t.eq(r.port, 7700)
    end },

    { "the model router needs a sidecar and a known mode", function()
      t.errors(function() load({ AIGW_ROUTER = "model" }) end, "AIGW_ROUTER_ADDR")
      t.errors(function() load({ AIGW_ROUTER = "classifier" }) end, "AIGW_ROUTER")
      t.errors(function() load({ AIGW_ROUTER = "model", AIGW_ROUTER_ADDR = "nowhere" }) end, "AIGW_ROUTER_ADDR")
    end },

    { "the model router serves every consumer unless a canary share is set", function()
      local base = { AIGW_ROUTER = "model", AIGW_CACHE_ADDR = "unix:/tmp/s.sock" }
      t.eq(load(base).router.share, 1)
      base.AIGW_ROUTER_MODEL_SHARE = "0.1"
      t.eq(load(base).router.share, 0.1)
      base.AIGW_ROUTER_MODEL_SHARE = "1.5"
      t.errors(function() load(base) end, "AIGW_ROUTER_MODEL_SHARE")
    end },

    { "the shadow router must differ from the active one", function()
      t.eq(load().router_shadow, nil)
      local s = load({ AIGW_ROUTER_SHADOW = "model", AIGW_CACHE_ADDR = "unix:/tmp/s.sock" }).router_shadow
      t.eq(s.mode, "model")
      t.eq(s.unix, "unix:/tmp/s.sock")
      t.eq(load({ AIGW_ROUTER = "model", AIGW_ROUTER_SHADOW = "heuristic", AIGW_CACHE_ADDR = "unix:/tmp/s.sock" })
        .router_shadow.mode, "heuristic")
      t.errors(function() load({ AIGW_ROUTER_SHADOW = "heuristic" }) end, "must differ")
      t.errors(function() load({ AIGW_ROUTER_SHADOW = "model" }) end, "AIGW_ROUTER_ADDR")
      t.errors(function() load({ AIGW_ROUTER_SHADOW = "both" }) end, "AIGW_ROUTER_SHADOW")
    end },

    { "the guard is off by default and the regex mode needs no sidecar", function()
      t.eq(load().guard, nil)
      local g = load({ AIGW_GUARD = "regex" }).guard
      t.eq(g.mode, "regex")
      t.eq(g.action, "block")
    end },

    { "the model guard uses the cache sidecar and checks its settings", function()
      local g = load({ AIGW_GUARD = "model", AIGW_GUARD_ACTION = "flag", AIGW_CACHE_ADDR = "unix:/tmp/s.sock" }).guard
      t.eq(g.unix, "unix:/tmp/s.sock")
      t.eq(g.action, "flag")
      t.eq(g.timeout, 200)
      t.errors(function() load({ AIGW_GUARD = "model" }) end, "AIGW_GUARD_ADDR")
      t.errors(function() load({ AIGW_GUARD = "strict" }) end, "AIGW_GUARD")
      t.errors(function() load({ AIGW_GUARD = "regex", AIGW_GUARD_ACTION = "drop" }) end, "AIGW_GUARD_ACTION")
    end },

    { "nginx.conf lists every variable with the env directive", function(args)
      local f = assert(io.open(args.nginx_conf, "r"))
      local conf = f:read("*a")
      f:close()
      for name in pairs(config.VARS) do
        t.truthy(conf:find("\nenv " .. name .. ";", 1, true), "nginx.conf has no 'env " .. name .. ";'")
      end
    end },
  },
}
