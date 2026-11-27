-- Prometheus metrics kept in a shared dictionary.
--
-- lua_shared_dict memory is shared by all NGINX workers, so the counters are
-- global to the gateway. Each key is a full series name with its labels, for
-- example: aigw_requests_total{tier="cheap",status="2xx"}

local _M = {}

local DICT = "aigw_metrics"

-- Help text per metric family. A family that ends in _seconds is a summary
-- (series _sum and _count). The others are counters.
local HELP = {
  aigw_requests_total = "Chat completion requests by tier and HTTP status class.",
  aigw_route_decisions_total = "Tier decisions by reason (router, router_model, router_fallback, pinned or router_error).",
  aigw_tokens_total = "Tokens by tier, kind and source (upstream usage or local estimate).",
  aigw_upstream_header_seconds = "Upstream time to response headers (time to first byte).",
  aigw_upstream_response_seconds = "Upstream time to the full response.",
  aigw_cache_lookups_total = "Semantic cache lookups by result (hit, miss, skip, bypass, error).",
  aigw_guard_decisions_total = "Prompt guard results (pass, block, flag, error) by source (regex or model).",
  aigw_router_shadow_total = "Auto requests by active tier and the tier of the shadow router (or error).",
  aigw_cache_lookup_seconds = "Time of a semantic cache lookup, embedding included.",
  aigw_metrics_errors_total = "Failed metric updates, for example a full dictionary.",
}

local function dict()
  return ngx.shared[DICT]
end

local function labels(t)
  local keys = {}
  for k in pairs(t) do keys[#keys + 1] = k end
  table.sort(keys)
  local out = {}
  for i, k in ipairs(keys) do
    local value = tostring(t[k]):gsub('[\\"\n]', "_")
    out[i] = ('%s="%s"'):format(k, value)
  end
  return "{" .. table.concat(out, ",") .. "}"
end

-- Family name and type of a series name.
local function family(name)
  local base = name:match("^(.-)_sum$") or name:match("^(.-)_count$")
  if base then return base, "summary" end
  return name, "counter"
end

local function incr(name, lbl, value)
  local key = name .. (lbl and labels(lbl) or "")
  local _, err = dict():incr(key, value or 1, 0)
  if err then
    dict():incr("aigw_metrics_errors_total", 1, 0)
    ngx.log(ngx.WARN, "aigw metrics: cannot update ", key, ": ", err)
  end
end
_M.incr = incr

-- $upstream_*_time holds "0.123", or "0.100, 0.023" after a retry. Sum the parts.
local function upstream_seconds(v)
  if not v or v == "" or v == "-" then return nil end
  local total, any = 0, false
  for part in v:gmatch("[%d%.]+") do
    total = total + tonumber(part)
    any = true
  end
  return any and total or nil
end

-- Record one finished request. `a` is ngx.ctx.aigw (see handler.lua).
function _M.record(a)
  local status = ngx.status or 0
  incr("aigw_requests_total", { tier = a.tier_name or "none", status = math.floor(status / 100) .. "xx" })
  if a.guard then
    incr("aigw_guard_decisions_total", { result = a.guard.result, source = a.guard.source })
  end
  if not a.tier_name then return end

  incr("aigw_route_decisions_total", { tier = a.tier_name, reason = a.reason })
  if a.cache then
    incr("aigw_cache_lookups_total", { tier = a.tier_name, result = a.cache })
    if a.cache == "hit" or a.cache == "miss" or a.cache == "error" then
      incr("aigw_cache_lookup_seconds_sum", { tier = a.tier_name }, a.cache_seconds or 0)
      incr("aigw_cache_lookup_seconds_count", { tier = a.tier_name })
    end
  end
  if a.token_source and a.token_source ~= "none" then
    incr("aigw_tokens_total", { tier = a.tier_name, kind = "prompt", source = a.token_source }, a.prompt_tokens)
    incr("aigw_tokens_total", { tier = a.tier_name, kind = "completion", source = a.token_source }, a.completion_tokens)
  end

  local header_s = upstream_seconds(ngx.var.upstream_header_time)
  if header_s then
    incr("aigw_upstream_header_seconds_sum", { tier = a.tier_name }, header_s)
    incr("aigw_upstream_header_seconds_count", { tier = a.tier_name })
  end
  local response_s = upstream_seconds(ngx.var.upstream_response_time)
  if response_s then
    incr("aigw_upstream_response_seconds_sum", { tier = a.tier_name }, response_s)
    incr("aigw_upstream_response_seconds_count", { tier = a.tier_name })
  end
end

-- content_by_lua handler for GET /metrics.
function _M.expose()
  local keys = dict():get_keys(0)
  table.sort(keys)
  local out, last_family = {}, nil
  for _, key in ipairs(keys) do
    local fam, kind = family(key:match("^[%w_]+"))
    if fam ~= last_family then
      out[#out + 1] = ("# HELP %s %s"):format(fam, HELP[fam] or fam)
      out[#out + 1] = ("# TYPE %s %s"):format(fam, kind)
      last_family = fam
    end
    out[#out + 1] = key .. " " .. tostring(dict():get(key))
  end
  ngx.header.content_type = "text/plain; version=0.0.4"
  ngx.print(table.concat(out, "\n"), "\n")
end

return _M
