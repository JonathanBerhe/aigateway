-- LuaJIT FFI binding to libaigw_core (the Rust crate in core/).
--
-- The declarations below repeat core/include/aigw_core.h. Change both together.
--
-- Each call is synchronous and runs on the NGINX worker event loop. That is
-- fine for the microsecond heuristics of phase 1. A model that needs
-- milliseconds blocks every other request on the worker, so phase 2 must
-- measure that cost before it puts a model behind this binding.

local ffi = require("ffi")

ffi.cdef [[
typedef struct {
    uint32_t tier;
    float score;
    uint32_t prompt_tokens;
} aigw_route_result;

const char *aigw_version(void);
uint32_t aigw_estimate_tokens(const char *text, size_t len);
int32_t aigw_route(const char *text, size_t len, float threshold, aigw_route_result *out);
int32_t aigw_guard(const char *text, size_t len, int32_t *rule);
const char *aigw_guard_rule_name(int32_t rule);
]]

local _M = {}

local lib
-- One result buffer per worker. A worker runs one Lua call at a time, and
-- the value is copied out before the next call, so reuse is safe and avoids
-- a GC allocation per request.
local result = ffi.new("aigw_route_result")
local rule = ffi.new("int32_t[1]")

-- Load the shared library. Call once, from init_by_lua or a test.
function _M.init(path)
  local ok, l = pcall(ffi.load, path)
  if not ok then
    error(("cannot load the native core from %q: %s"):format(path, l))
  end
  lib = l
  return _M.version()
end

function _M.version()
  return ffi.string(lib.aigw_version())
end

function _M.estimate_tokens(text)
  return tonumber(lib.aigw_estimate_tokens(text, #text))
end

-- Returns tier, score, prompt_tokens. On error returns nil and a message.
function _M.route(text, threshold)
  local rc = lib.aigw_route(text, #text, threshold, result)
  if rc ~= 0 then
    return nil, "aigw_route failed with code " .. tostring(rc)
  end
  return tonumber(result.tier), tonumber(result.score), tonumber(result.prompt_tokens)
end

-- Returns the name of the first guard rule that matches, or false.
-- On error returns nil and a message.
function _M.guard(text)
  local rc = lib.aigw_guard(text, #text, rule)
  if rc ~= 0 then
    return nil, "aigw_guard failed with code " .. tostring(rc)
  end
  if rule[0] < 0 then return false end
  local name = lib.aigw_guard_rule_name(rule[0])
  return name ~= nil and ffi.string(name) or ("rule_" .. tonumber(rule[0]))
end

_M.RESULT_SIZE = ffi.sizeof("aigw_route_result")

return _M
