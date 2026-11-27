-- Unit test runner for the gateway Lua code. It needs nothing beyond LuaJIT.
--
-- With OpenResty (all specs):
--   resty -I gateway/lua -I tests/unit tests/unit/run.lua <core lib> <nginx.conf>
-- With plain LuaJIT (specs that need cjson are skipped):
--   LUA_PATH='gateway/lua/?.lua;tests/unit/?.lua;;' luajit tests/unit/run.lua <core lib> <nginx.conf>

local args = { core_lib = arg[1], nginx_conf = arg[2] }
if not args.core_lib or not args.nginx_conf then
  io.stderr:write("usage: run.lua <path to libaigw_core> <path to nginx.conf>\n")
  os.exit(2)
end

local has_cjson = pcall(require, "cjson.safe")
local SPECS = { "config_spec", "core_spec", "request_spec", "sse_spec", "anthropic_spec" }

local passed, failed, skipped = 0, 0, 0
for _, name in ipairs(SPECS) do
  local spec = require(name)
  print(name)
  if spec.needs_cjson and not has_cjson then
    print("  skip  needs cjson, run with resty")
    skipped = skipped + 1
  else
    if spec.setup then spec.setup(args) end
    for _, case in ipairs(spec.cases) do
      local ok, err = pcall(case[2], args)
      if ok then
        passed = passed + 1
        print("  ok    " .. case[1])
      else
        failed = failed + 1
        print("  FAIL  " .. case[1] .. "\n        " .. tostring(err))
      end
    end
  end
end

print(("\npassed: %d, failed: %d, skipped specs: %d"):format(passed, failed, skipped))
os.exit(failed == 0 and 0 or 1)
