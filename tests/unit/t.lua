-- Tiny assertion helpers for the unit specs.
local t = {}

local function fail(msg, level)
  error(msg, (level or 1) + 2)
end

function t.eq(actual, expected, what)
  if actual ~= expected then
    fail(("%sexpected %s, got %s"):format(what and (what .. ": ") or "",
      tostring(expected), tostring(actual)))
  end
end

function t.truthy(value, what)
  if not value then fail(what or "expected a true value") end
end

function t.errors(fn, pattern)
  local ok, err = pcall(fn)
  if ok then fail("expected an error matching '" .. pattern .. "'") end
  if not tostring(err):find(pattern, 1, true) then
    fail(("expected an error matching '%s', got: %s"):format(pattern, tostring(err)))
  end
end

return t
