local t = require("t")
local sse = require("aigw.sse")

local STREAM = table.concat({
  ": keep-alive comment",
  'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}',
  "",
  'data: {"choices":[{"index":0,"delta":{"content":"Hello"}}]}',
  "",
  'data:{"choices":[{"index":0,"delta":{"content":", world"}}]}', -- no space after "data:"
  "",
  "data: not json",
  "",
  'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}}',
  "",
  "data: [DONE]",
  "",
  "",
}, "\n")

local function check(p, label)
  t.eq(p:content(), "Hello, world", label .. ": content")
  t.eq(p.usage and p.usage.prompt_tokens, 7, label .. ": prompt tokens")
  t.eq(p.usage and p.usage.completion_tokens, 3, label .. ": completion tokens")
  t.eq(p.done, true, label .. ": done")
  t.eq(p.events, 4, label .. ": decoded events")
end

return {
  needs_cjson = true,

  cases = {
    { "a stream in one chunk", function()
      local p = sse.new(require("cjson.safe").decode)
      p:feed(STREAM)
      check(p, "one chunk")
    end },

    { "a stream split in two at every byte", function()
      local decode = require("cjson.safe").decode
      for i = 0, #STREAM do
        local p = sse.new(decode)
        p:feed(STREAM:sub(1, i))
        p:feed(STREAM:sub(i + 1))
        check(p, "split at " .. i)
      end
    end },

    { "a stream fed one byte at a time", function()
      local p = sse.new(require("cjson.safe").decode)
      for i = 1, #STREAM do p:feed(STREAM:sub(i, i)) end
      check(p, "bytewise")
    end },

    { "CRLF line endings", function()
      local p = sse.new(require("cjson.safe").decode)
      p:feed((STREAM:gsub("\n", "\r\n")))
      check(p, "crlf")
    end },

    { "content is capped at max_content bytes", function()
      local p = sse.new(require("cjson.safe").decode, 4)
      p:feed(STREAM)
      t.eq(p:content(), "Hell")
    end },

    { "an endless line is dropped to bound memory", function()
      local p = sse.new(require("cjson.safe").decode)
      p:feed("data: " .. string.rep("x", 1024 * 1024 + 10))
      t.eq(p.pending, "")
      t.eq(p.dropped_lines, 1)
    end },
  },
}
