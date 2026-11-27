local t = require("t")
local request = require("aigw.request")

-- Stand-in for cjson.null, which is a light userdata.
local NULL = newproxy()

return {
  cases = {
    { "string contents are joined, the last user turn is kept", function()
      local all, last = request.prompt_text({ messages = {
        { role = "system", content = "Be brief." },
        { role = "user", content = "first" },
        { role = "assistant", content = "ok" },
        { role = "user", content = "second" },
      } })
      t.eq(all, "Be brief.\nfirst\nok\nsecond")
      t.eq(last, "second")
    end },

    { "text parts are read and image parts are skipped", function()
      local all, last = request.prompt_text({ messages = {
        { role = "user", content = {
          { type = "text", text = "what is in" },
          { type = "image_url", image_url = { url = "data:..." } },
          { type = "text", text = "this picture?" },
        } },
      } })
      t.eq(all, "what is in\nthis picture?")
      t.eq(last, all)
    end },

    { "null content from a tool call gives empty text", function()
      local all = request.prompt_text({ messages = {
        { role = "user", content = "x" },
        { role = "assistant", content = NULL, tool_calls = {} },
      } })
      t.eq(all, "x\n")
    end },

    { "without a user turn, all text is used for routing", function()
      local all, last = request.prompt_text({ messages = { { role = "system", content = "only system" } } })
      t.eq(last, all)
    end },

    { "bad shapes return an error message", function()
      local all, err = request.prompt_text("nope")
      t.eq(all, nil)
      t.truthy(err:find("JSON object", 1, true))
      _, err = request.prompt_text({ messages = {} })
      t.truthy(err:find("non-empty", 1, true))
      _, err = request.prompt_text({ messages = { { content = "no role" } } })
      t.truthy(err:find("messages[0]", 1, true))
    end },
  },
}
