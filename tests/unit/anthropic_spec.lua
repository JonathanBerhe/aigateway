local t = require("t")
local anthropic = require("aigw.anthropic")
local cjson = require("cjson.safe")

local function body(messages, extra)
  local b = { model = "auto", messages = messages }
  for k, v in pairs(extra or {}) do b[k] = v end
  return b
end

return {
  needs_cjson = true,
  cases = {
    { "system messages move to the system field", function()
      local out = anthropic.request(body({
        { role = "system", content = "Be terse." },
        { role = "developer", content = "Answer in English." },
        { role = "user", content = "hi" },
      }), "claude-test", 1024)
      t.eq(out.model, "claude-test")
      t.eq(out.system, "Be terse.\n\nAnswer in English.")
      t.eq(#out.messages, 1)
      t.eq(out.messages[1].role, "user")
      t.eq(out.max_tokens, 1024, "the tier default when the client sends none")
    end },

    { "request fields map to their Anthropic names", function()
      local out = anthropic.request(body({ { role = "user", content = { { type = "text", text = "a" }, { type = "text", text = "b" } } } },
        { max_completion_tokens = 50, max_tokens = 99, temperature = 0.2, stop = "END", stream = true }), "m", 1024)
      t.eq(out.messages[1].content, "a\nb")
      t.eq(out.max_tokens, 50, "max_completion_tokens wins over max_tokens")
      t.eq(out.temperature, 0.2)
      t.eq(out.stop_sequences[1], "END")
      t.eq(out.stream, true)
    end },

    { "tools, images, n and the tool role are refused", function()
      local user = { { role = "user", content = "hi" } }
      t.eq(select(2, anthropic.request(body(user, { tools = {} }), "m", 1)), "tools are not supported for this model")
      t.truthy(select(2, anthropic.request(body(user, { n = 2 }), "m", 1)):find("'n'"))
      local image = { { role = "user", content = { { type = "image_url", image_url = { url = "x" } } } } }
      t.truthy(select(2, anthropic.request(body(image), "m", 1)):find("only text"))
      t.truthy(select(2, anthropic.request(body({ { role = "tool", content = "x" } }), "m", 1)):find("role 'tool'"))
    end },

    { "a message maps to a chat completion", function()
      local out = anthropic.response({
        id = "msg_1", model = "claude-test", stop_reason = "max_tokens",
        content = { { type = "text", text = "Hello " }, { type = "text", text = "world" } },
        usage = { input_tokens = 10, output_tokens = 3 },
      }, 1700000000)
      t.eq(out.object, "chat.completion")
      t.eq(out.choices[1].message.content, "Hello world")
      t.eq(out.choices[1].finish_reason, "length")
      t.eq(out.usage.prompt_tokens, 10)
      t.eq(out.usage.total_tokens, 13)
    end },

    { "an error maps to an OpenAI error", function()
      local out = anthropic.error({ type = "error", error = { type = "overloaded_error", message = "busy" } })
      t.eq(out.error.type, "overloaded_error")
      t.eq(out.error.message, "busy")
    end },

    { "a stream maps event by event, also across chunks", function()
      local events = {
        { "message_start", { message = { id = "msg_1", model = "m", usage = { input_tokens = 7, output_tokens = 1 } } } },
        { "ping", {} },
        { "content_block_delta", { delta = { type = "text_delta", text = "Hi" } } },
        { "message_delta", { delta = { stop_reason = "end_turn" }, usage = { output_tokens = 2 } } },
        { "message_stop", {} },
      }
      local raw = {}
      for _, e in ipairs(events) do
        e[2].type = e[1]
        raw[#raw + 1] = "event: " .. e[1] .. "\r\ndata: " .. cjson.encode(e[2]) .. "\r\n\r\n"
      end
      raw = table.concat(raw)
      local s = anthropic.stream(1700000000, true)
      -- Feed in pieces of 7 bytes: events and lines span chunks.
      local out = {}
      for i = 1, #raw, 7 do out[#out + 1] = s:feed(raw:sub(i, i + 6)) end
      out = table.concat(out)
      local chunks = {}
      for data in out:gmatch("data: ([^\n]+)\n\n") do chunks[#chunks + 1] = data end
      t.eq(#chunks, 5, out)
      t.eq(cjson.decode(chunks[1]).choices[1].delta.role, "assistant")
      t.eq(cjson.decode(chunks[2]).choices[1].delta.content, "Hi")
      t.eq(cjson.decode(chunks[3]).choices[1].finish_reason, "stop")
      local usage = cjson.decode(chunks[4])
      t.eq(#usage.choices, 0)
      t.eq(usage.usage.prompt_tokens, 7)
      t.eq(usage.usage.completion_tokens, 2)
      t.eq(chunks[5], "[DONE]")
    end },
  },
}
