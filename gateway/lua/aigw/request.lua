-- Helpers that read an OpenAI chat completions request body.
-- No NGINX API is used here, so the unit tests run this module directly.

local _M = {}

local concat = table.concat

-- Text of one message `content` field. It is a string, or an array of parts
-- such as {type = "text", text = "..."}. Image and audio parts are skipped.
-- Any other value (cjson.null for an assistant tool call) gives "".
local function content_text(content)
  if type(content) == "string" then
    return content
  end
  if type(content) ~= "table" then
    return ""
  end
  local out = {}
  for _, part in ipairs(content) do
    if type(part) == "table" and part.type == "text" and type(part.text) == "string" then
      out[#out + 1] = part.text
    end
  end
  return concat(out, "\n")
end

-- Validate `body.messages` and extract text from it.
-- Returns all_text, last_user_text. On error returns nil and a message.
--   all_text: every message, used to estimate prompt tokens.
--   last_user_text: the latest user turn, used for the routing decision,
--   so that a long system prompt does not decide the tier of every request.
function _M.prompt_text(body)
  if type(body) ~= "table" then
    return nil, "request body must be a JSON object"
  end
  local messages = body.messages
  if type(messages) ~= "table" or #messages == 0 then
    return nil, "'messages' must be a non-empty array"
  end

  local all, last_user = {}, nil
  for i, m in ipairs(messages) do
    if type(m) ~= "table" or type(m.role) ~= "string" then
      return nil, ("'messages[%d]' must be an object with a 'role'"):format(i - 1)
    end
    local text = content_text(m.content)
    all[#all + 1] = text
    if m.role == "user" then
      last_user = text
    end
  end
  local all_text = concat(all, "\n")
  return all_text, last_user or all_text
end

return _M
