--[[
book-extras.lua: boxes appended to the end of every HTML page of the book.

  - "Use this book with your AI assistant" (growth experiment E3): every page,
    home page included.

The boxes are written into the page at render time (no JavaScript needed to
see them), and only for HTML output: the PDF and EPUB are left alone. Styles
and the copy button live in _includes/book-extras.html.
]]

local MCP_ENDPOINT = "https://research-methods-mcp.vercel.app/mcp?src=book-credit-score"
local MCP_SITE = "https://research-methods-mcp.vercel.app/"
-- Same install-link format as README.md in research-methods-mcp, with this
-- book's ?src= endpoint inside the config.
local CURSOR_INSTALL = "https://cursor.com/en/install-mcp?name=research-methods&config="
  .. "eyJ1cmwiOiJodHRwczovL3Jlc2VhcmNoLW1ldGhvZHMtbWNwLnZlcmNlbC5hcHAvbWNwP3NyYz1ib29rLWNyZWRpdC1zY29yZSJ9"
local VSCODE_INSTALL = "https://insiders.vscode.dev/redirect/mcp/install?name=research-methods&config="
  .. "%7B%22type%22%3A%22http%22%2C%22url%22%3A%22https%3A%2F%2Fresearch-methods-mcp.vercel.app"
  .. "%2Fmcp%3Fsrc%3Dbook-credit-score%22%7D"

local function ai_box()
  return table.concat({
    '<div class="book-extras-box book-extras-ai" role="note" aria-label="Use this book with your AI assistant">',
    '<p class="book-extras-title">Use this book with your AI assistant</p>',
    '<p>Ask questions about this book in Claude, Cursor, VS Code, or ChatGPT, with answers that cite the section.</p>',
    '<p class="book-extras-endpoint"><code>', MCP_ENDPOINT, '</code>',
    '<button type="button" class="book-extras-copy" data-copy="', MCP_ENDPOINT,
    '" aria-label="Copy the MCP endpoint">Copy</button></p>',
    '<p class="book-extras-install">',
    '<a href="', CURSOR_INSTALL, '"><img src="https://cursor.com/deeplink/mcp-install-dark.svg" ',
    'alt="Add to Cursor" height="24" loading="lazy"></a> ',
    '<a href="', VSCODE_INSTALL, '"><img ',
    'src="https://img.shields.io/badge/VS_Code-Install_server-0098FF?logo=visualstudiocode&amp;logoColor=white" ',
    'alt="Install in VS Code" height="24" loading="lazy"></a> ',
    '<a href="', MCP_SITE, '">Other apps</a>',
    '</p>',
    '</div>',
  })
end

function Pandoc(doc)
  if not quarto.doc.is_format("html:js") then
    return nil
  end
  local html = '<div class="book-extras">' .. ai_box() .. '</div>'
  doc.blocks:insert(pandoc.RawBlock("html", html))
  return doc
end
