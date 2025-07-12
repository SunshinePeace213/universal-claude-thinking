# External APIs

## Clear-thought MCP API
- **Purpose:** Structured thinking operations
- **Documentation:** MCP protocol specification
- **Base URL(s):** Local MCP server
- **Authentication:** MCP session tokens
- **Rate Limits:** None (local server)

**Key Endpoints Used:**
- `sequentialthinking` - Step-by-step reasoning
- `mentalmodel` - Apply mental frameworks
- `debuggingapproach` - Systematic debugging
- `scientificmethod` - Hypothesis testing

**Integration Notes:** Support for nested invocations, state preservation across calls

## WebSearch/Tavily-MCP APIs
- **Purpose:** Current information retrieval
- **Documentation:** Tavily API docs
- **Base URL(s):** https://api.tavily.com
- **Authentication:** API key
- **Rate Limits:** Per account limits

**Key Endpoints Used:**
- `tavily-search` - Web search with parameters
- `tavily-extract` - Content extraction
- `tavily-crawl` - Site crawling

**Integration Notes:** SIA optimizes search parameters, SEIQF validates results
