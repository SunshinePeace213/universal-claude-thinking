# Debug Monitor

## Debug Header Generation

### Core Debug Class

```javascript
class DebugMonitor {
  constructor() {
    this.activeModules = [];
    this.classification = null;
    this.tokenCounts = {
      orchestrator: 450,
      modules: 0,
      total: 450,
      budget: 5000
    };
    this.loadTimes = new Map();
    this.startTime = Date.now();
  }
  
  setClassification(category, confidence) {
    this.classification = { category, confidence };
  }
  
  addModule(name, tokens, loadTime) {
    this.activeModules.push({ name, tokens });
    this.tokenCounts.modules += tokens;
    this.tokenCounts.total = this.tokenCounts.orchestrator + this.tokenCounts.modules;
    this.loadTimes.set(name, loadTime);
  }
  
  getModuleStatus(name) {
    const module = this.activeModules.find(m => m.name === name);
    if (!module) return '❌';
    
    const loadTime = this.loadTimes.get(name);
    if (loadTime > 100) return '⚠️';
    return '✅';
  }
  
  generateHeader() {
    const modules = this.activeModules
      .map(m => `${m.name} (${this.formatTokens(m.tokens)})`)
      .join(', ');
    
    const classification = this.classification
      ? `${this.classification.category} (${this.classification.confidence.toFixed(2)} confidence)`
      : 'unclassified';
    
    const totalLoadTime = Array.from(this.loadTimes.values())
      .reduce((sum, time) => sum + time, 0);
    
    const header = [
      `🎯 Active Modules: ${modules || 'none'}`,
      `⚡ Classification: ${classification}`,
      `📊 Total Tokens: ${this.tokenCounts.total.toLocaleString()} / ${this.tokenCounts.budget.toLocaleString()}`,
      `🕒 Load Time: ${totalLoadTime}ms`
    ];
    
    // Add MCP tool status if available
    if (this.hasMcpTools()) {
      header.push(`🔧 MCP Tools: ${this.getMcpStatus()}`);
    }
    
    return header.join('\n');
  }
  
  formatTokens(count) {
    if (count >= 1000) {
      return `${(count / 1000).toFixed(1)}K`;
    }
    return count.toString();
  }
  
  hasMcpTools() {
    // Check if Clear-thought MCP is available
    return typeof global.mcpTools !== 'undefined';
  }
  
  getMcpStatus() {
    // Get MCP tool status
    const tools = ['sequentialthinking', 'mentalmodel', 'debuggingapproach'];
    const available = tools.filter(tool => this.isMcpToolAvailable(tool));
    return `${available.length}/${tools.length} active`;
  }
  
  isMcpToolAvailable(toolName) {
    // Check if specific MCP tool is available
    return global.mcpTools?.includes(toolName) || false;
  }
}
```

### Module Status Indicators

```javascript
const MODULE_INDICATORS = {
  // Status icons
  loaded: '✅',
  loading: '⏳',
  failed: '❌',
  fallback: '⚠️',
  
  // Module type icons
  thinking: '🧠',
  search: '🔍',
  code: '💻',
  analysis: '📊',
  meta: '🤔',
  
  // Performance icons
  fast: '⚡',
  normal: '✓',
  slow: '🐌'
};

function getModuleIcon(module) {
  const typeMap = {
    'SAGE': MODULE_INDICATORS.thinking,
    'SEIQF': MODULE_INDICATORS.analysis,
    'SIA': MODULE_INDICATORS.search,
    'response-formats': MODULE_INDICATORS.meta,
    'cognitive-tools': MODULE_INDICATORS.analysis
  };
  
  for (const [key, icon] of Object.entries(typeMap)) {
    if (module.name.includes(key)) {
      return icon;
    }
  }
  
  return MODULE_INDICATORS.thinking;
}
```

### Token Tracking

```javascript
class TokenTracker {
  constructor(budget = 5000) {
    this.budget = budget;
    this.usage = new Map();
    this.warnings = [];
  }
  
  track(component, tokens) {
    this.usage.set(component, tokens);
    
    const total = this.getTotal();
    if (total > this.budget * 0.8) {
      this.warnings.push(`Token usage at ${Math.round(total / this.budget * 100)}%`);
    }
    
    if (total > this.budget) {
      this.warnings.push(`Token budget exceeded by ${total - this.budget}`);
    }
  }
  
  getTotal() {
    return Array.from(this.usage.values())
      .reduce((sum, tokens) => sum + tokens, 0);
  }
  
  getBreakdown() {
    const breakdown = {};
    for (const [component, tokens] of this.usage) {
      breakdown[component] = {
        tokens,
        percentage: Math.round(tokens / this.getTotal() * 100)
      };
    }
    return breakdown;
  }
  
  shouldTruncate() {
    return this.getTotal() > this.budget * 0.95;
  }
}
```

### Compact Debug Format

```javascript
function generateCompactDebug(monitor) {
  // Keep under 200 tokens
  const modules = monitor.activeModules
    .map(m => `${getModuleIcon(m)}${m.name.split('/').pop()}`)
    .join(' ');
  
  const tokens = `${monitor.tokenCounts.total}/${monitor.tokenCounts.budget}`;
  const perf = monitor.getTotalLoadTime() > 100 ? '🐌' : '⚡';
  
  return `[${modules}] ${monitor.classification.category} ${tokens} ${perf}`;
}
```