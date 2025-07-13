---
version: 2.1.0
tokens: 500
---

# Claude Thinking Orchestrator

## Request Classification

@import "./request-classifier.md"

Use the enhanced classifier to categorize requests:
- **simple**: Direct questions, basic explanations
- **complex**: Multi-step reasoning, analysis tasks
- **search**: Finding information, code exploration
- **code**: Writing, refactoring, debugging code
- **meta**: Self-reflection, process questions

Returns RequestClassification with:
- category, confidence, requiredModules
- suggestedAgents, mcpTools, estimatedTokens

Confidence threshold: 0.8

## Module Loading Infrastructure

@import "./module-loader.md"

Initialize module loading system:
1. Create ModuleLoader with security validation
2. Setup TokenTracker with 5K budget
3. Initialize ModuleRegistry from metadata.yaml
4. Configure health checks and monitoring

## Dynamic Module Loading

```typescript
// Classify request
const classification = await classifier.classify(userRequest);

// Initialize infrastructure
const registry = new ModuleRegistry('./.claude/config/metadata.yaml');
await registry.initialize();

const validator = new SecurityValidator();
const tokenTracker = new TokenTracker(5000);
const healthChecker = new ModuleHealthChecker();

// Configure module loader
const loader = new ModuleLoader({
  basePath: './.claude',
  tokenBudget: 5000,
  securityValidator: validator,
  moduleRegistry: registry
});

// Resolve dependencies and load modules
const resolver = new DependencyResolver(registry);
const loadOrder = await resolver.resolveDependencies(classification.requiredModules);

// Load modules with health checks
const loadedModules = [];
for (const moduleId of loadOrder) {
  // Reserve tokens
  const metadata = await registry.getMetadata(moduleId);
  if (!tokenTracker.reserveTokens(moduleId, metadata.tokenCount)) {
    console.warn(`Token budget exceeded for ${moduleId}`);
    continue;
  }
  
  // Load and validate module
  try {
    const module = await loader.loadModule(moduleId);
    
    // Health check
    const health = await healthChecker.checkModuleHealth(
      moduleId,
      module.path,
      module.content
    );
    
    if (health.status === 'critical') {
      throw new Error(`Module ${moduleId} failed health check`);
    }
    
    loadedModules.push(module);
    tokenTracker.commitTokens(moduleId, metadata.tokenCount);
    
  } catch (error) {
    tokenTracker.releaseTokens(moduleId);
    console.error(`Failed to load ${moduleId}:`, error);
  }
}

// Activate loaded modules based on classification
activateModules(loadedModules, classification);
```

## Fallback Protocol

If module loading fails:
1. Continue with basic reasoning
2. Log failure in debug header
3. Use MCP tools directly if available

## Debug Header

```typescript
// Generate debug header with module loader stats
const debugHeader = {
  activeModules: loadedModules.map(m => m.id).join(', '),
  classification: `${classification.category} (${classification.confidence.toFixed(2)})`,
  tokenUsage: `${tokenTracker.getTotalTokensUsed()} / ${tokenTracker.budget.total}`,
  loadTime: `${totalLoadTime}ms`,
  telemetry: `${cacheHits} / ${totalRequests}`,
  suggestedAgents: classification.suggestedAgents.join(', '),
  mcpTools: classification.mcpTools.join(', '),
  health: unhealthyModules.length > 0 ? `⚠️ ${unhealthyModules.length} unhealthy` : '✅ All healthy'
};

console.log(`
🎯 Active Modules: ${debugHeader.activeModules}
⚡ Classification: ${debugHeader.classification}
📊 Total Tokens: ${debugHeader.tokenUsage}
🕒 Load Time: ${debugHeader.loadTime}
📈 Telemetry: ${debugHeader.telemetry}
🤖 Suggested Agents: ${debugHeader.suggestedAgents}
🔧 MCP Tools: ${debugHeader.mcpTools}
💚 Module Health: ${debugHeader.health}
`);
```

## Error Handling

- Invalid paths → Log and continue
- Missing modules → Use fallback
- Token overflow → Truncate gracefully
- MCP failures → Degrade to basic operation