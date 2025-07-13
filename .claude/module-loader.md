# Module Loader Infrastructure

## Overview

The Module Loader is responsible for dynamically loading thinking modules based on request classification, managing dependencies, tracking token usage, and ensuring security through hash validation.

## Core Components

### 1. Import Parser

Parses @import directives using Claude Code's native syntax:

```typescript
interface ImportDirective {
  path: string;          // Relative path from .claude/
  condition?: string;    // Optional conditional loading expression
  priority?: number;     // Load order priority (default: 0)
}

class ImportParser {
  parse(content: string): ImportDirective[] {
    // Extract @import directives from module content
    const importRegex = /@import\s+"([^"]+)"(?:\s+if\s+(.+))?/g;
    const imports: ImportDirective[] = [];
    
    let match;
    while ((match = importRegex.exec(content)) !== null) {
      imports.push({
        path: match[1],
        condition: match[2] || undefined,
        priority: 0
      });
    }
    
    return imports;
  }
}
```

### 2. Module Loader

Main class responsible for loading modules based on classification:

```typescript
interface ModuleLoaderConfig {
  basePath: string;           // Base directory (.claude/)
  tokenBudget: number;        // Total token budget (5K default)
  securityValidator: SecurityValidator;
  moduleRegistry: ModuleRegistry;
}

class ModuleLoader {
  private loadedModules: Map<string, LoadedModule> = new Map();
  private totalTokensUsed: number = 0;
  
  constructor(private config: ModuleLoaderConfig) {}
  
  async loadModules(classification: RequestClassification): Promise<LoadedModule[]> {
    const modules: LoadedModule[] = [];
    
    // Load required modules from classification
    for (const moduleId of classification.requiredModules) {
      try {
        const module = await this.loadModule(moduleId);
        modules.push(module);
      } catch (error) {
        console.error(`Failed to load module ${moduleId}:`, error);
        // Graceful degradation - continue with other modules
      }
    }
    
    return modules;
  }
  
  private async loadModule(moduleId: string): Promise<LoadedModule> {
    // Check if already loaded
    if (this.loadedModules.has(moduleId)) {
      return this.loadedModules.get(moduleId)!;
    }
    
    // Get module metadata
    const metadata = await this.config.moduleRegistry.getMetadata(moduleId);
    if (!metadata) {
      throw new Error(`Module ${moduleId} not found in registry`);
    }
    
    // Check token budget
    if (this.totalTokensUsed + metadata.tokenCount > this.config.tokenBudget) {
      throw new Error(`Token budget exceeded. Used: ${this.totalTokensUsed}, Required: ${metadata.tokenCount}, Budget: ${this.config.tokenBudget}`);
    }
    
    // Construct module path
    const modulePath = this.resolveModulePath(moduleId);
    
    // Validate module security
    await this.config.securityValidator.validateModule(modulePath, metadata.securityHash);
    
    // Load module content
    const content = await this.readModuleContent(modulePath);
    
    // Parse imports
    const parser = new ImportParser();
    const imports = parser.parse(content);
    
    // Create loaded module
    const loadedModule: LoadedModule = {
      id: moduleId,
      path: modulePath,
      content: content,
      metadata: metadata,
      imports: imports,
      loadTime: Date.now()
    };
    
    // Update tracking
    this.loadedModules.set(moduleId, loadedModule);
    this.totalTokensUsed += metadata.tokenCount;
    
    return loadedModule;
  }
  
  private resolveModulePath(moduleId: string): string {
    // Map module IDs to file paths
    const pathMappings: Record<string, string> = {
      'SAGE': 'thinking-modules/SAGE.md',
      'SEIQF': 'thinking-modules/SEIQF.md',
      'SIA': 'thinking-modules/SIA.md',
      'response-formats': 'thinking-modules/response-formats.md',
      'cognitive-tools/analysis': 'cognitive-tools/analysis.md',
      'cognitive-tools/search': 'cognitive-tools/search.md',
      'cognitive-tools/code-analysis': 'cognitive-tools/code-analysis.md',
      'cognitive-tools/meta-reasoning': 'cognitive-tools/meta-reasoning.md'
    };
    
    const relativePath = pathMappings[moduleId];
    if (!relativePath) {
      throw new Error(`Unknown module ID: ${moduleId}`);
    }
    
    return `${this.config.basePath}/${relativePath}`;
  }
  
  private async readModuleContent(path: string): Promise<string> {
    // In Claude Code context, this would use the native file reading capability
    // For now, return a placeholder that demonstrates the structure
    return `# Module Content for ${path}\n\n@import directives and module logic would be here`;
  }
}
```

### 3. Conditional Loading

Evaluates conditions for dynamic module inclusion:

```typescript
interface LoadContext {
  classification: RequestClassification;
  protocolState: any;
  mcpTools: string[];
}

class ConditionalLoader {
  evaluateCondition(condition: string, context: LoadContext): boolean {
    // Simple condition evaluation based on classification and context
    // Examples:
    // - "confidence > 0.8"
    // - "category === 'complex'"
    // - "hasTools(['github', 'playwright'])"
    
    try {
      // Safe evaluation using context variables
      const evalContext = {
        confidence: context.classification.confidence,
        category: context.classification.category,
        hasTools: (tools: string[]) => 
          tools.some(tool => context.classification.mcpTools.includes(tool))
      };
      
      // Parse and evaluate condition safely
      return this.safeEvaluate(condition, evalContext);
    } catch (error) {
      console.warn(`Failed to evaluate condition "${condition}":`, error);
      return false;
    }
  }
  
  private safeEvaluate(condition: string, context: any): boolean {
    // Implement safe condition evaluation without eval()
    // This is a simplified version - real implementation would use a proper parser
    
    // Check for simple comparisons
    if (condition.includes('>')) {
      const [left, right] = condition.split('>').map(s => s.trim());
      return context[left] > parseFloat(right);
    }
    
    if (condition.includes('===')) {
      const [left, right] = condition.split('===').map(s => s.trim());
      return context[left] === right.replace(/['"]/g, '');
    }
    
    // Check for function calls
    if (condition.includes('(') && condition.includes(')')) {
      const funcMatch = condition.match(/(\w+)\((.*)\)/);
      if (funcMatch) {
        const [, funcName, args] = funcMatch;
        if (typeof context[funcName] === 'function') {
          const parsedArgs = JSON.parse(`[${args}]`);
          return context[funcName](...parsedArgs);
        }
      }
    }
    
    return false;
  }
}
```

### 4. Error Handling

Comprehensive error handling for module loading failures:

```typescript
enum ModuleLoadError {
  NOT_FOUND = 'MODULE_NOT_FOUND',
  SECURITY_FAILED = 'SECURITY_VALIDATION_FAILED',
  TOKEN_EXCEEDED = 'TOKEN_BUDGET_EXCEEDED',
  CIRCULAR_DEPENDENCY = 'CIRCULAR_DEPENDENCY_DETECTED',
  PARSE_ERROR = 'MODULE_PARSE_ERROR'
}

class ModuleLoadException extends Error {
  constructor(
    public code: ModuleLoadError,
    public moduleId: string,
    message: string,
    public cause?: Error
  ) {
    super(message);
    this.name = 'ModuleLoadException';
  }
}

class ErrorHandler {
  handleLoadError(error: ModuleLoadException): void {
    // Log error with context
    console.error(`[ModuleLoader] ${error.code}: ${error.message}`, {
      moduleId: error.moduleId,
      cause: error.cause
    });
    
    // Emit telemetry event
    this.emitTelemetry({
      event: 'module_load_failed',
      code: error.code,
      moduleId: error.moduleId,
      timestamp: Date.now()
    });
    
    // Apply recovery strategy based on error type
    switch (error.code) {
      case ModuleLoadError.NOT_FOUND:
        // Try fallback module if available
        break;
      case ModuleLoadError.SECURITY_FAILED:
        // Quarantine module and alert
        break;
      case ModuleLoadError.TOKEN_EXCEEDED:
        // Suggest module prioritization
        break;
      default:
        // Log and continue with degraded functionality
    }
  }
  
  private emitTelemetry(event: any): void {
    // Telemetry implementation
  }
}
```

## Integration Points

### With Request Classifier

```typescript
// Example integration
const classification = await classifier.classify(request);
const modules = await moduleLoader.loadModules(classification);

// Module activation based on classification
for (const module of modules) {
  await activateModule(module);
}
```

### With Protocol State

```typescript
// Update protocol state with loaded modules
protocolState.loadedModules = modules.map(m => ({
  id: m.id,
  version: m.metadata.version,
  loadTime: m.loadTime,
  tokenCount: m.metadata.tokenCount
}));

protocolState.totalTokensUsed = moduleLoader.getTotalTokensUsed();
```

## Module Resolution Flow

1. **Classification** → Determines required modules
2. **Registry Lookup** → Get module metadata
3. **Token Check** → Ensure within budget
4. **Security Validation** → Verify module hash
5. **Content Loading** → Read module file
6. **Import Parsing** → Extract dependencies
7. **Conditional Evaluation** → Check dynamic conditions
8. **Activation** → Make module available
9. **State Update** → Track loaded modules

## Error Recovery Strategies

1. **Missing Module**: Use fallback or degrade gracefully
2. **Security Failure**: Quarantine and alert
3. **Token Exceeded**: Prioritize essential modules
4. **Parse Error**: Log and skip module
5. **Circular Dependency**: Break cycle and warn

## Performance Considerations

- Cache loaded modules to avoid re-parsing
- Parallel loading where dependencies allow
- Early token budget checks
- Lazy loading for optional modules
- Module load time target: <50ms per module

## Module Registry

Implementation of the module registry system:

```typescript
interface ModuleMetadata {
  id: string;              // Unique module identifier
  version: string;         // Semantic version
  tokenCount: number;      // Estimated token usage
  dependencies: string[];  // Required module IDs
  triggers: string[];      // Auto-activation keywords
  protocols: ('SAGE' | 'SEIQF' | 'SIA')[];  // Supported protocols
  securityHash: string;    // SHA-256 hash for validation
  lastModified: string;    // ISO date string
  description?: string;    // Module description
}

interface RegistryConfig {
  version: string;
  maxModules: number;
  tokenBudget: number;
  cacheEnabled: boolean;
  cacheTTL: number;
  validationRequired: boolean;
  quarantineEnabled: boolean;
  quarantineMaxRetries: number;
}

class ModuleRegistry {
  private modules: Map<string, ModuleMetadata> = new Map();
  private config: RegistryConfig;
  private cache: Map<string, { data: ModuleMetadata; timestamp: number }> = new Map();
  
  constructor(private registryPath: string) {}
  
  async initialize(): Promise<void> {
    // Load registry from metadata.yaml
    const registryData = await this.loadRegistryFile();
    
    // Parse and validate registry
    this.config = registryData.registry;
    
    // Load all module metadata
    for (const [moduleId, metadata] of Object.entries(registryData.modules)) {
      this.modules.set(moduleId, metadata as ModuleMetadata);
    }
    
    console.log(`Registry initialized with ${this.modules.size} modules`);
  }
  
  async getMetadata(moduleId: string): Promise<ModuleMetadata | null> {
    // Check cache first if enabled
    if (this.config.cacheEnabled) {
      const cached = this.cache.get(moduleId);
      if (cached && Date.now() - cached.timestamp < this.config.cacheTTL * 1000) {
        return cached.data;
      }
    }
    
    // Get from registry
    const metadata = this.modules.get(moduleId);
    if (!metadata) {
      return null;
    }
    
    // Update cache
    if (this.config.cacheEnabled) {
      this.cache.set(moduleId, {
        data: metadata,
        timestamp: Date.now()
      });
    }
    
    return metadata;
  }
  
  async updateMetadata(moduleId: string, updates: Partial<ModuleMetadata>): Promise<void> {
    const existing = await this.getMetadata(moduleId);
    if (!existing) {
      throw new Error(`Module ${moduleId} not found in registry`);
    }
    
    // Merge updates
    const updated = { ...existing, ...updates, lastModified: new Date().toISOString() };
    this.modules.set(moduleId, updated);
    
    // Invalidate cache
    this.cache.delete(moduleId);
    
    // Persist to file
    await this.saveRegistry();
  }
  
  async addModule(metadata: ModuleMetadata): Promise<void> {
    if (this.modules.size >= this.config.maxModules) {
      throw new Error(`Registry full. Max modules: ${this.config.maxModules}`);
    }
    
    // Validate module doesn't already exist
    if (this.modules.has(metadata.id)) {
      throw new Error(`Module ${metadata.id} already exists in registry`);
    }
    
    // Add to registry
    this.modules.set(metadata.id, metadata);
    
    // Persist to file
    await this.saveRegistry();
  }
  
  async removeModule(moduleId: string): Promise<void> {
    if (!this.modules.has(moduleId)) {
      throw new Error(`Module ${moduleId} not found in registry`);
    }
    
    // Check for dependencies
    const dependents = this.findDependents(moduleId);
    if (dependents.length > 0) {
      throw new Error(`Cannot remove module ${moduleId}. Required by: ${dependents.join(', ')}`);
    }
    
    // Remove from registry
    this.modules.delete(moduleId);
    this.cache.delete(moduleId);
    
    // Persist to file
    await this.saveRegistry();
  }
  
  findDependents(moduleId: string): string[] {
    const dependents: string[] = [];
    
    for (const [id, metadata] of this.modules.entries()) {
      if (metadata.dependencies.includes(moduleId)) {
        dependents.push(id);
      }
    }
    
    return dependents;
  }
  
  getModulesByProtocol(protocol: 'SAGE' | 'SEIQF' | 'SIA'): ModuleMetadata[] {
    const modules: ModuleMetadata[] = [];
    
    for (const metadata of this.modules.values()) {
      if (metadata.protocols.includes(protocol)) {
        modules.push(metadata);
      }
    }
    
    return modules;
  }
  
  getModulesByTrigger(keyword: string): ModuleMetadata[] {
    const modules: ModuleMetadata[] = [];
    const lowerKeyword = keyword.toLowerCase();
    
    for (const metadata of this.modules.values()) {
      if (metadata.triggers.some(trigger => 
        trigger.toLowerCase().includes(lowerKeyword)
      )) {
        modules.push(metadata);
      }
    }
    
    return modules;
  }
  
  private async loadRegistryFile(): Promise<any> {
    // In real implementation, this would read and parse the YAML file
    // For now, return a placeholder structure
    return {
      modules: {},
      registry: {
        version: "1.0.0",
        maxModules: 50,
        tokenBudget: 5000,
        cacheEnabled: true,
        cacheTTL: 3600,
        validationRequired: true,
        quarantineEnabled: true,
        quarantineMaxRetries: 3
      }
    };
  }
  
  private async saveRegistry(): Promise<void> {
    // Convert registry to YAML format
    const registryData = {
      modules: Object.fromEntries(this.modules),
      registry: this.config
    };
    
    // In real implementation, this would write to the YAML file
    // For now, just log
    console.log('Registry saved');
  }
}

// Usage example
const registry = new ModuleRegistry('./.claude/config/metadata.yaml');
await registry.initialize();

// Lookup module
const sageMetadata = await registry.getMetadata('SAGE');
console.log(`SAGE module: ${sageMetadata?.tokenCount} tokens`);
```

## Dependency Resolution

Implementation of topological sort for dependency resolution:

```typescript
interface DependencyNode {
  id: string;
  dependencies: string[];
  optional?: string[];
  visited?: boolean;
  visiting?: boolean;
}

interface DependencyGraph {
  nodes: Map<string, DependencyNode>;
  resolved: string[];
  cache: Map<string, string[]>;
}

class DependencyResolver {
  private graph: DependencyGraph = {
    nodes: new Map(),
    resolved: [],
    cache: new Map()
  };
  
  constructor(private registry: ModuleRegistry) {}
  
  async resolveDependencies(moduleIds: string[]): Promise<string[]> {
    // Check cache for complete resolution
    const cacheKey = moduleIds.sort().join(',');
    if (this.graph.cache.has(cacheKey)) {
      return this.graph.cache.get(cacheKey)!;
    }
    
    // Reset graph for new resolution
    this.graph.nodes.clear();
    this.graph.resolved = [];
    
    // Build dependency graph
    await this.buildGraph(moduleIds);
    
    // Detect circular dependencies
    const circular = this.detectCircularDependencies();
    if (circular.length > 0) {
      throw new ModuleLoadException(
        ModuleLoadError.CIRCULAR_DEPENDENCY,
        circular.join(' -> '),
        `Circular dependency detected: ${circular.join(' -> ')}`
      );
    }
    
    // Perform topological sort
    for (const moduleId of moduleIds) {
      await this.visit(moduleId);
    }
    
    // Cache result
    this.graph.cache.set(cacheKey, [...this.graph.resolved]);
    
    return this.graph.resolved;
  }
  
  private async buildGraph(moduleIds: string[]): Promise<void> {
    const visited = new Set<string>();
    const queue = [...moduleIds];
    
    while (queue.length > 0) {
      const moduleId = queue.shift()!;
      
      if (visited.has(moduleId)) {
        continue;
      }
      
      visited.add(moduleId);
      
      // Get module metadata
      const metadata = await this.registry.getMetadata(moduleId);
      if (!metadata) {
        console.warn(`Module ${moduleId} not found in registry`);
        continue;
      }
      
      // Create node
      const node: DependencyNode = {
        id: moduleId,
        dependencies: metadata.dependencies || [],
        optional: []  // Could be extended to support optional deps
      };
      
      this.graph.nodes.set(moduleId, node);
      
      // Add dependencies to queue
      queue.push(...node.dependencies);
    }
  }
  
  private detectCircularDependencies(): string[] {
    const circular: string[] = [];
    const path: string[] = [];
    const visited = new Set<string>();
    
    const detectCycle = (nodeId: string): boolean => {
      if (path.includes(nodeId)) {
        // Found cycle
        const cycleStart = path.indexOf(nodeId);
        circular.push(...path.slice(cycleStart), nodeId);
        return true;
      }
      
      if (visited.has(nodeId)) {
        return false;
      }
      
      const node = this.graph.nodes.get(nodeId);
      if (!node) {
        return false;
      }
      
      path.push(nodeId);
      
      for (const dep of node.dependencies) {
        if (detectCycle(dep)) {
          return true;
        }
      }
      
      path.pop();
      visited.add(nodeId);
      
      return false;
    };
    
    // Check all nodes
    for (const nodeId of this.graph.nodes.keys()) {
      if (detectCycle(nodeId)) {
        break;
      }
    }
    
    return circular;
  }
  
  private async visit(nodeId: string): Promise<void> {
    const node = this.graph.nodes.get(nodeId);
    if (!node) {
      // Module not in graph, might be external
      return;
    }
    
    if (node.visited) {
      return;
    }
    
    if (node.visiting) {
      // This shouldn't happen if circular deps were detected
      throw new Error(`Unexpected cycle at ${nodeId}`);
    }
    
    node.visiting = true;
    
    // Visit dependencies first (depth-first)
    for (const depId of node.dependencies) {
      await this.visit(depId);
    }
    
    // Visit optional dependencies if available
    if (node.optional) {
      for (const optId of node.optional) {
        try {
          await this.visit(optId);
        } catch (error) {
          // Optional dependency failed, continue
          console.warn(`Optional dependency ${optId} failed: ${error}`);
        }
      }
    }
    
    node.visiting = false;
    node.visited = true;
    
    // Add to resolved list (post-order)
    if (!this.graph.resolved.includes(nodeId)) {
      this.graph.resolved.push(nodeId);
    }
  }
  
  async getDependencyTree(moduleId: string): Promise<DependencyTree> {
    const tree: DependencyTree = {
      id: moduleId,
      children: []
    };
    
    const metadata = await this.registry.getMetadata(moduleId);
    if (!metadata) {
      return tree;
    }
    
    // Recursively build tree
    for (const depId of metadata.dependencies) {
      const childTree = await this.getDependencyTree(depId);
      tree.children.push(childTree);
    }
    
    return tree;
  }
  
  optimizeLoadOrder(modules: string[]): string[] {
    // Group modules by dependency depth
    const depths = new Map<string, number>();
    
    const calculateDepth = (moduleId: string, visited = new Set<string>()): number => {
      if (depths.has(moduleId)) {
        return depths.get(moduleId)!;
      }
      
      if (visited.has(moduleId)) {
        return 0; // Circular reference
      }
      
      visited.add(moduleId);
      
      const node = this.graph.nodes.get(moduleId);
      if (!node || node.dependencies.length === 0) {
        depths.set(moduleId, 0);
        return 0;
      }
      
      let maxDepth = 0;
      for (const depId of node.dependencies) {
        const depDepth = calculateDepth(depId, new Set(visited));
        maxDepth = Math.max(maxDepth, depDepth + 1);
      }
      
      depths.set(moduleId, maxDepth);
      return maxDepth;
    };
    
    // Calculate depths
    for (const moduleId of modules) {
      calculateDepth(moduleId);
    }
    
    // Sort by depth (dependencies first)
    return modules.sort((a, b) => {
      const depthA = depths.get(a) || 0;
      const depthB = depths.get(b) || 0;
      return depthA - depthB;
    });
  }
}

interface DependencyTree {
  id: string;
  children: DependencyTree[];
}

// Usage example
const resolver = new DependencyResolver(registry);

// Resolve dependencies for a set of modules
const loadOrder = await resolver.resolveDependencies(['SEIQF', 'cognitive-tools/code-analysis']);
console.log('Load order:', loadOrder);
// Output: ['SEIQF', 'cognitive-tools/code-analysis']

// Get dependency tree
const tree = await resolver.getDependencyTree('cognitive-tools/code-analysis');
console.log('Dependency tree:', JSON.stringify(tree, null, 2));
```

## Token Tracking System

Implementation of comprehensive token tracking:

```typescript
interface TokenMetrics {
  moduleId: string;
  allocated: number;
  used: number;
  percentage: number;
  timestamp: number;
}

interface TokenBudget {
  total: number;
  used: number;
  reserved: number;
  available: number;
  metrics: Map<string, TokenMetrics>;
}

class TokenTracker {
  private budget: TokenBudget = {
    total: 5000,  // Default 5K token budget
    used: 0,
    reserved: 0,
    available: 5000,
    metrics: new Map()
  };
  
  private alerts: TokenAlert[] = [];
  
  constructor(totalBudget: number = 5000) {
    this.budget.total = totalBudget;
    this.budget.available = totalBudget;
  }
  
  reserveTokens(moduleId: string, tokenCount: number): boolean {
    // Check if tokens are available
    if (tokenCount > this.budget.available) {
      this.emitAlert({
        type: 'BUDGET_EXCEEDED',
        moduleId,
        requested: tokenCount,
        available: this.budget.available,
        timestamp: Date.now()
      });
      return false;
    }
    
    // Reserve tokens
    this.budget.reserved += tokenCount;
    this.budget.available -= tokenCount;
    
    // Initialize metrics for module
    this.budget.metrics.set(moduleId, {
      moduleId,
      allocated: tokenCount,
      used: 0,
      percentage: 0,
      timestamp: Date.now()
    });
    
    return true;
  }
  
  commitTokens(moduleId: string, actualUsed: number): void {
    const metrics = this.budget.metrics.get(moduleId);
    if (!metrics) {
      console.warn(`No token reservation found for module ${moduleId}`);
      return;
    }
    
    // Update metrics
    metrics.used = actualUsed;
    metrics.percentage = (actualUsed / metrics.allocated) * 100;
    
    // Update budget
    this.budget.used += actualUsed;
    this.budget.reserved -= metrics.allocated;
    
    // Adjust available if actual usage was less than allocated
    const difference = metrics.allocated - actualUsed;
    if (difference > 0) {
      this.budget.available += difference;
    }
    
    // Check for overuse
    if (actualUsed > metrics.allocated) {
      this.emitAlert({
        type: 'MODULE_OVERUSE',
        moduleId,
        allocated: metrics.allocated,
        used: actualUsed,
        overage: actualUsed - metrics.allocated,
        timestamp: Date.now()
      });
    }
    
    // Log metrics to protocol state
    this.logMetrics(metrics);
  }
  
  releaseTokens(moduleId: string): void {
    const metrics = this.budget.metrics.get(moduleId);
    if (!metrics) {
      return;
    }
    
    // Release unused reserved tokens
    const unused = metrics.allocated - metrics.used;
    if (unused > 0 && metrics.used === 0) {
      this.budget.reserved -= metrics.allocated;
      this.budget.available += metrics.allocated;
      this.budget.metrics.delete(moduleId);
    }
  }
  
  getUsageReport(): TokenUsageReport {
    const moduleReports: ModuleTokenReport[] = [];
    
    for (const metrics of this.budget.metrics.values()) {
      moduleReports.push({
        moduleId: metrics.moduleId,
        allocated: metrics.allocated,
        used: metrics.used,
        efficiency: metrics.percentage,
        status: this.getModuleStatus(metrics)
      });
    }
    
    return {
      budget: {
        total: this.budget.total,
        used: this.budget.used,
        reserved: this.budget.reserved,
        available: this.budget.available,
        utilizationPercentage: (this.budget.used / this.budget.total) * 100
      },
      modules: moduleReports,
      alerts: this.alerts,
      recommendations: this.generateRecommendations()
    };
  }
  
  private getModuleStatus(metrics: TokenMetrics): string {
    if (metrics.percentage > 100) return 'OVERUSED';
    if (metrics.percentage > 90) return 'HIGH_USAGE';
    if (metrics.percentage > 70) return 'NORMAL';
    if (metrics.percentage > 0) return 'UNDERUTILIZED';
    return 'UNUSED';
  }
  
  private generateRecommendations(): string[] {
    const recommendations: string[] = [];
    
    // Check overall budget usage
    const usagePercentage = (this.budget.used / this.budget.total) * 100;
    
    if (usagePercentage > 90) {
      recommendations.push('Consider increasing token budget or optimizing module usage');
    }
    
    if (usagePercentage > 80) {
      recommendations.push('Token usage is high. Review module priorities');
    }
    
    // Check for inefficient modules
    for (const metrics of this.budget.metrics.values()) {
      if (metrics.percentage < 50 && metrics.allocated > 500) {
        recommendations.push(`Module ${metrics.moduleId} is underutilizing its token allocation`);
      }
    }
    
    // Check for token budget warnings
    if (this.budget.available < 500) {
      recommendations.push('Low token budget remaining. Prioritize essential modules only');
    }
    
    return recommendations;
  }
  
  private emitAlert(alert: TokenAlert): void {
    this.alerts.push(alert);
    
    // Log critical alerts
    if (alert.type === 'BUDGET_EXCEEDED' || alert.type === 'CRITICAL_LOW') {
      console.error(`[TokenTracker] ${alert.type}:`, alert);
    } else {
      console.warn(`[TokenTracker] ${alert.type}:`, alert);
    }
  }
  
  private logMetrics(metrics: TokenMetrics): void {
    // In real implementation, this would update protocol-state.json
    console.log(`[TokenTracker] Module ${metrics.moduleId}: ${metrics.used}/${metrics.allocated} tokens (${metrics.percentage.toFixed(1)}%)`);
  }
  
  enforceTokenBudget(moduleId: string, currentTokens: number): boolean {
    const metrics = this.budget.metrics.get(moduleId);
    if (!metrics) {
      return false;
    }
    
    // Check if module is exceeding its budget
    if (currentTokens > metrics.allocated) {
      // Emit warning but allow with grace period
      if (currentTokens > metrics.allocated * 1.1) {
        // 10% grace period exceeded
        this.emitAlert({
          type: 'BUDGET_ENFORCEMENT',
          moduleId,
          limit: metrics.allocated,
          current: currentTokens,
          timestamp: Date.now()
        });
        return false;
      }
    }
    
    return true;
  }
  
  getModuleTokenUsage(moduleId: string): TokenMetrics | null {
    return this.budget.metrics.get(moduleId) || null;
  }
  
  getTotalTokensUsed(): number {
    return this.budget.used;
  }
  
  getAvailableTokens(): number {
    return this.budget.available;
  }
}

interface TokenAlert {
  type: 'BUDGET_EXCEEDED' | 'MODULE_OVERUSE' | 'CRITICAL_LOW' | 'BUDGET_ENFORCEMENT';
  moduleId?: string;
  timestamp: number;
  [key: string]: any;
}

interface TokenUsageReport {
  budget: {
    total: number;
    used: number;
    reserved: number;
    available: number;
    utilizationPercentage: number;
  };
  modules: ModuleTokenReport[];
  alerts: TokenAlert[];
  recommendations: string[];
}

interface ModuleTokenReport {
  moduleId: string;
  allocated: number;
  used: number;
  efficiency: number;
  status: string;
}

// Usage example
const tokenTracker = new TokenTracker(5000);

// Reserve tokens for a module
if (tokenTracker.reserveTokens('SAGE', 2000)) {
  console.log('Tokens reserved for SAGE');
}

// Simulate module usage
tokenTracker.commitTokens('SAGE', 1800);

// Get usage report
const report = tokenTracker.getUsageReport();
console.log('Token usage report:', JSON.stringify(report, null, 2));
```

## Security Validation Framework

Implementation of comprehensive security validation:

```typescript
import * as crypto from 'crypto';
import * as path from 'path';

interface SecurityConfig {
  hashAlgorithm: 'sha256';
  quarantineEnabled: boolean;
  quarantineMaxRetries: number;
  auditEnabled: boolean;
  pathRestrictions: string[];
}

interface QuarantineEntry {
  moduleId: string;
  modulePath: string;
  reason: string;
  attempts: number;
  firstAttempt: number;
  lastAttempt: number;
  hash?: string;
}

interface AuditEvent {
  timestamp: number;
  eventType: 'VALIDATION_SUCCESS' | 'VALIDATION_FAILED' | 'QUARANTINE_ADD' | 
             'QUARANTINE_RELEASE' | 'PATH_VIOLATION' | 'MERKLE_VERIFICATION';
  moduleId?: string;
  details: any;
}

interface MerkleNode {
  hash: string;
  left?: MerkleNode;
  right?: MerkleNode;
  moduleId?: string;
}

class SecurityValidator {
  private config: SecurityConfig = {
    hashAlgorithm: 'sha256',
    quarantineEnabled: true,
    quarantineMaxRetries: 3,
    auditEnabled: true,
    pathRestrictions: ['.claude/', '.']
  };
  
  private quarantine: Map<string, QuarantineEntry> = new Map();
  private auditLog: AuditEvent[] = [];
  private merkleTree: MerkleNode | null = null;
  private moduleHashes: Map<string, string> = new Map();
  
  async validateModule(modulePath: string, expectedHash: string): Promise<boolean> {
    try {
      // Path traversal prevention
      if (!this.isPathSafe(modulePath)) {
        this.logAudit({
          timestamp: Date.now(),
          eventType: 'PATH_VIOLATION',
          details: { path: modulePath }
        });
        throw new Error(`Path traversal attempt detected: ${modulePath}`);
      }
      
      // Check quarantine
      const quarantineEntry = this.quarantine.get(modulePath);
      if (quarantineEntry && quarantineEntry.attempts >= this.config.quarantineMaxRetries) {
        throw new Error(`Module ${modulePath} is quarantined after ${quarantineEntry.attempts} failed attempts`);
      }
      
      // Compute module hash
      const actualHash = await this.computeHash(modulePath);
      
      // Validate hash
      if (actualHash !== expectedHash) {
        this.handleValidationFailure(modulePath, expectedHash, actualHash);
        return false;
      }
      
      // Success
      this.handleValidationSuccess(modulePath, actualHash);
      return true;
      
    } catch (error) {
      this.handleValidationError(modulePath, error);
      return false;
    }
  }
  
  private async computeHash(filePath: string): Promise<string> {
    // In real implementation, this would read file and compute SHA-256
    // For now, simulate hash computation
    const hash = crypto.createHash('sha256');
    hash.update(filePath); // In reality, would use file contents
    return hash.digest('hex');
  }
  
  private isPathSafe(modulePath: string): boolean {
    // Normalize path
    const normalized = path.normalize(modulePath);
    
    // Check for path traversal attempts
    if (normalized.includes('..')) {
      return false;
    }
    
    // Check if path is within allowed directories
    const isAllowed = this.config.pathRestrictions.some(restriction => 
      normalized.startsWith(restriction)
    );
    
    if (!isAllowed) {
      return false;
    }
    
    // Additional checks
    const forbidden = ['/etc', '/usr', '/bin', '/sbin', '/var', '/tmp'];
    for (const forbiddenPath of forbidden) {
      if (normalized.startsWith(forbiddenPath)) {
        return false;
      }
    }
    
    return true;
  }
  
  private handleValidationFailure(modulePath: string, expectedHash: string, actualHash: string): void {
    // Update quarantine
    const existing = this.quarantine.get(modulePath);
    const now = Date.now();
    
    if (existing) {
      existing.attempts++;
      existing.lastAttempt = now;
    } else {
      this.quarantine.set(modulePath, {
        moduleId: path.basename(modulePath, '.md'),
        modulePath,
        reason: 'Hash mismatch',
        attempts: 1,
        firstAttempt: now,
        lastAttempt: now,
        hash: actualHash
      });
    }
    
    // Log audit event
    this.logAudit({
      timestamp: now,
      eventType: 'VALIDATION_FAILED',
      moduleId: path.basename(modulePath, '.md'),
      details: {
        expectedHash,
        actualHash,
        attempts: existing ? existing.attempts + 1 : 1
      }
    });
    
    // Check if quarantine threshold reached
    const entry = this.quarantine.get(modulePath)!;
    if (entry.attempts >= this.config.quarantineMaxRetries) {
      this.logAudit({
        timestamp: now,
        eventType: 'QUARANTINE_ADD',
        moduleId: entry.moduleId,
        details: {
          reason: entry.reason,
          attempts: entry.attempts
        }
      });
    }
  }
  
  private handleValidationSuccess(modulePath: string, hash: string): void {
    // Remove from quarantine if present
    if (this.quarantine.has(modulePath)) {
      this.quarantine.delete(modulePath);
      this.logAudit({
        timestamp: Date.now(),
        eventType: 'QUARANTINE_RELEASE',
        moduleId: path.basename(modulePath, '.md'),
        details: { reason: 'Validation succeeded' }
      });
    }
    
    // Store hash for merkle tree
    this.moduleHashes.set(modulePath, hash);
    
    // Log success
    this.logAudit({
      timestamp: Date.now(),
      eventType: 'VALIDATION_SUCCESS',
      moduleId: path.basename(modulePath, '.md'),
      details: { hash }
    });
  }
  
  private handleValidationError(modulePath: string, error: any): void {
    console.error(`Validation error for ${modulePath}:`, error);
    
    // Add to quarantine with error
    const now = Date.now();
    this.quarantine.set(modulePath, {
      moduleId: path.basename(modulePath, '.md'),
      modulePath,
      reason: error.message,
      attempts: 1,
      firstAttempt: now,
      lastAttempt: now
    });
  }
  
  buildMerkleTree(moduleIds: string[]): MerkleNode | null {
    if (moduleIds.length === 0) return null;
    
    // Create leaf nodes
    const leaves: MerkleNode[] = moduleIds.map(moduleId => {
      const hash = this.moduleHashes.get(moduleId) || this.computeHashSync(moduleId);
      return {
        hash,
        moduleId
      };
    });
    
    // Build tree
    this.merkleTree = this.buildMerkleTreeRecursive(leaves);
    return this.merkleTree;
  }
  
  private buildMerkleTreeRecursive(nodes: MerkleNode[]): MerkleNode {
    if (nodes.length === 1) return nodes[0];
    
    const nextLevel: MerkleNode[] = [];
    
    for (let i = 0; i < nodes.length; i += 2) {
      const left = nodes[i];
      const right = nodes[i + 1] || left; // Duplicate last node if odd number
      
      const combinedHash = this.combineHashes(left.hash, right.hash);
      nextLevel.push({
        hash: combinedHash,
        left,
        right
      });
    }
    
    return this.buildMerkleTreeRecursive(nextLevel);
  }
  
  private combineHashes(hash1: string, hash2: string): string {
    const hash = crypto.createHash('sha256');
    hash.update(hash1 + hash2);
    return hash.digest('hex');
  }
  
  private computeHashSync(data: string): string {
    const hash = crypto.createHash('sha256');
    hash.update(data);
    return hash.digest('hex');
  }
  
  verifyMerkleProof(moduleId: string, proof: string[]): boolean {
    const moduleHash = this.moduleHashes.get(moduleId);
    if (!moduleHash || !this.merkleTree) return false;
    
    let currentHash = moduleHash;
    
    for (const proofHash of proof) {
      currentHash = this.combineHashes(currentHash, proofHash);
    }
    
    const verified = currentHash === this.merkleTree.hash;
    
    this.logAudit({
      timestamp: Date.now(),
      eventType: 'MERKLE_VERIFICATION',
      moduleId,
      details: { verified, proofLength: proof.length }
    });
    
    return verified;
  }
  
  getQuarantinedModules(): QuarantineEntry[] {
    return Array.from(this.quarantine.values());
  }
  
  releaseFromQuarantine(modulePath: string): boolean {
    if (this.quarantine.has(modulePath)) {
      const entry = this.quarantine.get(modulePath)!;
      this.quarantine.delete(modulePath);
      
      this.logAudit({
        timestamp: Date.now(),
        eventType: 'QUARANTINE_RELEASE',
        moduleId: entry.moduleId,
        details: { manual: true }
      });
      
      return true;
    }
    return false;
  }
  
  private logAudit(event: AuditEvent): void {
    if (!this.config.auditEnabled) return;
    
    this.auditLog.push(event);
    
    // In production, would also write to persistent audit log
    console.log(`[Security Audit] ${event.eventType}:`, event.details);
  }
  
  getAuditLog(filters?: { eventType?: string; moduleId?: string; since?: number }): AuditEvent[] {
    let events = [...this.auditLog];
    
    if (filters) {
      if (filters.eventType) {
        events = events.filter(e => e.eventType === filters.eventType);
      }
      if (filters.moduleId) {
        events = events.filter(e => e.moduleId === filters.moduleId);
      }
      if (filters.since) {
        events = events.filter(e => e.timestamp >= filters.since);
      }
    }
    
    return events;
  }
}

// Usage example
const validator = new SecurityValidator();

// Validate a module
const isValid = await validator.validateModule(
  './.claude/thinking-modules/SAGE.md',
  'a1b2c3d4e5f6789012345678901234567890123456789012345678901234'
);

// Build merkle tree for bulk validation
const merkleRoot = validator.buildMerkleTree(['SAGE', 'SEIQF', 'SIA']);
console.log('Merkle root:', merkleRoot?.hash);

// Check quarantined modules
const quarantined = validator.getQuarantinedModules();
console.log('Quarantined modules:', quarantined);
```

## Module Health Checks

Implementation of comprehensive module health monitoring:

```typescript
interface HealthCheckResult {
  moduleId: string;
  status: 'healthy' | 'warning' | 'unhealthy' | 'critical';
  checks: {
    syntax: boolean;
    headers: boolean;
    dependencies: boolean;
    performance: boolean;
    security: boolean;
  };
  metrics: {
    loadTime: number;
    size: number;
    complexity: number;
  };
  issues: string[];
  timestamp: number;
}

interface ModuleHeader {
  id?: string;
  version?: string;
  dependencies?: string[];
  tokenCount?: number;
  protocols?: string[];
}

interface HealthCheckConfig {
  syntaxValidation: boolean;
  headerValidation: boolean;
  dependencyValidation: boolean;
  performanceThresholds: {
    loadTime: number;      // ms
    maxSize: number;       // bytes
    maxComplexity: number; // cyclomatic complexity
  };
  requiredHeaders: string[];
}

class ModuleHealthChecker {
  private config: HealthCheckConfig = {
    syntaxValidation: true,
    headerValidation: true,
    dependencyValidation: true,
    performanceThresholds: {
      loadTime: 50,        // 50ms target
      maxSize: 100000,     // 100KB max
      maxComplexity: 20    // Complexity threshold
    },
    requiredHeaders: ['id', 'version', 'dependencies', 'tokenCount']
  };
  
  private healthHistory: Map<string, HealthCheckResult[]> = new Map();
  private healthStatus: Map<string, HealthCheckResult> = new Map();
  
  async checkModuleHealth(
    moduleId: string, 
    modulePath: string, 
    content?: string
  ): Promise<HealthCheckResult> {
    const startTime = Date.now();
    const issues: string[] = [];
    
    // Initialize result
    const result: HealthCheckResult = {
      moduleId,
      status: 'healthy',
      checks: {
        syntax: false,
        headers: false,
        dependencies: false,
        performance: false,
        security: false
      },
      metrics: {
        loadTime: 0,
        size: 0,
        complexity: 0
      },
      issues,
      timestamp: startTime
    };
    
    try {
      // Get or load module content
      const moduleContent = content || await this.loadModuleContent(modulePath);
      result.metrics.size = new TextEncoder().encode(moduleContent).length;
      
      // 1. Syntax validation
      if (this.config.syntaxValidation) {
        result.checks.syntax = await this.validateSyntax(moduleContent, issues);
      }
      
      // 2. Header validation
      if (this.config.headerValidation) {
        result.checks.headers = await this.validateHeaders(moduleContent, issues);
      }
      
      // 3. Dependency validation
      if (this.config.dependencyValidation) {
        const headers = this.extractHeaders(moduleContent);
        result.checks.dependencies = await this.validateDependencies(
          headers.dependencies || [], 
          issues
        );
      }
      
      // 4. Performance checks
      result.metrics.loadTime = Date.now() - startTime;
      result.checks.performance = this.checkPerformance(result.metrics, issues);
      
      // 5. Security checks (basic)
      result.checks.security = this.performSecurityChecks(moduleContent, issues);
      
      // Calculate complexity
      result.metrics.complexity = this.calculateComplexity(moduleContent);
      
      // Determine overall status
      result.status = this.determineHealthStatus(result.checks, issues);
      
    } catch (error) {
      issues.push(`Health check error: ${error.message}`);
      result.status = 'critical';
    }
    
    // Store result
    this.healthStatus.set(moduleId, result);
    this.addToHistory(moduleId, result);
    
    return result;
  }
  
  private async validateSyntax(content: string, issues: string[]): Promise<boolean> {
    let valid = true;
    
    // Check for balanced braces/brackets
    const braceBalance = this.checkBraceBalance(content);
    if (braceBalance !== 0) {
      issues.push(`Unbalanced braces: ${braceBalance > 0 ? 'missing closing' : 'missing opening'}`);
      valid = false;
    }
    
    // Check for proper @import syntax
    const importRegex = /@import\s+"([^"]+)"/g;
    let match;
    while ((match = importRegex.exec(content)) !== null) {
      const importPath = match[1];
      if (!importPath.endsWith('.md')) {
        issues.push(`Invalid import path: ${importPath} (must end with .md)`);
        valid = false;
      }
      if (importPath.includes('..')) {
        issues.push(`Security risk in import: ${importPath} (path traversal)`);
        valid = false;
      }
    }
    
    // Check for required markdown structure
    if (!content.includes('#')) {
      issues.push('Missing markdown headers');
      valid = false;
    }
    
    return valid;
  }
  
  private async validateHeaders(content: string, issues: string[]): Promise<boolean> {
    const headers = this.extractHeaders(content);
    let valid = true;
    
    // Check required headers
    for (const required of this.config.requiredHeaders) {
      if (!(required in headers)) {
        issues.push(`Missing required header: ${required}`);
        valid = false;
      }
    }
    
    // Validate header values
    if (headers.version && !this.isValidSemver(headers.version)) {
      issues.push(`Invalid version format: ${headers.version}`);
      valid = false;
    }
    
    if (headers.tokenCount && (isNaN(headers.tokenCount) || headers.tokenCount < 0)) {
      issues.push(`Invalid token count: ${headers.tokenCount}`);
      valid = false;
    }
    
    return valid;
  }
  
  private async validateDependencies(
    dependencies: string[], 
    issues: string[]
  ): Promise<boolean> {
    let valid = true;
    
    // Check for circular dependencies (simplified check)
    const seen = new Set<string>();
    for (const dep of dependencies) {
      if (seen.has(dep)) {
        issues.push(`Duplicate dependency: ${dep}`);
        valid = false;
      }
      seen.add(dep);
    }
    
    // Validate dependency format
    for (const dep of dependencies) {
      if (!this.isValidModuleId(dep)) {
        issues.push(`Invalid dependency ID: ${dep}`);
        valid = false;
      }
    }
    
    return valid;
  }
  
  private checkPerformance(
    metrics: HealthCheckResult['metrics'], 
    issues: string[]
  ): boolean {
    let valid = true;
    
    if (metrics.loadTime > this.config.performanceThresholds.loadTime) {
      issues.push(`Load time exceeded: ${metrics.loadTime}ms > ${this.config.performanceThresholds.loadTime}ms`);
      valid = false;
    }
    
    if (metrics.size > this.config.performanceThresholds.maxSize) {
      issues.push(`Module size exceeded: ${metrics.size} bytes > ${this.config.performanceThresholds.maxSize} bytes`);
      valid = false;
    }
    
    if (metrics.complexity > this.config.performanceThresholds.maxComplexity) {
      issues.push(`Complexity too high: ${metrics.complexity} > ${this.config.performanceThresholds.maxComplexity}`);
      valid = false;
    }
    
    return valid;
  }
  
  private performSecurityChecks(content: string, issues: string[]): boolean {
    let valid = true;
    
    // Check for dangerous patterns
    const dangerousPatterns = [
      /eval\s*\(/,
      /Function\s*\(/,
      /require\s*\(['"]\.\./,
      /<script/i,
      /javascript:/i
    ];
    
    for (const pattern of dangerousPatterns) {
      if (pattern.test(content)) {
        issues.push(`Security risk: dangerous pattern detected (${pattern})`);
        valid = false;
      }
    }
    
    return valid;
  }
  
  private extractHeaders(content: string): ModuleHeader {
    const headers: ModuleHeader = {};
    
    // Simple header extraction (in real implementation, would parse YAML frontmatter)
    const headerRegex = /^---\n([\s\S]*?)\n---/;
    const match = content.match(headerRegex);
    
    if (match) {
      // Parse YAML-like headers
      const headerContent = match[1];
      const lines = headerContent.split('\n');
      
      for (const line of lines) {
        const [key, value] = line.split(':').map(s => s.trim());
        if (key && value) {
          headers[key] = value;
        }
      }
    }
    
    return headers;
  }
  
  private calculateComplexity(content: string): number {
    // Simplified complexity calculation
    let complexity = 1;
    
    // Count decision points
    const decisionPatterns = [
      /if\s*\(/g,
      /else\s*{/g,
      /for\s*\(/g,
      /while\s*\(/g,
      /case\s+/g,
      /\?\s*:/g  // Ternary operators
    ];
    
    for (const pattern of decisionPatterns) {
      const matches = content.match(pattern);
      if (matches) {
        complexity += matches.length;
      }
    }
    
    return complexity;
  }
  
  private checkBraceBalance(content: string): number {
    let balance = 0;
    const braces = { '{': 1, '}': -1, '[': 1, ']': -1, '(': 1, ')': -1 };
    
    for (const char of content) {
      if (char in braces) {
        balance += braces[char];
      }
    }
    
    return balance;
  }
  
  private isValidSemver(version: string): boolean {
    return /^\d+\.\d+\.\d+(-[\w.]+)?(\+[\w.]+)?$/.test(version);
  }
  
  private isValidModuleId(id: string): boolean {
    return /^[a-zA-Z0-9-_/]+$/.test(id);
  }
  
  private determineHealthStatus(
    checks: HealthCheckResult['checks'], 
    issues: string[]
  ): HealthCheckResult['status'] {
    const failedChecks = Object.values(checks).filter(v => !v).length;
    
    if (failedChecks === 0 && issues.length === 0) {
      return 'healthy';
    } else if (failedChecks <= 1 && issues.length <= 2) {
      return 'warning';
    } else if (failedChecks <= 3) {
      return 'unhealthy';
    } else {
      return 'critical';
    }
  }
  
  private async loadModuleContent(path: string): Promise<string> {
    // In real implementation, would read file
    return `# Module Content\n\n@import "./test.md"`;
  }
  
  private addToHistory(moduleId: string, result: HealthCheckResult): void {
    if (!this.healthHistory.has(moduleId)) {
      this.healthHistory.set(moduleId, []);
    }
    
    const history = this.healthHistory.get(moduleId)!;
    history.push(result);
    
    // Keep only last 10 results
    if (history.length > 10) {
      history.shift();
    }
  }
  
  getHealthStatus(moduleId?: string): Map<string, HealthCheckResult> | HealthCheckResult | null {
    if (moduleId) {
      return this.healthStatus.get(moduleId) || null;
    }
    return new Map(this.healthStatus);
  }
  
  getHealthHistory(moduleId: string): HealthCheckResult[] {
    return this.healthHistory.get(moduleId) || [];
  }
  
  getUnhealthyModules(): { moduleId: string; result: HealthCheckResult }[] {
    const unhealthy: { moduleId: string; result: HealthCheckResult }[] = [];
    
    for (const [moduleId, result] of this.healthStatus) {
      if (result.status !== 'healthy') {
        unhealthy.push({ moduleId, result });
      }
    }
    
    return unhealthy;
  }
}

// Usage example
const healthChecker = new ModuleHealthChecker();

// Check module health
const health = await healthChecker.checkModuleHealth(
  'SAGE',
  './.claude/thinking-modules/SAGE.md'
);

console.log('Module health:', health);

// Get unhealthy modules
const unhealthy = healthChecker.getUnhealthyModules();
console.log('Unhealthy modules:', unhealthy);
```

## Hot-Reloading Support

Implementation of module hot-reloading with state preservation:

```typescript
interface FileWatcherConfig {
  enabled: boolean;
  debounceMs: number;
  maxRetries: number;
  preserveState: boolean;
  watchPaths: string[];
}

interface ReloadEvent {
  moduleId: string;
  path: string;
  changeType: 'create' | 'update' | 'delete';
  timestamp: number;
  previousHash?: string;
  newHash?: string;
}

interface ReloadResult {
  success: boolean;
  moduleId: string;
  duration: number;
  statePreserved: boolean;
  error?: Error;
}

class HotReloadManager {
  private config: FileWatcherConfig = {
    enabled: true,
    debounceMs: 300,      // Wait 300ms after last change
    maxRetries: 3,
    preserveState: true,
    watchPaths: [
      './.claude/thinking-modules/',
      './.claude/cognitive-tools/',
      './.claude/config/'
    ]
  };
  
  private watchers: Map<string, any> = new Map();
  private reloadQueue: Map<string, ReloadEvent> = new Map();
  private reloadInProgress: Set<string> = new Set();
  private stateSnapshot: Map<string, any> = new Map();
  
  constructor(
    private moduleLoader: ModuleLoader,
    private registry: ModuleRegistry,
    private healthChecker: ModuleHealthChecker,
    private tokenTracker: TokenTracker
  ) {}
  
  async startWatching(): Promise<void> {
    if (!this.config.enabled) return;
    
    console.log('[HotReload] Starting file watchers...');
    
    for (const watchPath of this.config.watchPaths) {
      await this.watchDirectory(watchPath);
    }
    
    // Start processing reload queue
    this.processReloadQueue();
  }
  
  async stopWatching(): Promise<void> {
    console.log('[HotReload] Stopping file watchers...');
    
    for (const [path, watcher] of this.watchers) {
      await this.closeWatcher(watcher);
    }
    
    this.watchers.clear();
    this.reloadQueue.clear();
  }
  
  private async watchDirectory(dirPath: string): Promise<void> {
    // In real implementation, would use fs.watch or chokidar
    const watcher = {
      path: dirPath,
      on: (event: string, callback: Function) => {
        // Simulated file watcher
        console.log(`[HotReload] Watching ${dirPath} for ${event}`);
      }
    };
    
    // Watch for file changes
    watcher.on('change', async (filePath: string) => {
      await this.handleFileChange(filePath, 'update');
    });
    
    watcher.on('add', async (filePath: string) => {
      await this.handleFileChange(filePath, 'create');
    });
    
    watcher.on('unlink', async (filePath: string) => {
      await this.handleFileChange(filePath, 'delete');
    });
    
    this.watchers.set(dirPath, watcher);
  }
  
  private async handleFileChange(
    filePath: string, 
    changeType: 'create' | 'update' | 'delete'
  ): Promise<void> {
    // Determine module ID from file path
    const moduleId = this.getModuleIdFromPath(filePath);
    if (!moduleId) return;
    
    // Check if module is currently loaded
    if (!this.moduleLoader.isModuleLoaded(moduleId) && changeType !== 'create') {
      return; // Don't reload modules that aren't currently loaded
    }
    
    // Create reload event
    const event: ReloadEvent = {
      moduleId,
      path: filePath,
      changeType,
      timestamp: Date.now()
    };
    
    // Add to reload queue (debounced)
    this.reloadQueue.set(moduleId, event);
  }
  
  private async processReloadQueue(): Promise<void> {
    setInterval(async () => {
      const now = Date.now();
      
      for (const [moduleId, event] of this.reloadQueue) {
        // Check debounce period
        if (now - event.timestamp < this.config.debounceMs) {
          continue;
        }
        
        // Skip if already reloading
        if (this.reloadInProgress.has(moduleId)) {
          continue;
        }
        
        // Process reload
        this.reloadQueue.delete(moduleId);
        await this.reloadModule(event);
      }
    }, 100); // Check every 100ms
  }
  
  private async reloadModule(event: ReloadEvent): Promise<ReloadResult> {
    const startTime = Date.now();
    const result: ReloadResult = {
      success: false,
      moduleId: event.moduleId,
      duration: 0,
      statePreserved: false
    };
    
    this.reloadInProgress.add(event.moduleId);
    
    try {
      console.log(`[HotReload] Reloading module ${event.moduleId}...`);
      
      // 1. Preserve state if enabled
      if (this.config.preserveState) {
        await this.preserveModuleState(event.moduleId);
        result.statePreserved = true;
      }
      
      // 2. Validate new module content
      const newContent = await this.readModuleContent(event.path);
      const health = await this.healthChecker.checkModuleHealth(
        event.moduleId,
        event.path,
        newContent
      );
      
      if (health.status === 'critical') {
        throw new Error(`Module health check failed: ${health.issues.join(', ')}`);
      }
      
      // 3. Update registry if metadata changed
      if (event.path.includes('metadata.yaml')) {
        await this.registry.initialize(); // Reload entire registry
      }
      
      // 4. Clear module from loader cache
      this.moduleLoader.clearModuleCache(event.moduleId);
      
      // 5. Reload module
      const classification = this.getModuleClassification(event.moduleId);
      const reloadedModules = await this.moduleLoader.loadModules(classification);
      
      // 6. Restore state if preserved
      if (this.config.preserveState && this.stateSnapshot.has(event.moduleId)) {
        await this.restoreModuleState(event.moduleId);
      }
      
      // 7. Verify reload success
      if (reloadedModules.some(m => m.id === event.moduleId)) {
        result.success = true;
        console.log(`[HotReload] Successfully reloaded ${event.moduleId}`);
      } else {
        throw new Error('Module reload verification failed');
      }
      
    } catch (error) {
      console.error(`[HotReload] Failed to reload ${event.moduleId}:`, error);
      result.error = error as Error;
      
      // Attempt recovery
      await this.attemptRecovery(event.moduleId, error as Error);
      
    } finally {
      this.reloadInProgress.delete(event.moduleId);
      result.duration = Date.now() - startTime;
    }
    
    // Emit reload event for monitoring
    this.emitReloadEvent(result);
    
    return result;
  }
  
  private async preserveModuleState(moduleId: string): Promise<void> {
    // Get current module state from protocol state
    const protocolState = await this.readProtocolState();
    const moduleState = protocolState.moduleStates?.[moduleId];
    
    if (moduleState) {
      this.stateSnapshot.set(moduleId, {
        ...moduleState,
        preservedAt: Date.now()
      });
    }
  }
  
  private async restoreModuleState(moduleId: string): Promise<void> {
    const preservedState = this.stateSnapshot.get(moduleId);
    if (!preservedState) return;
    
    // Restore to protocol state
    const protocolState = await this.readProtocolState();
    protocolState.moduleStates = protocolState.moduleStates || {};
    protocolState.moduleStates[moduleId] = preservedState;
    
    await this.writeProtocolState(protocolState);
    
    // Clean up snapshot
    this.stateSnapshot.delete(moduleId);
  }
  
  private async attemptRecovery(moduleId: string, error: Error): Promise<void> {
    console.log(`[HotReload] Attempting recovery for ${moduleId}...`);
    
    let attempts = 0;
    while (attempts < this.config.maxRetries) {
      attempts++;
      
      try {
        // Wait before retry
        await new Promise(resolve => setTimeout(resolve, 1000 * attempts));
        
        // Try to load previous version from cache
        const cachedModule = this.moduleLoader.getCachedModule(moduleId);
        if (cachedModule) {
          console.log(`[HotReload] Recovered ${moduleId} from cache`);
          return;
        }
        
        // Try to reload from disk
        const classification = this.getModuleClassification(moduleId);
        await this.moduleLoader.loadModules(classification);
        
        console.log(`[HotReload] Recovery successful for ${moduleId}`);
        return;
        
      } catch (recoveryError) {
        console.error(`[HotReload] Recovery attempt ${attempts} failed:`, recoveryError);
      }
    }
    
    // Recovery failed - module remains in failed state
    console.error(`[HotReload] Recovery failed for ${moduleId} after ${attempts} attempts`);
  }
  
  private getModuleIdFromPath(filePath: string): string | null {
    // Extract module ID from file path
    const match = filePath.match(/\/(thinking-modules|cognitive-tools)\/([^/]+)\.md$/);
    if (match) {
      const [, type, name] = match;
      return type === 'cognitive-tools' ? `cognitive-tools/${name}` : name;
    }
    
    // Check if it's metadata.yaml
    if (filePath.includes('metadata.yaml')) {
      return 'registry';
    }
    
    return null;
  }
  
  private getModuleClassification(moduleId: string): any {
    // Create minimal classification for single module reload
    return {
      category: 'reload',
      confidence: 1.0,
      requiredModules: [moduleId],
      suggestedAgents: [],
      mcpTools: [],
      estimatedTokens: 0
    };
  }
  
  private async readModuleContent(path: string): Promise<string> {
    // In real implementation, would read from file system
    return `# Module content from ${path}`;
  }
  
  private async readProtocolState(): Promise<any> {
    // In real implementation, would read from protocol-state.json
    return {};
  }
  
  private async writeProtocolState(state: any): Promise<void> {
    // In real implementation, would write to protocol-state.json
  }
  
  private closeWatcher(watcher: any): void {
    // Clean up watcher resources
    if (watcher && typeof watcher.close === 'function') {
      watcher.close();
    }
  }
  
  private emitReloadEvent(result: ReloadResult): void {
    // Emit event for monitoring/logging
    console.log('[HotReload] Event:', {
      moduleId: result.moduleId,
      success: result.success,
      duration: `${result.duration}ms`,
      statePreserved: result.statePreserved
    });
  }
  
  // Public API for manual reload
  async reloadModuleManual(moduleId: string): Promise<ReloadResult> {
    const metadata = await this.registry.getMetadata(moduleId);
    if (!metadata) {
      throw new Error(`Module ${moduleId} not found in registry`);
    }
    
    const event: ReloadEvent = {
      moduleId,
      path: this.moduleLoader.getModulePath(moduleId),
      changeType: 'update',
      timestamp: Date.now()
    };
    
    return this.reloadModule(event);
  }
  
  getReloadStatus(): {
    enabled: boolean;
    watching: string[];
    inProgress: string[];
    queuedReloads: string[];
  } {
    return {
      enabled: this.config.enabled,
      watching: Array.from(this.watchers.keys()),
      inProgress: Array.from(this.reloadInProgress),
      queuedReloads: Array.from(this.reloadQueue.keys())
    };
  }
}

// Extension to ModuleLoader for hot-reload support
interface ModuleLoader {
  isModuleLoaded(moduleId: string): boolean;
  clearModuleCache(moduleId: string): void;
  getCachedModule(moduleId: string): any;
  getModulePath(moduleId: string): string;
}

// Usage example
const hotReloadManager = new HotReloadManager(
  moduleLoader,
  registry,
  healthChecker,
  tokenTracker
);

// Start watching for changes
await hotReloadManager.startWatching();

// Get reload status
const status = hotReloadManager.getReloadStatus();
console.log('Hot-reload status:', status);

// Manual reload
const result = await hotReloadManager.reloadModuleManual('SAGE');
console.log('Manual reload result:', result);

// Stop watching
await hotReloadManager.stopWatching();
```