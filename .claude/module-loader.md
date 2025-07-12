# Module Loading System

## Dynamic Import Templates

### Core Loading Logic

```javascript
class ModuleLoader {
  constructor() {
    this.loadedModules = new Set();
    this.moduleCache = new Map();
    this.loadTimes = new Map();
  }
  
  async loadModule(modulePath) {
    const startTime = Date.now();
    
    try {
      // Check cache first
      if (this.moduleCache.has(modulePath)) {
        return this.moduleCache.get(modulePath);
      }
      
      // Validate path security
      if (!this.isValidPath(modulePath)) {
        throw new Error(`Invalid module path: ${modulePath}`);
      }
      
      // Load module (simulated @import)
      const module = await this.performImport(modulePath);
      
      // Cache and track
      this.moduleCache.set(modulePath, module);
      this.loadedModules.add(modulePath);
      this.loadTimes.set(modulePath, Date.now() - startTime);
      
      return module;
    } catch (error) {
      return this.handleLoadFailure(modulePath, error);
    }
  }
  
  isValidPath(path) {
    // Must start with ./ and be within .claude directory
    return path.startsWith('./') && 
           path.includes('.claude/') &&
           !path.includes('..') &&
           path.endsWith('.md');
  }
  
  async performImport(path) {
    // Simulate @import functionality
    return `@import "${path}"`;
  }
  
  handleLoadFailure(path, error) {
    console.error(`Failed to load module ${path}:`, error);
    
    // Return fallback content
    return {
      status: 'failed',
      module: path,
      fallback: true,
      error: error.message
    };
  }
}
```

### Conditional Loading Based on Classification

```javascript
async function loadModulesForRequest(classification) {
  const loader = new ModuleLoader();
  const modules = [];
  
  // Get required modules based on category
  const requiredModules = getModulesByCategory(classification.category);
  
  // Load modules in parallel for performance
  const loadPromises = requiredModules.map(moduleName => {
    const modulePath = resolveModulePath(moduleName);
    return loader.loadModule(modulePath);
  });
  
  const results = await Promise.allSettled(loadPromises);
  
  // Process results
  results.forEach((result, index) => {
    if (result.status === 'fulfilled') {
      modules.push({
        name: requiredModules[index],
        content: result.value,
        status: 'loaded'
      });
    } else {
      modules.push({
        name: requiredModules[index],
        status: 'failed',
        error: result.reason
      });
    }
  });
  
  return {
    modules,
    totalLoadTime: loader.getTotalLoadTime(),
    loadedCount: modules.filter(m => m.status === 'loaded').length
  };
}
```

### Module Path Resolution

```javascript
function resolveModulePath(moduleName) {
  const baseDir = './.claude/';
  
  // Handle nested paths
  if (moduleName.includes('/')) {
    return `${baseDir}${moduleName}.md`;
  }
  
  // Default to thinking-modules directory
  return `${baseDir}thinking-modules/${moduleName}.md`;
}

function getModulesByCategory(category) {
  const moduleMap = {
    A: ['response-formats'],
    B: ['SAGE', 'SEIQF', 'cognitive-tools/analysis'],
    C: ['SIA', 'cognitive-tools/search'],
    D: ['SEIQF', 'cognitive-tools/code-analysis'],
    E: ['SAGE', 'cognitive-tools/meta-reasoning']
  };
  
  return moduleMap[category] || moduleMap.A;
}
```

### Fallback Mechanism

```javascript
class FallbackHandler {
  static getFallbackModules(failedModules) {
    const fallbacks = {
      'SAGE': 'basic-reasoning',
      'SEIQF': 'simple-analysis',
      'SIA': 'basic-search',
      'cognitive-tools/analysis': 'basic-analysis',
      'cognitive-tools/search': 'basic-search',
      'cognitive-tools/code-analysis': 'basic-code',
      'cognitive-tools/meta-reasoning': 'basic-meta'
    };
    
    return failedModules.map(module => {
      return fallbacks[module] || 'basic-reasoning';
    });
  }
  
  static async loadFallbacks(failedModules, loader) {
    const fallbackModules = this.getFallbackModules(failedModules);
    const results = [];
    
    for (const fallback of fallbackModules) {
      try {
        const module = await loader.loadModule(
          resolveModulePath(fallback)
        );
        results.push(module);
      } catch (error) {
        // If fallback also fails, use inline basic reasoning
        results.push(this.getInlineFallback());
      }
    }
    
    return results;
  }
  
  static getInlineFallback() {
    return {
      name: 'inline-fallback',
      content: 'Use basic reasoning without specialized modules',
      status: 'fallback'
    };
  }
}
```