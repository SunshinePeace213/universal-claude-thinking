# Request Classifier

## Module Info
- **Version**: 2.0.0
- **Token Budget**: 500
- **Purpose**: Classify incoming requests with confidence scoring and module recommendations

## Data Models

### RequestClassification Interface
```typescript
interface RequestClassification {
  category: 'simple' | 'complex' | 'search' | 'code' | 'meta';
  confidence: number;
  requiredModules: string[];
  suggestedAgents: string[];
  mcpTools: string[];
  estimatedTokens: number;
}
```

## Classification Logic

### Category Mappings
- **simple** (A): Direct questions, basic explanations
- **complex** (B): Multi-step reasoning, analysis tasks  
- **search** (C): Finding information, code exploration
- **code** (D): Writing, refactoring, debugging code
- **meta** (E): Self-reflection, process questions

### Constants and Configuration

```javascript
// Confidence weight constants for scoring
const CONFIDENCE_WEIGHTS = {
  INDICATOR_MATCH: 0.3,
  CODE_INDICATOR_MATCH: 0.4,
  FILE_INDICATOR_MATCH: 0.3,
  SELF_REFERENCE_MATCH: 0.3,
  FALLBACK_THRESHOLD: 0.8,
  FALLBACK_CONFIDENCE: 0.8
};

// Cache configuration
const CACHE_CONFIG = {
  MAX_SIZE: 100,
  KEY_PREFIX: 'clf_'
};

// Telemetry configuration
const TELEMETRY_CONFIG = {
  MAX_ENTRIES: 1000,
  PERSIST_THRESHOLD: 100
};
```

### LRU Cache Implementation
```javascript
class LRUCache {
  constructor(maxSize = CACHE_CONFIG.MAX_SIZE) {
    this.maxSize = maxSize;
    this.cache = new Map();
  }
  
  get(key) {
    if (!this.cache.has(key)) return null;
    const value = this.cache.get(key);
    this.cache.delete(key);
    this.cache.set(key, value);
    return value;
  }
  
  set(key, value) {
    if (this.cache.has(key)) {
      this.cache.delete(key);
    } else if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    this.cache.set(key, value);
  }
}

const classificationCache = new LRUCache();

// Secure cache key generation
function generateSecureCacheKey(request) {
  const normalized = request.trim().toLowerCase();
  // Simple hash function for environments without crypto
  let hash = 0;
  for (let i = 0; i < normalized.length; i++) {
    const char = normalized.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32-bit integer
  }
  return CACHE_CONFIG.KEY_PREFIX + Math.abs(hash).toString(36);
}
```

### Enhanced Classification Function

```javascript
function classifyRequest(request) {
  const startTime = Date.now();
  const cacheKey = generateSecureCacheKey(request);
  
  // Check cache first
  const cached = classificationCache.get(cacheKey);
  if (cached) {
    logTelemetry('cache_hit', { category: cached.category });
    return cached;
  }
  
  const text = request.toLowerCase();
  const wordCount = text.split(/\s+/).length;
  
  const patterns = {
    simple: {
      patterns: [/what is/i, /how do/i, /explain/i, /define/i],
      maxWords: 50,
      baseConfidence: 0.9,
      agentTypes: ['responder'],
      mcpTools: [],
      tokenEstimate: 1000
    },
    complex: {
      patterns: [/analyze/i, /compare/i, /evaluate/i, /design/i],
      indicators: ['multiple', 'steps', 'consider'],
      baseConfidence: 0.85,
      agentTypes: ['analyzer', 'strategist'],
      mcpTools: ['sequentialthinking', 'systemsthinking'],
      tokenEstimate: 3000
    },
    search: {
      patterns: [/find/i, /search/i, /locate/i, /where is/i, /list all/i],
      fileIndicators: ['file', 'directory', 'folder', 'path', 'implementation', 'files'],
      baseConfidence: 0.95,
      agentTypes: ['searcher'],
      mcpTools: ['tavily-search', 'context7'],
      tokenEstimate: 2000
    },
    code: {
      patterns: [/write/i, /implement/i, /fix/i, /debug/i, /refactor/i, /create a function/i, /build/i],
      codeIndicators: ['function', 'class', 'algorithm', 'module', 'component'],
      fileExtensions: ['.js', '.ts', '.py', '.java', '.cpp', '.go'],
      extensionWeights: {
        '.js': 0.3, '.ts': 0.3, '.py': 0.25,
        '.java': 0.2, '.cpp': 0.15, '.go': 0.15
      },
      baseConfidence: 0.9,
      agentTypes: ['developer'],
      mcpTools: ['debuggingapproach', 'sequentialthinking'],
      tokenEstimate: 2500
    },
    meta: {
      patterns: [/how do you/i, /what is your process/i, /why did you/i, /explain your/i, /your reasoning/i],
      selfReference: ['your', 'you', 'claude'],
      baseConfidence: 0.85,
      agentTypes: ['meta-reasoner'],
      mcpTools: ['metacognitivemonitoring'],
      tokenEstimate: 1500
    }
  };
  
  let bestMatch = { 
    category: 'simple', 
    confidence: 0.0,
    requiredModules: [],
    suggestedAgents: [],
    mcpTools: [],
    estimatedTokens: 1000
  };
  
  // Process categories in priority order to handle conflicts
  const categoryOrder = ['search', 'code', 'meta', 'complex', 'simple'];
  const orderedPatterns = {};
  categoryOrder.forEach(cat => {
    if (patterns[cat]) orderedPatterns[cat] = patterns[cat];
  });
  
  for (const [category, config] of Object.entries(orderedPatterns)) {
    let score = 0;
    let patternMatches = 0;
    let totalPossiblePatterns = (config.patterns?.length || 0);
    
    // Pattern matching (primary signal)
    config.patterns?.forEach(pattern => {
      if (pattern.test(text)) {
        score += config.baseConfidence;
        patternMatches++;
      }
    });
    
    // Early exit if no pattern matches for this category
    if (patternMatches === 0 && totalPossiblePatterns > 0) {
      continue;
    }
    
    // Indicator matching (secondary signals)
    config.indicators?.forEach(indicator => {
      if (text.includes(indicator)) {
        score += CONFIDENCE_WEIGHTS.INDICATOR_MATCH;
      }
    });
    
    // Code indicators (for code category)
    config.codeIndicators?.forEach(indicator => {
      if (text.includes(indicator)) {
        score += CONFIDENCE_WEIGHTS.CODE_INDICATOR_MATCH;
      }
    });
    
    // File extension weighted scoring (for code category)
    if (category === 'code' && config.fileExtensions) {
      config.fileExtensions.forEach(ext => {
        if (text.includes(ext)) {
          const weight = config.extensionWeights[ext] || 0.1;
          score += weight;
        }
      });
    }
    
    // File indicators (for search category)
    config.fileIndicators?.forEach(indicator => {
      if (text.includes(indicator)) {
        score += CONFIDENCE_WEIGHTS.FILE_INDICATOR_MATCH;
      }
    });
    
    // Self-reference indicators (for meta category)
    config.selfReference?.forEach(ref => {
      if (text.includes(ref)) {
        score += CONFIDENCE_WEIGHTS.SELF_REFERENCE_MATCH;
      }
    });
    
    // Calculate final confidence
    // Use the total score but cap it at 1.0
    const finalScore = Math.min(score, 1.0);
    
    if (finalScore > bestMatch.confidence) {
      bestMatch = {
        category,
        confidence: finalScore,
        requiredModules: getRequiredModules(category),
        suggestedAgents: config.agentTypes,
        mcpTools: config.mcpTools,
        estimatedTokens: config.tokenEstimate
      };
    }
  }
  
  // Fallback to simple if confidence too low
  if (bestMatch.confidence < CONFIDENCE_WEIGHTS.FALLBACK_THRESHOLD) {
    logTelemetry('low_confidence_fallback', { 
      originalCategory: bestMatch.category,
      confidence: bestMatch.confidence 
    });
    bestMatch = {
      category: 'simple',
      confidence: CONFIDENCE_WEIGHTS.FALLBACK_CONFIDENCE,
      requiredModules: getRequiredModules('simple'),
      suggestedAgents: ['responder'],
      mcpTools: [],
      estimatedTokens: 1000
    };
  }
  
  // Cache the result
  classificationCache.set(cacheKey, bestMatch);
  
  // Log telemetry
  const elapsedTime = Date.now() - startTime;
  logTelemetry('classification_complete', {
    category: bestMatch.category,
    confidence: bestMatch.confidence,
    elapsedMs: elapsedTime,
    wordCount: wordCount
  });
  
  return bestMatch;
}

// Module mapping with priority scores
const moduleMapping = {
  simple: {
    modules: ['response-formats'],
    priority: 1.0
  },
  complex: {
    modules: ['SAGE', 'SEIQF', 'cognitive-tools/analysis'],
    priority: 0.9
  },
  search: {
    modules: ['SIA', 'cognitive-tools/search'],
    priority: 0.95
  },
  code: {
    modules: ['SEIQF', 'cognitive-tools/code-analysis'],
    priority: 0.95
  },
  meta: {
    modules: ['SAGE', 'cognitive-tools/meta-reasoning'],
    priority: 0.85
  }
};

function getRequiredModules(category) {
  const mapping = moduleMapping[category] || moduleMapping.simple;
  return mapping.modules;
}

function getModulePriority(category) {
  const mapping = moduleMapping[category] || moduleMapping.simple;
  return mapping.priority;
}

// Telemetry logging with size limits
class TelemetryLogger {
  constructor(maxEntries = TELEMETRY_CONFIG.MAX_ENTRIES) {
    this.entries = [];
    this.maxEntries = maxEntries;
  }
  
  log(event, data) {
    const entry = {
      timestamp: new Date().toISOString(),
      event,
      data
    };
    
    // Implement circular buffer
    if (this.entries.length >= this.maxEntries) {
      this.entries.shift(); // Remove oldest entry
    }
    this.entries.push(entry);
    
    // Persist to protocol state if available
    if (typeof updateProtocolState === 'function' && 
        this.entries.length % TELEMETRY_CONFIG.PERSIST_THRESHOLD === 0) {
      updateProtocolState({
        telemetry: this.entries.slice(-TELEMETRY_CONFIG.PERSIST_THRESHOLD)
      });
    }
  }
  
  getEntries() {
    return [...this.entries]; // Return copy to prevent external modification
  }
  
  clear() {
    this.entries = [];
  }
}

const telemetryLogger = new TelemetryLogger();

function logTelemetry(event, data) {
  telemetryLogger.log(event, data);
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    classifyRequest,
    getRequiredModules,
    getModulePriority,
    logTelemetry,
    LRUCache,
    TelemetryLogger,
    generateSecureCacheKey,
    CONFIDENCE_WEIGHTS,
    CACHE_CONFIG,
    TELEMETRY_CONFIG
  };
}
```