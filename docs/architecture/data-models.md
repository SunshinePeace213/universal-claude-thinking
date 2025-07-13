# Data Models

## ModuleMetadata

**Purpose:** Defines metadata for each thinking module

**Key Attributes:**

- id: string - Unique module identifier
- version: string - Semantic version
- tokenCount: number - Estimated token usage
- dependencies: string[] - Required module IDs
- triggers: string[] - Auto-activation keywords
- protocols: string[] - Supported protocols (SAGE, SEIQF, SIA)

**TypeScript Interface:**

```typescript
interface ModuleMetadata {
  id: string;
  version: string;
  tokenCount: number;
  dependencies: string[];
  triggers: string[];
  protocols: ('SAGE' | 'SEIQF' | 'SIA')[];
  securityHash: string;
  lastModified: Date;
}
```

**Relationships:**

- Referenced by ModuleRegistry
- Dependencies point to other ModuleMetadata

## RequestClassification

**Purpose:** Classification result for incoming requests

**Key Attributes:**

- category: string - Request type (A/B/C/D/E)
- confidence: number - Classification confidence (0-1)
- requiredModules: string[] - Module IDs to load
- suggestedAgents: string[] - Virtual agent IDs

**TypeScript Interface:**

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

**Relationships:**

- Triggers ModuleLoader
- Informs VirtualAgent selection

## ProtocolState

**Purpose:** Shared state across protocols and agents

**Key Attributes:**

- sageStatus: object - SAGE bias detection state
- seiqfQuality: object - Information quality metrics
- siaIntent: object - Semantic intent analysis
- activeAlerts: Alert[] - Current warnings/errors

**TypeScript Interface:**

```typescript
interface ProtocolState {
  sageStatus: {
    biasLevel: 'none' | 'low' | 'medium' | 'high' | 'critical';
    detectedBiases: string[];
    mitigationApplied: boolean;
  };
  seiqfQuality: {
    overallScore: number;
    sourcesEvaluated: number;
    credibilityFlags: string[];
  };
  siaIntent: {
    primaryIntent: string;
    confidence: number;
    expansions: string[];
  };
  activeAlerts: Alert[];
  thinkingLog: ThinkingEntry[];
}
```

**Relationships:**

- Shared across all VirtualAgents
- Updated by thinking protocols
