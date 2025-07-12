# Testing Strategy

## Testing Pyramid
```
         Migration Tests
        /              \
    Integration Tests    
   /                  \
Module Tests    Performance Tests
```

## Test Organization

### Module Tests
```
tests/
├── unit/
│   ├── classifier.test.ts
│   ├── module-loader.test.ts
│   └── state-manager.test.ts
├── modules/
│   ├── SAGE.test.md
│   ├── SEIQF.test.md
│   └── SIA.test.md
└── fixtures/
    └── test-requests.yaml
```

### Integration Tests
```
tests/
└── integration/
    ├── mcp-integration.test.ts
    ├── agent-pipeline.test.ts
    └── nested-invocation.test.ts
```

### Performance Tests
```
tests/
└── performance/
    ├── token-usage.bench.ts
    ├── classification.bench.ts
    └── parallel-execution.bench.ts
```

## Test Examples

### Module Test
```typescript
describe('SAGE Module', () => {
  it('should detect confirmation bias', async () => {
    const result = await loadAndExecute('SAGE', {
      input: 'All swans are white because I\'ve only seen white swans',
      state: createMockState()
    });
    
    expect(result.biasLevel).toBe('medium');
    expect(result.detectedBiases).toContain('confirmation');
  });
});
```

### Integration Test
```typescript
describe('Nested Tool Invocation', () => {
  it('should handle 3-level nested calls', async () => {
    const result = await orchestrator.execute({
      request: 'Analyze latest ML research on transformers',
      maxDepth: 3
    });
    
    expect(result.toolInvocations).toHaveLength(3);
    expect(result.recursionDepth).toBeLessThanOrEqual(3);
  });
});
```

### Performance Test
```typescript
describe('Token Usage Benchmark', () => {
  it('should stay under 5K tokens for complex requests', async () => {
    const metrics = await benchmark.run('complex-analysis.yaml');
    
    expect(metrics.avgTokens).toBeLessThan(5000);
    expect(metrics.p95Tokens).toBeLessThan(7000);
  });
});
```

### Hook Test
```typescript
describe('Module Validation Hook', () => {
  it('should prevent modules exceeding token limit', async () => {
    const largeModule = generateLargeModule(6000); // 6000 tokens
    
    const result = await hookExecutor.execute('module-validation', {
      path: '/test/large-module.md',
      content: largeModule
    });
    
    expect(result.success).toBe(false);
    expect(result.error).toContain('exceeds 5000 token limit');
  });
  
  it('should update merkle tree on valid module', async () => {
    const validModule = generateValidModule();
    const merkleTreeBefore = await getMerkleRoot();
    
    await hookExecutor.execute('module-validation', {
      path: '/test/valid-module.md',
      content: validModule
    });
    
    const merkleTreeAfter = await getMerkleRoot();
    expect(merkleTreeAfter).not.toBe(merkleTreeBefore);
  });
});
```
