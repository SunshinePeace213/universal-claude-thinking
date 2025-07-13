# Error Handling

When errors occur:

1. Log to thinking visibility
2. Apply fallback strategy
3. Continue with degraded functionality
4. Notify user of limitations

Example:
⚠️ SEIQF: Source validation timeout - using cached credibility scores

````

## Orchestrator Error Handling
```typescript
class ModuleOrchestrator {
  async loadWithFallback(moduleId: string): Promise<Module> {
    try {
      return await this.loader.load(moduleId);
    } catch (error) {
      this.log.warn(`Module ${moduleId} failed: ${error.message}`);

      if (this.isEssential(moduleId)) {
        return this.loadEssentialVersion(moduleId);
      }

      return this.createStub(moduleId);
    }
  }
}
````
