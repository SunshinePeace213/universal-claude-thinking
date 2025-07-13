// Mock the @import functionality (used in other test files)
// const mockImport = jest.fn();

describe('Module Loading Integration', () => {
  let moduleLoader;

  beforeEach(() => {
    jest.clearAllMocks();

    // Mock module loader
    moduleLoader = {
      loadedModules: new Set(),
      loadTimes: new Map(),

      async loadModule(modulePath) {
        const startTime = Date.now();

        // Validate path
        if (
          !modulePath.startsWith('./') ||
          !modulePath.includes('.claude/') ||
          modulePath.includes('..')
        ) {
          throw new Error(`Invalid module path: ${modulePath}`);
        }

        // Simulate loading
        await new Promise(resolve => setTimeout(resolve, 20));

        this.loadedModules.add(modulePath);
        this.loadTimes.set(modulePath, Date.now() - startTime);

        return { path: modulePath, loaded: true };
      },

      getTotalLoadTime() {
        return Array.from(this.loadTimes.values()).reduce(
          (sum, time) => sum + time,
          0
        );
      }
    };
  });

  describe('Module Loading with Classification', () => {
    test('should load correct modules for category A (simple)', async () => {
      const modules = await loadModulesForCategory('A', moduleLoader);

      expect(modules).toHaveLength(1);
      expect(modules[0].path).toBe(
        './.claude/thinking-modules/response-formats.md'
      );
    });

    test('should load correct modules for category B (complex)', async () => {
      const modules = await loadModulesForCategory('B', moduleLoader);

      expect(modules).toHaveLength(3);
      expect(modules.map(m => m.path)).toContain(
        './.claude/thinking-modules/SAGE.md'
      );
      expect(modules.map(m => m.path)).toContain(
        './.claude/thinking-modules/SEIQF.md'
      );
      expect(modules.map(m => m.path)).toContain(
        './.claude/cognitive-tools/analysis.md'
      );
    });

    test('should load correct modules for category C (search)', async () => {
      const modules = await loadModulesForCategory('C', moduleLoader);

      expect(modules).toHaveLength(2);
      expect(modules.map(m => m.path)).toContain(
        './.claude/thinking-modules/SIA.md'
      );
      expect(modules.map(m => m.path)).toContain(
        './.claude/cognitive-tools/search.md'
      );
    });

    test('should load correct modules for category D (code)', async () => {
      const modules = await loadModulesForCategory('D', moduleLoader);

      expect(modules).toHaveLength(2);
      expect(modules.map(m => m.path)).toContain(
        './.claude/thinking-modules/SEIQF.md'
      );
      expect(modules.map(m => m.path)).toContain(
        './.claude/cognitive-tools/code-analysis.md'
      );
    });

    test('should load correct modules for category E (meta)', async () => {
      const modules = await loadModulesForCategory('E', moduleLoader);

      expect(modules).toHaveLength(2);
      expect(modules.map(m => m.path)).toContain(
        './.claude/thinking-modules/SAGE.md'
      );
      expect(modules.map(m => m.path)).toContain(
        './.claude/cognitive-tools/meta-reasoning.md'
      );
    });
  });

  describe('Fallback Mechanisms', () => {
    test('should use fallback when module loading fails', async () => {
      const failingLoader = {
        ...moduleLoader,
        async loadModule(modulePath) {
          if (modulePath.includes('SAGE')) {
            throw new Error('Module not found');
          }
          return moduleLoader.loadModule(modulePath);
        }
      };

      const result = await loadModulesWithFallback(['SAGE'], failingLoader);

      expect(result.failed).toContain('SAGE');
      expect(result.fallbacks).toContain('basic-reasoning');
    });

    test('should handle multiple module failures', async () => {
      const failingLoader = {
        ...moduleLoader,
        async loadModule() {
          throw new Error('All modules failed');
        }
      };

      const result = await loadModulesWithFallback(
        ['SAGE', 'SEIQF', 'SIA'],
        failingLoader
      );

      expect(result.failed).toHaveLength(3);
      expect(result.fallbacks).toHaveLength(3);
      expect(result.fallbacks).toContain('basic-reasoning');
      expect(result.fallbacks).toContain('simple-analysis');
      expect(result.fallbacks).toContain('basic-search');
    });

    test('should use inline fallback when fallback module also fails', async () => {
      const failingLoader = {
        ...moduleLoader,
        async loadModule() {
          throw new Error('Everything fails');
        }
      };

      const result = await loadModulesWithFallback(
        ['SAGE'],
        failingLoader,
        true
      );

      expect(result.inlineFallback).toBe(true);
      expect(result.fallbackContent).toContain(
        'basic reasoning without specialized modules'
      );
    });
  });

  describe('Performance Requirements', () => {
    test('should load modules within 50ms each', async () => {
      const modules = ['SAGE', 'SEIQF', 'SIA'];

      for (const module of modules) {
        const startTime = Date.now();
        await moduleLoader.loadModule(
          `./.claude/thinking-modules/${module}.md`
        );
        const loadTime = Date.now() - startTime;

        expect(loadTime).toBeLessThan(50);
      }
    });

    test('should handle parallel loading efficiently', async () => {
      const modules = ['SAGE', 'SEIQF', 'SIA', 'response-formats'];
      const paths = modules.map(m => `./.claude/thinking-modules/${m}.md`);

      const startTime = Date.now();
      await Promise.all(paths.map(path => moduleLoader.loadModule(path)));
      const totalTime = Date.now() - startTime;

      // Parallel loading should be faster than sequential
      expect(totalTime).toBeLessThan(modules.length * 50);
    });
  });

  describe('Path Validation', () => {
    test('should reject paths outside .claude directory', async () => {
      await expect(moduleLoader.loadModule('../outside.md')).rejects.toThrow(
        'Invalid module path'
      );
      await expect(
        moduleLoader.loadModule('/absolute/path.md')
      ).rejects.toThrow('Invalid module path');
      await expect(
        moduleLoader.loadModule('./other-dir/module.md')
      ).rejects.toThrow('Invalid module path');
    });

    test('should reject paths with directory traversal', async () => {
      await expect(
        moduleLoader.loadModule('./.claude/../../../etc/passwd')
      ).rejects.toThrow('Invalid module path');
      await expect(
        moduleLoader.loadModule('./.claude/modules/../../../secret.md')
      ).rejects.toThrow('Invalid module path');
    });

    test('should accept valid .claude paths', async () => {
      const validPaths = [
        './.claude/thinking-modules/SAGE.md',
        './.claude/cognitive-tools/analysis.md',
        './.claude/response-formats.md'
      ];

      for (const path of validPaths) {
        await expect(moduleLoader.loadModule(path)).resolves.toBeTruthy();
      }
    });
  });
});

// Helper functions
async function loadModulesForCategory(category, loader) {
  const moduleMap = {
    A: ['response-formats'],
    B: ['SAGE', 'SEIQF', 'cognitive-tools/analysis'],
    C: ['SIA', 'cognitive-tools/search'],
    D: ['SEIQF', 'cognitive-tools/code-analysis'],
    E: ['SAGE', 'cognitive-tools/meta-reasoning']
  };

  const moduleNames = moduleMap[category] || [];
  const modules = [];

  for (const moduleName of moduleNames) {
    const path = moduleName.includes('/')
      ? `./.claude/${moduleName}.md`
      : `./.claude/thinking-modules/${moduleName}.md`;

    try {
      const module = await loader.loadModule(path);
      modules.push(module);
    } catch (err) {
      modules.push({ path, error: err.message });
    }
  }

  return modules;
}

async function loadModulesWithFallback(
  moduleNames,
  loader,
  testInlineFallback = false
) {
  const failed = [];
  const fallbacks = [];
  let inlineFallback = false;
  let fallbackContent = '';

  const fallbackMap = {
    SAGE: 'basic-reasoning',
    SEIQF: 'simple-analysis',
    SIA: 'basic-search'
  };

  for (const moduleName of moduleNames) {
    try {
      await loader.loadModule(`./.claude/thinking-modules/${moduleName}.md`);
    } catch {
      failed.push(moduleName);
      const fallbackName = fallbackMap[moduleName] || 'basic-reasoning';

      if (testInlineFallback) {
        try {
          await loader.loadModule(
            `./.claude/thinking-modules/${fallbackName}.md`
          );
          fallbacks.push(fallbackName);
        } catch {
          inlineFallback = true;
          fallbackContent = 'Use basic reasoning without specialized modules';
        }
      } else {
        fallbacks.push(fallbackName);
      }
    }
  }

  return { failed, fallbacks, inlineFallback, fallbackContent };
}
