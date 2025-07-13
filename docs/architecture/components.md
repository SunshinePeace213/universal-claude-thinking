# Components

## Request Classifier

**Responsibility:** Analyze incoming requests and determine required modules

**Key Interfaces:**

- classifyRequest(input: string): RequestClassification
- updateClassificationRules(rules: ClassificationRule[]): void

**Dependencies:** triggers.yaml, pattern matching engine

**Technology Stack:** TypeScript patterns, YAML configuration

## Module Loader & Security Validator

**Responsibility:** Securely load and validate thinking modules

**Key Interfaces:**

- loadModule(moduleId: string): Promise<Module>
- validateIntegrity(module: Module): boolean
- checkDependencies(moduleId: string): string[]

**Dependencies:** Module Registry, File System, Crypto

**Technology Stack:** Claude Code @import, SHA-256 hashing

## Virtual Agent Orchestrator

**Responsibility:** Manage agent execution with shared state

**Key Interfaces:**

- createAgentPipeline(agents: string[]): Pipeline
- executeWithState(pipeline: Pipeline, input: string): Result
- synchronizeState(updates: StateUpdate[]): void

**Dependencies:** Virtual Agents, Protocol State Manager

**Technology Stack:** Event-driven state management, JSON

## Parallel MCP Executor

**Responsibility:** Execute multiple MCP tools concurrently

**Key Interfaces:**

- analyzeDependencies(operations: Operation[]): DependencyGraph
- executeParallel(operations: Operation[]): Promise<Results[]>
- mergeResults(results: Results[], strategy: MergeStrategy): Result

**Dependencies:** Clear-thought MCP, External Tools

**Technology Stack:** Promise.all(), dependency analysis

## Universal Dynamic Information Gatherer

**Responsibility:** Enable nested tool invocations during thinking

**Key Interfaces:**

- detectInformationGap(context: ThinkingContext): Gap[]
- selectTool(gap: Gap): string
- invokeNested(tool: string, params: any, depth: number): Result

**Dependencies:** MCP Tools, SEIQF quality gates

**Technology Stack:** Recursive invocation with depth tracking
