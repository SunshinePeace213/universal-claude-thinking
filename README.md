# Universal Claude Thinking

A modular thinking system for Claude Code that provides structured cognitive tools and reusable thinking modules to enhance AI reasoning capabilities.

## Overview

Universal Claude Thinking is a framework that enables Claude Code to utilize specialized thinking modules and cognitive tools through a standardized interface. The system provides:

- **Thinking Modules**: Reusable protocol modules for different thinking patterns
- **Cognitive Tools**: Specialized operations for complex reasoning tasks
- **Context Fields**: Structured data fields for maintaining context across operations

## Installation

### Prerequisites

- Node.js (v16 or higher)
- npm (v7 or higher)
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/universal-claude-thinking.git
cd universal-claude-thinking
```

2. Install dependencies:
```bash
npm install
```

3. Configure environment (optional):
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

### Directory Structure

```
.claude/
├── thinking-modules/    # Protocol modules for thinking patterns
├── cognitive-tools/     # Tools for cognitive operations
└── context-fields/      # Context data structures
```

### Creating a Thinking Module

1. Create a new module file in `.claude/thinking-modules/`
2. Define the module metadata in YAML format
3. Implement the module protocol

Example:
```yaml
name: analytical-thinking
version: 1.0.0
description: Module for analytical reasoning patterns
```

### Using Cognitive Tools

Cognitive tools are available in `.claude/cognitive-tools/` and can be imported into your Claude Code sessions.

### Running Tests

```bash
npm test
```

## Development

### Project Structure

See `docs/architecture/unified-project-structure.md` for detailed project organization.

### Coding Standards

This project follows the coding standards defined in `docs/architecture/coding-standards.md`:

- Use kebab-case for module and config files
- Follow security best practices
- Maintain comprehensive error handling

## Contributing

[Contribution guidelines to be added]

## License

[License information to be added]

## Support

For issues and feature requests, please use the GitHub issue tracker.