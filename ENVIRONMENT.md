# Environment Setup

This project uses `uv` for Python package management with automatic virtual environment activation via `direnv`.

## Quick Start

### Prerequisites
1. **Python 3.12+** (managed via pyenv)
2. **uv** - Fast Python package manager
3. **direnv** - Auto-activate virtual environment

### Installation

#### Install uv (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Install direnv (if not already installed)
```bash
brew install direnv
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc  # For zsh
# or
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc  # For bash
```

### Project Setup
```bash
# Clone the repository
git clone <repository-url>
cd universal-claude-thinking-v2

# Allow direnv to load the .envrc file
direnv allow .

# Sync all dependencies (creates .venv automatically)
uv sync --all-extras

# Verify installation
uv run python -c "import torch; print('✓ Environment ready')"
```

## Auto-Activation

The virtual environment automatically activates when you `cd` into the project directory thanks to direnv.

- **Entering project**: `.venv` activates automatically
- **Leaving project**: `.venv` deactivates automatically
- **Manual activation**: `source .venv/bin/activate` (if needed)

## Managing Dependencies

### Adding Dependencies
```bash
# Add a production dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Add with specific version
uv add "package-name>=1.0.0"
```

### Updating Dependencies
```bash
# Update all packages
uv sync --upgrade

# Update specific package
uv add package-name --upgrade
```

### Exporting Requirements
```bash
# Generate requirements.txt for pip compatibility
uv pip compile pyproject.toml -o requirements.txt

# Generate dev requirements
uv pip compile pyproject.toml --extra dev -o requirements-dev.txt
```

## Running Code

### With uv (Recommended)
```bash
# Run Python scripts
uv run python your_script.py

# Run tests
uv run pytest

# Run linting
uv run ruff check .

# Run formatting
uv run black .
```

### With Activated venv
```bash
# After auto-activation via direnv
python your_script.py
pytest
ruff check .
black .
```

## Project Structure

```
universal-claude-thinking-v2/
├── .venv/                # Virtual environment (auto-managed by uv)
├── .envrc                # direnv configuration for auto-activation
├── pyproject.toml        # Project configuration and dependencies
├── uv.lock              # Lock file for reproducible installs
├── requirements.txt      # Pip-compatible requirements (generated)
├── requirements-dev.txt  # Dev requirements (generated)
└── src/                 # Source code
```

## Key Files

- **pyproject.toml**: Source of truth for dependencies and project metadata
- **uv.lock**: Ensures reproducible installations across all environments
- **requirements.txt**: For environments without uv (auto-generated)
- **.envrc**: Configures automatic virtual environment activation

## Environment Variables

The following environment variables are automatically set when entering the project:

- `VIRTUAL_ENV`: Path to the virtual environment
- `PROJECT_NAME`: "universal-claude-thinking-v2"
- `PYTHONPATH`: Includes the `src/` directory

## Troubleshooting

### Virtual environment not activating
```bash
# Ensure direnv is allowed
direnv allow .

# Reload shell configuration
source ~/.zshrc  # or ~/.bashrc
```

### Package installation issues
```bash
# Clear uv cache
uv cache clean

# Recreate virtual environment
rm -rf .venv
uv venv --python 3.12
uv sync --all-extras
```

### Python version mismatch
```bash
# Ensure Python 3.12 is available
pyenv install 3.12.11
pyenv local 3.12.11

# Recreate venv with correct Python
uv venv --python 3.12
```

## Development Workflow

1. **Enter project directory** - Virtual environment activates automatically
2. **Make changes** to code
3. **Add dependencies** with `uv add package-name`
4. **Run tests** with `uv run pytest`
5. **Format code** with `uv run black .`
6. **Lint code** with `uv run ruff check .`
7. **Commit changes** including updated `uv.lock`

## CI/CD Integration

For CI/CD pipelines without uv:
```bash
# Use the generated requirements.txt
pip install -r requirements.txt

# Or for development environment
pip install -r requirements-dev.txt
```

## Performance Notes

- **uv** is 10-100x faster than pip for dependency resolution
- Uses a global cache to avoid re-downloading packages
- Parallel downloads and installations
- Optimized for modern Python projects

## Clean Up

To clean up global Python packages after migration:
```bash
# List user-installed packages
pip list --user

# Uninstall specific packages (review carefully)
pip uninstall --user package-name

# Or bulk cleanup (REVIEW LIST FIRST)
# pip freeze --user | xargs pip uninstall -y
```

## Support

For issues related to:
- **uv**: https://github.com/astral-sh/uv
- **direnv**: https://direnv.net/
- **Project**: See main README.md