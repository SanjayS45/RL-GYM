# Contributing to RL-GYM

Thank you for your interest in contributing to RL-GYM! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Adding New Features](#adding-new-features)

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributions from everyone regardless of experience level.

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- Git

### Setting Up Development Environment

1. **Fork the repository**
   ```bash
   # Click the "Fork" button on GitHub
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/RL-GYM.git
   cd RL-GYM
   ```

3. **Set up the upstream remote**
   ```bash
   git remote add upstream https://github.com/SanjayS45/RL-GYM.git
   ```

4. **Install dependencies**
   ```bash
   make install
   ```

5. **Start development servers**
   ```bash
   make dev
   ```

## Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code
   - Add tests
   - Update documentation

3. **Test your changes**
   ```bash
   make test
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**

## Coding Standards

### Python (Backend)

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions focused and under 50 lines when possible

```python
def calculate_reward(
    state: np.ndarray,
    action: int,
    next_state: np.ndarray
) -> float:
    """
    Calculate the reward for a state transition.
    
    Args:
        state: Current state array
        action: Action taken
        next_state: Resulting state array
        
    Returns:
        Calculated reward value
    """
    # Implementation
    pass
```

### TypeScript (Frontend)

- Use TypeScript strict mode
- Prefer functional components with hooks
- Use meaningful variable and function names
- Add JSDoc comments for complex functions

```typescript
/**
 * Calculate the exponential moving average of training rewards.
 * @param data - Array of reward values
 * @param alpha - Smoothing factor (0-1)
 * @returns Smoothed reward values
 */
function exponentialMovingAverage(data: number[], alpha = 0.1): number[] {
  // Implementation
}
```

### React Components

- Use functional components with hooks
- Keep components small and focused
- Extract reusable logic into custom hooks
- Use proper TypeScript types for props

```tsx
interface ButtonProps {
  label: string;
  onClick: () => void;
  variant?: 'primary' | 'secondary';
  disabled?: boolean;
}

export const Button: React.FC<ButtonProps> = ({
  label,
  onClick,
  variant = 'primary',
  disabled = false,
}) => {
  // Implementation
};
```

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements

### Examples

```bash
feat(rl-core): add dueling DQN architecture
fix(api): handle missing session ID gracefully
docs(readme): update installation instructions
test(environments): add GridWorld unit tests
```

## Pull Request Process

1. **Update your branch with latest changes**
   ```bash
   git fetch upstream
   git rebase upstream/master
   ```

2. **Ensure all tests pass**
   ```bash
   make test
   ```

3. **Create a clear PR description**
   - Describe what changes you made
   - Link to any related issues
   - Include screenshots for UI changes

4. **Address review feedback**
   - Make requested changes
   - Push updates to your branch

5. **Merge**
   - A maintainer will merge your PR once approved

## Adding New Features

### Adding a New RL Algorithm

1. Create a new file in `backend/rl_core/algorithms/`
2. Extend the `BasePolicy` class
3. Implement required methods:
   - `act(state, deterministic)` - Select an action
   - `learn(batch)` - Update the policy
   - `save(path)` - Save model weights
   - `load(path)` - Load model weights
4. Export from `__init__.py`
5. Add to the API routes
6. Add example configuration

### Adding a New Environment

1. Create a new file in `backend/environments/`
2. Extend `gymnasium.Env`
3. Implement required methods:
   - `reset()` - Reset the environment
   - `step(action)` - Take a step
   - `render()` - Render the environment
4. Define observation and action spaces
5. Export from `__init__.py`
6. Add to the API routes
7. Update frontend visualization if needed

### Adding a New UI Component

1. Create a new file in `frontend/src/components/`
2. Use TypeScript and proper types
3. Add Tailwind CSS for styling
4. Export from the appropriate index file
5. Add to relevant panels or pages

## Testing

### Backend Tests

```bash
cd backend
python -m pytest tests/ -v
```

### Frontend Tests

```bash
cd frontend
npm test
```

### Integration Tests

```bash
make test-integration
```

## Questions?

Feel free to:
- Open an issue for bugs or feature requests
- Start a discussion for questions
- Reach out to maintainers

Thank you for contributing! 🎉

