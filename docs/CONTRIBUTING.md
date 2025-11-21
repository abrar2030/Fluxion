# Contributing to Fluxion

Thank you for your interest in contributing to Fluxion! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## How to Contribute

### 1. Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/abrar2030/Fluxion.git
cd Fluxion

# Install dependencies
cd blockchain && forge install
cd ../backend && pip install -r requirements.txt
cd ../frontend && npm install

# Set up pre-commit hooks
pre-commit install
```

### 2. Branching Strategy

- `main` - Production-ready code
- `develop` - Development branch
- `feature/*` - New features
- `bugfix/*` - Bug fixes
- `hotfix/*` - Critical fixes for production

### 3. Development Workflow

1. Create a new branch

```bash
git checkout -b feature/your-feature-name
```

2. Make your changes
3. Run tests

```bash
# Smart Contracts
forge test

# Backend
pytest

# Frontend
npm test
```

4. Submit a pull request

### 4. Pull Request Process

1. Update documentation
2. Add tests for new features
3. Ensure CI passes
4. Get two approvals from maintainers
5. Squash commits before merging

## Coding Standards

### Solidity

- Follow [Solidity Style Guide](https://docs.soliditylang.org/en/latest/style-guide.html)
- Use latest stable Solidity version
- Document functions with NatSpec
- Include comprehensive test coverage

```solidity
/// @notice Calculate the optimal swap amount
/// @param amount Input amount
/// @param price Current price
/// @return Optimal swap amount
function calculateOptimalSwap(
    uint256 amount,
    uint256 price
) public pure returns (uint256) {
    // Implementation
}
```

### Python

- Follow PEP 8
- Use type hints
- Document with docstrings
- Maximum line length: 88 characters (Black formatter)

```python
def calculate_liquidity_score(
    volume: float,
    depth: float,
    volatility: float
) -> float:
    """
    Calculate liquidity score based on market metrics.

    Args:
        volume: 24h trading volume
        depth: Order book depth
        volatility: Price volatility

    Returns:
        float: Liquidity score between 0 and 1
    """
    # Implementation
```

### TypeScript/React

- Use ESLint and Prettier
- Follow Airbnb Style Guide
- Use functional components
- Implement proper error boundaries

```typescript
interface TradingViewProps {
  pair: string;
  timeframe: string;
  onPriceUpdate: (price: number) => void;
}

const TradingView: React.FC<TradingViewProps> = ({
  pair,
  timeframe,
  onPriceUpdate,
}) => {
  // Implementation
};
```

## Testing Guidelines

### Smart Contract Testing

- Unit tests for each function
- Integration tests for complex flows
- Fuzzing tests for edge cases
- Gas optimization tests

### Backend Testing

- Unit tests with pytest
- Integration tests with TestClient
- Mock external services
- Performance benchmarks

### Frontend Testing

- Unit tests with Jest
- Component tests with React Testing Library
- E2E tests with Cypress
- Visual regression tests

## Documentation

### Required Documentation

1. Technical specifications
2. API documentation
3. Architecture diagrams
4. Deployment instructions
5. Security considerations

### Documentation Style

- Clear and concise
- Include examples
- Keep up-to-date
- Use proper formatting

## Review Process

### Code Review Checklist

- [ ] Follows coding standards
- [ ] Includes tests
- [ ] Updates documentation
- [ ] No security vulnerabilities
- [ ] Optimized for performance
- [ ] Proper error handling

### Security Review

- Smart contract audits
- Dependency scanning
- Static analysis
- Penetration testing

## Release Process

### Version Control

- Semantic versioning
- Changelog updates
- Release notes
- Migration guides

### Deployment

1. Test on testnet
2. Security audit
3. Community review
4. Mainnet deployment
5. Monitoring

## Getting Help

- Join our [Discord](https://discord.gg/fluxion)
- Check [Issues](https://github.com/abrar2030/Fluxion/issues)
- Read [Documentation](https://docs.fluxion.exchange)
- Email: support@fluxion.exchange

## Recognition

Contributors will be:

- Listed in CONTRIBUTORS.md
- Eligible for bounties
- Given priority for grants
- Recognized in release notes
