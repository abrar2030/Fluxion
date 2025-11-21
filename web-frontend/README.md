# Web Frontend Directory

## Overview

The `web-frontend` directory contains the React-based web application for the Fluxion synthetic asset liquidity engine. This web interface provides users with a comprehensive platform to interact with Fluxion's features, including synthetic asset trading, portfolio management, analytics, and governance.

## Directory Structure

- `__mocks__/`: Mock files for testing
- `src/`: Source code for the web application
- `babel.config.js`: Babel configuration for JavaScript transpilation
- `index.html`: Main HTML entry point
- `package.json`: NPM package configuration and dependencies
- `vite.config.js`: Vite build tool configuration

## Key Components

### Source Code

The `src` directory contains the application's source code organized into:

- `components/`: Reusable UI components
- `pages/`: Page components for different sections of the application
- `hooks/`: Custom React hooks
- `services/`: API and blockchain interaction services
- `store/`: State management (Redux/Context API)
- `utils/`: Utility functions and helpers
- `assets/`: Static assets including images, icons, and fonts
- `styles/`: Global styles and theme configuration
- `constants/`: Application constants and configuration
- `contexts/`: React context providers

### Configuration

Key configuration files include:

- `vite.config.js`: Vite bundler configuration for development and production builds
- `babel.config.js`: JavaScript transpilation settings
- `package.json`: NPM dependencies and scripts

### Testing

The `__mocks__` directory contains mock implementations for:

- External services
- API responses
- Web3 providers
- Browser APIs

## Development Setup

1. Install dependencies:

   ```bash
   npm install
   ```

2. Start the development server:

   ```bash
   npm run dev
   ```

3. Run tests:

   ```bash
   npm test
   ```

4. Build for production:
   ```bash
   npm run build
   ```

## Features

The web application provides:

- Interactive trading interface for synthetic assets
- Advanced charting and technical analysis tools
- Portfolio management dashboard
- Order book visualization
- Liquidity pool management
- Governance participation interface
- User account management
- Cross-chain asset bridge interface
- Analytics and reporting tools
- Dark/light theme support
- Responsive design for desktop and mobile browsers

## Technology Stack

- React 18 with TypeScript
- Vite for fast development and optimized builds
- Recharts for data visualization
- ethers.js for blockchain interaction
- TailwindCSS for styling
- React Router for navigation
- React Query for data fetching
- Jest and Testing Library for testing

## Best Practices

- Follow component-based architecture
- Implement responsive design for all screen sizes
- Ensure accessibility compliance
- Optimize for performance with code splitting
- Implement comprehensive error handling
- Follow security best practices for Web3 applications
