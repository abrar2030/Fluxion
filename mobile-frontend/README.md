# Mobile Frontend Directory

## Overview

The `mobile-frontend` directory contains the React Native mobile application for the Fluxion synthetic asset liquidity engine. This mobile app provides users with a portable interface to interact with the Fluxion platform, enabling synthetic asset trading, portfolio management, and analytics on mobile devices.

## Directory Structure

- `.expo/`: Expo configuration files and cache
- `assets/`: Static assets including images, fonts, and other media files
- `src/`: Source code for the mobile application
- `App.js`: Main application entry point
- `app.json`: Expo application configuration
- `index.js`: JavaScript entry point
- `mobile_frontend_design.md`: Design documentation for the mobile frontend
- `package.json`: NPM package configuration and dependencies
- `yarn.lock`: Yarn dependency lock file

## Key Components

### Source Code

The `src` directory contains the application's source code organized into:

- `components/`: Reusable UI components
- `screens/`: Screen components for different app sections
- `navigation/`: Navigation configuration and components
- `hooks/`: Custom React hooks
- `services/`: API and blockchain interaction services
- `store/`: State management (Redux/Context)
- `utils/`: Utility functions and helpers
- `constants/`: Application constants and configuration

### Assets

The `assets` directory contains:

- Brand images and logos
- UI icons and illustrations
- Font files
- Animation files
- Splash screen and app icons

### Configuration

Key configuration files include:

- `app.json`: Expo configuration including app name, version, and build settings
- `package.json`: NPM dependencies and scripts
- `mobile_frontend_design.md`: Design guidelines and UI/UX documentation

## Development Setup

1. Install dependencies:

   ```bash
   yarn install
   ```

2. Start the development server:

   ```bash
   yarn start
   ```

3. Run on iOS simulator:

   ```bash
   yarn ios
   ```

4. Run on Android emulator:
   ```bash
   yarn android
   ```

## Features

The mobile application provides:

- Synthetic asset trading interface
- Portfolio management and tracking
- Real-time price charts and market data
- Order creation and management
- Wallet integration and management
- Push notifications for market events
- Biometric authentication
- Dark/light theme support

## Testing

The mobile application can be tested using:

- Jest for unit and component testing
- Detox for end-to-end testing
- Manual testing on physical devices

Run tests with:

```bash
yarn test
```

## Building for Production

To create production builds:

1. For iOS:

   ```bash
   expo build:ios
   ```

2. For Android:
   ```bash
   expo build:android
   ```
