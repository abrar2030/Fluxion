// Mock AsyncStorage
jest.mock('@react-native-async-storage/async-storage', () => ({
    setItem: jest.fn(() => Promise.resolve()),
    getItem: jest.fn(() => Promise.resolve(null)),
    removeItem: jest.fn(() => Promise.resolve()),
    multiRemove: jest.fn(() => Promise.resolve()),
    clear: jest.fn(() => Promise.resolve()),
    getAllKeys: jest.fn(() => Promise.resolve([])),
    multiGet: jest.fn(() => Promise.resolve([])),
    multiSet: jest.fn(() => Promise.resolve()),
}));

// Mock expo-constants
jest.mock('expo-constants', () => ({
    default: {
        expoConfig: null,
        manifest: null,
    },
}));

// Mock expo-modules-core EventEmitter
jest.mock('expo-modules-core', () => ({
    EventEmitter: jest.fn(),
    Subscription: jest.fn(),
}));

// Mock react-native-vector-icons
jest.mock('react-native-vector-icons/MaterialCommunityIcons', () => 'MaterialCommunityIcons');

// Global mocks for React Native
global.__DEV__ = true;

// Silence console warnings during tests
global.console = {
    ...console,
    error: jest.fn(),
    warn: jest.fn(),
    log: jest.fn(),
};
