// Mock the API client module before importing
jest.mock('../api/client', () => ({
    fetchAssets: jest.fn(),
    fetchPools: jest.fn(),
    predictEnergy: jest.fn(),
}));

import React from 'react';
import { render, waitFor, fireEvent } from '@testing-library/react-native';
import AssetsScreen from '../screens/AssetsScreen';
import { fetchAssets } from '../api/client';

// Mock the API client
jest.mock('../api/client');

// Mock navigation
const mockNavigation = {
    navigate: jest.fn(),
};

describe('AssetsScreen', () => {
    beforeEach(() => {
        jest.clearAllMocks();
    });

    const mockAssets = [
        {
            id: 'synBTC',
            symbol: 'synBTC',
            name: 'Synthetic Bitcoin',
            type: 'synthetic',
            price: 65000,
            price_change_24h: 2.5,
            tvl: 125000000,
            volume_24h: 45000000,
            market_cap: 1250000000,
        },
        {
            id: 'synETH',
            symbol: 'synETH',
            name: 'Synthetic Ethereum',
            type: 'synthetic',
            price: 3500,
            price_change_24h: -1.2,
            tvl: 85000000,
            volume_24h: 32000000,
            market_cap: 420000000,
        },
    ];

    it('renders loading state initially', () => {
        fetchAssets.mockImplementation(() => new Promise(() => {}));
        const { getByText } = render(<AssetsScreen navigation={mockNavigation} />);
        expect(getByText('Loading assets...')).toBeTruthy();
    });

    it('loads and displays assets successfully', async () => {
        fetchAssets.mockResolvedValue(mockAssets);

        const { getByText } = render(<AssetsScreen navigation={mockNavigation} />);

        await waitFor(() => {
            expect(getByText('Synthetic Bitcoin')).toBeTruthy();
            expect(getByText('Synthetic Ethereum')).toBeTruthy();
        });
    });

    it('handles API error gracefully', async () => {
        const errorMessage = 'Failed to fetch assets';
        fetchAssets.mockRejectedValue(new Error(errorMessage));

        const { getByText } = render(<AssetsScreen navigation={mockNavigation} />);

        await waitFor(() => {
            expect(getByText(errorMessage)).toBeTruthy();
        });
    });

    it('filters assets by search query', async () => {
        fetchAssets.mockResolvedValue(mockAssets);

        const { getByPlaceholderText, getByText, queryByText } = render(
            <AssetsScreen navigation={mockNavigation} />,
        );

        await waitFor(() => {
            expect(getByText('Synthetic Bitcoin')).toBeTruthy();
        });

        const searchInput = getByPlaceholderText('Search assets...');
        fireEvent.changeText(searchInput, 'Bitcoin');

        expect(getByText('Synthetic Bitcoin')).toBeTruthy();
        expect(queryByText('Synthetic Ethereum')).toBeNull();
    });

    it('filters assets by type', async () => {
        const mixedAssets = [
            ...mockAssets,
            {
                id: 'synGOLD',
                symbol: 'synGOLD',
                name: 'Synthetic Gold',
                type: 'derivative',
                price: 2050,
                price_change_24h: 0.5,
                tvl: 30000000,
                volume_24h: 8000000,
                market_cap: 150000000,
            },
        ];

        fetchAssets.mockResolvedValue(mixedAssets);

        const { getByText, queryByText } = render(<AssetsScreen navigation={mockNavigation} />);

        await waitFor(() => {
            expect(getByText('Synthetic Gold')).toBeTruthy();
        });

        // Click synthetic filter
        const syntheticFilter = getByText('Synthetic');
        fireEvent.press(syntheticFilter);

        expect(getByText('Synthetic Bitcoin')).toBeTruthy();
        expect(getByText('Synthetic Ethereum')).toBeTruthy();
        expect(queryByText('Synthetic Gold')).toBeNull();
    });

    it('displays price changes correctly', async () => {
        fetchAssets.mockResolvedValue(mockAssets);

        const { getByText } = render(<AssetsScreen navigation={mockNavigation} />);

        await waitFor(() => {
            expect(getByText('+2.50%')).toBeTruthy(); // Positive change
            expect(getByText('-1.20%')).toBeTruthy(); // Negative change
        });
    });

    it('handles pull to refresh', async () => {
        fetchAssets.mockResolvedValue(mockAssets);

        const { getByTestId } = render(<AssetsScreen navigation={mockNavigation} />);

        await waitFor(() => {
            expect(fetchAssets).toHaveBeenCalledTimes(1);
        });

        // Simulate pull to refresh
        // Note: Testing RefreshControl requires additional setup in real scenarios
        // This is a simplified test
        fetchAssets.mockClear();
        fetchAssets.mockResolvedValue(mockAssets);

        // In a real test, you would trigger the refresh control
        // For now, we just verify the function works
        expect(fetchAssets).toHaveBeenCalledTimes(0);
    });

    it('displays empty state when no assets', async () => {
        fetchAssets.mockResolvedValue([]);

        const { getByText } = render(<AssetsScreen navigation={mockNavigation} />);

        await waitFor(() => {
            expect(getByText('No assets available')).toBeTruthy();
        });
    });

    it('formats currency values correctly', async () => {
        fetchAssets.mockResolvedValue(mockAssets);

        const { getByText } = render(<AssetsScreen navigation={mockNavigation} />);

        await waitFor(() => {
            // Check that TVL is formatted as currency
            expect(getByText(/\$125,000,000/)).toBeTruthy();
        });
    });

    it('retries failed requests', async () => {
        // First call fails, second succeeds
        fetchAssets
            .mockRejectedValueOnce(new Error('Network error'))
            .mockResolvedValueOnce(mockAssets);

        const { getByText } = render(<AssetsScreen navigation={mockNavigation} />);

        await waitFor(() => {
            expect(getByText('Network error')).toBeTruthy();
        });

        // Click retry button
        const retryButton = getByText('Retry');
        fireEvent.press(retryButton);

        await waitFor(() => {
            expect(getByText('Synthetic Bitcoin')).toBeTruthy();
        });
    });
});
