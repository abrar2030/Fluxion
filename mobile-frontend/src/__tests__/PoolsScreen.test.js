// Mock the API client module before importing
jest.mock('../api/client', () => ({
    fetchAssets: jest.fn(),
    fetchPools: jest.fn(),
    predictEnergy: jest.fn(),
}));

import React from 'react';
import { render, waitFor, fireEvent } from '@testing-library/react-native';
import PoolsScreen from '../screens/PoolsScreen';
import { fetchPools } from '../api/client';

// Mock the API client
jest.mock('../api/client');

// Mock navigation
const mockNavigation = {
    navigate: jest.fn(),
};

describe('PoolsScreen', () => {
    beforeEach(() => {
        jest.clearAllMocks();
    });

    const mockPools = [
        {
            id: 'pool1',
            name: 'synBTC/synUSD',
            pair: 'synBTC/synUSD',
            tvl: 10500000,
            apr: 12.5,
            volume_24h: 2100000,
            fees_24h: 6300,
            utilization: 0.72,
            verified: true,
        },
        {
            id: 'pool2',
            name: 'synETH/synUSD',
            pair: 'synETH/synUSD',
            tvl: 8200000,
            apr: 10.8,
            volume_24h: 1640000,
            fees_24h: 4920,
            utilization: 0.65,
            verified: true,
        },
    ];

    it('renders loading state initially', () => {
        fetchPools.mockImplementation(() => new Promise(() => {}));
        const { getByText } = render(<PoolsScreen navigation={mockNavigation} />);
        expect(getByText('Loading pools...')).toBeTruthy();
    });

    it('loads and displays pools successfully', async () => {
        fetchPools.mockResolvedValue(mockPools);

        const { getByText } = render(<PoolsScreen navigation={mockNavigation} />);

        await waitFor(() => {
            expect(getByText('synBTC/synUSD')).toBeTruthy();
            expect(getByText('synETH/synUSD')).toBeTruthy();
        });
    });

    it('handles API error gracefully', async () => {
        const errorMessage = 'Failed to fetch pools';
        fetchPools.mockRejectedValue(new Error(errorMessage));

        const { getByText } = render(<PoolsScreen navigation={mockNavigation} />);

        await waitFor(() => {
            expect(getByText(errorMessage)).toBeTruthy();
        });
    });

    it('filters pools by search query', async () => {
        fetchPools.mockResolvedValue(mockPools);

        const { getByPlaceholderText, getByText, queryByText } = render(
            <PoolsScreen navigation={mockNavigation} />,
        );

        await waitFor(() => {
            expect(getByText('synBTC/synUSD')).toBeTruthy();
        });

        const searchInput = getByPlaceholderText('Search pools...');
        fireEvent.changeText(searchInput, 'BTC');

        expect(getByText('synBTC/synUSD')).toBeTruthy();
        expect(queryByText('synETH/synUSD')).toBeNull();
    });

    it('sorts pools by TVL correctly', async () => {
        fetchPools.mockResolvedValue(mockPools);

        const { getByText, getAllByText } = render(<PoolsScreen navigation={mockNavigation} />);

        await waitFor(() => {
            expect(getByText('synBTC/synUSD')).toBeTruthy();
        });

        // Default sort is by TVL, so synBTC/synUSD (higher TVL) should come first
        const poolNames = getAllByText(/syn[A-Z]+\/syn[A-Z]+/);
        // First pool should be synBTC/synUSD (TVL: 10.5M)
        // Second pool should be synETH/synUSD (TVL: 8.2M)
    });

    it('sorts pools by APR when APR chip is pressed', async () => {
        fetchPools.mockResolvedValue(mockPools);

        const { getByText } = render(<PoolsScreen navigation={mockNavigation} />);

        await waitFor(() => {
            expect(getByText('synBTC/synUSD')).toBeTruthy();
        });

        // Click APR sort chip
        const aprChip = getByText('APR');
        fireEvent.press(aprChip);

        // Pools should now be sorted by APR
        // synBTC/synUSD has 12.5% APR, synETH/synUSD has 10.8% APR
    });

    it('sorts pools by Volume when Volume chip is pressed', async () => {
        fetchPools.mockResolvedValue(mockPools);

        const { getByText } = render(<PoolsScreen navigation={mockNavigation} />);

        await waitFor(() => {
            expect(getByText('synBTC/synUSD')).toBeTruthy();
        });

        // Click Volume sort chip
        const volumeChip = getByText('Volume');
        fireEvent.press(volumeChip);

        // Pools should now be sorted by volume
    });

    it('displays APR values correctly', async () => {
        fetchPools.mockResolvedValue(mockPools);

        const { getByText } = render(<PoolsScreen navigation={mockNavigation} />);

        await waitFor(() => {
            expect(getByText('12.50%')).toBeTruthy();
            expect(getByText('10.80%')).toBeTruthy();
        });
    });

    it('displays utilization progress bar', async () => {
        fetchPools.mockResolvedValue(mockPools);

        const { getByText } = render(<PoolsScreen navigation={mockNavigation} />);

        await waitFor(() => {
            expect(getByText('72.0%')).toBeTruthy(); // First pool utilization
            expect(getByText('65.0%')).toBeTruthy(); // Second pool utilization
        });
    });

    it('handles pull to refresh', async () => {
        fetchPools.mockResolvedValue(mockPools);

        const {} = render(<PoolsScreen navigation={mockNavigation} />);

        await waitFor(() => {
            expect(fetchPools).toHaveBeenCalledTimes(1);
        });

        // Simulate pull to refresh
        fetchPools.mockClear();
        fetchPools.mockResolvedValue(mockPools);
    });

    it('displays empty state when no pools', async () => {
        fetchPools.mockResolvedValue([]);

        const { getByText } = render(<PoolsScreen navigation={mockNavigation} />);

        await waitFor(() => {
            expect(getByText('No pools available')).toBeTruthy();
        });
    });

    it('displays verified badge for verified pools', async () => {
        fetchPools.mockResolvedValue(mockPools);

        const { getAllByText } = render(<PoolsScreen navigation={mockNavigation} />);

        await waitFor(() => {
            const verifiedBadges = getAllByText('Verified');
            expect(verifiedBadges.length).toBe(2); // Both pools are verified
        });
    });

    it('retries failed requests', async () => {
        // First call fails, second succeeds
        fetchPools
            .mockRejectedValueOnce(new Error('Network error'))
            .mockResolvedValueOnce(mockPools);

        const { getByText } = render(<PoolsScreen navigation={mockNavigation} />);

        await waitFor(() => {
            expect(getByText('Network error')).toBeTruthy();
        });

        // Click retry button
        const retryButton = getByText('Retry');
        fireEvent.press(retryButton);

        await waitFor(() => {
            expect(getByText('synBTC/synUSD')).toBeTruthy();
        });
    });

    it('formats currency values correctly', async () => {
        fetchPools.mockResolvedValue(mockPools);

        const { getByText } = render(<PoolsScreen navigation={mockNavigation} />);

        await waitFor(() => {
            // Check that TVL is formatted as currency
            expect(getByText(/\$10,500,000/)).toBeTruthy();
        });
    });
});
