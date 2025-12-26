import {
    isValidEthereumAddress,
    shortenAddress,
    formatNumber,
    formatCompactNumber,
    formatPercentage,
    timeAgo,
    isValidJSON,
    safeJSONParse,
    clamp,
} from '../utils/helpers';

describe('Helper Functions', () => {
    describe('isValidEthereumAddress', () => {
        it('validates correct Ethereum addresses', () => {
            expect(isValidEthereumAddress('0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb')).toBe(false); // Missing last char
            expect(isValidEthereumAddress('0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb9')).toBe(true);
            expect(isValidEthereumAddress('0xABCDEF1234567890ABCDEF1234567890ABCDEF12')).toBe(true);
        });

        it('rejects invalid addresses', () => {
            expect(isValidEthereumAddress('')).toBe(false);
            expect(isValidEthereumAddress(null)).toBe(false);
            expect(isValidEthereumAddress('0xinvalid')).toBe(false);
            expect(isValidEthereumAddress('742d35Cc6634C0532925a3b844Bc9e7595f0bEb9')).toBe(false); // Missing 0x
        });
    });

    describe('shortenAddress', () => {
        it('shortens valid addresses correctly', () => {
            const address = '0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb9';
            expect(shortenAddress(address)).toBe('0x742d...bEb9');
            expect(shortenAddress(address, 6)).toBe('0x742d35...f0bEb9');
        });

        it('returns invalid addresses unchanged', () => {
            expect(shortenAddress('invalid')).toBe('invalid');
            expect(shortenAddress('')).toBe('');
        });
    });

    describe('formatNumber', () => {
        it('formats numbers with commas', () => {
            expect(formatNumber(1000)).toBe('1,000.00');
            expect(formatNumber(1234567.89)).toBe('1,234,567.89');
            expect(formatNumber(42, 0)).toBe('42');
        });

        it('handles edge cases', () => {
            expect(formatNumber(0)).toBe('0.00');
            expect(formatNumber(null)).toBe('0');
            expect(formatNumber(undefined)).toBe('0');
            expect(formatNumber(NaN)).toBe('0');
        });
    });

    describe('formatCompactNumber', () => {
        it('formats large numbers with suffixes', () => {
            expect(formatCompactNumber(1000)).toBe('1.00K');
            expect(formatCompactNumber(1500000)).toBe('1.50M');
            expect(formatCompactNumber(2500000000)).toBe('2.50B');
            expect(formatCompactNumber(1000000000000)).toBe('1.00T');
        });

        it('handles small numbers', () => {
            expect(formatCompactNumber(500)).toBe('500.00');
            expect(formatCompactNumber(0)).toBe('0.00');
        });

        it('handles edge cases', () => {
            expect(formatCompactNumber(null)).toBe('0');
            expect(formatCompactNumber(undefined)).toBe('0');
        });
    });

    describe('formatPercentage', () => {
        it('formats percentages correctly', () => {
            expect(formatPercentage(0.15)).toBe('15.00%');
            expect(formatPercentage(0.5)).toBe('50.00%');
            expect(formatPercentage(1.25)).toBe('125.00%');
            expect(formatPercentage(0.123, 1)).toBe('12.3%');
        });

        it('handles edge cases', () => {
            expect(formatPercentage(0)).toBe('0.00%');
            expect(formatPercentage(null)).toBe('0%');
        });
    });

    describe('timeAgo', () => {
        beforeEach(() => {
            jest.useFakeTimers();
            jest.setSystemTime(new Date('2024-01-01T12:00:00'));
        });

        afterEach(() => {
            jest.useRealTimers();
        });

        it('calculates seconds ago', () => {
            const timestamp = new Date('2024-01-01T11:59:30');
            expect(timeAgo(timestamp)).toBe('30 seconds ago');
        });

        it('calculates minutes ago', () => {
            const timestamp = new Date('2024-01-01T11:55:00');
            expect(timeAgo(timestamp)).toBe('5 minutes ago');
        });

        it('calculates hours ago', () => {
            const timestamp = new Date('2024-01-01T09:00:00');
            expect(timeAgo(timestamp)).toBe('3 hours ago');
        });

        it('calculates days ago', () => {
            const timestamp = new Date('2023-12-30T12:00:00');
            expect(timeAgo(timestamp)).toBe('2 days ago');
        });

        it('uses singular for 1 unit', () => {
            const timestamp = new Date('2024-01-01T11:00:00');
            expect(timeAgo(timestamp)).toBe('1 hour ago');
        });
    });

    describe('isValidJSON', () => {
        it('validates valid JSON', () => {
            expect(isValidJSON('{}')).toBe(true);
            expect(isValidJSON('{"key": "value"}')).toBe(true);
            expect(isValidJSON('[]')).toBe(true);
            expect(isValidJSON('[1, 2, 3]')).toBe(true);
        });

        it('rejects invalid JSON', () => {
            expect(isValidJSON('')).toBe(false);
            expect(isValidJSON('{')).toBe(false);
            expect(isValidJSON('not json')).toBe(false);
            expect(isValidJSON('{key: value}')).toBe(false);
        });
    });

    describe('safeJSONParse', () => {
        it('parses valid JSON', () => {
            expect(safeJSONParse('{"key": "value"}')).toEqual({ key: 'value' });
            expect(safeJSONParse('[1, 2, 3]')).toEqual([1, 2, 3]);
        });

        it('returns fallback for invalid JSON', () => {
            expect(safeJSONParse('invalid', {})).toEqual({});
            expect(safeJSONParse('', null)).toBeNull();
            expect(safeJSONParse('bad json', { default: true })).toEqual({ default: true });
        });
    });

    describe('clamp', () => {
        it('clamps values within range', () => {
            expect(clamp(5, 0, 10)).toBe(5);
            expect(clamp(-5, 0, 10)).toBe(0);
            expect(clamp(15, 0, 10)).toBe(10);
        });

        it('works with negative ranges', () => {
            expect(clamp(0, -10, 10)).toBe(0);
            expect(clamp(-15, -10, 10)).toBe(-10);
            expect(clamp(15, -10, 10)).toBe(10);
        });
    });
});
