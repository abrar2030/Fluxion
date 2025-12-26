/**
 * Validates if a string is a valid Ethereum address
 * @param {string} address - Address to validate
 * @returns {boolean} True if valid Ethereum address
 */
export const isValidEthereumAddress = (address) => {
    if (!address) return false;
    return /^0x[a-fA-F0-9]{40}$/.test(address);
};

/**
 * Shortens an Ethereum address for display
 * @param {string} address - Full Ethereum address
 * @param {number} chars - Number of characters to show on each side (default: 4)
 * @returns {string} Shortened address (e.g., "0x1234...5678")
 */
export const shortenAddress = (address, chars = 4) => {
    if (!isValidEthereumAddress(address)) return address;
    return `${address.substring(0, chars + 2)}...${address.substring(address.length - chars)}`;
};

/**
 * Formats a number with commas for thousands
 * @param {number} value - Number to format
 * @param {number} decimals - Number of decimal places (default: 2)
 * @returns {string} Formatted number
 */
export const formatNumber = (value, decimals = 2) => {
    if (value === null || value === undefined || isNaN(value)) return '0';
    return Number(value).toLocaleString('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
    });
};

/**
 * Formats a large number with suffix (K, M, B, T)
 * @param {number} value - Number to format
 * @param {number} decimals - Number of decimal places (default: 2)
 * @returns {string} Formatted number with suffix
 */
export const formatCompactNumber = (value, decimals = 2) => {
    if (value === null || value === undefined || isNaN(value)) return '0';

    const suffixes = ['', 'K', 'M', 'B', 'T'];
    const tier = Math.floor(Math.log10(Math.abs(value)) / 3);

    if (tier === 0) return formatNumber(value, decimals);

    const suffix = suffixes[tier];
    const scale = Math.pow(10, tier * 3);
    const scaled = value / scale;

    return `${scaled.toFixed(decimals)}${suffix}`;
};

/**
 * Formats a percentage value
 * @param {number} value - Percentage value (e.g., 0.15 for 15%)
 * @param {number} decimals - Number of decimal places (default: 2)
 * @returns {string} Formatted percentage
 */
export const formatPercentage = (value, decimals = 2) => {
    if (value === null || value === undefined || isNaN(value)) return '0%';
    return `${(value * 100).toFixed(decimals)}%`;
};

/**
 * Calculates time ago from a timestamp
 * @param {Date|string|number} timestamp - Timestamp to calculate from
 * @returns {string} Time ago string (e.g., "5 minutes ago")
 */
export const timeAgo = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const seconds = Math.floor((now - date) / 1000);

    if (seconds < 60) return `${seconds} seconds ago`;

    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes} minute${minutes !== 1 ? 's' : ''} ago`;

    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours} hour${hours !== 1 ? 's' : ''} ago`;

    const days = Math.floor(hours / 24);
    if (days < 30) return `${days} day${days !== 1 ? 's' : ''} ago`;

    const months = Math.floor(days / 30);
    if (months < 12) return `${months} month${months !== 1 ? 's' : ''} ago`;

    const years = Math.floor(months / 12);
    return `${years} year${years !== 1 ? 's' : ''} ago`;
};

/**
 * Validates if a string is valid JSON
 * @param {string} str - String to validate
 * @returns {boolean} True if valid JSON
 */
export const isValidJSON = (str) => {
    try {
        JSON.parse(str);
        return true;
    } catch (e) {
        return false;
    }
};

/**
 * Debounce function to limit how often a function is called
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @returns {Function} Debounced function
 */
export const debounce = (func, wait = 300) => {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
};

/**
 * Throttle function to limit how often a function can be called
 * @param {Function} func - Function to throttle
 * @param {number} limit - Time limit in milliseconds
 * @returns {Function} Throttled function
 */
export const throttle = (func, limit = 300) => {
    let inThrottle;
    return function executedFunction(...args) {
        if (!inThrottle) {
            func(...args);
            inThrottle = true;
            setTimeout(() => (inThrottle = false), limit);
        }
    };
};

/**
 * Safely parse JSON with fallback value
 * @param {string} str - JSON string to parse
 * @param {*} fallback - Fallback value if parsing fails
 * @returns {*} Parsed JSON or fallback value
 */
export const safeJSONParse = (str, fallback = null) => {
    try {
        return JSON.parse(str);
    } catch (e) {
        return fallback;
    }
};

/**
 * Clamps a number between min and max values
 * @param {number} value - Value to clamp
 * @param {number} min - Minimum value
 * @param {number} max - Maximum value
 * @returns {number} Clamped value
 */
export const clamp = (value, min, max) => {
    return Math.min(Math.max(value, min), max);
};
