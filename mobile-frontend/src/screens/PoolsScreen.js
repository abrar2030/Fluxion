import React, { useState, useEffect, useCallback } from 'react';
import { View, StyleSheet, FlatList, RefreshControl, TouchableOpacity } from 'react-native';
import {
    Text,
    Card,
    Title,
    Paragraph,
    useTheme,
    Searchbar,
    Chip,
    ActivityIndicator,
    Snackbar,
    IconButton,
    ProgressBar,
} from 'react-native-paper';
import { fetchPools } from '../api/client';
import { formatCurrency } from '../utils/formatters';

const PoolsScreen = ({ navigation }) => {
    const theme = useTheme();
    const [pools, setPools] = useState([]);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [error, setError] = useState(null);
    const [searchQuery, setSearchQuery] = useState('');
    const [sortBy, setSortBy] = useState('tvl'); // tvl, apr, volume

    // Load pools on mount
    useEffect(() => {
        loadPools();

        // Set up auto-refresh every 30 seconds
        const interval = setInterval(() => {
            loadPools(true); // Silent refresh
        }, 30000);

        return () => clearInterval(interval);
    }, []);

    // Load pools from API
    const loadPools = async (silent = false) => {
        if (!silent) setLoading(true);
        setError(null);

        try {
            const data = await fetchPools({
                limit: 100,
                offset: 0,
            });

            // Handle both array response and paginated response
            const poolsArray = Array.isArray(data) ? data : data.items || data.pools || [];
            setPools(poolsArray);
        } catch (err) {
            console.error('Error loading pools:', err);
            setError(err.message || 'Failed to load pools');

            // Use mock data in development mode as fallback
            if (__DEV__) {
                console.log('Using mock data as fallback');
                setPools(getMockPools());
            }
        } finally {
            setLoading(false);
            setRefreshing(false);
        }
    };

    // Pull to refresh handler
    const onRefresh = useCallback(() => {
        setRefreshing(true);
        loadPools();
    }, []);

    // Search handler
    const onChangeSearch = (query) => {
        setSearchQuery(query);
    };

    // Filter and sort pools
    const getFilteredAndSortedPools = () => {
        // Filter by search
        let filtered = pools.filter((pool) => {
            if (!searchQuery) return true;

            const searchLower = searchQuery.toLowerCase();
            return (
                pool.name?.toLowerCase().includes(searchLower) ||
                pool.pair?.toLowerCase().includes(searchLower) ||
                pool.id?.toLowerCase().includes(searchLower) ||
                pool.token0?.toLowerCase().includes(searchLower) ||
                pool.token1?.toLowerCase().includes(searchLower)
            );
        });

        // Sort
        filtered.sort((a, b) => {
            switch (sortBy) {
                case 'tvl':
                    return (
                        (b.tvl || b.total_value_locked || 0) - (a.tvl || a.total_value_locked || 0)
                    );
                case 'apr':
                    return (b.apr || b.estimated_apr || 0) - (a.apr || a.estimated_apr || 0);
                case 'volume':
                    return (b.volume_24h || b.volume || 0) - (a.volume_24h || a.volume || 0);
                default:
                    return 0;
            }
        });

        return filtered;
    };

    const filteredPools = getFilteredAndSortedPools();

    // Render individual pool card
    const renderPoolCard = ({ item }) => {
        const apr = item.apr || item.estimated_apr || 0;
        const tvl = item.tvl || item.total_value_locked || 0;
        const volume = item.volume_24h || item.volume || 0;
        const fees = item.fees_24h || item.fees || 0;
        const utilization = item.utilization || 0;
        const poolName = item.name || item.pair || `${item.token0}/${item.token1}`;

        return (
            <TouchableOpacity onPress={() => handlePoolPress(item)}>
                <Card style={styles.card}>
                    <Card.Content>
                        <View style={styles.poolHeader}>
                            <View style={styles.poolInfo}>
                                <Title style={styles.poolName}>{poolName}</Title>
                                {item.protocol && (
                                    <Paragraph style={styles.poolProtocol}>
                                        {item.protocol}
                                    </Paragraph>
                                )}
                            </View>
                            <View style={styles.aprContainer}>
                                <Text style={[styles.aprValue, { color: theme.colors.primary }]}>
                                    {apr.toFixed(2)}%
                                </Text>
                                <Text style={styles.aprLabel}>APR</Text>
                            </View>
                        </View>

                        <View style={styles.statsGrid}>
                            <View style={styles.statItem}>
                                <Text style={styles.statLabel}>TVL</Text>
                                <Text style={styles.statValue}>{formatCurrency(tvl)}</Text>
                            </View>
                            <View style={styles.statItem}>
                                <Text style={styles.statLabel}>24h Volume</Text>
                                <Text style={styles.statValue}>{formatCurrency(volume)}</Text>
                            </View>
                            <View style={styles.statItem}>
                                <Text style={styles.statLabel}>24h Fees</Text>
                                <Text style={styles.statValue}>{formatCurrency(fees)}</Text>
                            </View>
                        </View>

                        {utilization > 0 && (
                            <View style={styles.utilizationContainer}>
                                <View style={styles.utilizationHeader}>
                                    <Text style={styles.utilizationLabel}>Utilization</Text>
                                    <Text style={styles.utilizationValue}>
                                        {(utilization * 100).toFixed(1)}%
                                    </Text>
                                </View>
                                <ProgressBar
                                    progress={utilization}
                                    color={theme.colors.primary}
                                    style={styles.progressBar}
                                />
                            </View>
                        )}

                        <View style={styles.tagsContainer}>
                            {item.stable && (
                                <Chip
                                    mode="outlined"
                                    style={styles.tagChip}
                                    textStyle={styles.tagChipText}
                                >
                                    Stable
                                </Chip>
                            )}
                            {item.verified && (
                                <Chip
                                    icon="check-circle"
                                    mode="outlined"
                                    style={[styles.tagChip, { borderColor: theme.colors.primary }]}
                                    textStyle={[
                                        styles.tagChipText,
                                        { color: theme.colors.primary },
                                    ]}
                                >
                                    Verified
                                </Chip>
                            )}
                            {item.incentivized && (
                                <Chip
                                    icon="gift"
                                    mode="outlined"
                                    style={styles.tagChip}
                                    textStyle={styles.tagChipText}
                                >
                                    Rewards
                                </Chip>
                            )}
                        </View>
                    </Card.Content>
                </Card>
            </TouchableOpacity>
        );
    };

    // Handle pool press (navigate to detail screen if implemented)
    const handlePoolPress = (pool) => {
        // For now, just log the pool
        console.log('Pool pressed:', pool);
        // In a full implementation, you would navigate to a pool detail screen:
        // navigation.navigate('PoolDetail', { poolId: pool.id });
    };

    // Render empty state
    const renderEmptyState = () => (
        <View style={styles.emptyState}>
            <IconButton icon="waves" size={64} iconColor={theme.colors.outline} />
            <Text style={[styles.emptyText, { color: theme.colors.onSurfaceVariant }]}>
                {searchQuery ? 'No pools found matching your search' : 'No pools available'}
            </Text>
            <Text style={[styles.emptySubtext, { color: theme.colors.onSurfaceVariant }]}>
                {error ? 'There was an error loading pools' : 'Pull down to refresh'}
            </Text>
        </View>
    );

    // Render loading state
    if (loading && !refreshing) {
        return (
            <View style={[styles.centerContainer, { backgroundColor: theme.colors.background }]}>
                <ActivityIndicator size="large" color={theme.colors.primary} />
                <Text style={[styles.loadingText, { color: theme.colors.onSurfaceVariant }]}>
                    Loading pools...
                </Text>
            </View>
        );
    }

    return (
        <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
            {/* Search Bar */}
            <Searchbar
                placeholder="Search pools..."
                onChangeText={onChangeSearch}
                value={searchQuery}
                style={styles.searchBar}
                elevation={0}
            />

            {/* Sort Chips */}
            <View style={styles.sortContainer}>
                <Text style={styles.sortLabel}>Sort by:</Text>
                <Chip
                    selected={sortBy === 'tvl'}
                    onPress={() => setSortBy('tvl')}
                    style={styles.sortChip}
                    mode={sortBy === 'tvl' ? 'flat' : 'outlined'}
                >
                    TVL
                </Chip>
                <Chip
                    selected={sortBy === 'apr'}
                    onPress={() => setSortBy('apr')}
                    style={styles.sortChip}
                    mode={sortBy === 'apr' ? 'flat' : 'outlined'}
                >
                    APR
                </Chip>
                <Chip
                    selected={sortBy === 'volume'}
                    onPress={() => setSortBy('volume')}
                    style={styles.sortChip}
                    mode={sortBy === 'volume' ? 'flat' : 'outlined'}
                >
                    Volume
                </Chip>
            </View>

            {/* Pools List */}
            <FlatList
                data={filteredPools}
                renderItem={renderPoolCard}
                keyExtractor={(item) => item.id || item.pair || item.name}
                contentContainerStyle={styles.listContainer}
                refreshControl={
                    <RefreshControl
                        refreshing={refreshing}
                        onRefresh={onRefresh}
                        colors={[theme.colors.primary]}
                        tintColor={theme.colors.primary}
                    />
                }
                ListEmptyComponent={renderEmptyState}
            />

            {/* Error Snackbar */}
            <Snackbar
                visible={!!error}
                onDismiss={() => setError(null)}
                action={{
                    label: 'Retry',
                    onPress: () => loadPools(),
                }}
                duration={5000}
            >
                {error}
            </Snackbar>
        </View>
    );
};

// Mock data for development/testing
const getMockPools = () => [
    {
        id: 'pool1',
        name: 'synBTC/synUSD',
        pair: 'synBTC/synUSD',
        token0: 'synBTC',
        token1: 'synUSD',
        protocol: 'Fluxion AMM',
        tvl: 10500000,
        apr: 12.5,
        volume_24h: 2100000,
        fees_24h: 6300,
        utilization: 0.72,
        stable: false,
        verified: true,
        incentivized: true,
    },
    {
        id: 'pool2',
        name: 'synETH/synUSD',
        pair: 'synETH/synUSD',
        token0: 'synETH',
        token1: 'synUSD',
        protocol: 'Fluxion AMM',
        tvl: 8200000,
        apr: 10.8,
        volume_24h: 1640000,
        fees_24h: 4920,
        utilization: 0.65,
        stable: false,
        verified: true,
        incentivized: true,
    },
    {
        id: 'pool3',
        name: 'synETH/synBTC',
        pair: 'synETH/synBTC',
        token0: 'synETH',
        token1: 'synBTC',
        protocol: 'Fluxion AMM',
        tvl: 5100000,
        apr: 8.2,
        volume_24h: 816000,
        fees_24h: 2448,
        utilization: 0.58,
        stable: false,
        verified: true,
        incentivized: false,
    },
    {
        id: 'pool4',
        name: 'synUSD/USDC',
        pair: 'synUSD/USDC',
        token0: 'synUSD',
        token1: 'USDC',
        protocol: 'Fluxion Stable',
        tvl: 15000000,
        apr: 5.5,
        volume_24h: 3000000,
        fees_24h: 900,
        utilization: 0.85,
        stable: true,
        verified: true,
        incentivized: true,
    },
    {
        id: 'pool5',
        name: 'synGOLD/synUSD',
        pair: 'synGOLD/synUSD',
        token0: 'synGOLD',
        token1: 'synUSD',
        protocol: 'Fluxion AMM',
        tvl: 3500000,
        apr: 15.2,
        volume_24h: 700000,
        fees_24h: 2100,
        utilization: 0.45,
        stable: false,
        verified: true,
        incentivized: true,
    },
];

const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
    centerContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 16,
    },
    searchBar: {
        margin: 16,
        marginBottom: 8,
    },
    sortContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingHorizontal: 16,
        paddingBottom: 8,
    },
    sortLabel: {
        fontSize: 14,
        marginRight: 8,
        opacity: 0.7,
    },
    sortChip: {
        marginRight: 8,
    },
    listContainer: {
        padding: 16,
        paddingTop: 8,
    },
    card: {
        marginBottom: 12,
        elevation: 2,
    },
    poolHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        marginBottom: 12,
    },
    poolInfo: {
        flex: 1,
    },
    poolName: {
        fontSize: 18,
        fontWeight: 'bold',
        marginBottom: 4,
    },
    poolProtocol: {
        fontSize: 12,
        opacity: 0.6,
    },
    aprContainer: {
        alignItems: 'flex-end',
        paddingLeft: 12,
    },
    aprValue: {
        fontSize: 24,
        fontWeight: 'bold',
    },
    aprLabel: {
        fontSize: 12,
        opacity: 0.6,
    },
    statsGrid: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        paddingVertical: 12,
        borderTopWidth: 1,
        borderTopColor: '#eee',
        borderBottomWidth: 1,
        borderBottomColor: '#eee',
        marginBottom: 12,
    },
    statItem: {
        flex: 1,
    },
    statLabel: {
        fontSize: 11,
        opacity: 0.6,
        marginBottom: 4,
    },
    statValue: {
        fontSize: 13,
        fontWeight: '600',
    },
    utilizationContainer: {
        marginBottom: 12,
    },
    utilizationHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginBottom: 6,
    },
    utilizationLabel: {
        fontSize: 12,
        opacity: 0.6,
    },
    utilizationValue: {
        fontSize: 12,
        fontWeight: '600',
    },
    progressBar: {
        height: 6,
        borderRadius: 3,
    },
    tagsContainer: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: 6,
    },
    tagChip: {
        height: 24,
        marginRight: 6,
    },
    tagChipText: {
        fontSize: 11,
    },
    loadingText: {
        marginTop: 16,
        fontSize: 16,
    },
    emptyState: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        paddingVertical: 60,
    },
    emptyText: {
        fontSize: 16,
        fontWeight: '600',
        marginTop: 16,
        textAlign: 'center',
    },
    emptySubtext: {
        fontSize: 14,
        marginTop: 8,
        textAlign: 'center',
    },
});

export default PoolsScreen;
