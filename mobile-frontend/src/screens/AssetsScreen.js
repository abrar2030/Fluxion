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
} from 'react-native-paper';
import { fetchAssets } from '../api/client';
import { formatCurrency } from '../utils/formatters';

const AssetsScreen = ({ navigation }) => {
    const theme = useTheme();
    const [assets, setAssets] = useState([]);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [error, setError] = useState(null);
    const [searchQuery, setSearchQuery] = useState('');
    const [filterType, setFilterType] = useState('all'); // all, synthetic, derivative

    // Load assets on mount
    useEffect(() => {
        loadAssets();

        // Set up auto-refresh every 30 seconds
        const interval = setInterval(() => {
            loadAssets(true); // Silent refresh
        }, 30000);

        return () => clearInterval(interval);
    }, []);

    // Load assets from API
    const loadAssets = async (silent = false) => {
        if (!silent) setLoading(true);
        setError(null);

        try {
            const data = await fetchAssets({
                limit: 100,
                offset: 0,
            });

            // Handle both array response and paginated response
            const assetsArray = Array.isArray(data) ? data : data.items || data.assets || [];
            setAssets(assetsArray);
        } catch (err) {
            console.error('Error loading assets:', err);
            setError(err.message || 'Failed to load assets');

            // Use mock data in development mode as fallback
            if (__DEV__) {
                console.log('Using mock data as fallback');
                setAssets(getMockAssets());
            }
        } finally {
            setLoading(false);
            setRefreshing(false);
        }
    };

    // Pull to refresh handler
    const onRefresh = useCallback(() => {
        setRefreshing(true);
        loadAssets();
    }, []);

    // Search handler
    const onChangeSearch = (query) => {
        setSearchQuery(query);
    };

    // Filter assets based on search and type
    const filteredAssets = assets.filter((asset) => {
        // Search filter
        const matchesSearch =
            !searchQuery ||
            asset.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
            asset.symbol?.toLowerCase().includes(searchQuery.toLowerCase()) ||
            asset.id?.toLowerCase().includes(searchQuery.toLowerCase());

        // Type filter
        const matchesType =
            filterType === 'all' || asset.type?.toLowerCase() === filterType.toLowerCase();

        return matchesSearch && matchesType;
    });

    // Render individual asset card
    const renderAssetCard = ({ item }) => {
        const priceChange = item.price_change_24h || item.change_24h || 0;
        const isPositive = priceChange >= 0;
        const changeColor = isPositive ? theme.colors.success || '#4caf50' : theme.colors.error;

        return (
            <TouchableOpacity onPress={() => handleAssetPress(item)}>
                <Card style={styles.card}>
                    <Card.Content>
                        <View style={styles.assetHeader}>
                            <View style={styles.assetInfo}>
                                <Title style={styles.assetName}>
                                    {item.name || 'Unknown Asset'}
                                </Title>
                                <Paragraph style={styles.assetSymbol}>
                                    {item.symbol || item.id}
                                </Paragraph>
                            </View>
                            <View style={styles.assetStats}>
                                <Text style={styles.assetPrice}>
                                    {formatCurrency(item.price || item.current_price || 0)}
                                </Text>
                                <Chip
                                    icon={isPositive ? 'trending-up' : 'trending-down'}
                                    style={[
                                        styles.changeChip,
                                        { backgroundColor: changeColor + '20' },
                                    ]}
                                    textStyle={[styles.changeText, { color: changeColor }]}
                                >
                                    {isPositive ? '+' : ''}
                                    {priceChange.toFixed(2)}%
                                </Chip>
                            </View>
                        </View>
                        <View style={styles.assetDetails}>
                            <View style={styles.detailItem}>
                                <Text style={styles.detailLabel}>TVL</Text>
                                <Text style={styles.detailValue}>
                                    {formatCurrency(item.tvl || item.total_value_locked || 0)}
                                </Text>
                            </View>
                            <View style={styles.detailItem}>
                                <Text style={styles.detailLabel}>24h Volume</Text>
                                <Text style={styles.detailValue}>
                                    {formatCurrency(item.volume_24h || item.volume || 0)}
                                </Text>
                            </View>
                            <View style={styles.detailItem}>
                                <Text style={styles.detailLabel}>Market Cap</Text>
                                <Text style={styles.detailValue}>
                                    {formatCurrency(item.market_cap || 0)}
                                </Text>
                            </View>
                        </View>
                        {item.type && (
                            <Chip
                                mode="outlined"
                                style={styles.typeChip}
                                textStyle={styles.typeChipText}
                            >
                                {item.type}
                            </Chip>
                        )}
                    </Card.Content>
                </Card>
            </TouchableOpacity>
        );
    };

    // Handle asset press (navigate to detail screen if implemented)
    const handleAssetPress = (asset) => {
        // For now, just log the asset
        console.log('Asset pressed:', asset);
        // In a full implementation, you would navigate to an asset detail screen:
        // navigation.navigate('AssetDetail', { assetId: asset.id });
    };

    // Render empty state
    const renderEmptyState = () => (
        <View style={styles.emptyState}>
            <IconButton icon="package-variant" size={64} iconColor={theme.colors.outline} />
            <Text style={[styles.emptyText, { color: theme.colors.onSurfaceVariant }]}>
                {searchQuery ? 'No assets found matching your search' : 'No assets available'}
            </Text>
            <Text style={[styles.emptySubtext, { color: theme.colors.onSurfaceVariant }]}>
                {error ? 'There was an error loading assets' : 'Pull down to refresh'}
            </Text>
        </View>
    );

    // Render loading state
    if (loading && !refreshing) {
        return (
            <View style={[styles.centerContainer, { backgroundColor: theme.colors.background }]}>
                <ActivityIndicator size="large" color={theme.colors.primary} />
                <Text style={[styles.loadingText, { color: theme.colors.onSurfaceVariant }]}>
                    Loading assets...
                </Text>
            </View>
        );
    }

    return (
        <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
            {/* Search Bar */}
            <Searchbar
                placeholder="Search assets..."
                onChangeText={onChangeSearch}
                value={searchQuery}
                style={styles.searchBar}
                elevation={0}
            />

            {/* Filter Chips */}
            <View style={styles.filterContainer}>
                <Chip
                    selected={filterType === 'all'}
                    onPress={() => setFilterType('all')}
                    style={styles.filterChip}
                    mode={filterType === 'all' ? 'flat' : 'outlined'}
                >
                    All
                </Chip>
                <Chip
                    selected={filterType === 'synthetic'}
                    onPress={() => setFilterType('synthetic')}
                    style={styles.filterChip}
                    mode={filterType === 'synthetic' ? 'flat' : 'outlined'}
                >
                    Synthetic
                </Chip>
                <Chip
                    selected={filterType === 'derivative'}
                    onPress={() => setFilterType('derivative')}
                    style={styles.filterChip}
                    mode={filterType === 'derivative' ? 'flat' : 'outlined'}
                >
                    Derivative
                </Chip>
            </View>

            {/* Assets List */}
            <FlatList
                data={filteredAssets}
                renderItem={renderAssetCard}
                keyExtractor={(item) => item.id || item.symbol}
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
                    onPress: () => loadAssets(),
                }}
                duration={5000}
            >
                {error}
            </Snackbar>
        </View>
    );
};

// Mock data for development/testing
const getMockAssets = () => [
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
        price_change_24h: 1.8,
        tvl: 85000000,
        volume_24h: 32000000,
        market_cap: 420000000,
    },
    {
        id: 'synUSD',
        symbol: 'synUSD',
        name: 'Synthetic USD',
        type: 'synthetic',
        price: 1.0,
        price_change_24h: 0.0,
        tvl: 50000000,
        volume_24h: 15000000,
        market_cap: 50000000,
    },
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
    {
        id: 'synSP500',
        symbol: 'synSP500',
        name: 'Synthetic S&P 500',
        type: 'derivative',
        price: 4500,
        price_change_24h: 1.2,
        tvl: 60000000,
        volume_24h: 20000000,
        market_cap: 300000000,
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
    filterContainer: {
        flexDirection: 'row',
        paddingHorizontal: 16,
        paddingBottom: 8,
        gap: 8,
    },
    filterChip: {
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
    assetHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        marginBottom: 12,
    },
    assetInfo: {
        flex: 1,
    },
    assetName: {
        fontSize: 18,
        fontWeight: 'bold',
        marginBottom: 4,
    },
    assetSymbol: {
        fontSize: 14,
        opacity: 0.7,
    },
    assetStats: {
        alignItems: 'flex-end',
    },
    assetPrice: {
        fontSize: 20,
        fontWeight: 'bold',
        marginBottom: 4,
    },
    changeChip: {
        height: 24,
        paddingHorizontal: 8,
    },
    changeText: {
        fontSize: 12,
        fontWeight: 'bold',
    },
    assetDetails: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginBottom: 8,
        paddingTop: 8,
        borderTopWidth: 1,
        borderTopColor: '#eee',
    },
    detailItem: {
        flex: 1,
    },
    detailLabel: {
        fontSize: 12,
        opacity: 0.6,
        marginBottom: 2,
    },
    detailValue: {
        fontSize: 14,
        fontWeight: '600',
    },
    typeChip: {
        alignSelf: 'flex-start',
        height: 24,
    },
    typeChipText: {
        fontSize: 11,
        textTransform: 'uppercase',
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

export default AssetsScreen;
