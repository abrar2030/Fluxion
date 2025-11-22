import React from 'react';
import { View, StyleSheet } from 'react-native';
import { Text, Card, Title, Paragraph, useTheme } from 'react-native-paper';

const AssetsScreen = () => {
    const theme = useTheme();

    // Placeholder data - replace with actual API call
    const assets = [
        {
            id: 'synBTC',
            name: 'Synthetic Bitcoin',
            price: '$65,000',
            change: '+2.5%',
        },
        { id: 'synETH', name: 'Synthetic Ether', price: '$3,500', change: '+1.8%' },
        { id: 'synUSD', name: 'Synthetic USD', price: '$1.00', change: '+0.0%' },
    ];

    return (
        <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
            <Title style={styles.title}>Synthetic Assets</Title>
            {assets.map((asset) => (
                <Card key={asset.id} style={styles.card}>
                    <Card.Content>
                        <Title>
                            {asset.name} ({asset.id})
                        </Title>
                        <Paragraph>Price: {asset.price}</Paragraph>
                        <Paragraph>24h Change: {asset.change}</Paragraph>
                    </Card.Content>
                </Card>
            ))}
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        padding: 16,
    },
    title: {
        marginBottom: 16,
        textAlign: 'center',
    },
    card: {
        marginBottom: 12,
    },
});

export default AssetsScreen;
