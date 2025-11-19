import React from 'react';
import { View, StyleSheet } from 'react-native';
import { Text, Card, Title, Paragraph, useTheme } from 'react-native-paper';

const PoolsScreen = () => {
  const theme = useTheme();

  // Placeholder data - replace with actual API call
  const pools = [
    { id: 'pool1', name: 'synBTC/synUSD', tvl: '$10.5M', apr: '12.5%' },
    { id: 'pool2', name: 'synETH/synUSD', tvl: '$8.2M', apr: '10.8%' },
    { id: 'pool3', name: 'synETH/synBTC', tvl: '$5.1M', apr: '8.2%' },
  ];

  return (
    <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
      <Title style={styles.title}>Liquidity Pools</Title>
      {pools.map(pool => (
        <Card key={pool.id} style={styles.card}>
          <Card.Content>
            <Title>{pool.name}</Title>
            <Paragraph>Total Value Locked (TVL): {pool.tvl}</Paragraph>
            <Paragraph>Estimated APR: {pool.apr}</Paragraph>
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

export default PoolsScreen;
