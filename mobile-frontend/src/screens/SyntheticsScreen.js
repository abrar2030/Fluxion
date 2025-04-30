import React, { useState, useEffect } from 'react';
import { Box, Text, VStack, Heading, ScrollView, Card, HStack, Icon, Spinner, FlatList, Button, Input } from 'native-base';
import { Ionicons } from '@expo/vector-icons';
import { getSynthetics } from '../services/api'; // Placeholder API call

const SyntheticsScreen = () => {
  const [loading, setLoading] = useState(true);
  const [synthetics, setSynthetics] = useState([]);
  const [assetToTrade, setAssetToTrade] = useState('');
  const [amount, setAmount] = useState('');

  useEffect(() => {
    const fetchSynthetics = async () => {
      try {
        // Replace with actual API call when available
        // const response = await getSynthetics();
        // setSynthetics(response.data);

        // Placeholder data
        const mockSynthetics = [
          { id: 's1', symbol: 'sUSD', name: 'Synthetic USD', price: '$1.00', change: '+0.01%' },
          { id: 's2', symbol: 'sETH', name: 'Synthetic ETH', price: '$3,000.50', change: '-1.20%' },
          { id: 's3', symbol: 'sBTC', name: 'Synthetic BTC', price: '$40,500.75', change: '+0.85%' },
          { id: 's4', symbol: 'sAAPL', name: 'Synthetic Apple', price: '$175.20', change: '+2.10%' },
        ];
        setSynthetics(mockSynthetics);

      } catch (error) {
        console.error('Error fetching synthetics:', error);
      }
      setLoading(false);
    };

    setTimeout(() => {
      fetchSynthetics();
    }, 1200);

  }, []);

  const handleTrade = () => {
    if (!assetToTrade || !amount) {
      console.warn('Please select an asset and enter an amount');
      return;
    }
    console.log(`Trading ${amount} of ${assetToTrade}`);
    // Add actual trade logic here (integration with backend/blockchain)
  };

  const renderSyntheticItem = ({ item }) => (
    <Card bg="gray.800" p={3} rounded="lg" mb={3}>
      <HStack justifyContent="space-between" alignItems="center">
        <VStack>
          <Text color="white" fontWeight="bold">{item.symbol} <Text color="gray.400" fontSize="xs">({item.name})</Text></Text>
        </VStack>
        <VStack alignItems="flex-end">
          <Text color="white">{item.price}</Text>
          <Text color={item.change.startsWith('+') ? 'green.400' : 'red.400'} fontSize="sm">{item.change}</Text>
        </VStack>
      </HStack>
    </Card>
  );

  return (
    <ScrollView flex={1} bg="gray.950">
      <VStack space={4} p={4} alignItems="stretch">
        <Heading color="white" textAlign="center">Synthetic Assets</Heading>

        {/* Trading Interface Placeholder */}
        <Card bg="gray.800" p={4} rounded="lg">
          <VStack space={3}>
            <Heading size="sm" color="white">Trade Synthetics</Heading>
            {/* Basic Input fields - Replace with proper asset selection later */}
            <Input
              placeholder="Asset Symbol (e.g., sETH)"
              value={assetToTrade}
              onChangeText={setAssetToTrade}
              color="white"
              placeholderTextColor="gray.500"
            />
            <Input
              placeholder="Amount"
              value={amount}
              onChangeText={setAmount}
              keyboardType="numeric"
              color="white"
              placeholderTextColor="gray.500"
            />
            <Button colorScheme="primary" onPress={handleTrade}>
              Execute Trade
            </Button>
          </VStack>
        </Card>

        {/* List of Synthetics */}
        <Heading size="md" color="white" mt={4}>Available Assets</Heading>
        {loading ? (
          <Spinner color="primary.500" size="lg" mt={5} />
        ) : (
          <FlatList
            data={synthetics}
            renderItem={renderSyntheticItem}
            keyExtractor={(item) => item.id}
            ListEmptyComponent={<Text color="gray.500" textAlign="center" mt={5}>No synthetic assets found.</Text>}
            // Disable FlatList scrolling within ScrollView
            scrollEnabled={false}
          />
        )}
      </VStack>
    </ScrollView>
  );
};

export default SyntheticsScreen;

