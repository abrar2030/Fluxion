import React, { useState, useEffect } from 'react';
import { Box, Text, VStack, Heading, ScrollView, Card, HStack, Icon, Spinner, FlatList, Button } from 'native-base';
import { Ionicons } from '@expo/vector-icons';
import { getPools } from '../services/api'; // Placeholder API call

const PoolsScreen = () => {
  const [loading, setLoading] = useState(true);
  const [pools, setPools] = useState([]);

  useEffect(() => {
    const fetchPools = async () => {
      try {
        // Replace with actual API call when available
        // const response = await getPools();
        // setPools(response.data);

        // Placeholder data
        const mockPools = [
          { id: '1', name: 'ETH/USDC', tvl: '$1.2M', apr: '15.5%' },
          { id: '2', name: 'WBTC/ETH', tvl: '$800K', apr: '12.1%' },
          { id: '3', name: 'LINK/USDT', tvl: '$500K', apr: '18.2%' },
          { id: '4', name: 'MATIC/DAI', tvl: '$350K', apr: '22.0%' },
        ];
        setPools(mockPools);

      } catch (error) {
        console.error('Error fetching pools:', error);
        // Handle error state appropriately
      }
      setLoading(false);
    };

    // Simulate loading
    setTimeout(() => {
      fetchPools();
    }, 1000);

  }, []);

  const renderPoolItem = ({ item }) => (
    <Card bg="gray.800" p={4} rounded="lg" mb={3}>
      <HStack justifyContent="space-between" alignItems="center">
        <VStack>
          <Text color="white" fontWeight="bold" fontSize="md">{item.name}</Text>
          <Text color="gray.400" fontSize="sm">TVL: {item.tvl}</Text>
        </VStack>
        <VStack alignItems="flex-end">
          <Text color="green.400" fontWeight="bold">APR: {item.apr}</Text>
          <Button size="sm" variant="outline" colorScheme="primary" mt={1} onPress={() => console.log('Manage Pool:', item.id)}>
            Manage
          </Button>
        </VStack>
      </HStack>
    </Card>
  );

  return (
    <Box flex={1} bg="gray.950">
      <VStack space={4} p={4} flex={1}>
        <HStack justifyContent="space-between" alignItems="center">
           <Heading color="white">Liquidity Pools</Heading>
           <Button
             size="sm"
             colorScheme="primary"
             leftIcon={<Icon as={Ionicons} name="add-circle-outline" size="sm" />}
             onPress={() => console.log('Create New Pool')}
           >
             Create Pool
           </Button>
        </HStack>

        {loading ? (
          <Spinner color="primary.500" size="lg" mt={10} />
        ) : (
          <FlatList
            data={pools}
            renderItem={renderPoolItem}
            keyExtractor={(item) => item.id}
            ListEmptyComponent={<Text color="gray.500" textAlign="center" mt={10}>No pools found.</Text>}
          />
        )}
      </VStack>
    </Box>
  );
};

export default PoolsScreen;

