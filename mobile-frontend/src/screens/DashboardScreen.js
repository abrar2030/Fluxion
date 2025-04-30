import React, { useState, useEffect } from 'react';
import { Box, Text, VStack, Heading, ScrollView, Card, HStack, Icon, Spinner } from 'native-base';
import { Ionicons } from '@expo/vector-icons';
import { getHealth } from '../services/api'; // Example API call

const DashboardScreen = () => {
  const [loading, setLoading] = useState(true);
  const [backendStatus, setBackendStatus] = useState('Checking...');

  useEffect(() => {
    // Example: Check backend health on load
    const checkBackend = async () => {
      try {
        const response = await getHealth();
        if (response.data.status === 'healthy') {
          setBackendStatus('Operational');
        } else {
          setBackendStatus('Error');
        }
      } catch (error) {
        console.error('Error checking backend status:', error);
        setBackendStatus('Offline');
      }
      setLoading(false);
    };

    // Simulate loading and check status
    // In a real app, fetch actual dashboard data here
    setTimeout(() => {
       // checkBackend(); // Uncomment when backend is running and accessible
       setBackendStatus('Simulated Operational'); // Placeholder status
       setLoading(false); // Simulate loading finished
    }, 1500);

  }, []);

  return (
    <ScrollView flex={1} bg="gray.950">
      <VStack space={4} p={4} alignItems="stretch">
        <Heading color="white" textAlign="center">Dashboard</Heading>

        {loading ? (
          <Spinner color="primary.500" size="lg" />
        ) : (
          <VStack space={4}>
            {/* Status Card */}
            <Card bg="gray.800" p={4} rounded="lg">
              <HStack space={3} alignItems="center">
                <Icon as={Ionicons} name={backendStatus === 'Operational' || backendStatus === 'Simulated Operational' ? "checkmark-circle" : "warning"} size="md" color={backendStatus === 'Operational' || backendStatus === 'Simulated Operational' ? "green.500" : "red.500"} />
                <VStack>
                  <Text color="gray.400" fontSize="sm">Backend Status</Text>
                  <Text color="white" fontWeight="bold">{backendStatus}</Text>
                </VStack>
              </HStack>
            </Card>

            {/* Placeholder Cards for other data */}
            <Card bg="gray.800" p={4} rounded="lg">
              <VStack space={2}>
                <Text color="gray.400" fontSize="sm">Total Value Locked (TVL)</Text>
                <Text color="white" fontSize="xl" fontWeight="bold">$ ---</Text>
                <Text color="gray.500">(Data loading...)</Text>
              </VStack>
            </Card>

            <Card bg="gray.800" p={4} rounded="lg">
              <VStack space={2}>
                <Text color="gray.400" fontSize="sm">Active Pools</Text>
                <Text color="white" fontSize="xl" fontWeight="bold">--</Text>
                 <Text color="gray.500">(Data loading...)</Text>
             </VStack>
            </Card>

             <Card bg="gray.800" p={4} rounded="lg">
              <VStack space={2}>
                <Text color="gray.400" fontSize="sm">Recent Transactions</Text>
                <Text color="gray.500">(Transaction list placeholder)</Text>
              </VStack>
            </Card>
          </VStack>
        )}
      </VStack>
    </ScrollView>
  );
};

export default DashboardScreen;

