import React, { useState, useEffect } from 'react';
import {
  Box,
  Heading,
  SimpleGrid,
  Flex,
  Text,
  Card,
  CardBody,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Badge,
  Button,
  useColorModeValue,
  Icon,
} from '@chakra-ui/react';
import { FiTrendingUp, FiDollarSign, FiActivity, FiDroplet } from 'react-icons/fi';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useWeb3 } from .../lib/web3-config.jsx';
import { useData } from '../lib/data-context';
import { useUI } from '../lib/ui-context';
import StatCard from '../components/ui/StatCard';

const Dashboard = () => {
  const { isConnected, account, connectWallet } = useWeb3();
  const { marketData, poolsData, analyticsData, fetchMarketData, fetchPoolsData, isLoading } = useData();
  const { addNotification } = useUI();
  const [chartData, setChartData] = useState([]);

  const cardBg = useColorModeValue('gray.700', 'gray.700');
  const borderColor = useColorModeValue('gray.600', 'gray.600');

  useEffect(() => {
    // Fetch data when component mounts
    fetchMarketData();
    fetchPoolsData();

    // Set up chart data
    setChartData(analyticsData.tvlData);

    // Show welcome notification
    addNotification({
      title: 'Welcome to Fluxion',
      message: 'Your decentralized liquidity and synthetic asset platform',
      type: 'info',
      duration: 5000
    });
  }, []);

  const handleConnectWallet = async () => {
    try {
      await connectWallet();
      addNotification({
        title: 'Wallet Connected',
        message: 'Your wallet has been successfully connected.',
        type: 'success'
      });
    } catch (error) {
      addNotification({
        title: 'Connection Failed',
        message: error.message || 'Failed to connect wallet. Please try again.',
        type: 'error'
      });
    }
  };

  return (
    <Box maxW="7xl" mx="auto" pt={5} px={{ base: 2, sm: 12, md: 17 }} className="fade-in">
      <Heading as="h1" mb={6} fontSize="3xl" color="white">
        Dashboard
      </Heading>

      {!isConnected && (
        <Card bg={cardBg} borderColor={borderColor} borderWidth="1px" borderRadius="lg" mb={8} className="slide-up">
          <CardBody>
            <Flex direction={{ base: 'column', md: 'row' }} align="center" justify="space-between">
              <Box mb={{ base: 4, md: 0 }}>
                <Heading size="md" color="white" mb={2}>
                  Welcome to Fluxion
                </Heading>
                <Text color="gray.300">
                  Connect your wallet to start creating and managing liquidity pools and synthetic assets.
                </Text>
              </Box>
              <Button colorScheme="blue" onClick={handleConnectWallet}>
                Connect Wallet
              </Button>
            </Flex>
          </CardBody>
        </Card>
      )}

      <SimpleGrid columns={{ base: 1, md: 4 }} spacing={{ base: 5, lg: 8 }}>
        <StatCard
          title="Total Value Locked"
          value={marketData.tvl}
          icon={FiDollarSign}
          type={{ type: 'increase', value: '23.36' }}
          isLoading={isLoading}
        />
        <StatCard
          title="24h Volume"
          value={marketData.volume24h}
          icon={FiActivity}
          type={{ type: 'increase', value: '12.5' }}
          isLoading={isLoading}
        />
        <StatCard
          title="Active Pools"
          value={marketData.activePools.toString()}
          icon={FiDroplet}
          helpText="Total number of active liquidity pools"
          isLoading={isLoading}
        />
        <StatCard
          title="APY (avg)"
          value={marketData.avgApy}
          icon={FiTrendingUp}
          type={{ type: 'decrease', value: '1.2' }}
          isLoading={isLoading}
        />
      </SimpleGrid>

      <Box
        mt={10}
        bg={cardBg}
        shadow="xl"
        border="1px solid"
        borderColor={borderColor}
        rounded="lg"
        p={6}
        className="slide-up"
      >
        <Heading as="h2" size="md" mb={6} color="white">
          Platform Overview
        </Heading>
        <Box height="300px">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={chartData}
              margin={{
                top: 5,
                right: 30,
                left: 20,
                bottom: 5,
              }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#444" />
              <XAxis dataKey="name" stroke="#999" />
              <YAxis stroke="#999" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#333',
                  borderColor: '#555',
                  color: 'white'
                }}
              />
              <Line type="monotone" dataKey="tvl" stroke="#0080ff" activeDot={{ r: 8 }} />
              <Line type="monotone" dataKey="volume" stroke="#ff8c00" />
            </LineChart>
          </ResponsiveContainer>
        </Box>
      </Box>

      <Box
        mt={10}
        bg={cardBg}
        shadow="xl"
        border="1px solid"
        borderColor={borderColor}
        rounded="lg"
        p={6}
        className="slide-up"
      >
        <Flex justifyContent="space-between" alignItems="center" mb={6}>
          <Heading as="h2" size="md" color="white">
            Top Performing Pools
          </Heading>
          <Button colorScheme="blue" size="sm" onClick={() => window.location.href = '/pools'}>
            View All
          </Button>
        </Flex>
        <Box overflowX="auto">
          <Table variant="simple" colorScheme="whiteAlpha">
            <Thead>
              <Tr>
                <Th>Pool</Th>
                <Th>TVL</Th>
                <Th>24h Volume</Th>
                <Th>APY</Th>
                <Th>Risk Level</Th>
              </Tr>
            </Thead>
            <Tbody>
              {isLoading ? (
                <Tr>
                  <Td colSpan={5} textAlign="center" py={4}>Loading pool data...</Td>
                </Tr>
              ) : poolsData.length > 0 ? (
                poolsData.slice(0, 4).map((pool) => (
                  <Tr key={pool.id}>
                    <Td fontWeight="bold">{pool.id}</Td>
                    <Td>{pool.tvl}</Td>
                    <Td>{pool.volume24h}</Td>
                    <Td color="green.400">{pool.apy}</Td>
                    <Td>
                      <Badge
                        colorScheme={
                          pool.risk === 'Low' ? 'green' : pool.risk === 'Medium' ? 'yellow' : 'red'
                        }
                      >
                        {pool.risk}
                      </Badge>
                    </Td>
                  </Tr>
                ))
              ) : (
                <Tr>
                  <Td colSpan={5} textAlign="center" py={4}>No pools available</Td>
                </Tr>
              )}
            </Tbody>
          </Table>
        </Box>
      </Box>
    </Box>
  );
};

export default Dashboard;
