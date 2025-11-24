import React, { useState, useEffect } from 'react';
import {
  Box,
  Heading,
  SimpleGrid,
  Card,
  CardHeader,
  CardBody,
  CardFooter,
  Text,
  Button,
  Badge,
  Flex,
  Progress,
  HStack,
  VStack,
  Avatar,
  Input,
  InputGroup,
  InputLeftElement,
  useColorModeValue,
  Icon,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  Divider
} from '@chakra-ui/react';
import { FiDroplet, FiTrendingUp, FiDollarSign, FiUsers, FiSearch, FiFilter, FiChevronDown } from 'react-icons/fi';
import { useWeb3 } from ../lib/web3-config.jsx';
import { useData } from '../lib/data-context';
import { useUI } from '../lib/ui-context';
import { useNavigate } from 'react-router-dom';
import StatCard from '../components/ui/StatCard';

const PoolCard = ({ pool, onViewDetails }) => {
  const cardBg = useColorModeValue('gray.700', 'gray.700');
  const borderColor = useColorModeValue('gray.600', 'gray.600');

  return (
    <Card
      bg={cardBg}
      borderColor={borderColor}
      borderWidth="1px"
      borderRadius="lg"
      overflow="hidden"
      transition="all 0.3s"
      _hover={{ transform: 'translateY(-5px)', shadow: 'xl' }}
      className="slide-up"
    >
      <CardHeader pb={0}>
        <Flex justify="space-between" align="center">
          <Heading size="md" color="white">{pool.id}</Heading>
          <Badge
            colorScheme={
              pool.risk === 'Low' ? 'green' :
              pool.risk === 'Medium' ? 'yellow' : 'red'
            }
          >
            {pool.risk} Risk
          </Badge>
        </Flex>
      </CardHeader>
      <CardBody>
        <VStack spacing={3} align="stretch">
          <Flex justify="space-between">
            <Text color="gray.400">TVL:</Text>
            <Text color="white" fontWeight="bold">{pool.tvl}</Text>
          </Flex>
          <Flex justify="space-between">
            <Text color="gray.400">24h Volume:</Text>
            <Text color="white">{pool.volume24h}</Text>
          </Flex>
          <Flex justify="space-between">
            <Text color="gray.400">APY:</Text>
            <Text color="green.400" fontWeight="bold">{pool.apy}</Text>
          </Flex>

          <Box pt={2}>
            <Text color="gray.400" mb={1}>Asset Weights:</Text>
            <HStack spacing={2}>
              {pool.assets.map((asset, index) => (
                <Flex key={asset} align="center">
                  <Avatar size="xs" name={asset} bg={index === 0 ? 'brand.500' : 'accent.500'} mr={1} />
                  <Text color="white">{asset} ({pool.weights[index]}%)</Text>
                </Flex>
              ))}
            </HStack>
          </Box>

          <Box pt={2}>
            <Flex justify="space-between" mb={1}>
              <Text color="gray.400">Utilization:</Text>
              <Text color="white">{pool.utilization}%</Text>
            </Flex>
            <Progress
              value={pool.utilization}
              colorScheme={
                pool.utilization < 70 ? 'green' :
                pool.utilization < 85 ? 'yellow' : 'red'
              }
              borderRadius="md"
              size="sm"
            />
          </Box>
        </VStack>
      </CardBody>
      <Divider borderColor="gray.600" />
      <CardFooter>
        <Button variant="solid" colorScheme="blue" size="sm" width="full" onClick={() => onViewDetails(pool.id)}>
          View Details
        </Button>
      </CardFooter>
    </Card>
  );
};

const Pools = () => {
  const navigate = useNavigate();
  const { isConnected, account } = useWeb3();
  const { poolsData, fetchPoolsData, isLoading } = useData();
  const { addNotification } = useUI();

  const [filter, setFilter] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredPools, setFilteredPools] = useState([]);

  useEffect(() => {
    // Fetch pools data when component mounts
    fetchPoolsData();
  }, []);

  // Filter and search pools
  useEffect(() => {
    let result = [...poolsData];

    // Apply risk filter
    if (filter !== 'all') {
      result = result.filter(pool => pool.risk.toLowerCase() === filter.toLowerCase());
    }

    // Apply search term
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      result = result.filter(pool =>
        pool.id.toLowerCase().includes(term) ||
        pool.assets.some(asset => asset.toLowerCase().includes(term))
      );
    }

    setFilteredPools(result);
  }, [poolsData, filter, searchTerm]);

  const handleViewDetails = (poolId) => {
    // In a real app, this would navigate to a pool details page
    addNotification({
      title: 'Pool Details',
      message: `Viewing details for pool ${poolId}`,
      type: 'info'
    });
  };

  const handleCreatePool = () => {
    navigate('/create-pool');
  };

  return (
    <Box maxW="7xl" mx="auto" pt={5} px={{ base: 2, sm: 12, md: 17 }} className="fade-in">
      <Heading as="h1" mb={6} fontSize="3xl" color="white">
        Liquidity Pools
      </Heading>

      <SimpleGrid columns={{ base: 1, md: 4 }} spacing={{ base: 5, lg: 8 }} mb={8}>
        <StatCard
          title="Total Pools"
          value={poolsData.length.toString()}
          icon={FiDroplet}
          helpText="Total number of liquidity pools"
          isLoading={isLoading}
        />

        <StatCard
          title="Total TVL"
          value="$5.9M"
          icon={FiDollarSign}
          type={{ type: 'increase', value: '23.36' }}
          isLoading={isLoading}
        />

        <StatCard
          title="Avg APY"
          value="5.8%"
          icon={FiTrendingUp}
          type={{ type: 'decrease', value: '1.2' }}
          isLoading={isLoading}
        />

        <StatCard
          title="Active Users"
          value="1,248"
          icon={FiUsers}
          type={{ type: 'increase', value: '12.5' }}
          isLoading={isLoading}
        />
      </SimpleGrid>

      <Flex
        mb={6}
        justifyContent="space-between"
        alignItems="center"
        flexDirection={{ base: 'column', md: 'row' }}
        gap={{ base: 4, md: 0 }}
      >
        <HStack spacing={2} flexWrap="wrap">
          <Button
            size="sm"
            colorScheme={filter === 'all' ? 'blue' : 'gray'}
            onClick={() => setFilter('all')}
          >
            All Pools
          </Button>
          <Button
            size="sm"
            colorScheme={filter === 'low' ? 'green' : 'gray'}
            onClick={() => setFilter('low')}
          >
            Low Risk
          </Button>
          <Button
            size="sm"
            colorScheme={filter === 'medium' ? 'yellow' : 'gray'}
            onClick={() => setFilter('medium')}
          >
            Medium Risk
          </Button>
          <Button
            size="sm"
            colorScheme={filter === 'high' ? 'red' : 'gray'}
            onClick={() => setFilter('high')}
          >
            High Risk
          </Button>
        </HStack>

        <HStack>
          <InputGroup maxW="250px">
            <InputLeftElement pointerEvents="none">
              <Icon as={FiSearch} color="gray.400" />
            </InputLeftElement>
            <Input
              placeholder="Search pools..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              bg="gray.800"
              color="white"
              borderColor="gray.600"
            />
          </InputGroup>

          <Menu>
            <MenuButton as={Button} rightIcon={<FiChevronDown />} variant="outline">
              Sort By
            </MenuButton>
            <MenuList bg="gray.800" borderColor="gray.700">
              <MenuItem _hover={{ bg: 'gray.700' }} color="white">TVL (High to Low)</MenuItem>
              <MenuItem _hover={{ bg: 'gray.700' }} color="white">TVL (Low to High)</MenuItem>
              <MenuItem _hover={{ bg: 'gray.700' }} color="white">APY (High to Low)</MenuItem>
              <MenuItem _hover={{ bg: 'gray.700' }} color="white">APY (Low to High)</MenuItem>
            </MenuList>
          </Menu>

          <Button colorScheme="blue" leftIcon={<FiDroplet />} onClick={handleCreatePool}>
            Create New Pool
          </Button>
        </HStack>
      </Flex>

      {isLoading ? (
        <Flex justify="center" align="center" minH="300px">
          <Text color="gray.400">Loading pools data...</Text>
        </Flex>
      ) : filteredPools.length > 0 ? (
        <SimpleGrid columns={{ base: 1, md: 3 }} spacing={6}>
          {filteredPools.map(pool => (
            <PoolCard key={pool.id} pool={pool} onViewDetails={handleViewDetails} />
          ))}
        </SimpleGrid>
      ) : (
        <Flex justify="center" align="center" minH="300px" direction="column">
          <Text color="gray.400" mb={4}>No pools found matching your criteria</Text>
          <Button colorScheme="blue" onClick={() => {
            setFilter('all');
            setSearchTerm('');
          }}>
            Reset Filters
          </Button>
        </Flex>
      )}
    </Box>
  );
};

export default Pools;
