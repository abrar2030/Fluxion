import React, { useState } from "react";
import {
  Box,
  Flex,
  Heading,
  Text,
  SimpleGrid,
  Button,
  HStack,
  VStack,
  Icon,
  useColorModeValue,
  Badge,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  Card,
  CardBody,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  IconButton,
} from "@chakra-ui/react";
import {
  FiTrendingUp,
  FiDroplet,
  FiDollarSign,
  FiActivity,
  FiPlus,
} from "react-icons/fi";
import { Link as RouterLink } from "react-router-dom";
import { useWeb3 } from "../../lib/web3-config.jsx";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";

// Mock data for charts
const tvlData = [
  { name: "Jan", tvl: 4000 },
  { name: "Feb", tvl: 3000 },
  { name: "Mar", tvl: 2000 },
  { name: "Apr", tvl: 2780 },
  { name: "May", tvl: 1890 },
  { name: "Jun", tvl: 2390 },
  { name: "Jul", tvl: 3490 },
];

const volumeData = [
  { name: "Jan", volume: 2400 },
  { name: "Feb", volume: 1398 },
  { name: "Mar", volume: 9800 },
  { name: "Apr", volume: 3908 },
  { name: "May", volume: 4800 },
  { name: "Jun", volume: 3800 },
  { name: "Jul", volume: 4300 },
];

const poolDistribution = [
  { name: "ETH-USDC", value: 400 },
  { name: "WBTC-ETH", value: 300 },
  { name: "ETH-DAI", value: 300 },
  { name: "USDC-DAI", value: 200 },
];

const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042"];

const Dashboard = () => {
  const { isConnected, pools } = useWeb3();
  const cardBg = useColorModeValue("gray.800", "gray.700");
  const borderColor = useColorModeValue("gray.700", "gray.600");
  const textColor = useColorModeValue("white", "white");
  const subTextColor = useColorModeValue("gray.400", "gray.400");

  // State for active tab
  const [activeTab, setActiveTab] = useState(0);

  // Mock user positions
  const userPositions = [
    {
      id: "position-1",
      pool: "ETH-USDC",
      liquidity: "$25,000",
      share: "0.5%",
      earnings: "$1,250",
      apy: "8.4%",
    },
    {
      id: "position-2",
      pool: "WBTC-ETH",
      liquidity: "$12,500",
      share: "0.2%",
      earnings: "$450",
      apy: "7.2%",
    },
  ];

  // Mock recent transactions
  const recentTransactions = [
    {
      id: "tx-1",
      type: "Add Liquidity",
      pool: "ETH-USDC",
      amount: "$10,000",
      time: "2 hours ago",
      status: "Completed",
    },
    {
      id: "tx-2",
      type: "Swap",
      pool: "WBTC-ETH",
      amount: "0.5 BTC → 7.5 ETH",
      time: "5 hours ago",
      status: "Completed",
    },
    {
      id: "tx-3",
      type: "Remove Liquidity",
      pool: "ETH-DAI",
      amount: "$5,000",
      time: "1 day ago",
      status: "Completed",
    },
  ];

  return (
    <Box>
      {/* Hero Section */}
      <Box
        p={8}
        borderRadius="xl"
        mb={8}
        bgGradient="linear(to-br, gray.900, gray.800)"
        boxShadow="xl"
        border="1px solid"
        borderColor={borderColor}
        position="relative"
        overflow="hidden"
      >
        {/* Decorative elements */}
        <Box
          position="absolute"
          top="-50px"
          right="-50px"
          w="200px"
          h="200px"
          borderRadius="full"
          bg="brand.500"
          opacity="0.1"
        />
        <Box
          position="absolute"
          bottom="-30px"
          left="30%"
          w="100px"
          h="100px"
          borderRadius="full"
          bg="accent.500"
          opacity="0.1"
        />

        <Flex
          direction={{ base: "column", md: "row" }}
          justify="space-between"
          align={{ base: "flex-start", md: "center" }}
        >
          <Box mb={{ base: 6, md: 0 }}>
            <Heading as="h1" size="xl" mb={4}>
              Welcome to Fluxion
            </Heading>
            <Text fontSize="lg" color={subTextColor} maxW="600px">
              The next-generation decentralized liquidity protocol. Provide
              liquidity, earn fees, and participate in the future of DeFi.
            </Text>
          </Box>

          <HStack spacing={4}>
            <RouterLink to="/pools">
              <Button
                leftIcon={<FiDroplet />}
                colorScheme="brand"
                size="lg"
                variant="outline"
                _hover={{
                  transform: "translateY(-2px)",
                  boxShadow: "lg",
                }}
              >
                Explore Pools
              </Button>
            </RouterLink>

            <RouterLink to="/pools/create">
              <Button
                leftIcon={<FiPlus />}
                colorScheme="brand"
                size="lg"
                bgGradient="linear(to-r, brand.500, accent.500)"
                _hover={{
                  bgGradient: "linear(to-r, brand.600, accent.600)",
                  transform: "translateY(-2px)",
                  boxShadow: "lg",
                }}
              >
                Create Pool
              </Button>
            </RouterLink>
          </HStack>
        </Flex>
      </Box>

      {/* Stats Overview */}
      <SimpleGrid columns={{ base: 1, md: 4 }} spacing={6} mb={8}>
        <Box
          bg={cardBg}
          p={6}
          borderRadius="lg"
          boxShadow="md"
          border="1px solid"
          borderColor={borderColor}
        >
          <Stat>
            <StatLabel>Total Value Locked</StatLabel>
            <StatNumber>$142.5M</StatNumber>
            <StatHelpText>
              <StatArrow type="increase" />
              23.36%
            </StatHelpText>
          </Stat>
        </Box>

        <Box
          bg={cardBg}
          p={6}
          borderRadius="lg"
          boxShadow="md"
          border="1px solid"
          borderColor={borderColor}
        >
          <Stat>
            <StatLabel>24h Volume</StatLabel>
            <StatNumber>$28.4M</StatNumber>
            <StatHelpText>
              <StatArrow type="increase" />
              12.05%
            </StatHelpText>
          </Stat>
        </Box>

        <Box
          bg={cardBg}
          p={6}
          borderRadius="lg"
          boxShadow="md"
          border="1px solid"
          borderColor={borderColor}
        >
          <Stat>
            <StatLabel>Active Pools</StatLabel>
            <StatNumber>247</StatNumber>
            <StatHelpText>
              <StatArrow type="increase" />8 new today
            </StatHelpText>
          </Stat>
        </Box>

        <Box
          bg={cardBg}
          p={6}
          borderRadius="lg"
          boxShadow="md"
          border="1px solid"
          borderColor={borderColor}
        >
          <Stat>
            <StatLabel>Average APY</StatLabel>
            <StatNumber>8.74%</StatNumber>
            <StatHelpText>
              <StatArrow type="decrease" />
              0.42%
            </StatHelpText>
          </Stat>
        </Box>
      </SimpleGrid>

      {/* Charts Section */}
      <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={8} mb={8}>
        <Box
          bg={cardBg}
          p={6}
          borderRadius="lg"
          boxShadow="md"
          border="1px solid"
          borderColor={borderColor}
        >
          <Heading size="md" mb={6}>
            Total Value Locked
          </Heading>
          <Box h="300px">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={tvlData}
                margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
              >
                <defs>
                  <linearGradient id="colorTvl" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#0080ff" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="#0080ff" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                <XAxis dataKey="name" stroke="#888" />
                <YAxis stroke="#888" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "rgba(23, 25, 35, 0.9)",
                    border: "1px solid #333",
                    borderRadius: "4px",
                    color: "white",
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="tvl"
                  stroke="#0080ff"
                  fillOpacity={1}
                  fill="url(#colorTvl)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </Box>
        </Box>

        <Box
          bg={cardBg}
          p={6}
          borderRadius="lg"
          boxShadow="md"
          border="1px solid"
          borderColor={borderColor}
        >
          <Heading size="md" mb={6}>
            Trading Volume
          </Heading>
          <Box h="300px">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={volumeData}
                margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                <XAxis dataKey="name" stroke="#888" />
                <YAxis stroke="#888" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "rgba(23, 25, 35, 0.9)",
                    border: "1px solid #333",
                    borderRadius: "4px",
                    color: "white",
                  }}
                />
                <Bar dataKey="volume" fill="#ff7000" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Box>
        </Box>
      </SimpleGrid>

      {/* Pool Distribution */}
      <Box
        bg={cardBg}
        p={6}
        borderRadius="lg"
        boxShadow="md"
        border="1px solid"
        borderColor={borderColor}
        mb={8}
      >
        <Heading size="md" mb={6}>
          Pool Distribution
        </Heading>
        <SimpleGrid columns={{ base: 1, md: 2 }} spacing={8}>
          <Box h="300px">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={poolDistribution}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  outerRadius={120}
                  fill="#8884d8"
                  dataKey="value"
                  label={({ name, percent }) =>
                    `${name} ${(percent * 100).toFixed(0)}%`
                  }
                >
                  {poolDistribution.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={COLORS[index % COLORS.length]}
                    />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: "rgba(23, 25, 35, 0.9)",
                    border: "1px solid #333",
                    borderRadius: "4px",
                    color: "white",
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </Box>

          <Box>
            <Heading size="sm" mb={4}>
              Top Pools by TVL
            </Heading>
            <VStack spacing={4} align="stretch">
              {poolDistribution.map((pool, index) => (
                <Card key={index} bg="gray.700" variant="outline">
                  <CardBody>
                    <Flex justify="space-between" align="center">
                      <HStack>
                        <Box
                          bg={COLORS[index % COLORS.length]}
                          borderRadius="md"
                          w="12px"
                          h="12px"
                        />
                        <Text fontWeight="bold">{pool.name}</Text>
                      </HStack>
                      <Text>${(pool.value / 10).toFixed(1)}M</Text>
                    </Flex>
                  </CardBody>
                </Card>
              ))}
            </VStack>
          </Box>
        </SimpleGrid>
      </Box>

      {/* User Dashboard or Connect Wallet */}
      {isConnected ? (
        <Box
          bg={cardBg}
          p={6}
          borderRadius="lg"
          boxShadow="md"
          border="1px solid"
          borderColor={borderColor}
        >
          <Tabs
            variant="soft-rounded"
            colorScheme="brand"
            index={activeTab}
            onChange={(index) => setActiveTab(index)}
          >
            <TabList mb={6}>
              <Tab>My Positions</Tab>
              <Tab>Recent Transactions</Tab>
              <Tab>Rewards</Tab>
            </TabList>

            <TabPanels>
              <TabPanel px={0}>
                <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6}>
                  {userPositions.map((position, index) => (
                    <Box
                      key={index}
                      p={4}
                      borderRadius="md"
                      bg="gray.700"
                      border="1px solid"
                      borderColor="gray.600"
                      _hover={{
                        borderColor: "brand.500",
                        transform: "translateY(-2px)",
                        transition: "all 0.3s ease",
                      }}
                    >
                      <Flex justify="space-between" mb={3}>
                        <Heading size="md">{position.pool}</Heading>
                        <Badge colorScheme="green" px={2} py={1}>
                          {position.apy} APY
                        </Badge>
                      </Flex>

                      <SimpleGrid columns={2} spacing={4} mb={4}>
                        <Box>
                          <Text color={subTextColor}>My Liquidity</Text>
                          <Text fontWeight="bold">{position.liquidity}</Text>
                        </Box>
                        <Box>
                          <Text color={subTextColor}>Pool Share</Text>
                          <Text fontWeight="bold">{position.share}</Text>
                        </Box>
                        <Box>
                          <Text color={subTextColor}>Earnings</Text>
                          <Text fontWeight="bold" color="green.400">
                            {position.earnings}
                          </Text>
                        </Box>
                      </SimpleGrid>

                      <HStack spacing={2} mt={4}>
                        <Button size="sm" variant="outline" flex="1">
                          Add Liquidity
                        </Button>
                        <Button size="sm" colorScheme="brand" flex="1">
                          Manage
                        </Button>
                      </HStack>
                    </Box>
                  ))}
                </SimpleGrid>
              </TabPanel>

              <TabPanel px={0}>
                <VStack spacing={4} align="stretch">
                  {recentTransactions.map((tx, index) => (
                    <Card key={index} bg="gray.700" variant="outline">
                      <CardBody>
                        <Flex justify="space-between" align="center">
                          <HStack>
                            <Badge
                              colorScheme={
                                tx.type === "Add Liquidity"
                                  ? "green"
                                  : tx.type === "Remove Liquidity"
                                    ? "red"
                                    : "blue"
                              }
                              px={2}
                              py={1}
                            >
                              {tx.type}
                            </Badge>
                            <VStack align="start" spacing={0}>
                              <Text fontWeight="bold">{tx.pool}</Text>
                              <Text fontSize="sm" color={subTextColor}>
                                {tx.time}
                              </Text>
                            </VStack>
                          </HStack>
                          <Text>{tx.amount}</Text>
                        </Flex>
                      </CardBody>
                    </Card>
                  ))}
                </VStack>
              </TabPanel>

              <TabPanel px={0}>
                <Box textAlign="center" py={10}>
                  <Icon as={FiActivity} boxSize={12} color="brand.500" mb={4} />
                  <Heading size="md" mb={2}>
                    Rewards Program Coming Soon
                  </Heading>
                  <Text color={subTextColor} maxW="600px" mx="auto">
                    We're working on an exciting rewards program for liquidity
                    providers. Stay tuned for updates!
                  </Text>
                </Box>
              </TabPanel>
            </TabPanels>
          </Tabs>
        </Box>
      ) : (
        <Box
          bg={cardBg}
          p={8}
          borderRadius="lg"
          boxShadow="md"
          border="1px solid"
          borderColor={borderColor}
          textAlign="center"
        >
          <Icon as={FiDollarSign} boxSize={12} color="brand.500" mb={4} />
          <Heading size="lg" mb={4}>
            Connect Your Wallet
          </Heading>
          <Text color={subTextColor} maxW="600px" mx="auto" mb={6}>
            Connect your wallet to view your positions, track your earnings, and
            participate in the Fluxion ecosystem.
          </Text>
          <Button
            size="lg"
            colorScheme="brand"
            bgGradient="linear(to-r, brand.500, accent.500)"
            _hover={{
              bgGradient: "linear(to-r, brand.600, accent.600)",
              transform: "translateY(-2px)",
              boxShadow: "lg",
            }}
          >
            Connect Wallet
          </Button>
        </Box>
      )}
    </Box>
  );
};

export default Dashboard;
