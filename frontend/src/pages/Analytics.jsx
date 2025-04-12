import React from 'react';
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
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  useColorModeValue,
  Icon,
  Select,
  Button,
  HStack
} from '@chakra-ui/react';
import { FiBarChart2, FiTrendingUp, FiTrendingDown, FiDollarSign } from 'react-icons/fi';
import { 
  LineChart, 
  Line, 
  AreaChart, 
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from 'recharts';

// Mock data for analytics
const volumeData = [
  { name: 'Jan', volume: 2400 },
  { name: 'Feb', volume: 1398 },
  { name: 'Mar', volume: 9800 },
  { name: 'Apr', volume: 3908 },
  { name: 'May', volume: 4800 },
  { name: 'Jun', volume: 3800 },
  { name: 'Jul', volume: 4300 },
];

const tvlData = [
  { name: 'Jan', tvl: 4000 },
  { name: 'Feb', tvl: 3000 },
  { name: 'Mar', tvl: 2000 },
  { name: 'Apr', tvl: 2780 },
  { name: 'May', tvl: 1890 },
  { name: 'Jun', tvl: 2390 },
  { name: 'Jul', tvl: 3490 },
];

const poolDistributionData = [
  { name: 'ETH-USDC', value: 2400 },
  { name: 'BTC-ETH', value: 1800 },
  { name: 'LINK-ETH', value: 950 },
  { name: 'UNI-USDT', value: 750 },
  { name: 'Others', value: 1200 },
];

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

const Analytics = () => {
  const cardBg = useColorModeValue('gray.700', 'gray.700');
  const borderColor = useColorModeValue('gray.600', 'gray.600');

  return (
    <Box maxW="7xl" mx="auto" pt={5} px={{ base: 2, sm: 12, md: 17 }} className="fade-in">
      <Heading as="h1" mb={6} fontSize="3xl" color="white">
        Analytics
      </Heading>
      
      <SimpleGrid columns={{ base: 1, md: 4 }} spacing={{ base: 5, lg: 8 }} mb={8}>
        <Stat
          px={{ base: 2, md: 4 }}
          py="5"
          shadow="xl"
          border="1px solid"
          borderColor={borderColor}
          rounded="lg"
          bg={cardBg}
          className="slide-up"
        >
          <Flex justifyContent="space-between">
            <Box pl={{ base: 2, md: 4 }}>
              <StatLabel fontWeight="medium" color="gray.300">Total Volume (24h)</StatLabel>
              <StatNumber fontSize="2xl" fontWeight="medium" color="white">$840K</StatNumber>
              <StatHelpText color="green.400">
                <StatArrow type="increase" />
                12.5%
              </StatHelpText>
            </Box>
            <Box my="auto" color="brand.500" alignContent="center">
              <Icon as={FiBarChart2} w={8} h={8} />
            </Box>
          </Flex>
        </Stat>
        
        <Stat
          px={{ base: 2, md: 4 }}
          py="5"
          shadow="xl"
          border="1px solid"
          borderColor={borderColor}
          rounded="lg"
          bg={cardBg}
          className="slide-up"
        >
          <Flex justifyContent="space-between">
            <Box pl={{ base: 2, md: 4 }}>
              <StatLabel fontWeight="medium" color="gray.300">TVL</StatLabel>
              <StatNumber fontSize="2xl" fontWeight="medium" color="white">$5.9M</StatNumber>
              <StatHelpText color="green.400">
                <StatArrow type="increase" />
                23.36%
              </StatHelpText>
            </Box>
            <Box my="auto" color="brand.500" alignContent="center">
              <Icon as={FiDollarSign} w={8} h={8} />
            </Box>
          </Flex>
        </Stat>
        
        <Stat
          px={{ base: 2, md: 4 }}
          py="5"
          shadow="xl"
          border="1px solid"
          borderColor={borderColor}
          rounded="lg"
          bg={cardBg}
          className="slide-up"
        >
          <Flex justifyContent="space-between">
            <Box pl={{ base: 2, md: 4 }}>
              <StatLabel fontWeight="medium" color="gray.300">Highest APY</StatLabel>
              <StatNumber fontSize="2xl" fontWeight="medium" color="white">8.2%</StatNumber>
              <StatHelpText color="green.400">
                <StatArrow type="increase" />
                0.5%
              </StatHelpText>
            </Box>
            <Box my="auto" color="brand.500" alignContent="center">
              <Icon as={FiTrendingUp} w={8} h={8} />
            </Box>
          </Flex>
        </Stat>
        
        <Stat
          px={{ base: 2, md: 4 }}
          py="5"
          shadow="xl"
          border="1px solid"
          borderColor={borderColor}
          rounded="lg"
          bg={cardBg}
          className="slide-up"
        >
          <Flex justifyContent="space-between">
            <Box pl={{ base: 2, md: 4 }}>
              <StatLabel fontWeight="medium" color="gray.300">Lowest APY</StatLabel>
              <StatNumber fontSize="2xl" fontWeight="medium" color="white">4.8%</StatNumber>
              <StatHelpText color="red.400">
                <StatArrow type="decrease" />
                0.3%
              </StatHelpText>
            </Box>
            <Box my="auto" color="brand.500" alignContent="center">
              <Icon as={FiTrendingDown} w={8} h={8} />
            </Box>
          </Flex>
        </Stat>
      </SimpleGrid>
      
      <Tabs variant="soft-rounded" colorScheme="blue" mb={8}>
        <TabList mb={4}>
          <Tab color="gray.300" _selected={{ color: 'white', bg: 'brand.500' }}>Volume</Tab>
          <Tab color="gray.300" _selected={{ color: 'white', bg: 'brand.500' }}>TVL</Tab>
          <Tab color="gray.300" _selected={{ color: 'white', bg: 'brand.500' }}>Pool Distribution</Tab>
          <Tab color="gray.300" _selected={{ color: 'white', bg: 'brand.500' }}>Risk Analysis</Tab>
        </TabList>
        
        <TabPanels>
          <TabPanel p={0}>
            <Card bg={cardBg} borderColor={borderColor} borderWidth="1px" borderRadius="lg" className="slide-up">
              <CardBody>
                <Flex justify="space-between" align="center" mb={4}>
                  <Heading size="md" color="white">Trading Volume</Heading>
                  <HStack>
                    <Select 
                      size="sm" 
                      width="150px" 
                      bg="gray.800" 
                      color="white" 
                      borderColor="gray.600"
                      defaultValue="7d"
                    >
                      <option value="24h">Last 24 hours</option>
                      <option value="7d">Last 7 days</option>
                      <option value="30d">Last 30 days</option>
                      <option value="90d">Last 90 days</option>
                    </Select>
                    <Button size="sm" colorScheme="blue">Export</Button>
                  </HStack>
                </Flex>
                <Box height="400px">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={volumeData}
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
                      <Legend />
                      <Bar dataKey="volume" fill="#0080ff" name="Volume" />
                    </BarChart>
                  </ResponsiveContainer>
                </Box>
              </CardBody>
            </Card>
          </TabPanel>
          
          <TabPanel p={0}>
            <Card bg={cardBg} borderColor={borderColor} borderWidth="1px" borderRadius="lg" className="slide-up">
              <CardBody>
                <Flex justify="space-between" align="center" mb={4}>
                  <Heading size="md" color="white">Total Value Locked</Heading>
                  <HStack>
                    <Select 
                      size="sm" 
                      width="150px" 
                      bg="gray.800" 
                      color="white" 
                      borderColor="gray.600"
                      defaultValue="7d"
                    >
                      <option value="24h">Last 24 hours</option>
                      <option value="7d">Last 7 days</option>
                      <option value="30d">Last 30 days</option>
                      <option value="90d">Last 90 days</option>
                    </Select>
                    <Button size="sm" colorScheme="blue">Export</Button>
                  </HStack>
                </Flex>
                <Box height="400px">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart
                      data={tvlData}
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
                      <Legend />
                      <Area type="monotone" dataKey="tvl" stroke="#0080ff" fill="#0080ff" fillOpacity={0.3} name="TVL" />
                    </AreaChart>
                  </ResponsiveContainer>
                </Box>
              </CardBody>
            </Card>
          </TabPanel>
          
          <TabPanel p={0}>
            <Card bg={cardBg} borderColor={borderColor} borderWidth="1px" borderRadius="lg" className="slide-up">
              <CardBody>
                <Flex justify="space-between" align="center" mb={4}>
                  <Heading size="md" color="white">Pool Distribution</Heading>
                  <Button size="sm" colorScheme="blue">Export</Button>
                </Flex>
                <Box height="400px">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={poolDistributionData}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        outerRadius={150}
                        fill="#8884d8"
                        dataKey="value"
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      >
                        {poolDistributionData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#333', 
                          borderColor: '#555',
                          color: 'white'
                        }} 
                      />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </Box>
              </CardBody>
            </Card>
          </TabPanel>
          
          <TabPanel p={0}>
            <Card bg={cardBg} borderColor={borderColor} borderWidth="1px" borderRadius="lg" className="slide-up">
              <CardBody>
                <Flex justify="space-between" align="center" mb={4}>
                  <Heading size="md" color="white">Risk Analysis</Heading>
                  <HStack>
                    <Select 
                      size="sm" 
                      width="150px" 
                      bg="gray.800" 
                      color="white" 
                      borderColor="gray.600"
                      defaultValue="all"
                    >
                      <option value="all">All Pools</option>
                      <option value="low">Low Risk</option>
                      <option value="medium">Medium Risk</option>
                      <option value="high">High Risk</option>
                    </Select>
                    <Button size="sm" colorScheme="blue">Export</Button>
                  </HStack>
                </Flex>
                <Box height="400px">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={[
                        { name: 'ETH-USDC', var: 0.02, cvar: 0.03, sharpe: 1.8 },
                        { name: 'BTC-ETH', var: 0.04, cvar: 0.06, sharpe: 1.5 },
                        { name: 'LINK-ETH', var: 0.05, cvar: 0.08, sharpe: 1.2 },
                        { name: 'UNI-USDT', var: 0.03, cvar: 0.04, sharpe: 1.6 },
                        { name: 'AAVE-WBTC', var: 0.04, cvar: 0.07, sharpe: 1.4 },
                        { name: 'SNX-ETH', var: 0.06, cvar: 0.09, sharpe: 1.1 },
                      ]}
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
                      <Legend />
                      <Line type="monotone" dataKey="var" stroke="#0080ff" name="VaR (95%)" />
                      <Line type="monotone" dataKey="cvar" stroke="#ff8c00" name="CVaR" />
                      <Line type="monotone" dataKey="sharpe" stroke="#00C49F" name="Sharpe Ratio" />
                    </LineChart>
                  </ResponsiveContainer>
                </Box>
              </CardBody>
            </Card>
          </TabPanel>
        </TabPanels>
      </Tabs>
      
      <SimpleGrid columns={{ base: 1, md: 2 }} spacing={8}>
        <Card bg={cardBg} borderColor={borderColor} borderWidth="1px" borderRadius="lg" className="slide-up">
          <CardBody>
            <Heading size="md" mb={4} color="white">Top Performing Pools</Heading>
            <Box>
              {[
                { name: 'SNX-ETH', apy: '8.2%', volume: '$92K', change: '+2.3%' },
                { name: 'LINK-ETH', apy: '7.3%', volume: '$120K', change: '+1.5%' },
                { name: 'UNI-USDT', apy: '6.1%', volume: '$85K', change: '+0.8%' },
              ].map((pool, index) => (
                <Flex 
                  key={index} 
                  justify="space-between" 
                  align="center" 
                  p={3} 
                  borderBottom={index < 2 ? '1px solid' : 'none'} 
                  borderColor="gray.600"
                >
                  <Text color="white" fontWeight="bold">{pool.name}</Text>
                  <HStack spacing={4}>
                    <Text color="gray.300">{pool.volume}</Text>
                    <Text color="green.400" fontWeight="bold">{pool.apy}</Text>
                    <Text color="green.400">{pool.change}</Text>
                  </HStack>
                </Flex>
              ))}
            </Box>

(Content truncated due to size limit. Use line ranges to read in chunks)