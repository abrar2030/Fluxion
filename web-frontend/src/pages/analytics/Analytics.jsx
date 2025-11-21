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
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  Card,
  CardBody,
  Divider,
  Tooltip,
  useDisclosure,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalFooter,
  ModalBody,
  ModalCloseButton,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Progress,
} from "@chakra-ui/react";
import {
  FiTrendingUp,
  FiTrendingDown,
  FiBarChart2,
  FiActivity,
  FiDollarSign,
  FiInfo,
  FiExternalLink,
  FiDownload,
  FiShare2,
} from "react-icons/fi";
import { Link as RouterLink } from "react-router-dom";
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
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

// Mock data for charts
const tvlData = [
  { name: "Jan", tvl: 85 },
  { name: "Feb", tvl: 92 },
  { name: "Mar", tvl: 105 },
  { name: "Apr", tvl: 120 },
  { name: "May", tvl: 132 },
  { name: "Jun", tvl: 142 },
  { name: "Jul", tvl: 138 },
];

const volumeData = [
  { name: "Jan", volume: 12 },
  { name: "Feb", volume: 18 },
  { name: "Mar", volume: 15 },
  { name: "Apr", volume: 21 },
  { name: "May", volume: 24 },
  { name: "Jun", volume: 28 },
  { name: "Jul", volume: 26 },
];

const poolTypeData = [
  { name: "Weighted", value: 65 },
  { name: "Stable", value: 25 },
  { name: "Boosted", value: 10 },
];

const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042"];

const topPools = [
  {
    name: "ETH-USDC",
    tvl: "$42.5M",
    volume: "$8.2M",
    apy: "8.4%",
    trend: "up",
  },
  {
    name: "WBTC-ETH",
    tvl: "$38.7M",
    volume: "$7.5M",
    apy: "7.2%",
    trend: "down",
  },
  {
    name: "USDC-DAI-USDT",
    tvl: "$32.5M",
    volume: "$5.8M",
    apy: "4.5%",
    trend: "up",
  },
  {
    name: "ETH-LINK",
    tvl: "$12.8M",
    volume: "$3.2M",
    apy: "9.7%",
    trend: "up",
  },
];

const Analytics = () => {
  const cardBg = useColorModeValue("gray.800", "gray.700");
  const borderColor = useColorModeValue("gray.700", "gray.600");
  const textColor = useColorModeValue("white", "white");
  const subTextColor = useColorModeValue("gray.400", "gray.400");

  // State for chart modal
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [selectedChart, setSelectedChart] = useState(null);

  // Handle chart expansion
  const handleExpandChart = (chartType) => {
    setSelectedChart(chartType);
    onOpen();
  };

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

        <Heading as="h1" size="xl" mb={4}>
          Analytics Dashboard
        </Heading>
        <Text fontSize="lg" color={subTextColor} maxW="800px">
          Comprehensive analytics and insights for the Fluxion ecosystem.
          Monitor performance, track trends, and make data-driven decisions.
        </Text>
      </Box>

      {/* Stats Overview */}
      <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={6} mb={8}>
        <Stat
          px={6}
          py={5}
          bg={cardBg}
          borderRadius="lg"
          boxShadow="md"
          border="1px solid"
          borderColor={borderColor}
          _hover={{
            transform: "translateY(-5px)",
            transition: "transform 0.3s ease",
            boxShadow: "xl",
          }}
        >
          <StatLabel
            fontWeight="medium"
            isTruncated
            d="flex"
            alignItems="center"
          >
            <Icon as={FiDollarSign} mr={2} color="brand.500" /> Total Value
            Locked
          </StatLabel>
          <StatNumber fontSize="3xl" fontWeight="bold">
            $142.5M
          </StatNumber>
          <StatHelpText>
            <StatArrow type="increase" />
            23.36%
          </StatHelpText>
        </Stat>

        <Stat
          px={6}
          py={5}
          bg={cardBg}
          borderRadius="lg"
          boxShadow="md"
          border="1px solid"
          borderColor={borderColor}
          _hover={{
            transform: "translateY(-5px)",
            transition: "transform 0.3s ease",
            boxShadow: "xl",
          }}
        >
          <StatLabel
            fontWeight="medium"
            isTruncated
            d="flex"
            alignItems="center"
          >
            <Icon as={FiBarChart2} mr={2} color="accent.500" /> 24h Volume
          </StatLabel>
          <StatNumber fontSize="3xl" fontWeight="bold">
            $28.4M
          </StatNumber>
          <StatHelpText>
            <StatArrow type="decrease" />
            5.14%
          </StatHelpText>
        </Stat>

        <Stat
          px={6}
          py={5}
          bg={cardBg}
          borderRadius="lg"
          boxShadow="md"
          border="1px solid"
          borderColor={borderColor}
          _hover={{
            transform: "translateY(-5px)",
            transition: "transform 0.3s ease",
            boxShadow: "xl",
          }}
        >
          <StatLabel
            fontWeight="medium"
            isTruncated
            d="flex"
            alignItems="center"
          >
            <Icon as={FiActivity} mr={2} color="green.400" /> Active Pools
          </StatLabel>
          <StatNumber fontSize="3xl" fontWeight="bold">
            247
          </StatNumber>
          <StatHelpText>
            <StatArrow type="increase" />
            12.05%
          </StatHelpText>
        </Stat>

        <Stat
          px={6}
          py={5}
          bg={cardBg}
          borderRadius="lg"
          boxShadow="md"
          border="1px solid"
          borderColor={borderColor}
          _hover={{
            transform: "translateY(-5px)",
            transition: "transform 0.3s ease",
            boxShadow: "xl",
          }}
        >
          <StatLabel
            fontWeight="medium"
            isTruncated
            d="flex"
            alignItems="center"
          >
            <Icon as={FiTrendingUp} mr={2} color="purple.400" /> Average APY
          </StatLabel>
          <StatNumber fontSize="3xl" fontWeight="bold">
            8.74%
          </StatNumber>
          <StatHelpText>
            <StatArrow type="increase" />
            2.31%
          </StatHelpText>
        </Stat>
      </SimpleGrid>

      {/* Main Analytics Content */}
      <Tabs variant="soft-rounded" colorScheme="brand" mb={8}>
        <TabList mb={6}>
          <Tab>Overview</Tab>
          <Tab>Pools</Tab>
          <Tab>Volume</Tab>
          <Tab>Historical</Tab>
        </TabList>

        <TabPanels>
          <TabPanel px={0}>
            {/* TVL and Volume Charts */}
            <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={8} mb={8}>
              <Box
                bg={cardBg}
                p={6}
                borderRadius="lg"
                boxShadow="md"
                border="1px solid"
                borderColor={borderColor}
                position="relative"
              >
                <Flex justify="space-between" align="center" mb={4}>
                  <Heading size="md">TVL Trend (Millions $)</Heading>
                  <HStack>
                    <Tooltip label="Expand Chart" placement="top">
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => handleExpandChart("tvl")}
                      >
                        <Icon as={FiExternalLink} />
                      </Button>
                    </Tooltip>
                    <Tooltip label="Download Data" placement="top">
                      <Button size="sm" variant="ghost">
                        <Icon as={FiDownload} />
                      </Button>
                    </Tooltip>
                  </HStack>
                </Flex>

                <Box h="300px">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart
                      data={tvlData}
                      margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                    >
                      <defs>
                        <linearGradient
                          id="colorTvl"
                          x1="0"
                          y1="0"
                          x2="0"
                          y2="1"
                        >
                          <stop
                            offset="5%"
                            stopColor="#0080ff"
                            stopOpacity={0.8}
                          />
                          <stop
                            offset="95%"
                            stopColor="#0080ff"
                            stopOpacity={0}
                          />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                      <XAxis dataKey="name" stroke="#888" />
                      <YAxis stroke="#888" />
                      <RechartsTooltip
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

                <HStack mt={4} justify="space-between">
                  <Text color={subTextColor}>Last 6 months</Text>
                  <Badge colorScheme="green" px={2} py={1}>
                    +67.5%
                  </Badge>
                </HStack>
              </Box>

              <Box
                bg={cardBg}
                p={6}
                borderRadius="lg"
                boxShadow="md"
                border="1px solid"
                borderColor={borderColor}
                position="relative"
              >
                <Flex justify="space-between" align="center" mb={4}>
                  <Heading size="md">Volume Trend (Millions $)</Heading>
                  <HStack>
                    <Tooltip label="Expand Chart" placement="top">
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => handleExpandChart("volume")}
                      >
                        <Icon as={FiExternalLink} />
                      </Button>
                    </Tooltip>
                    <Tooltip label="Download Data" placement="top">
                      <Button size="sm" variant="ghost">
                        <Icon as={FiDownload} />
                      </Button>
                    </Tooltip>
                  </HStack>
                </Flex>

                <Box h="300px">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={volumeData}
                      margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                      <XAxis dataKey="name" stroke="#888" />
                      <YAxis stroke="#888" />
                      <RechartsTooltip
                        contentStyle={{
                          backgroundColor: "rgba(23, 25, 35, 0.9)",
                          border: "1px solid #333",
                          borderRadius: "4px",
                          color: "white",
                        }}
                      />
                      <Bar dataKey="volume" fill="#ff8c00" />
                    </BarChart>
                  </ResponsiveContainer>
                </Box>

                <HStack mt={4} justify="space-between">
                  <Text color={subTextColor}>Last 6 months</Text>
                  <Badge colorScheme="green" px={2} py={1}>
                    +116.7%
                  </Badge>
                </HStack>
              </Box>
            </SimpleGrid>

            {/* Pool Distribution and Top Pools */}
            <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={8}>
              <Box
                bg={cardBg}
                p={6}
                borderRadius="lg"
                boxShadow="md"
                border="1px solid"
                borderColor={borderColor}
              >
                <Flex justify="space-between" align="center" mb={4}>
                  <Heading size="md">Pool Type Distribution</Heading>
                  <Tooltip label="Expand Chart" placement="top">
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => handleExpandChart("distribution")}
                    >
                      <Icon as={FiExternalLink} />
                    </Button>
                  </Tooltip>
                </Flex>

                <Box h="300px">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={poolTypeData}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        outerRadius={100}
                        fill="#8884d8"
                        dataKey="value"
                        label={({ name, percent }) =>
                          `${name} ${(percent * 100).toFixed(0)}%`
                        }
                      >
                        {poolTypeData.map((entry, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={COLORS[index % COLORS.length]}
                          />
                        ))}
                      </Pie>
                      <RechartsTooltip
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

                <SimpleGrid columns={3} spacing={4} mt={4}>
                  {poolTypeData.map((type, index) => (
                    <Box key={index}>
                      <HStack>
                        <Box
                          w="3"
                          h="3"
                          borderRadius="full"
                          bg={COLORS[index]}
                        />
                        <Text fontSize="sm">{type.name}</Text>
                      </HStack>
                      <Text fontWeight="bold">{type.value}%</Text>
                    </Box>
                  ))}
                </SimpleGrid>
              </Box>

              <Box
                bg={cardBg}
                p={6}
                borderRadius="lg"
                boxShadow="md"
                border="1px solid"
                borderColor={borderColor}
              >
                <Flex justify="space-between" align="center" mb={4}>
                  <Heading size="md">Top Performing Pools</Heading>
                  <Button
                    as={RouterLink}
                    to="/pools"
                    size="sm"
                    variant="outline"
                    rightIcon={<FiExternalLink />}
                  >
                    View All
                  </Button>
                </Flex>

                <Table variant="simple" size="sm">
                  <Thead>
                    <Tr>
                      <Th>Pool</Th>
                      <Th isNumeric>TVL</Th>
                      <Th isNumeric>Volume</Th>
                      <Th isNumeric>APY</Th>
                    </Tr>
                  </Thead>
                  <Tbody>
                    {topPools.map((pool, index) => (
                      <Tr key={index}>
                        <Td fontWeight="medium">{pool.name}</Td>
                        <Td isNumeric>{pool.tvl}</Td>
                        <Td isNumeric>{pool.volume}</Td>
                        <Td isNumeric>
                          <HStack justify="flex-end">
                            <Text
                              fontWeight="bold"
                              color={
                                pool.trend === "up" ? "green.400" : "red.400"
                              }
                            >
                              {pool.apy}
                            </Text>
                            <Icon
                              as={
                                pool.trend === "up"
                                  ? FiTrendingUp
                                  : FiTrendingDown
                              }
                              color={
                                pool.trend === "up" ? "green.400" : "red.400"
                              }
                            />
                          </HStack>
                        </Td>
                      </Tr>
                    ))}
                  </Tbody>
                </Table>

                <Divider my={4} />

                <SimpleGrid columns={2} spacing={4}>
                  <Card bg="gray.700" variant="outline">
                    <CardBody py={3}>
                      <Text fontSize="sm" color={subTextColor}>
                        Highest APY
                      </Text>
                      <Text fontWeight="bold" fontSize="lg">
                        ETH-LINK
                      </Text>
                      <HStack>
                        <Text color="green.400" fontWeight="bold">
                          9.7%
                        </Text>
                        <Icon as={FiTrendingUp} color="green.400" />
                      </HStack>
                    </CardBody>
                  </Card>

                  <Card bg="gray.700" variant="outline">
                    <CardBody py={3}>
                      <Text fontSize="sm" color={subTextColor}>
                        Highest Volume
                      </Text>
                      <Text fontWeight="bold" fontSize="lg">
                        ETH-USDC
                      </Text>
                      <Text fontWeight="bold">$8.2M (24h)</Text>
                    </CardBody>
                  </Card>
                </SimpleGrid>
              </Box>
            </SimpleGrid>
          </TabPanel>

          <TabPanel px={0}>
            {/* Pools Tab Content */}
            <Text>Detailed pool analytics would be shown here.</Text>
          </TabPanel>

          <TabPanel px={0}>
            {/* Volume Tab Content */}
            <Text>Detailed volume analytics would be shown here.</Text>
          </TabPanel>

          <TabPanel px={0}>
            {/* Historical Tab Content */}
            <Text>Historical data and trends would be shown here.</Text>
          </TabPanel>
        </TabPanels>
      </Tabs>

      {/* Chart Modal */}
      <Modal isOpen={isOpen} onClose={onClose} size="4xl">
        <ModalOverlay backdropFilter="blur(10px)" />
        <ModalContent bg="gray.800" color="white">
          <ModalHeader>
            {selectedChart === "tvl" && "TVL Trend (Millions $)"}
            {selectedChart === "volume" && "Volume Trend (Millions $)"}
            {selectedChart === "distribution" && "Pool Type Distribution"}
          </ModalHeader>
          <ModalCloseButton />
          <ModalBody pb={6}>
            <Box h="500px">
              {selectedChart === "tvl" && (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart
                    data={tvlData}
                    margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                  >
                    <defs>
                      <linearGradient
                        id="colorTvlModal"
                        x1="0"
                        y1="0"
                        x2="0"
                        y2="1"
                      >
                        <stop
                          offset="5%"
                          stopColor="#0080ff"
                          stopOpacity={0.8}
                        />
                        <stop
                          offset="95%"
                          stopColor="#0080ff"
                          stopOpacity={0}
                        />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                    <XAxis dataKey="name" stroke="#888" />
                    <YAxis stroke="#888" />
                    <RechartsTooltip
                      contentStyle={{
                        backgroundColor: "rgba(23, 25, 35, 0.9)",
                        border: "1px solid #333",
                        borderRadius: "4px",
                        color: "white",
                      }}
                    />
                    <Legend />
                    <Area
                      type="monotone"
                      dataKey="tvl"
                      stroke="#0080ff"
                      fillOpacity={1}
                      fill="url(#colorTvlModal)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              )}

              {selectedChart === "volume" && (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={volumeData}
                    margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                    <XAxis dataKey="name" stroke="#888" />
                    <YAxis stroke="#888" />
                    <RechartsTooltip
                      contentStyle={{
                        backgroundColor: "rgba(23, 25, 35, 0.9)",
                        border: "1px solid #333",
                        borderRadius: "4px",
                        color: "white",
                      }}
                    />
                    <Legend />
                    <Bar dataKey="volume" fill="#ff8c00" />
                  </BarChart>
                </ResponsiveContainer>
              )}

              {selectedChart === "distribution" && (
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={poolTypeData}
                      cx="50%"
                      cy="50%"
                      labelLine={true}
                      outerRadius={180}
                      fill="#8884d8"
                      dataKey="value"
                      label={({ name, percent }) =>
                        `${name} ${(percent * 100).toFixed(0)}%`
                      }
                    >
                      {poolTypeData.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={COLORS[index % COLORS.length]}
                        />
                      ))}
                    </Pie>
                    <RechartsTooltip
                      contentStyle={{
                        backgroundColor: "rgba(23, 25, 35, 0.9)",
                        border: "1px solid #333",
                        borderRadius: "4px",
                        color: "white",
                      }}
                    />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              )}
            </Box>
          </ModalBody>

          <ModalFooter>
            <Button leftIcon={<FiDownload />} colorScheme="brand" mr={3}>
              Download Data
            </Button>
            <Button leftIcon={<FiShare2 />} variant="outline">
              Share
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </Box>
  );
};

export default Analytics;
