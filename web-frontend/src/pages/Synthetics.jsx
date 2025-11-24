import React, { useState } from 'react';
import {
    Box,
    Heading,
    SimpleGrid,
    Card,
    CardBody,
    Text,
    Button,
    Flex,
    HStack,
    VStack,
    Input,
    FormControl,
    FormLabel,
    Select,
    NumberInput,
    NumberInputField,
    NumberInputStepper,
    NumberIncrementStepper,
    NumberDecrementStepper,
    Divider,
    Badge,
    Icon,
    Table,
    Thead,
    Tbody,
    Tr,
    Th,
    Td,
    useColorModeValue,
    useToast,
} from '@chakra-ui/react';
import { FiPlus, FiExternalLink, FiAlertCircle, FiCheckCircle } from 'react-icons/fi';

// Mock data for synthetic assets
const syntheticAssets = [
    {
        id: 'sUSD',
        name: 'Synthetic USD',
        price: '$1.00',
        oracle: 'Chainlink',
        collateral: '150%',
        status: 'active',
    },
    {
        id: 'sBTC',
        name: 'Synthetic Bitcoin',
        price: '$63,245.78',
        oracle: 'Chainlink',
        collateral: '175%',
        status: 'active',
    },
    {
        id: 'sETH',
        name: 'Synthetic Ethereum',
        price: '$3,125.42',
        oracle: 'Chainlink',
        collateral: '165%',
        status: 'active',
    },
    {
        id: 'sGOLD',
        name: 'Synthetic Gold',
        price: '$2,345.67',
        oracle: 'Band Protocol',
        collateral: '180%',
        status: 'active',
    },
    {
        id: 'sTSLA',
        name: 'Synthetic Tesla',
        price: '$187.32',
        oracle: 'API3',
        collateral: '200%',
        status: 'inactive',
    },
];

const Synthetics = () => {
    const toast = useToast();
    const [activeTab, setActiveTab] = useState('explore');
    const [newAsset, setNewAsset] = useState({
        id: '',
        name: '',
        oracle: 'Chainlink',
        collateral: 150,
    });

    const cardBg = useColorModeValue('gray.700', 'gray.700');
    const borderColor = useColorModeValue('gray.600', 'gray.600');

    const handleCreateAsset = () => {
        if (!newAsset.id || !newAsset.name) {
            toast({
                title: 'Missing information',
                description: 'Please fill in all required fields',
                status: 'error',
                duration: 3000,
                isClosable: true,
            });
            return;
        }

        if (syntheticAssets.some((asset) => asset.id === newAsset.id)) {
            toast({
                title: 'Duplicate asset ID',
                description: 'An asset with this ID already exists',
                status: 'error',
                duration: 3000,
                isClosable: true,
            });
            return;
        }

        toast({
            title: 'Asset created',
            description: `Synthetic asset ${newAsset.name} has been created`,
            status: 'success',
            duration: 3000,
            isClosable: true,
        });

        setNewAsset({ id: '', name: '', oracle: 'Chainlink', collateral: 150 });
        setActiveTab('explore');
    };

    return (
        <Box maxW="7xl" mx="auto" pt={5} px={{ base: 2, sm: 12, md: 17 }} className="fade-in">
            <Heading as="h1" mb={6} fontSize="3xl" color="white">
                Synthetic Assets
            </Heading>

            <HStack spacing={4} mb={8}>
                <Button
                    colorScheme={activeTab === 'explore' ? 'blue' : 'gray'}
                    onClick={() => setActiveTab('explore')}
                    size="lg"
                >
                    Explore Assets
                </Button>
                <Button
                    colorScheme={activeTab === 'create' ? 'blue' : 'gray'}
                    onClick={() => setActiveTab('create')}
                    size="lg"
                    leftIcon={<FiPlus />}
                >
                    Create Asset
                </Button>
            </HStack>

            {activeTab === 'explore' ? (
                <Box>
                    {/* Assets Table */}
                    <Card
                        bg={cardBg}
                        borderColor={borderColor}
                        borderWidth="1px"
                        borderRadius="lg"
                        mb={8}
                        className="slide-up"
                    >
                        <CardBody>
                            <Heading size="md" mb={4} color="white">
                                Available Synthetic Assets
                            </Heading>
                            <Box overflowX="auto">
                                <Table variant="simple" colorScheme="whiteAlpha">
                                    <Thead>
                                        <Tr>
                                            <Th>Asset ID</Th>
                                            <Th>Name</Th>
                                            <Th>Price</Th>
                                            <Th>Oracle</Th>
                                            <Th>Collateral Ratio</Th>
                                            <Th>Status</Th>
                                            <Th>Actions</Th>
                                        </Tr>
                                    </Thead>
                                    <Tbody>
                                        {syntheticAssets.map((asset) => (
                                            <Tr key={asset.id}>
                                                <Td fontWeight="bold" color="white">
                                                    {asset.id}
                                                </Td>
                                                <Td>{asset.name}</Td>
                                                <Td>{asset.price}</Td>
                                                <Td>{asset.oracle}</Td>
                                                <Td>{asset.collateral}</Td>
                                                <Td>
                                                    <Badge
                                                        colorScheme={
                                                            asset.status === 'active'
                                                                ? 'green'
                                                                : 'red'
                                                        }
                                                    >
                                                        {asset.status === 'active'
                                                            ? 'Active'
                                                            : 'Inactive'}
                                                    </Badge>
                                                </Td>
                                                <Td>
                                                    <HStack spacing={2}>
                                                        <Button
                                                            size="sm"
                                                            colorScheme="blue"
                                                            variant="outline"
                                                        >
                                                            Trade
                                                        </Button>
                                                        <Button
                                                            size="sm"
                                                            colorScheme="gray"
                                                            variant="outline"
                                                        >
                                                            Details
                                                        </Button>
                                                    </HStack>
                                                </Td>
                                            </Tr>
                                        ))}
                                    </Tbody>
                                </Table>
                            </Box>
                        </CardBody>
                    </Card>

                    {/* Market & Oracle Overview */}
                    <SimpleGrid columns={{ base: 1, md: 2 }} spacing={8}>
                        <Card
                            bg={cardBg}
                            borderColor={borderColor}
                            borderWidth="1px"
                            borderRadius="lg"
                            className="slide-up"
                        >
                            <CardBody>
                                <Heading size="md" mb={4} color="white">
                                    Market Overview
                                </Heading>
                                <VStack spacing={4} align="stretch">
                                    {[
                                        ['Total Synthetic Assets', '5'],
                                        ['Active Assets', '4'],
                                        ['Total Market Cap', '$12.4M'],
                                        ['24h Trading Volume', '$3.2M'],
                                        ['Total Collateral Locked', '$18.7M'],
                                    ].map(([label, value], idx) => (
                                        <Flex key={idx} justify="space-between" align="center">
                                            <Text color="gray.300">{label}:</Text>
                                            <Text color="white" fontWeight="bold">
                                                {value}
                                            </Text>
                                        </Flex>
                                    ))}
                                </VStack>
                            </CardBody>
                        </Card>

                        <Card
                            bg={cardBg}
                            borderColor={borderColor}
                            borderWidth="1px"
                            borderRadius="lg"
                            className="slide-up"
                        >
                            <CardBody>
                                <Heading size="md" mb={4} color="white">
                                    Oracle Providers
                                </Heading>
                                <VStack spacing={4} align="stretch">
                                    {[
                                        {
                                            name: 'Chainlink',
                                            status: 'Connected',
                                            lastUpdate: '2 min ago',
                                        },
                                        {
                                            name: 'Band Protocol',
                                            status: 'Connected',
                                            lastUpdate: '5 min ago',
                                        },
                                        {
                                            name: 'API3',
                                            status: 'Connected',
                                            lastUpdate: '8 min ago',
                                        },
                                        {
                                            name: 'Tellor',
                                            status: 'Not Connected',
                                            lastUpdate: 'N/A',
                                        },
                                    ].map((oracle, idx) => (
                                        <Flex
                                            key={idx}
                                            justify="space-between"
                                            align="center"
                                            p={3}
                                            borderRadius="md"
                                            bg="gray.800"
                                        >
                                            <HStack>
                                                <Icon
                                                    as={
                                                        oracle.status === 'Connected'
                                                            ? FiCheckCircle
                                                            : FiAlertCircle
                                                    }
                                                    color={
                                                        oracle.status === 'Connected'
                                                            ? 'green.400'
                                                            : 'red.400'
                                                    }
                                                />
                                                <Text color="white">{oracle.name}</Text>
                                            </HStack>
                                            <HStack>
                                                <Badge
                                                    colorScheme={
                                                        oracle.status === 'Connected'
                                                            ? 'green'
                                                            : 'red'
                                                    }
                                                >
                                                    {oracle.status}
                                                </Badge>
                                                <Text color="gray.400" fontSize="sm">
                                                    {oracle.lastUpdate}
                                                </Text>
                                            </HStack>
                                        </Flex>
                                    ))}
                                </VStack>
                            </CardBody>
                        </Card>
                    </SimpleGrid>
                </Box>
            ) : (
                <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={8}>
                    {/* Create Asset Form */}
                    <VStack spacing={6} align="stretch">
                        <Card
                            bg={cardBg}
                            borderColor={borderColor}
                            borderWidth="1px"
                            borderRadius="lg"
                            className="slide-up"
                        >
                            <CardBody>
                                <Heading size="md" mb={4} color="white">
                                    Create Synthetic Asset
                                </Heading>
                                <FormControl mb={4}>
                                    <FormLabel color="gray.300">Asset ID</FormLabel>
                                    <Input
                                        placeholder="e.g., sGOLD"
                                        value={newAsset.id}
                                        onChange={(e) =>
                                            setNewAsset({ ...newAsset, id: e.target.value })
                                        }
                                        bg="gray.800"
                                        color="white"
                                        borderColor="gray.600"
                                    />
                                </FormControl>
                                <FormControl mb={4}>
                                    <FormLabel color="gray.300">Asset Name</FormLabel>
                                    <Input
                                        placeholder="e.g., Synthetic Gold"
                                        value={newAsset.name}
                                        onChange={(e) =>
                                            setNewAsset({ ...newAsset, name: e.target.value })
                                        }
                                        bg="gray.800"
                                        color="white"
                                        borderColor="gray.600"
                                    />
                                </FormControl>
                                <FormControl mb={4}>
                                    <FormLabel color="gray.300">Oracle Provider</FormLabel>
                                    <Select
                                        value={newAsset.oracle}
                                        onChange={(e) =>
                                            setNewAsset({ ...newAsset, oracle: e.target.value })
                                        }
                                        bg="gray.800"
                                        color="white"
                                        borderColor="gray.600"
                                    >
                                        <option value="Chainlink">Chainlink</option>
                                        <option value="Band Protocol">Band Protocol</option>
                                        <option value="API3">API3</option>
                                        <option value="Tellor">Tellor</option>
                                    </Select>
                                </FormControl>
                                <FormControl mb={4}>
                                    <FormLabel color="gray.300">Collateral Ratio (%)</FormLabel>
                                    <NumberInput
                                        value={newAsset.collateral}
                                        min={120}
                                        max={300}
                                        onChange={(v) =>
                                            setNewAsset({ ...newAsset, collateral: parseInt(v) })
                                        }
                                    >
                                        <NumberInputField
                                            bg="gray.800"
                                            color="white"
                                            borderColor="gray.600"
                                        />
                                        <NumberInputStepper>
                                            <NumberIncrementStepper color="gray.400" />
                                            <NumberDecrementStepper color="gray.400" />
                                        </NumberInputStepper>
                                    </NumberInput>
                                </FormControl>
                                <Button
                                    colorScheme="blue"
                                    size="lg"
                                    onClick={handleCreateAsset}
                                    isFullWidth
                                    mt={4}
                                >
                                    Create Synthetic Asset
                                </Button>
                            </CardBody>
                        </Card>
                    </VStack>

                    {/* Asset Preview */}
                    <VStack spacing={6} align="stretch">
                        <Card
                            bg={cardBg}
                            borderColor={borderColor}
                            borderWidth="1px"
                            borderRadius="lg"
                            className="slide-up"
                        >
                            <CardBody>
                                <Heading size="md" mb={4} color="white">
                                    Asset Preview
                                </Heading>
                                <Box
                                    p={4}
                                    borderRadius="md"
                                    bg="gray.800"
                                    borderWidth="1px"
                                    borderColor="gray.600"
                                    mb={4}
                                >
                                    <Flex justify="space-between" align="center" mb={2}>
                                        <Heading size="md" color="white">
                                            {newAsset.id || 'sXXX'}
                                        </Heading>
                                        <Badge colorScheme="green">Preview</Badge>
                                    </Flex>
                                    <Text color="gray.300" mb={3}>
                                        {newAsset.name || 'Synthetic Asset Name'}
                                    </Text>
                                    <Divider mb={3} borderColor="gray.600" />
                                    <VStack spacing={2} align="stretch">
                                        <Flex justify="space-between">
                                            <Text color="gray.400">Oracle:</Text>
                                            <Text color="white">{newAsset.oracle}</Text>
                                        </Flex>
                                        <Flex justify="space-between">
                                            <Text color="gray.400">Collateral Ratio:</Text>
                                            <Text color="white">{newAsset.collateral}%</Text>
                                        </Flex>
                                        <Flex justify="space-between">
                                            <Text color="gray.400">Status:</Text>
                                            <Badge colorScheme="yellow">Pending</Badge>
                                        </Flex>
                                    </VStack>
                                </Box>

                                <Heading size="sm" mb={3} color="white">
                                    Risk Assessment
                                </Heading>
                                <Box
                                    p={4}
                                    borderRadius="md"
                                    bg="gray.800"
                                    borderWidth="1px"
                                    borderColor="gray.600"
                                >
                                    <VStack spacing={3} align="stretch">
                                        <Flex>
                                            <Icon
                                                as={FiAlertCircle}
                                                color="yellow.400"
                                                mt={1}
                                                mr={2}
                                            />
                                            <Text color="gray.300">
                                                <Text as="span" color="white" fontWeight="medium">
                                                    Collateral Risk:
                                                </Text>{' '}
                                                {newAsset.collateral >= 180
                                                    ? 'Low'
                                                    : newAsset.collateral >= 150
                                                      ? 'Medium'
                                                      : 'High'}
                                            </Text>
                                        </Flex>
                                        <Flex>
                                            <Icon
                                                as={FiAlertCircle}
                                                color="green.400"
                                                mt={1}
                                                mr={2}
                                            />
                                            <Text color="gray.300">
                                                <Text as="span" color="white" fontWeight="medium">
                                                    Oracle Risk:
                                                </Text>{' '}
                                                {newAsset.oracle === 'Chainlink' ? 'Low' : 'Medium'}
                                            </Text>
                                        </Flex>
                                        <Flex>
                                            <Icon
                                                as={FiExternalLink}
                                                color="blue.400"
                                                mt={1}
                                                mr={2}
                                            />
                                            <Text color="gray.300">
                                                <Text as="span" color="white" fontWeight="medium">
                                                    Market Liquidity:
                                                </Text>{' '}
                                                To be determined after launch
                                            </Text>
                                        </Flex>
                                    </VStack>
                                </Box>
                            </CardBody>
                        </Card>
                    </VStack>
                </SimpleGrid>
            )}
        </Box>
    );
};

export default Synthetics;
