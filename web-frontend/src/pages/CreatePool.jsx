import React, { useState, useEffect } from 'react';
import {
    Box,
    Heading,
    FormControl,
    FormLabel,
    Input,
    Select,
    Button,
    VStack,
    HStack,
    SimpleGrid,
    Slider,
    SliderTrack,
    SliderFilledTrack,
    SliderThumb,
    Text,
    Flex,
    NumberInput,
    NumberInputField,
    NumberInputStepper,
    NumberIncrementStepper,
    NumberDecrementStepper,
    Card,
    CardBody,
    Divider,
    Badge,
    Icon,
    useColorModeValue,
    useToast,
} from '@chakra-ui/react';
import { FiPlus, FiMinus, FiInfo, FiAlertCircle } from 'react-icons/fi';
import { useWeb3 } from '../lib/web3-config.jsx';
import { useUI } from '../lib/ui-context';
import { useNavigate } from 'react-router-dom';

// Mock token data
const availableTokens = [
    { symbol: 'ETH', name: 'Ethereum', balance: '2.5' },
    { symbol: 'USDC', name: 'USD Coin', balance: '5000' },
    { symbol: 'WBTC', name: 'Wrapped Bitcoin', balance: '0.15' },
    { symbol: 'DAI', name: 'Dai Stablecoin', balance: '3500' },
    { symbol: 'LINK', name: 'Chainlink', balance: '120' },
    { symbol: 'UNI', name: 'Uniswap', balance: '200' },
    { symbol: 'AAVE', name: 'Aave', balance: '45' },
    { symbol: 'SNX', name: 'Synthetix', balance: '150' },
];

const CreatePool = () => {
    const navigate = useNavigate();
    const { isConnected, createPool } = useWeb3();
    const { addNotification, setLoadingState } = useUI();
    const toast = useToast();

    const [poolAssets, setPoolAssets] = useState([
        { token: '', weight: 50 },
        { token: '', weight: 50 },
    ]);
    const [fee, setFee] = useState(0.3);
    const [amplification, setAmplification] = useState(100);
    const [isSubmitting, setIsSubmitting] = useState(false);

    const cardBg = useColorModeValue('gray.700', 'gray.700');
    const borderColor = useColorModeValue('gray.600', 'gray.600');

    useEffect(() => {
        // Check if user is connected
        if (!isConnected) {
            addNotification({
                title: 'Wallet Required',
                message: 'Please connect your wallet to create a pool',
                type: 'warning',
            });
        }
    }, [isConnected]);

    const handleAddAsset = () => {
        if (poolAssets.length < 8) {
            const newAssets = [...poolAssets];
            const equalWeight = Math.floor(100 / (poolAssets.length + 1));

            // Redistribute weights
            const newPoolAssets = newAssets.map((asset) => ({
                ...asset,
                weight: equalWeight,
            }));

            newPoolAssets.push({ token: '', weight: equalWeight });

            // Adjust to ensure sum is 100
            const sum = newPoolAssets.reduce((acc, asset) => acc + asset.weight, 0);
            if (sum < 100) {
                newPoolAssets[0].weight += 100 - sum;
            }

            setPoolAssets(newPoolAssets);
        } else {
            toast({
                title: 'Maximum assets reached',
                description: 'You can add up to 8 assets in a pool',
                status: 'warning',
                duration: 3000,
                isClosable: true,
            });
        }
    };

    const handleRemoveAsset = (index) => {
        if (poolAssets.length > 2) {
            const newAssets = [...poolAssets];
            newAssets.splice(index, 1);

            // Redistribute weights
            const equalWeight = Math.floor(100 / newAssets.length);
            const newPoolAssets = newAssets.map((asset) => ({
                ...asset,
                weight: equalWeight,
            }));

            // Adjust to ensure sum is 100
            const sum = newPoolAssets.reduce((acc, asset) => acc + asset.weight, 0);
            if (sum < 100) {
                newPoolAssets[0].weight += 100 - sum;
            }

            setPoolAssets(newPoolAssets);
        } else {
            toast({
                title: 'Minimum assets required',
                description: 'A pool must have at least 2 assets',
                status: 'warning',
                duration: 3000,
                isClosable: true,
            });
        }
    };

    const handleTokenChange = (index, value) => {
        const newAssets = [...poolAssets];
        newAssets[index].token = value;
        setPoolAssets(newAssets);
    };

    const handleWeightChange = (index, value) => {
        const newAssets = [...poolAssets];

        // Calculate the difference
        const oldWeight = newAssets[index].weight;
        const diff = value - oldWeight;

        // Find another asset to adjust
        let adjustIndex = index === 0 ? 1 : 0;

        // Make sure the adjustment doesn't make any weight negative
        if (newAssets[adjustIndex].weight - diff < 1) {
            toast({
                title: 'Invalid weight distribution',
                description: 'Cannot reduce weight below 1%',
                status: 'error',
                duration: 3000,
                isClosable: true,
            });
            return;
        }

        // Update weights
        newAssets[index].weight = value;
        newAssets[adjustIndex].weight -= diff;

        setPoolAssets(newAssets);
    };

    const handleCreatePool = async () => {
        // Validate inputs
        const hasEmptyToken = poolAssets.some((asset) => !asset.token);
        if (hasEmptyToken) {
            toast({
                title: 'Missing token selection',
                description: 'Please select tokens for all assets',
                status: 'error',
                duration: 3000,
                isClosable: true,
            });
            return;
        }

        // Check for duplicate tokens
        const tokens = poolAssets.map((asset) => asset.token);
        const uniqueTokens = new Set(tokens);
        if (uniqueTokens.size !== tokens.length) {
            toast({
                title: 'Duplicate tokens',
                description: 'Each token can only be used once in a pool',
                status: 'error',
                duration: 3000,
                isClosable: true,
            });
            return;
        }

        if (!isConnected) {
            addNotification({
                title: 'Wallet Required',
                message: 'Please connect your wallet to create a pool',
                type: 'warning',
            });
            return;
        }

        try {
            setIsSubmitting(true);
            setLoadingState('createPool', true);

            // Prepare pool config
            const poolConfig = {
                assets: poolAssets.map((asset) => asset.token),
                weights: poolAssets.map((asset) => asset.weight),
                fee,
                amplification,
            };

            // Call the createPool function from web3 context
            const newPool = await createPool(poolConfig);

            addNotification({
                title: 'Pool Created Successfully',
                message: `Your liquidity pool has been created with ID: ${newPool.id}`,
                type: 'success',
                duration: 5000,
            });

            // Reset form
            setPoolAssets([
                { token: '', weight: 50 },
                { token: '', weight: 50 },
            ]);
            setFee(0.3);
            setAmplification(100);

            // Navigate to pools page
            setTimeout(() => {
                navigate('/pools');
            }, 2000);
        } catch (error) {
            addNotification({
                title: 'Pool Creation Failed',
                message: error.message || 'Failed to create pool. Please try again.',
                type: 'error',
                duration: 5000,
            });
        } finally {
            setIsSubmitting(false);
            setLoadingState('createPool', false);
        }
    };

    return (
        <Box maxW="7xl" mx="auto" pt={5} px={{ base: 2, sm: 12, md: 17 }} className="fade-in">
            <Heading as="h1" mb={6} fontSize="3xl" color="white">
                Create Liquidity Pool
            </Heading>

            <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={8}>
                <VStack spacing={6} align="stretch">
                    <Card bg={cardBg} borderColor={borderColor} borderWidth="1px" borderRadius="lg">
                        <CardBody>
                            <Heading size="md" mb={4} color="white">
                                Pool Assets
                            </Heading>

                            {poolAssets.map((asset, index) => (
                                <Box key={index} mb={4}>
                                    <Flex justify="space-between" align="center" mb={2}>
                                        <Heading size="sm" color="white">
                                            Asset {index + 1}
                                        </Heading>
                                        {index > 1 && (
                                            <Button
                                                size="sm"
                                                colorScheme="red"
                                                variant="ghost"
                                                onClick={() => handleRemoveAsset(index)}
                                            >
                                                <Icon as={FiMinus} />
                                            </Button>
                                        )}
                                    </Flex>

                                    <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
                                        <FormControl>
                                            <FormLabel color="gray.300">Token</FormLabel>
                                            <Select
                                                placeholder="Select token"
                                                value={asset.token}
                                                onChange={(e) =>
                                                    handleTokenChange(index, e.target.value)
                                                }
                                                bg="gray.800"
                                                color="white"
                                                borderColor="gray.600"
                                            >
                                                {availableTokens.map((token) => (
                                                    <option key={token.symbol} value={token.symbol}>
                                                        {token.symbol} - {token.name}
                                                    </option>
                                                ))}
                                            </Select>
                                        </FormControl>

                                        <FormControl>
                                            <FormLabel color="gray.300">Weight (%)</FormLabel>
                                            <NumberInput
                                                value={asset.weight}
                                                min={1}
                                                max={99}
                                                onChange={(valueString) =>
                                                    handleWeightChange(index, parseInt(valueString))
                                                }
                                                bg="gray.800"
                                                color="white"
                                                borderColor="gray.600"
                                            >
                                                <NumberInputField />
                                                <NumberInputStepper>
                                                    <NumberIncrementStepper color="gray.400" />
                                                    <NumberDecrementStepper color="gray.400" />
                                                </NumberInputStepper>
                                            </NumberInput>
                                        </FormControl>
                                    </SimpleGrid>
                                </Box>
                            ))}

                            <Button
                                leftIcon={<FiPlus />}
                                colorScheme="blue"
                                variant="outline"
                                onClick={handleAddAsset}
                                mt={2}
                                isFullWidth
                            >
                                Add Asset
                            </Button>
                        </CardBody>
                    </Card>

                    <Card bg={cardBg} borderColor={borderColor} borderWidth="1px" borderRadius="lg">
                        <CardBody>
                            <Heading size="md" mb={4} color="white">
                                Pool Parameters
                            </Heading>

                            <FormControl mb={4}>
                                <FormLabel color="gray.300">Swap Fee (%)</FormLabel>
                                <HStack spacing={4}>
                                    <Slider
                                        value={fee * 100}
                                        min={0.01}
                                        max={1}
                                        step={0.01}
                                        onChange={(v) => setFee(v / 100)}
                                        flex="1"
                                        colorScheme="blue"
                                    >
                                        <SliderTrack>
                                            <SliderFilledTrack />
                                        </SliderTrack>
                                        <SliderThumb boxSize={6} />
                                    </Slider>
                                    <Text
                                        color="white"
                                        fontWeight="bold"
                                        w="60px"
                                        textAlign="right"
                                    >
                                        {fee.toFixed(2)}%
                                    </Text>
                                </HStack>
                            </FormControl>

                            <FormControl>
                                <FormLabel color="gray.300">
                                    <Flex align="center">
                                        Amplification Factor
                                        <Icon as={FiInfo} ml={1} color="gray.400" />
                                    </Flex>
                                </FormLabel>
                                <HStack spacing={4}>
                                    <Slider
                                        value={amplification}
                                        min={1}
                                        max={500}
                                        step={1}
                                        onChange={(v) => setAmplification(v)}
                                        flex="1"
                                        colorScheme="blue"
                                    >
                                        <SliderTrack>
                                            <SliderFilledTrack />
                                        </SliderTrack>
                                        <SliderThumb boxSize={6} />
                                    </Slider>
                                    <Text
                                        color="white"
                                        fontWeight="bold"
                                        w="60px"
                                        textAlign="right"
                                    >
                                        {amplification}
                                    </Text>
                                </HStack>
                                <Text color="gray.400" fontSize="sm" mt={1}>
                                    Higher values increase price stability for stablecoins (20-100
                                    recommended for volatile assets)
                                </Text>
                            </FormControl>
                        </CardBody>
                    </Card>

                    <Button
                        colorScheme="blue"
                        size="lg"
                        onClick={handleCreatePool}
                        isLoading={isSubmitting}
                        loadingText="Creating Pool"
                        isDisabled={!isConnected}
                    >
                        Create Pool
                    </Button>
                </VStack>

                <VStack spacing={6} align="stretch">
                    <Card bg={cardBg} borderColor={borderColor} borderWidth="1px" borderRadius="lg">
                        <CardBody>
                            <Heading size="md" mb={4} color="white">
                                Pool Preview
                            </Heading>

                            <Box mb={4}>
                                <Text color="gray.300" mb={1}>
                                    Asset Distribution
                                </Text>
                                {poolAssets.map((asset, index) =>
                                    asset.token ? (
                                        <Flex key={index} align="center" mb={2}>
                                            <Box
                                                w="10px"
                                                h="10px"
                                                borderRadius="full"
                                                bg={`hsl(${index * 40}, 70%, 60%)`}
                                                mr={2}
                                            />
                                            <Text color="white" fontWeight="medium" flex="1">
                                                {asset.token}
                                            </Text>
                                            <Badge colorScheme="blue">{asset.weight}%</Badge>
                                        </Flex>
                                    ) : (
                                        <Flex key={index} align="center" mb={2}>
                                            <Box
                                                w="10px"
                                                h="10px"
                                                borderRadius="full"
                                                bg="gray.500"
                                                mr={2}
                                            />
                                            <Text color="gray.400" flex="1">
                                                Select Token
                                            </Text>
                                            <Badge colorScheme="gray">{asset.weight}%</Badge>
                                        </Flex>
                                    ),
                                )}
                            </Box>

                            <Divider my={4} borderColor="gray.600" />

                            <VStack spacing={3} align="stretch">
                                <Text color="gray.300">
                                    - Asset diversification: Avoid concentrating weights too heavily
                                    on a single token.
                                </Text>
                                <Text color="gray.300">
                                    - Risk exposure: Stablecoins have lower volatility compared to
                                    volatile assets.
                                </Text>
                                <Text color="gray.300">
                                    - Swap fee impact: Higher fees may reduce trading volume but
                                    increase pool earnings.
                                </Text>
                            </VStack>
                        </CardBody>
                    </Card>
                </VStack>
            </SimpleGrid>
        </Box>
    );
};

export default CreatePool;
