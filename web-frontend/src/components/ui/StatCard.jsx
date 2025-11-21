import React from "react";
import {
  Box,
  Button,
  Flex,
  Text,
  Icon,
  useColorModeValue,
  Tooltip,
} from "@chakra-ui/react";
import { FiInfo } from "react-icons/fi";

const StatCard = ({ title, value, icon, helpText, type, isLoading }) => {
  const bgColor = useColorModeValue("gray.700", "gray.700");
  const borderColor = useColorModeValue("gray.600", "gray.600");

  return (
    <Box
      p={5}
      bg={bgColor}
      borderWidth="1px"
      borderColor={borderColor}
      borderRadius="lg"
      shadow="xl"
      transition="all 0.3s"
      _hover={{ transform: "translateY(-5px)", shadow: "2xl" }}
      className="stat-card slide-up"
    >
      <Flex justifyContent="space-between" alignItems="center">
        <Box>
          <Flex alignItems="center">
            <Text fontSize="lg" fontWeight="medium" color="gray.300" mr={1}>
              {title}
            </Text>
            {helpText && (
              <Tooltip label={helpText} placement="top">
                <Box as="span">
                  <Icon as={FiInfo} color="gray.400" boxSize={4} />
                </Box>
              </Tooltip>
            )}
          </Flex>
          <Text fontSize="2xl" fontWeight="bold" color="white" mt={1}>
            {isLoading ? "—" : value}
          </Text>
          {type && (
            <Text
              fontSize="sm"
              color={type === "increase" ? "green.400" : "red.400"}
              mt={1}
            >
              {type === "increase" ? "↑" : "↓"}{" "}
              {type === "increase" ? "+" : "-"}
              {Math.abs(parseFloat(type.value || 0)).toFixed(2)}%
            </Text>
          )}
        </Box>
        <Box p={2} borderRadius="full" bg="gray.800" color="brand.500">
          <Icon as={icon} boxSize={6} />
        </Box>
      </Flex>
    </Box>
  );
};

export default StatCard;
