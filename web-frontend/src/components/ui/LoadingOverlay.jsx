import React from "react";
import { Box, Flex, Text, Icon, Spinner } from "@chakra-ui/react";

const LoadingOverlay = ({ message = "Loading..." }) => {
  return (
    <Flex
      position="fixed"
      top="0"
      left="0"
      right="0"
      bottom="0"
      bg="rgba(0, 0, 0, 0.7)"
      zIndex="9999"
      justifyContent="center"
      alignItems="center"
      flexDirection="column"
    >
      <Spinner
        thickness="4px"
        speed="0.65s"
        emptyColor="gray.700"
        color="brand.500"
        size="xl"
        mb={4}
      />
      <Text color="white" fontSize="lg" fontWeight="medium">
        {message}
      </Text>
    </Flex>
  );
};

export default LoadingOverlay;
