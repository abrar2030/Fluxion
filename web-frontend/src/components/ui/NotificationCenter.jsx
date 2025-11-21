import React from "react";
import {
  Box,
  Flex,
  Text,
  Icon,
  CloseButton,
  useColorModeValue,
} from "@chakra-ui/react";
import {
  FiCheckCircle,
  FiAlertCircle,
  FiInfo,
  FiAlertTriangle,
} from "react-icons/fi";
import { useUI } from "../../lib/ui-context";

const NotificationCenter = () => {
  const { notifications, dismissNotification } = useUI();

  if (notifications.length === 0) return null;

  return (
    <Box
      position="fixed"
      top="20px"
      right="20px"
      zIndex="toast"
      maxWidth="400px"
      maxHeight="calc(100vh - 40px)"
      overflowY="auto"
      className="notification-container"
    >
      {notifications.map((notification) => (
        <NotificationItem
          key={notification.id}
          notification={notification}
          onDismiss={() => dismissNotification(notification.id)}
        />
      ))}
    </Box>
  );
};

const NotificationItem = ({ notification, onDismiss }) => {
  const { title, message, type = "info" } = notification;

  const bgColor = useColorModeValue(
    {
      success: "green.500",
      error: "red.500",
      warning: "yellow.500",
      info: "blue.500",
    }[type],
    {
      success: "green.600",
      error: "red.600",
      warning: "yellow.600",
      info: "blue.600",
    }[type],
  );

  const IconComponent = {
    success: FiCheckCircle,
    error: FiAlertCircle,
    warning: FiAlertTriangle,
    info: FiInfo,
  }[type];

  return (
    <Flex
      bg="gray.800"
      borderLeft="4px solid"
      borderLeftColor={bgColor}
      boxShadow="md"
      borderRadius="md"
      p={4}
      mb={3}
      alignItems="flex-start"
      className="notification-item slide-in-right"
    >
      <Icon as={IconComponent} color={bgColor} boxSize={5} mt={0.5} mr={3} />
      <Box flex="1">
        {title && (
          <Text fontWeight="bold" color="white" mb={1}>
            {title}
          </Text>
        )}
        <Text color="gray.300">{message}</Text>
      </Box>
      <CloseButton color="gray.400" onClick={onDismiss} ml={2} />
    </Flex>
  );
};

export default NotificationCenter;
