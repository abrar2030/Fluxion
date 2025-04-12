import React, { useEffect } from 'react';
import { Box, VStack, Alert, AlertIcon, AlertTitle, AlertDescription, CloseButton } from '@chakra-ui/react';
import { useUI } from '../../contexts/UIContext';

const NotificationCenter = ({ notifications }) => {
  const { removeNotification } = useUI();

  useEffect(() => {
    const timeoutIds = notifications.map((notification) => {
      if (notification.autoClose !== false) {
        return setTimeout(() => {
          removeNotification(notification.id);
        }, notification.duration || 5000);
      }
      return null;
    });

    return () => {
      timeoutIds.forEach((id) => id && clearTimeout(id));
    };
  }, [notifications, removeNotification]);

  return (
    <Box
      position="fixed"
      bottom={4}
      right={4}
      zIndex={9999}
      maxW="400px"
      w="100%"
    >
      <VStack spacing={4} align="stretch">
        {notifications.map((notification) => (
          <Alert
            key={notification.id}
            status={notification.type || 'info'}
            variant="solid"
            borderRadius="md"
          >
            <AlertIcon />
            <Box flex="1">
              {notification.title && (
                <AlertTitle>{notification.title}</AlertTitle>
              )}
              <AlertDescription display="block">
                {notification.message}
              </AlertDescription>
            </Box>
            <CloseButton
              position="absolute"
              right="8px"
              top="8px"
              onClick={() => removeNotification(notification.id)}
            />
          </Alert>
        ))}
      </VStack>
    </Box>
  );
};

export default NotificationCenter; 