import React, { useState, useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import { Box, useDisclosure, Flex, Alert, AlertIcon } from '@chakra-ui/react';
import Navbar from './components/layout/Navbar';
import Sidebar from './components/layout/Sidebar';
import Footer from './components/layout/Footer';
import Dashboard from './pages/dashboard/Dashboard';
import Pools from './pages/pools/Pools';
import CreatePool from './pages/pools/CreatePool';
import Analytics from './pages/analytics/Analytics';
import Synthetics from './pages/synthetics/Synthetics';
import Settings from './pages/settings/Settings';
import NotificationCenter from './components/common/NotificationCenter';
import LoadingOverlay from './components/common/LoadingOverlay';
import { useUI } from './contexts/UIContext';
import { useWeb3 } from './hooks/useWeb3';
import ErrorBoundary from './components/common/ErrorBoundary';

function App() {
  const { sidebarOpen, toggleSidebar, notifications } = useUI();
  const { isLoading: web3Loading, error: web3Error } = useWeb3();
  const [isInitialized, setIsInitialized] = useState(false);

  // Initialize app
  useEffect(() => {
    // Simulate initialization process
    const timer = setTimeout(() => {
      setIsInitialized(true);
    }, 1000);
    
    return () => clearTimeout(timer);
  }, []);

  if (!isInitialized) {
    return <LoadingOverlay message="Initializing Fluxion..." />;
  }

  return (
    <ErrorBoundary>
      <Box display="flex" flexDirection="column" minHeight="100vh" bg="gray.900">
        <Navbar />
        <Box display="flex" flex="1">
          <Sidebar isOpen={sidebarOpen} onToggle={toggleSidebar} />
          <Box 
            as="main" 
            flex="1" 
            p={4} 
            ml={{ base: 0, md: sidebarOpen ? '60px' : '0' }}
            transition="margin-left 0.3s"
            className="fade-in"
          >
            {web3Loading && <LoadingOverlay />}
            {web3Error && (
              <Alert status="error" mb={4}>
                <AlertIcon />
                {web3Error.message}
              </Alert>
            )}
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/pools" element={<Pools />} />
              <Route path="/create-pool" element={<CreatePool />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/synthetics" element={<Synthetics />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </Box>
        </Box>
        <Footer />
        
        {/* Notification Center */}
        {notifications.length > 0 && (
          <NotificationCenter notifications={notifications} />
        )}
      </Box>
    </ErrorBoundary>
  );
}

export default App;
