import React, { useState, useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import { Box, useDisclosure, Flex } from '@chakra-ui/react';
import Navbar from './components/layout/Navbar';
import Sidebar from './components/layout/Sidebar';
import Footer from './components/layout/Footer';
import Dashboard from './pages/Dashboard';
import Pools from './pages/Pools';
import CreatePool from './pages/CreatePool';
import Analytics from './pages/Analytics';
import Synthetics from './pages/Synthetics';
import Settings from './pages/Settings';
import NotificationCenter from './components/ui/NotificationCenter';
import LoadingOverlay from './components/ui/LoadingOverlay';
import { useUI } from './lib/ui-context';
import { useWeb3 } from './lib/web3-config';

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
  );
}

export default App;
