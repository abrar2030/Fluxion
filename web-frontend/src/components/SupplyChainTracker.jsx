import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Container, 
  Typography, 
  Paper, 
  Grid, 
  Button, 
  TextField, 
  Select, 
  MenuItem, 
  FormControl, 
  InputLabel,
  CircularProgress,
  Snackbar,
  Alert,
  Stepper,
  Step,
  StepLabel,
  Card,
  CardContent,
  Divider,
  IconButton,
  useTheme
} from '@mui/material';
import {
  Add as AddIcon,
  Search as SearchIcon,
  FilterList as FilterIcon,
  ArrowForward as ArrowForwardIcon,
  QrCode as QrCodeIcon,
  LocationOn as LocationIcon,
  History as HistoryIcon
} from '@mui/icons-material';
import { MapContainer, TileLayer, Marker, Popup, Polyline } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Fix for Leaflet marker icons
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Custom marker icons for different statuses
const createStatusIcon = (color) => {
  return new L.Icon({
    iconUrl: 'https://cdn.rawgit.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-' + color + '.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41]
  });
};

const icons = {
  'Created': createStatusIcon('blue'),
  'InTransit': createStatusIcon('orange'),
  'Delivered': createStatusIcon('green'),
  'Rejected': createStatusIcon('red'),
  'Recalled': createStatusIcon('violet')
};

// Mock data for demonstration
const mockAssets = [
  { 
    id: 1, 
    metadata: 'Product XYZ-123', 
    currentCustodian: '0x1234567890abcdef1234567890abcdef12345678', 
    timestamp: '2023-05-18T14:30:00Z', 
    status: 'InTransit', 
    location: 'New York, USA',
    position: [40.7128, -74.0060]
  },
  { 
    id: 2, 
    metadata: 'Component ABC-456', 
    currentCustodian: '0xabcdef1234567890abcdef1234567890abcdef12', 
    timestamp: '2023-05-17T10:15:00Z', 
    status: 'Delivered', 
    location: 'Los Angeles, USA',
    position: [34.0522, -118.2437]
  },
  { 
    id: 3, 
    metadata: 'Raw Material DEF-789', 
    currentCustodian: '0x7890abcdef1234567890abcdef1234567890abcd', 
    timestamp: '2023-05-16T08:45:00Z', 
    status: 'Created', 
    location: 'Chicago, USA',
    position: [41.8781, -87.6298]
  },
  { 
    id: 4, 
    metadata: 'Shipment GHI-012', 
    currentCustodian: '0xef1234567890abcdef1234567890abcdef123456', 
    timestamp: '2023-05-15T16:20:00Z', 
    status: 'Rejected', 
    location: 'Houston, USA',
    position: [29.7604, -95.3698]
  },
];

const mockTransfers = [
  {
    assetId: 1,
    transferId: 0,
    from: '0x7890abcdef1234567890abcdef1234567890abcd',
    to: '0x1234567890abcdef1234567890abcdef12345678',
    timestamp: '2023-05-17T09:30:00Z',
    location: 'Boston, USA',
    proofHash: '0x123456789abcdef123456789abcdef123456789abcdef123456789abcdef1234'
  },
  {
    assetId: 2,
    transferId: 0,
    from: '0x7890abcdef1234567890abcdef1234567890abcd',
    to: '0xabcdef1234567890abcdef1234567890abcdef12',
    timestamp: '2023-05-16T11:45:00Z',
    location: 'San Francisco, USA',
    proofHash: '0xabcdef123456789abcdef123456789abcdef123456789abcdef123456789abcd'
  }
];

// Asset route for demonstration
const assetRoute = [
  [40.7128, -74.0060], // New York
  [39.9526, -75.1652], // Philadelphia
  [38.9072, -77.0369], // Washington DC
  [37.7749, -122.4194] // San Francisco
];

// Supply Chain Tracker component
const SupplyChainTracker = () => {
  const theme = useTheme();
  const [assets, setAssets] = useState([]);
  const [selectedAsset, setSelectedAsset] = useState(null);
  const [transfers, setTransfers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [notification, setNotification] = useState({ open: false, message: '', severity: 'info' });
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [transferDialogOpen, setTransferDialogOpen] = useState(false);
  const [newAsset, setNewAsset] = useState({
    metadata: '',
    initialCustodian: '',
    location: ''
  });
  const [transferDetails, setTransferDetails] = useState({
    to: '',
    location: '',
    proofHash: ''
  });
  const [activeStep, setActiveStep] = useState(0);

  // Fetch assets on component mount
  useEffect(() => {
    fetchAssets();
  }, []);

  // Fetch asset transfers when an asset is selected
  useEffect(() => {
    if (selectedAsset) {
      fetchTransfers(selectedAsset.id);
    }
  }, [selectedAsset]);

  // Fetch assets (mock implementation)
  const fetchAssets = () => {
    setLoading(true);
    // Simulate API call
    setTimeout(() => {
      setAssets(mockAssets);
      setLoading(false);
    }, 1000);
  };

  // Fetch transfers for a specific asset (mock implementation)
  const fetchTransfers = (assetId) => {
    setLoading(true);
    // Simulate API call
    setTimeout(() => {
      const assetTransfers = mockTransfers.filter(transfer => transfer.assetId === assetId);
      setTransfers(assetTransfers);
      setLoading(false);
    }, 800);
  };

  // Handle asset selection
  const handleAssetSelect = (asset) => {
    setSelectedAsset(asset);
  };

  // Handle search input change
  const handleSearchChange = (event) => {
    setSearchTerm(event.target.value);
  };

  // Handle status filter change
  const handleStatusFilterChange = (event) => {
    setStatusFilter(event.target.value);
  };

  // Filter assets based on search term and status filter
  const filteredAssets = assets.filter(asset => {
    const matchesSearch = asset.metadata.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         asset.location.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         asset.id.toString().includes(searchTerm);
    const matchesStatus = statusFilter === '' || asset.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  // Handle new asset form change
  const handleNewAssetChange = (field) => (event) => {
    setNewAsset({
      ...newAsset,
      [field]: event.target.value
    });
  };

  // Handle transfer details form change
  const handleTransferDetailsChange = (field) => (event) => {
    setTransferDetails({
      ...transferDetails,
      [field]: event.target.value
    });
  };

  // Create new asset (mock implementation)
  const handleCreateAsset = () => {
    setLoading(true);
    // Simulate API call
    setTimeout(() => {
      const newId = Math.max(...assets.map(a => a.id)) + 1;
      const createdAsset = {
        id: newId,
        metadata: newAsset.metadata,
        currentCustodian: newAsset.initialCustodian,
        timestamp: new Date().toISOString(),
        status: 'Created',
        location: newAsset.location,
        position: [40.7128, -74.0060] // Default to New York for demo
      };
      
      setAssets([...assets, createdAsset]);
      setLoading(false);
      setCreateDialogOpen(false);
      setNewAsset({
        metadata: '',
        initialCustodian: '',
        location: ''
      });
      setNotification({
        open: true,
        message: `Asset ${newId} created successfully`,
        severity: 'success'
      });
    }, 1500);
  };

  // Transfer asset (mock implementation)
  const handleTransferAsset = () => {
    if (!selectedAsset) return;
    
    setLoading(true);
    // Simulate API call
    setTimeout(() => {
      const newTransferId = transfers.length;
      const newTransfer = {
        assetId: selectedAsset.id,
        transferId: newTransferId,
        from: selectedAsset.currentCustodian,
        to: transferDetails.to,
        timestamp: new Date().toISOString(),
        location: transferDetails.location,
        proofHash: transferDetails.proofHash || '0x' + Array(64).fill(0).map(() => Math.floor(Math.random() * 16).toString(16)).join('')
      };
      
      // Update asset
      const updatedAssets = assets.map(asset => {
        if (asset.id === selectedAsset.id) {
          return {
            ...asset,
            currentCustodian: transferDetails.to,
            location: transferDetails.location,
            status: 'InTransit',
            timestamp: new Date().toISOString()
          };
        }
        return asset;
      });
      
      setAssets(updatedAssets);
      setTransfers([...transfers, newTransfer]);
      setSelectedAsset({
        ...selectedAsset,
        currentCustodian: transferDetails.to,
        location: transferDetails.location,
        status: 'InTransit',
        timestamp: new Date().toISOString()
      });
      
      setLoading(false);
      setTransferDialogOpen(false);
      setTransferDetails({
        to: '',
        location: '',
        proofHash: ''
      });
      setNotification({
        open: true,
        message: `Asset ${selectedAsset.id} transferred successfully`,
        severity: 'success'
      });
    }, 1500);
  };

  // Handle notification close
  const handleNotificationClose = () => {
    setNotification({ ...notification, open: false });
  };

  // Format address for display
  const formatAddress = (address) => {
    if (!address) return '';
    return `${address.substring(0, 6)}...${address.substring(address.length - 4)}`;
  };

  // Get status color
  const getStatusColor = (status) => {
    switch (status) {
      case 'Created': return theme.palette.info.main;
      case 'InTransit': return theme.palette.warning.main;
      case 'Delivered': return theme.palette.success.main;
      case 'Rejected': return theme.palette.error.main;
      case 'Recalled': return theme.palette.secondary.main;
      default: return theme.palette.text.primary;
    }
  };

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Supply Chain Tracker
      </Typography>
      <Typography variant="subtitle1" color="text.secondary" paragraph>
        Track and manage assets throughout the supply chain
      </Typography>

      {/* Search and filter bar */}
      <Paper sx={{ p: 2, mb: 3, display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
          <SearchIcon sx={{ color: 'action.active', mr: 1 }} />
          <TextField
            variant="outlined"
            size="small"
            placeholder="Search assets..."
            value={searchTerm}
            onChange={handleSearchChange}
            sx={{ flexGrow: 1 }}
          />
        </Box>
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <FormControl variant="outlined" size="small" sx={{ minWidth: 150 }}>
            <InputLabel id="status-filter-label">Status</InputLabel>
            <Select
              labelId="status-filter-label"
              value={statusFilter}
              onChange={handleStatusFilterChange}
              label="Status"
            >
              <MenuItem value="">All Statuses</MenuItem>
              <MenuItem value="Created">Created</MenuItem>
              <MenuItem value="InTransit">In Transit</MenuItem>
              <MenuItem value="Delivered">Delivered</MenuItem>
              <MenuItem value="Rejected">Rejected</MenuItem>
              <MenuItem value="Recalled">Recalled</MenuItem>
            </Select>
          </FormControl>
          
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => setCreateDialogOpen(true)}
          >
            New Asset
          </Button>
        </Box>
      </Paper>

      {/* Main content */}
      <Grid container spacing={3}>
        {/* Asset list */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ height: '70vh', overflow: 'auto' }}>
            {loading && assets.length === 0 ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                <CircularProgress />
              </Box>
            ) : filteredAssets.length === 0 ? (
              <Box sx={{ p: 3, textAlign: 'center' }}>
                <Typography variant="body1" color="text.secondary">
                  No assets found matching your criteria
                </Typography>
              </Box>
            ) : (
              <Box>
                {filteredAssets.map((asset) => (
                  <Box
                    key={asset.id}
                    sx={{
                      p: 2,
                      borderBottom: `1px solid ${theme.palette.divider}`,
                      cursor: 'pointer',
                      backgroundColor: selectedAsset?.id === asset.id ? theme.palette.action.selected : 'transparent',
                      '&:hover': {
                        backgroundColor: theme.palette.action.hover
                      }
                    }}
                    onClick={() => handleAssetSelect(asset)}
                  >
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                      <Typography variant="subtitle1" component="div" sx={{ fontWeight: 'bold' }}>
                        Asset #{asset.id}
                      </Typography>
                      <Box
                        sx={{
                          px: 1,
                          py: 0.5,
                          borderRadius: 1,
                          backgroundColor: `${getStatusColor(asset.status)}20`,
                          color: getStatusColor(asset.status),
                          fontSize: '0.75rem',
                          fontWeight: 'bold'
                        }}
                      >
                        {asset.status}
                      </Box>
                    </Box>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      {asset.metadata}
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <LocationIcon fontSize="small" color="action" />
                      <Typography variant="body2" color="text.secondary">
                        {asset.location}
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                      <HistoryIcon fontSize="small" color="action" />
                      <Typography variant="body2" color="text.secondary">
                        {new Date(asset.timestamp).toLocaleString()}
                      </Typography>
                    </Box>
                  </Box>
                ))}
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Asset details and map */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ height: '70vh', overflow: 'auto', p: 0 }}>
            {selectedAsset ? (
              <Box>
                {/* Asset details header */}
                <Box sx={{ p: 3, borderBottom: `1px solid ${theme.palette.divider}` }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="h6" component="div">
                      Asset #{selectedAsset.id} Details
                    </Typography>
                    <Button
                      variant="outlined"
                      startIcon={<ArrowForwardIcon />}
                      onClick={() => setTransferDialogOpen(true)}
                    >
                      Transfer Asset
                    </Button>
                  </Box>
                  
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                      <Typography variant="body2" color="text.secondary">Metadata</Typography>
                      <Typography variant="body1">{selectedAsset.metadata}</Typography>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <Typography variant="body2" color="text.secondary">Current Custodian</Typography>
                      <Typography variant="body1">{formatAddress(selectedAsset.currentCustodian)}</Typography>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <Typography variant="body2" color="text.secondary">Location</Typography>
                      <Typography variant="body1">{selectedAsset.location}</Typography>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <Typography variant="body2" color="text.secondary">Last Updated</Typography>
                      <Typography variant="body1">{new Date(selectedAsset.timestamp).toLocaleString()}</Typography>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <Typography variant="body2" color="text.secondary">Status</Typography>
                      <Box
                        sx={{
                          display: 'inline-block',
                          px: 1,
                          py: 0.5,
                          borderRadius: 1,
                          backgroundColor: `${getStatusColor(selectedAsset.status)}20`,
                          color: getStatusColor(selectedAsset.status),
                          fontWeight: 'bold'
                        }}
                      >
                        {selectedAsset.status}
                      </Box>
                    </Grid>
                  </Grid>
                </Box>

                {/* Map */}
                <Box sx={{ height: '300px', width: '100%' }}>
                  <MapContainer 
                    center={selectedAsset.position || [40.7128, -74.0060]} 
                    zoom={13} 
                    style={{ height: '100%', width: '100%' }}
                  >
                    <TileLayer
                      attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                      url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    />
                    <Marker 
                      position={selectedAsset.position || [40.7128, -74.0060]}
                      icon={icons[selectedAsset.status] || icons['Created']}
                    >
                      <Popup>
                        <Typography variant="subtitle1">Asset #{selectedAsset.id}</Typography>
                        <Typography variant="body2">{selectedAsset.metadata}</Typography>
                        <Typography variant="body2">Status: {selectedAsset.status}</Typography>
                        <Typography variant="body2">Location: {selectedAsset.location}</Typography>
                      </Popup>
                    </Marker>
                    <Polyline positions={assetRoute} color="blue" />
                  </MapContainer>
                </Box>

                {/* Transfer history */}
                <Box sx={{ p: 3 }}>
                  <Typography variant="h6" component="div" sx={{ mb: 2 }}>
                    Transfer History
                  </Typography>
                  
                  {loading ? (
                    <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                      <CircularProgress />
                    </Box>
                  ) : transfers.length === 0 ? (
                    <Typography variant="body2" color="text.secondary">
                      No transfer history available for this asset.
                    </Typography>
                  ) : (
                    <Box>
                      {transfers.map((transfer, index) => (
                        <Card key={index} variant="outlined" sx={{ mb: 2 }}>
                          <CardContent>
                            <Grid container spacing={2}>
                              <Grid item xs={12} sm={6}>
                                <Typography variant="body2" color="text.secondary">From</Typography>
                                <Typography variant="body1">{formatAddress(transfer.from)}</Typography>
                              </Grid>
                              <Grid item xs={12} sm={6}>
                                <Typography variant="body2" color="text.secondary">To</Typography>
                                <Typography variant="body1">{formatAddress(transfer.to)}</Typography>
                              </Grid>
                              <Grid item xs={12} sm={6}>
                                <Typography variant="body2" color="text.secondary">Location</Typography>
                                <Typography variant="body1">{transfer.location}</Typography>
                              </Grid>
                              <Grid item xs={12} sm={6}>
                                <Typography variant="body2" color="text.secondary">Timestamp</Typography>
                                <Typography variant="body1">{new Date(transfer.timestamp).toLocaleString()}</Typography>
                              </Grid>
                              <Grid item xs={12}>
                                <Typography variant="body2" color="text.secondary">Proof Hash</Typography>
                                <Typography variant="body1" sx={{ wordBreak: 'break-all' }}>{transfer.proofHash}</Typography>
                              </Grid>
                            </Grid>
                          </CardContent>
                        </Card>
                      ))}
                    </Box>
                  )}
                </Box>
              </Box>
            ) : (
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', flexDirection: 'column', p: 3 }}>
                <Typography variant="h6" color="text.secondary" sx={{ mb: 2 }}>
                  Select an asset to view details
                </Typography>
                <Typography variant="body2" color="text.secondary" align="center">
                  Click on any asset in the list to view its details, transfer history, and location on the map.
                </Typography>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>

      {/* Create Asset Dialog */}
      {createDialogOpen && (
        <Paper sx={{ position: 'fixed', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', width: '90%', maxWidth: 600, p: 4, zIndex: 1000 }}>
          <Typography variant="h6" component="div" sx={{ mb: 3 }}>
            Create New Asset
          </Typography>
          
          <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
            <Step>
              <StepLabel>Asset Information</StepLabel>
            </Step>
            <Step>
              <StepLabel>Custodian & Location</StepLabel>
            </Step>
            <Step>
              <StepLabel>Confirmation</StepLabel>
            </Step>
          </Stepper>
          
          {activeStep === 0 && (
            <Box>
              <TextField
                fullWidth
                label="Asset Metadata"
                variant="outlined"
                value={newAsset.metadata}
                onChange={handleNewAssetChange('metadata')}
                sx={{ mb: 3 }}
                placeholder="Enter product name, serial number, or other identifying information"
              />
              
              <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
                <Button onClick={() => setCreateDialogOpen(false)} sx={{ mr: 1 }}>
                  Cancel
                </Button>
                <Button 
                  variant="contained" 
                  onClick={() => setActiveStep(1)}
                  disabled={!newAsset.metadata}
                >
                  Next
                </Button>
              </Box>
            </Box>
          )}
          
          {activeStep === 1 && (
            <Box>
              <TextField
                fullWidth
                label="Initial Custodian Address"
                variant="outlined"
                value={newAsset.initialCustodian}
                onChange={handleNewAssetChange('initialCustodian')}
                sx={{ mb: 3 }}
                placeholder="0x..."
              />
              
              <TextField
                fullWidth
                label="Initial Location"
                variant="outlined"
                value={newAsset.location}
                onChange={handleNewAssetChange('location')}
                sx={{ mb: 3 }}
                placeholder="City, Country"
              />
              
              <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
                <Button onClick={() => setActiveStep(0)} sx={{ mr: 1 }}>
                  Back
                </Button>
                <Button 
                  variant="contained" 
                  onClick={() => setActiveStep(2)}
                  disabled={!newAsset.initialCustodian || !newAsset.location}
                >
                  Next
                </Button>
              </Box>
            </Box>
          )}
          
          {activeStep === 2 && (
            <Box>
              <Typography variant="subtitle1" gutterBottom>
                Please confirm the asset details:
              </Typography>
              
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={12} sm={6}>
                  <Typography variant="body2" color="text.secondary">Metadata</Typography>
                  <Typography variant="body1">{newAsset.metadata}</Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="body2" color="text.secondary">Initial Custodian</Typography>
                  <Typography variant="body1">{formatAddress(newAsset.initialCustodian)}</Typography>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="body2" color="text.secondary">Initial Location</Typography>
                  <Typography variant="body1">{newAsset.location}</Typography>
                </Grid>
              </Grid>
              
              <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
                <Button onClick={() => setActiveStep(1)} sx={{ mr: 1 }}>
                  Back
                </Button>
                <Button 
                  variant="contained" 
                  onClick={handleCreateAsset}
                  disabled={loading}
                >
                  {loading ? <CircularProgress size={24} /> : 'Create Asset'}
                </Button>
              </Box>
            </Box>
          )}
        </Paper>
      )}

      {/* Transfer Asset Dialog */}
      {transferDialogOpen && selectedAsset && (
        <Paper sx={{ position: 'fixed', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', width: '90%', maxWidth: 600, p: 4, zIndex: 1000 }}>
          <Typography variant="h6" component="div" sx={{ mb: 3 }}>
            Transfer Asset #{selectedAsset.id}
          </Typography>
          
          <TextField
            fullWidth
            label="Recipient Address"
            variant="outlined"
            value={transferDetails.to}
            onChange={handleTransferDetailsChange('to')}
            sx={{ mb: 3 }}
            placeholder="0x..."
          />
          
          <TextField
            fullWidth
            label="New Location"
            variant="outlined"
            value={transferDetails.location}
            onChange={handleTransferDetailsChange('location')}
            sx={{ mb: 3 }}
            placeholder="City, Country"
          />
          
          <TextField
            fullWidth
            label="Proof Hash (Optional)"
            variant="outlined"
            value={transferDetails.proofHash}
            onChange={handleTransferDetailsChange('proofHash')}
            sx={{ mb: 3 }}
            placeholder="0x..."
            helperText="Leave blank to generate automatically"
          />
          
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
            <Button onClick={() => setTransferDialogOpen(false)} sx={{ mr: 1 }}>
              Cancel
            </Button>
            <Button 
              variant="contained" 
              onClick={handleTransferAsset}
              disabled={loading || !transferDetails.to || !transferDetails.location}
            >
              {loading ? <CircularProgress size={24} /> : 'Transfer Asset'}
            </Button>
          </Box>
        </Paper>
      )}

      {/* Notification */}
      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={handleNotificationClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert onClose={handleNotificationClose} severity={notification.severity} sx={{ width: '100%' }}>
          {notification.message}
        </Alert>
      </Snackbar>
    </Container>
  );
};

export default SupplyChainTracker;
