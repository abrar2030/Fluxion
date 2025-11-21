import React, { useState, useEffect } from "react";
import {
  Box,
  Container,
  Grid,
  Paper,
  Typography,
  Button,
  CircularProgress,
  Tabs,
  Tab,
  Divider,
  useTheme,
  useMediaQuery,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  AppBar,
  Toolbar,
} from "@mui/material";
import {
  Dashboard as DashboardIcon,
  Timeline as TimelineIcon,
  Inventory as InventoryIcon,
  LocalShipping as ShippingIcon,
  Settings as SettingsIcon,
  Notifications as NotificationsIcon,
  Menu as MenuIcon,
  Search as SearchIcon,
  AccountCircle as AccountIcon,
  ChevronRight as ChevronRightIcon,
} from "@mui/icons-material";
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
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  Polyline,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";

// Fix for Leaflet marker icons
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png",
  iconUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png",
  shadowUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png",
});

// Mock data for demonstration
const liquidityData = [
  { name: "Jan", value: 4000 },
  { name: "Feb", value: 3000 },
  { name: "Mar", value: 2000 },
  { name: "Apr", value: 2780 },
  { name: "May", value: 1890 },
  { name: "Jun", value: 2390 },
  { name: "Jul", value: 3490 },
];

const supplyChainData = [
  { name: "Created", value: 400 },
  { name: "In Transit", value: 300 },
  { name: "Delivered", value: 300 },
  { name: "Rejected", value: 200 },
  { name: "Recalled", value: 100 },
];

const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042", "#8884d8"];

const assetLocations = [
  { id: 1, name: "Asset 1", position: [51.505, -0.09], status: "In Transit" },
  { id: 2, name: "Asset 2", position: [51.51, -0.1], status: "Delivered" },
  { id: 3, name: "Asset 3", position: [51.49, -0.08], status: "Created" },
  { id: 4, name: "Asset 4", position: [51.5, -0.05], status: "In Transit" },
];

const assetRoute = [
  [51.505, -0.09],
  [51.51, -0.1],
  [51.52, -0.12],
  [51.53, -0.11],
];

const recentTransactions = [
  {
    id: 1,
    asset: "Asset 1",
    from: "0x123...",
    to: "0x456...",
    timestamp: "2023-05-18 14:30",
    status: "Completed",
  },
  {
    id: 2,
    asset: "Asset 2",
    from: "0x789...",
    to: "0xabc...",
    timestamp: "2023-05-18 13:45",
    status: "Pending",
  },
  {
    id: 3,
    asset: "Asset 3",
    from: "0xdef...",
    to: "0x123...",
    timestamp: "2023-05-18 12:20",
    status: "Completed",
  },
  {
    id: 4,
    asset: "Asset 4",
    from: "0x456...",
    to: "0x789...",
    timestamp: "2023-05-18 11:10",
    status: "Failed",
  },
];

// Dashboard component
const Dashboard = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("md"));
  const [drawerOpen, setDrawerOpen] = useState(!isMobile);
  const [activeTab, setActiveTab] = useState(0);
  const [loading, setLoading] = useState(false);
  const [assetData, setAssetData] = useState([]);
  const [supplyChainStats, setSupplyChainStats] = useState(supplyChainData);

  useEffect(() => {
    // Simulate data loading
    setLoading(true);
    setTimeout(() => {
      setAssetData(assetLocations);
      setLoading(false);
    }, 1000);
  }, []);

  const handleDrawerToggle = () => {
    setDrawerOpen(!drawerOpen);
  };

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const drawerWidth = 240;

  const drawer = (
    <Box>
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          p: 2,
        }}
      >
        <Typography variant="h6" component="div" sx={{ fontWeight: "bold" }}>
          Fluxion
        </Typography>
        {isMobile && (
          <IconButton onClick={handleDrawerToggle}>
            <ChevronRightIcon />
          </IconButton>
        )}
      </Box>
      <Divider />
      <List>
        {[
          { text: "Dashboard", icon: <DashboardIcon /> },
          { text: "Supply Chain", icon: <ShippingIcon /> },
          { text: "Assets", icon: <InventoryIcon /> },
          { text: "Analytics", icon: <TimelineIcon /> },
          { text: "Settings", icon: <SettingsIcon /> },
        ].map((item, index) => (
          <ListItem
            button
            key={item.text}
            selected={index === 0}
            sx={{
              "&.Mui-selected": {
                backgroundColor: theme.palette.primary.main + "20",
                borderLeft: `4px solid ${theme.palette.primary.main}`,
              },
              "&.Mui-selected:hover": {
                backgroundColor: theme.palette.primary.main + "30",
              },
            }}
          >
            <ListItemIcon
              sx={{
                color: index === 0 ? theme.palette.primary.main : "inherit",
              }}
            >
              {item.icon}
            </ListItemIcon>
            <ListItemText
              primary={item.text}
              primaryTypographyProps={{
                fontWeight: index === 0 ? "bold" : "normal",
                color: index === 0 ? theme.palette.primary.main : "inherit",
              }}
            />
          </ListItem>
        ))}
      </List>
    </Box>
  );

  return (
    <Box sx={{ display: "flex", height: "100vh", overflow: "hidden" }}>
      {/* Sidebar */}
      <Drawer
        variant={isMobile ? "temporary" : "persistent"}
        open={drawerOpen}
        onClose={handleDrawerToggle}
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          "& .MuiDrawer-paper": {
            width: drawerWidth,
            boxSizing: "border-box",
          },
        }}
      >
        {drawer}
      </Drawer>

      {/* Main content */}
      <Box component="main" sx={{ flexGrow: 1, overflow: "auto" }}>
        {/* App bar */}
        <AppBar
          position="static"
          color="default"
          elevation={0}
          sx={{
            borderBottom: `1px solid ${theme.palette.divider}`,
            backgroundColor: theme.palette.background.paper,
          }}
        >
          <Toolbar>
            <IconButton
              edge="start"
              color="inherit"
              aria-label="menu"
              onClick={handleDrawerToggle}
              sx={{
                mr: 2,
                display: { sm: "block", md: !drawerOpen ? "block" : "none" },
              }}
            >
              <MenuIcon />
            </IconButton>
            <Box sx={{ flexGrow: 1 }} />
            <IconButton color="inherit">
              <SearchIcon />
            </IconButton>
            <IconButton color="inherit">
              <NotificationsIcon />
            </IconButton>
            <IconButton color="inherit">
              <AccountIcon />
            </IconButton>
          </Toolbar>
        </AppBar>

        {/* Dashboard content */}
        <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
          <Typography variant="h4" component="h1" gutterBottom>
            Dashboard
          </Typography>
          <Typography variant="subtitle1" color="text.secondary" paragraph>
            Overview of your supply chain and liquidity pools
          </Typography>

          {/* Summary cards */}
          <Grid container spacing={3} sx={{ mb: 4 }}>
            {[
              {
                title: "Total Assets",
                value: "1,234",
                color: theme.palette.primary.main,
              },
              {
                title: "In Transit",
                value: "256",
                color: theme.palette.warning.main,
              },
              {
                title: "Delivered",
                value: "789",
                color: theme.palette.success.main,
              },
              { title: "Issues", value: "12", color: theme.palette.error.main },
            ].map((card, index) => (
              <Grid item xs={12} sm={6} md={3} key={index}>
                <Paper
                  sx={{
                    p: 3,
                    display: "flex",
                    flexDirection: "column",
                    borderLeft: `4px solid ${card.color}`,
                    height: "100%",
                  }}
                >
                  <Typography
                    variant="h3"
                    component="div"
                    sx={{ fontWeight: "bold" }}
                  >
                    {card.value}
                  </Typography>
                  <Typography variant="subtitle1" color="text.secondary">
                    {card.title}
                  </Typography>
                </Paper>
              </Grid>
            ))}
          </Grid>

          {/* Tabs */}
          <Paper sx={{ mb: 4 }}>
            <Tabs
              value={activeTab}
              onChange={handleTabChange}
              indicatorColor="primary"
              textColor="primary"
              variant={isMobile ? "scrollable" : "fullWidth"}
              scrollButtons="auto"
            >
              <Tab label="Overview" />
              <Tab label="Supply Chain" />
              <Tab label="Liquidity" />
              <Tab label="Analytics" />
            </Tabs>
          </Paper>

          {/* Tab content */}
          <Box role="tabpanel" hidden={activeTab !== 0}>
            {activeTab === 0 && (
              <Grid container spacing={3}>
                {/* Liquidity chart */}
                <Grid item xs={12} md={8}>
                  <Paper sx={{ p: 3, height: "100%" }}>
                    <Typography variant="h6" gutterBottom>
                      Liquidity Overview
                    </Typography>
                    <ResponsiveContainer width="100%" height={300}>
                      <AreaChart data={liquidityData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Area
                          type="monotone"
                          dataKey="value"
                          stroke={theme.palette.primary.main}
                          fill={theme.palette.primary.main + "80"}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </Paper>
                </Grid>

                {/* Supply chain status */}
                <Grid item xs={12} md={4}>
                  <Paper sx={{ p: 3, height: "100%" }}>
                    <Typography variant="h6" gutterBottom>
                      Supply Chain Status
                    </Typography>
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie
                          data={supplyChainStats}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          outerRadius={80}
                          fill="#8884d8"
                          dataKey="value"
                          label={({ name, percent }) =>
                            `${name}: ${(percent * 100).toFixed(0)}%`
                          }
                        >
                          {supplyChainStats.map((entry, index) => (
                            <Cell
                              key={`cell-${index}`}
                              fill={COLORS[index % COLORS.length]}
                            />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </Paper>
                </Grid>

                {/* Map */}
                <Grid item xs={12}>
                  <Paper sx={{ p: 3 }}>
                    <Typography variant="h6" gutterBottom>
                      Asset Tracking Map
                    </Typography>
                    <Box sx={{ height: 400, width: "100%" }}>
                      {loading ? (
                        <Box
                          sx={{
                            display: "flex",
                            justifyContent: "center",
                            alignItems: "center",
                            height: "100%",
                          }}
                        >
                          <CircularProgress />
                        </Box>
                      ) : (
                        <MapContainer
                          center={[51.505, -0.09]}
                          zoom={13}
                          style={{ height: "100%", width: "100%" }}
                        >
                          <TileLayer
                            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                          />
                          {assetData.map((asset) => (
                            <Marker key={asset.id} position={asset.position}>
                              <Popup>
                                <Typography variant="subtitle1">
                                  {asset.name}
                                </Typography>
                                <Typography variant="body2">
                                  Status: {asset.status}
                                </Typography>
                              </Popup>
                            </Marker>
                          ))}
                          <Polyline positions={assetRoute} color="blue" />
                        </MapContainer>
                      )}
                    </Box>
                  </Paper>
                </Grid>

                {/* Recent transactions */}
                <Grid item xs={12}>
                  <Paper sx={{ p: 3 }}>
                    <Typography variant="h6" gutterBottom>
                      Recent Transactions
                    </Typography>
                    <Box sx={{ overflowX: "auto" }}>
                      <Box sx={{ minWidth: 650 }}>
                        <Box
                          sx={{
                            display: "flex",
                            fontWeight: "bold",
                            borderBottom: `1px solid ${theme.palette.divider}`,
                            py: 1.5,
                          }}
                        >
                          <Box sx={{ flex: "1 1 100px" }}>ID</Box>
                          <Box sx={{ flex: "1 1 150px" }}>Asset</Box>
                          <Box sx={{ flex: "1 1 150px" }}>From</Box>
                          <Box sx={{ flex: "1 1 150px" }}>To</Box>
                          <Box sx={{ flex: "1 1 200px" }}>Timestamp</Box>
                          <Box sx={{ flex: "1 1 100px" }}>Status</Box>
                        </Box>
                        {recentTransactions.map((tx) => (
                          <Box
                            key={tx.id}
                            sx={{
                              display: "flex",
                              borderBottom: `1px solid ${theme.palette.divider}`,
                              py: 1.5,
                            }}
                          >
                            <Box sx={{ flex: "1 1 100px" }}>{tx.id}</Box>
                            <Box sx={{ flex: "1 1 150px" }}>{tx.asset}</Box>
                            <Box sx={{ flex: "1 1 150px" }}>{tx.from}</Box>
                            <Box sx={{ flex: "1 1 150px" }}>{tx.to}</Box>
                            <Box sx={{ flex: "1 1 200px" }}>{tx.timestamp}</Box>
                            <Box sx={{ flex: "1 1 100px" }}>
                              <Box
                                component="span"
                                sx={{
                                  px: 1,
                                  py: 0.5,
                                  borderRadius: 1,
                                  fontSize: "0.75rem",
                                  backgroundColor:
                                    tx.status === "Completed"
                                      ? theme.palette.success.main + "20"
                                      : tx.status === "Pending"
                                        ? theme.palette.warning.main + "20"
                                        : theme.palette.error.main + "20",
                                  color:
                                    tx.status === "Completed"
                                      ? theme.palette.success.main
                                      : tx.status === "Pending"
                                        ? theme.palette.warning.main
                                        : theme.palette.error.main,
                                }}
                              >
                                {tx.status}
                              </Box>
                            </Box>
                          </Box>
                        ))}
                      </Box>
                    </Box>
                  </Paper>
                </Grid>
              </Grid>
            )}
          </Box>

          <Box role="tabpanel" hidden={activeTab !== 1}>
            {activeTab === 1 && (
              <Typography variant="h6">Supply Chain content</Typography>
            )}
          </Box>

          <Box role="tabpanel" hidden={activeTab !== 2}>
            {activeTab === 2 && (
              <Typography variant="h6">Liquidity content</Typography>
            )}
          </Box>

          <Box role="tabpanel" hidden={activeTab !== 3}>
            {activeTab === 3 && (
              <Typography variant="h6">Analytics content</Typography>
            )}
          </Box>
        </Container>
      </Box>
    </Box>
  );
};

export default Dashboard;
