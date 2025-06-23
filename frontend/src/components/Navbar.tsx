import React from "react";
import { AppBar, Toolbar, Typography, Box, Button, IconButton, Menu, MenuItem, useTheme } from "@mui/material";
import {
  Dashboard as DashboardIcon,
  ModelTraining as TrainingIcon,
  Assessment as BacktestIcon,
  MoreVert as MoreIcon,
} from "@mui/icons-material";
import { useNavigate, useLocation } from "react-router-dom";

const Navbar: React.FC = () => {
  const theme = useTheme();
  const navigate = useNavigate();
  const location = useLocation();
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);

  const handleMenuClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(anchorEl);
  };

  const navItems = [
    { path: "/", label: "Dashboard", icon: <DashboardIcon /> },
    // { path: "/data", label: "Data Explorer", icon: <DataIcon /> },
    { path: "/training", label: "Model Training", icon: <TrainingIcon /> },
    { path: "/backtesting", label: "Backtesting", icon: <BacktestIcon /> },
    // { path: "/live", label: "Live Trading", icon: <LiveIcon /> },
  ];

  const isActive = (path: string) => location.pathname === path;

  return (
    <AppBar
      position="sticky"
      elevation={0}
      sx={{
        background: "linear-gradient(135deg, #1a1d35 0%, #0a0e27 100%)",
        borderBottom: "1px solid rgba(255, 255, 255, 0.1)",
      }}
    >
      <Toolbar sx={{ px: 3 }}>
        {/* Logo and Title */}
        <Box sx={{ display: "flex", alignItems: "center", mr: 4 }}>
          <img src="/logo.svg" alt="logo" style={{ width: 64, height: 64, marginRight: 8 }} />
          <Typography
            variant="h5"
            component="div"
            sx={{
              fontWeight: 700,
              background: "linear-gradient(45deg, #1976d2 30%, #42a5f5 90%)",
              backgroundClip: "text",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}
          >
            AI Trader
          </Typography>
        </Box>

        {/* Navigation Items - Desktop */}
        <Box sx={{ flexGrow: 1, display: { xs: "none", md: "flex" }, gap: 1 }}>
          {navItems.map((item) => (
            <Button
              key={item.path}
              onClick={() => navigate(item.path)}
              startIcon={item.icon}
              sx={{
                color: isActive(item.path) ? theme.palette.primary.main : "rgba(255, 255, 255, 0.7)",
                backgroundColor: isActive(item.path) ? "rgba(25, 118, 210, 0.1)" : "transparent",
                borderRadius: 2,
                px: 2,
                py: 1,
                fontWeight: isActive(item.path) ? 600 : 400,
                border: isActive(item.path) ? `1px solid ${theme.palette.primary.main}` : "1px solid transparent",
                "&:hover": {
                  backgroundColor: "rgba(25, 118, 210, 0.1)",
                  color: theme.palette.primary.light,
                },
              }}
            >
              {item.label}
            </Button>
          ))}
        </Box>

        {/* Mobile Menu */}
        <Box sx={{ display: { xs: "flex", md: "none" }, ml: "auto" }}>
          <IconButton
            size="large"
            aria-label="navigation menu"
            aria-controls="mobile-menu"
            aria-haspopup="true"
            onClick={handleMenuClick}
            color="inherit"
          >
            <MoreIcon />
          </IconButton>
          <Menu
            id="mobile-menu"
            anchorEl={anchorEl}
            open={Boolean(anchorEl)}
            onClose={handleMenuClose}
            sx={{
              "& .MuiPaper-root": {
                backgroundColor: theme.palette.background.paper,
                border: `1px solid ${theme.palette.divider}`,
              },
            }}
          >
            {navItems.map((item) => (
              <MenuItem
                key={item.path}
                onClick={() => {
                  navigate(item.path);
                  handleMenuClose();
                }}
                sx={{
                  color: isActive(item.path) ? theme.palette.primary.main : "inherit",
                  backgroundColor: isActive(item.path) ? "rgba(25, 118, 210, 0.1)" : "transparent",
                }}
              >
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  {item.icon}
                  {item.label}
                </Box>
              </MenuItem>
            ))}
          </Menu>
        </Box>

        {/* Status Indicator */}
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            ml: 2,
            px: 2,
            py: 0.5,
            borderRadius: 2,
            backgroundColor: "rgba(46, 125, 50, 0.1)",
            border: "1px solid rgba(46, 125, 50, 0.3)",
          }}
        >
          <Box
            sx={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              backgroundColor: theme.palette.success.main,
              mr: 1,
              animation: "pulse 2s infinite",
              "@keyframes pulse": {
                "0%": { opacity: 1 },
                "50%": { opacity: 0.5 },
                "100%": { opacity: 1 },
              },
            }}
          />
          <Typography variant="caption" sx={{ color: theme.palette.success.main, fontWeight: 600 }}>
            Live
          </Typography>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;
