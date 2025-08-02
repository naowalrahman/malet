import React from "react";
import { AppBar, Toolbar, Typography, Box, Button, IconButton, Menu, MenuItem } from "@mui/material";
import {
  Dashboard as DashboardIcon,
  ModelTraining as TrainingIcon,
  Assessment as BacktestIcon,
  Psychology as PredictIcon,
  MoreVert as MoreIcon,
  GitHub as GitHubIcon,
} from "@mui/icons-material";
import { useNavigate, useLocation } from "react-router-dom";

function Navbar() {
  const navigate = useNavigate();
  const location = useLocation();
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);

  const handleMenuClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const navItems = [
    { path: "/", label: "Dashboard", icon: <DashboardIcon /> },
    // { path: "/data", label: "Data Explorer", icon: <DataIcon /> },
    { path: "/training", label: "Model Training", icon: <TrainingIcon /> },
    { path: "/backtesting", label: "Backtesting", icon: <BacktestIcon /> },
    { path: "/predict", label: "Predict", icon: <PredictIcon /> },
    // { path: "/live", label: "Live Trading", icon: <LiveIcon /> },
  ];

  const isActive = (path: string) => location.pathname === path;

  return (
    <AppBar
      position="sticky"
      elevation={0}
      sx={{
        background: "linear-gradient(135deg, #1a1a1a 0%, #262626 100%)",
        borderBottom: "1px solid rgba(148, 163, 184, 0.08)",
        backdropFilter: "blur(20px)",
        boxShadow: "0 1px 3px 0 rgba(0, 0, 0, 0.3), 0 1px 2px 0 rgba(0, 0, 0, 0.06)",
      }}
    >
      <Toolbar sx={{ px: 3, py: 1 }}>
        {/* Logo and Title */}
        <Box sx={{ display: "flex", alignItems: "center", mr: 4 }}>
          <img src="/logo.svg" alt="logo" style={{ width: 56, height: 56, marginRight: 12 }} />
          <Typography
            variant="h5"
            component="div"
            sx={{
              fontWeight: 700,
              fontSize: "1.5rem",
              background: "linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%)",
              backgroundClip: "text",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              letterSpacing: "-0.02em",
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
              variant={isActive(item.path) ? "contained" : "text"}
              sx={{
                borderRadius: 2.5,
                px: 2.5,
                py: 1,
                fontSize: "0.875rem",
                fontWeight: isActive(item.path) ? 600 : 500,
                "&:hover": {
                  transform: "translateY(-1px)",
                  boxShadow: "0 2px 8px 0 rgba(59, 130, 246, 0.2)",
                },
                "& .MuiButton-startIcon": {
                  marginRight: 1,
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
            sx={{
              color: "#cbd5e1",
              "&:hover": {
                backgroundColor: "rgba(59, 130, 246, 0.08)",
                color: "#60a5fa",
              },
            }}
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
                backgroundColor: "#1a1a1a",
                border: "1px solid rgba(148, 163, 184, 0.08)",
                borderRadius: 3,
                backdropFilter: "blur(20px)",
                boxShadow: "0 4px 20px rgba(0, 0, 0, 0.3), 0 1px 3px rgba(0, 0, 0, 0.4)",
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
                  color: isActive(item.path) ? "#ffffff" : "#cbd5e1",
                  backgroundColor: isActive(item.path) ? "#3b82f6" : "transparent",
                  borderRadius: 2,
                  mx: 1,
                  my: 0.5,
                  transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
                  "&:hover": {
                    backgroundColor: isActive(item.path) ? "#2563eb" : "rgba(59, 130, 246, 0.08)",
                    color: isActive(item.path) ? "#ffffff" : "#60a5fa",
                  },
                }}
              >
                <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
                  {item.icon}
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    {item.label}
                  </Typography>
                </Box>
              </MenuItem>
            ))}
          </Menu>
        </Box>

        <Button
          variant="outlined"
          color="inherit"
          href="https://github.com/naowalrahman/malet"
          target="_blank"
          sx={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            ml: 2,
          }}
        >
          <GitHubIcon />
        </Button>
      </Toolbar>
    </AppBar>
  );
}

export default Navbar;
