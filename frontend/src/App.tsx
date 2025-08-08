import { ThemeProvider, createTheme } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { Box } from "@mui/material";

import "@fontsource/inter/100.css";
import "@fontsource/inter/200.css";
import "@fontsource/inter/300.css";
import "@fontsource/inter/400.css";
import "@fontsource/inter/500.css";
import "@fontsource/inter/600.css";
import "@fontsource/inter/700.css";
import "@fontsource/inter/800.css";
import "@fontsource/inter/900.css";

import Navbar from "./components/Navbar";
import Dashboard from "./pages/Dashboard";
import DataExplorer from "./pages/DataExplorer";
import ModelTraining from "./pages/Training";
import Backtesting from "./pages/Backtesting";
import LiveTrading from "./pages/LiveTrading";
import Predict from "./pages/Predict";

// Modern dark theme with sleek gray-blue tones
const theme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: "#3b82f6", // Modern blue
      light: "#60a5fa",
      dark: "#2563eb",
      contrastText: "#ffffff",
    },
    secondary: {
      main: "#f59e0b", // Warm amber accent
      light: "#fbbf24",
      dark: "#d97706",
      contrastText: "#000000",
    },
    background: {
      default: "#0f0f0f", // Deep charcoal
      paper: "#1a1a1a", // Slightly lighter charcoal
    },
    success: {
      main: "#10b981", // Modern green
      light: "#34d399",
      dark: "#059669",
    },
    error: {
      main: "#ef4444", // Modern red
      light: "#f87171",
      dark: "#dc2626",
    },
    warning: {
      main: "#f59e0b", // Warm orange
      light: "#fbbf24",
      dark: "#d97706",
    },
    info: {
      main: "#06b6d4", // Modern cyan
      light: "#22d3ee",
      dark: "#0891b2",
    },
    text: {
      primary: "#f8fafc", // Near white
      secondary: "#cbd5e1", // Light gray
      disabled: "#64748b", // Medium gray
    },
    divider: "rgba(148, 163, 184, 0.12)", // Subtle divider
  },
  typography: {
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    allVariants: {
      fontWeight: 400,
      letterSpacing: "-0.01em",
    },
    h1: {
      fontWeight: 700,
      fontSize: "2.5rem",
      lineHeight: 1.2,
      letterSpacing: "-0.025em",
    },
    h2: {
      fontWeight: 700,
      fontSize: "2rem",
      lineHeight: 1.3,
      letterSpacing: "-0.025em",
    },
    h3: {
      fontWeight: 600,
      fontSize: "1.5rem",
      lineHeight: 1.4,
      letterSpacing: "-0.02em",
    },
    h4: {
      fontWeight: 600,
      fontSize: "1.25rem",
      lineHeight: 1.4,
      letterSpacing: "-0.02em",
    },
    h5: {
      fontWeight: 600,
      fontSize: "1.125rem",
      lineHeight: 1.4,
      letterSpacing: "-0.01em",
    },
    h6: {
      fontWeight: 600,
      fontSize: "1rem",
      lineHeight: 1.5,
      letterSpacing: "-0.01em",
    },
    body1: {
      fontSize: "0.875rem",
      lineHeight: 1.6,
    },
    body2: {
      fontSize: "0.8125rem",
      lineHeight: 1.5,
    },
  },
  shape: {
    borderRadius: 12,
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundColor: "#1a1a1a",
          borderRadius: 16,
          border: "1px solid rgba(148, 163, 184, 0.08)",
          backgroundImage: "none",
          boxShadow: "0 1px 3px 0 rgba(0, 0, 0, 0.3), 0 1px 2px 0 rgba(0, 0, 0, 0.06)",
          transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
          "&:hover": {
            boxShadow: "0 4px 12px 0 rgba(0, 0, 0, 0.4), 0 2px 4px 0 rgba(0, 0, 0, 0.1)",
            borderColor: "rgba(148, 163, 184, 0.12)",
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundColor: "#1a1a1a",
          backgroundImage: "none",
          borderRadius: 12,
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 10,
          textTransform: "none",
          fontWeight: 600,
          fontSize: "0.875rem",
          padding: "8px 16px",
          transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
          boxShadow: "none",
          "&:hover": {
            boxShadow: "0 2px 8px 0 rgba(59, 130, 246, 0.3)",
          },
        },
        contained: {
          background: "linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)",
          color: "#ffffff",
          "&:hover": {
            background: "linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%)",
            color: "#ffffff",
          },
          "&:disabled": {
            background: "rgba(148, 163, 184, 0.3)",
            color: "rgba(148, 163, 184, 0.7)",
          },
        },
        outlined: {
          borderColor: "rgba(148, 163, 184, 0.3)",
          color: "#cbd5e1",
          "&:hover": {
            borderColor: "#3b82f6",
            backgroundColor: "rgba(59, 130, 246, 0.08)",
            color: "#60a5fa",
          },
          "&:disabled": {
            borderColor: "rgba(148, 163, 184, 0.2)",
            color: "rgba(148, 163, 184, 0.5)",
          },
        },
        text: {
          color: "#cbd5e1",
          "&:hover": {
            backgroundColor: "rgba(59, 130, 246, 0.08)",
            color: "#60a5fa",
          },
          "&:disabled": {
            color: "rgba(148, 163, 184, 0.5)",
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          fontWeight: 500,
          fontSize: "0.8125rem",
        },
        colorPrimary: {
          backgroundColor: "#3b82f6",
          color: "#ffffff",
        },
        colorSecondary: {
          backgroundColor: "#f59e0b",
          color: "#000000",
        },
        colorSuccess: {
          backgroundColor: "#10b981",
          color: "#ffffff",
        },
        colorError: {
          backgroundColor: "#ef4444",
          color: "#ffffff",
        },
        colorWarning: {
          backgroundColor: "#f59e0b",
          color: "#000000",
        },
        colorInfo: {
          backgroundColor: "#06b6d4",
          color: "#ffffff",
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: "#1a1a1a",
          borderBottom: "1px solid rgba(148, 163, 184, 0.08)",
          boxShadow: "none",
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: "#1a1a1a",
          borderRight: "1px solid rgba(148, 163, 184, 0.08)",
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          "& .MuiOutlinedInput-root": {
            borderRadius: 10,
            "&:hover .MuiOutlinedInput-notchedOutline": {
              borderColor: "rgba(148, 163, 184, 0.3)",
            },
          },
        },
      },
    },
    MuiIconButton: {
      styleOverrides: {
        root: {
          color: "#cbd5e1",
          "&:hover": {
            backgroundColor: "rgba(59, 130, 246, 0.08)",
            color: "#60a5fa",
          },
        },
      },
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ display: "flex", flexDirection: "column", minHeight: "100vh" }}>
          <Navbar />
          <Box component="main" sx={{ flexGrow: 1 }}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/data" element={<DataExplorer />} />
              <Route path="/training" element={<ModelTraining />} />
              <Route path="/backtesting" element={<Backtesting />} />
              <Route path="/predict" element={<Predict />} />
              <Route path="/live" element={<LiveTrading />} />
            </Routes>
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;
