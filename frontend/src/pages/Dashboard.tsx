import React, { useState, useEffect } from "react";
import {
  Container,
  Card,
  CardContent,
  Typography,
  Box,
  Button,
  Chip,
  LinearProgress,
  Alert,
  IconButton,
  Tooltip,
  Grid,
  ToggleButton,
  ToggleButtonGroup,
  Divider,
} from "@mui/material";
import {
  TrendingUp,
  TrendingDown,
  Assessment,
  ModelTraining,
  Refresh,
  Psychology,
  Timeline,
  QuestionMarkRounded,
} from "@mui/icons-material";
import { useTheme } from "@mui/material/styles";
import { useNavigate } from "react-router-dom";
import { apiService } from "../services/api";
import type { TrainedModelDetails, MarketAnalysis } from "../services/api";
import AIAnalysis from "../components/AIAnalysis";

interface MetricCardProps {
  title: string;
  value: string | number;
  change?: number;
  icon: React.ReactNode;
  color: "primary" | "secondary" | "success" | "error" | "warning" | "info";
}

const MetricCard: React.FC<MetricCardProps> = ({ title, value, change, icon, color }) => {
  const theme = useTheme();

  return (
    <Card sx={{ height: "100%", position: "relative", overflow: "visible" }}>
      <CardContent>
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <Box>
            <Typography variant="h6" sx={{ mb: 1, fontWeight: 600 }}>
              {title}
            </Typography>
            <Typography variant="h4" sx={{ fontWeight: 700, color: theme.palette[color].main }}>
              {value}
            </Typography>
            {change !== undefined && (
              <Box sx={{ display: "flex", alignItems: "center", mt: 1 }}>
                {change >= 0 ? (
                  <TrendingUp
                    sx={{
                      fontSize: 16,
                      color: theme.palette.success.main,
                      mr: 0.5,
                    }}
                  />
                ) : (
                  <TrendingDown
                    sx={{
                      fontSize: 16,
                      color: theme.palette.error.main,
                      mr: 0.5,
                    }}
                  />
                )}
                <Typography
                  variant="body2"
                  sx={{
                    color: change >= 0 ? theme.palette.success.main : theme.palette.error.main,
                    fontWeight: 600,
                  }}
                >
                  {Math.abs(change).toFixed(2)}%
                </Typography>
              </Box>
            )}
          </Box>
          <Box
            sx={{
              p: 2,
              borderRadius: 2,
              backgroundColor: `${theme.palette[color].main}20`,
              color: theme.palette[color].main,
            }}
          >
            {icon}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

function Dashboard() {
  const theme = useTheme();
  const navigate = useNavigate();
  const [models, setModels] = useState<TrainedModelDetails[]>([]);
  const [marketAnalysis, setMarketAnalysis] = useState<MarketAnalysis[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState<string>("SPY");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const watchedSymbols = ["SPY", "DIA", "QQQ"];

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch models
      const modelsResponse = await apiService.getTrainedModels();
      setModels(modelsResponse.models);

      // Fetch market analysis for SPY, DIA, and QQQ
      try {
        const analysisResponse = await apiService.getMarketAnalysis(watchedSymbols);
        setMarketAnalysis(analysisResponse);
      } catch (analysisError) {
        console.warn("Could not fetch market analysis:", analysisError);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch data");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const activeModels = models.filter((model) => model.accuracy > 0.5);
  const avgAccuracy = models.length > 0 ? models.reduce((sum, model) => sum + model.accuracy, 0) / models.length : 0;

  // Get the selected symbol's market analysis
  const selectedAnalysis = marketAnalysis.find((analysis) => analysis.symbol === selectedSymbol);

  const getSignalColor = (signal: number) => {
    if (signal > 0) return theme.palette.success.main;
    if (signal < 0) return theme.palette.error.main;
    return theme.palette.warning.main;
  };

  const getSignalText = (signal: number) => {
    if (signal > 0) return "BULLISH";
    if (signal < 0) return "BEARISH";
    return "NEUTRAL";
  };

  const handleSymbolChange = (_event: React.MouseEvent<HTMLElement>, newSymbol: string | null) => {
    if (newSymbol !== null) {
      setSelectedSymbol(newSymbol);
    }
  };

  if (loading) {
    return (
      <Container maxWidth="xl" sx={{ py: 4 }}>
        <Box sx={{ width: "100%" }}>
          <LinearProgress />
          <Typography sx={{ mt: 2, textAlign: "center" }}>Loading dashboard...</Typography>
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4, textAlign: "center" }}>
        <Typography variant="h3" component="h1" gutterBottom fontWeight={700}>
          Trading Dashboard
        </Typography>
        <Typography variant="body1" color="text.secondary">
          AI-powered trading insights and analytics.
        </Typography>
        <Box sx={{ mt: 2 }}>
          <Tooltip title="Refresh Data">
            <IconButton onClick={fetchData}>
              <Refresh />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Key Metrics */}
      <Grid container spacing={3} sx={{ mb: 4, justifyContent: "center" }}>
        <Grid size={{ xs: 12, sm: 6, md: 4 }}>
          <MetricCard
            title="Active Models"
            value={activeModels.length}
            icon={<Psychology sx={{ fontSize: 32 }} />}
            color="primary"
          />
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 4 }}>
          <MetricCard
            title="Avg Accuracy"
            value={`${(avgAccuracy * 100).toFixed(1)}%`}
            icon={<Assessment sx={{ fontSize: 32 }} />}
            color="success"
          />
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 4 }}>
          <MetricCard
            title={`${selectedSymbol} Price`}
            value={selectedAnalysis ? `$${selectedAnalysis.current_price.toFixed(2)}` : "$0.00"}
            change={selectedAnalysis ? selectedAnalysis.price_change_pct : undefined}
            icon={<Timeline sx={{ fontSize: 32 }} />}
            color="info"
          />
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* Market Overview */}
        <Grid size={{ xs: 12, lg: 8 }}>
          <Card>
            <CardContent>
              <Box
                sx={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  mb: 3,
                  flexWrap: "wrap",
                  gap: 2,
                }}
              >
                <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                  <Typography variant="h5" sx={{ fontWeight: 600 }}>
                    Market Overview
                  </Typography>

                  <ToggleButtonGroup
                    value={selectedSymbol}
                    exclusive
                    onChange={handleSymbolChange}
                    size="small"
                    sx={{
                      "& .MuiToggleButton-root": {
                        px: 2,
                        py: 0.5,
                        fontSize: "0.875rem",
                        fontWeight: 600,
                      },
                    }}
                  >
                    {watchedSymbols.map((symbol) => (
                      <ToggleButton key={symbol} value={symbol}>
                        {symbol}
                      </ToggleButton>
                    ))}
                  </ToggleButtonGroup>
                </Box>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  {selectedAnalysis && (
                    <Chip
                      label={getSignalText(selectedAnalysis.combined_signal)}
                      sx={{
                        bgcolor: `${getSignalColor(selectedAnalysis.combined_signal)}20`,
                        color: getSignalColor(selectedAnalysis.combined_signal),
                        fontWeight: 600,
                      }}
                    />
                  )}
                  <Tooltip
                    title={
                      <Box sx={{ maxWidth: 320, p: 1 }}>
                        <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 0.5 }}>
                          Market Overview
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          This section analyzes data from the past 60 days to provide a summary of key market indicators
                          and technical signals for your selected symbol, using Google Gemini for AI-powered
                          interpretation. Market analysis is cached for 24 hours. Use the symbol buttons to switch
                          between different symbols.
                        </Typography>
                      </Box>
                    }
                    placement="top"
                    arrow
                  >
                    <IconButton sx={{ p: 0.1 }}>
                      <QuestionMarkRounded sx={{ fontSize: 16 }} />
                    </IconButton>
                  </Tooltip>
                </Box>
              </Box>

              {selectedAnalysis ? (
                <>
                  <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
                    <Assessment sx={{ color: theme.palette.primary.main, mr: 1 }} />
                    <Typography
                      variant="h6"
                      sx={{
                        fontWeight: 700,
                        letterSpacing: 1,
                        color: theme.palette.primary.main,
                        textShadow: "0 1px 4px rgba(0,0,0,0.08)",
                        textTransform: "uppercase",
                      }}
                    >
                      {selectedSymbol} -{" "}
                      {
                        {
                          SPY: "S&P 500 ETF",
                          DIA: "Dow Jones ETF",
                          QQQ: "NASDAQ ETF",
                        }[selectedAnalysis.symbol]
                      }
                    </Typography>
                  </Box>
                  <Grid container spacing={3}>
                    <Grid size={{ xs: 12, md: 6 }}>
                      <Box sx={{ mb: 2 }}>
                        <Typography variant="subtitle2" sx={{ color: "text.secondary", mb: 1 }}>
                          RSI (Relative Strength Index)
                        </Typography>
                        <Box sx={{ display: "flex", alignItems: "center" }}>
                          <Box sx={{ width: "100%", mr: 1 }}>
                            <LinearProgress
                              variant="determinate"
                              value={selectedAnalysis.rsi}
                              sx={{
                                height: 8,
                                borderRadius: 4,
                                bgcolor: "rgba(255, 255, 255, 0.1)",
                                "& .MuiLinearProgress-bar": {
                                  bgcolor:
                                    selectedAnalysis.rsi > 70
                                      ? theme.palette.error.main
                                      : selectedAnalysis.rsi < 30
                                        ? theme.palette.success.main
                                        : theme.palette.warning.main,
                                },
                              }}
                            />
                          </Box>
                          <Typography variant="body2" sx={{ minWidth: 35, fontWeight: 600 }}>
                            {selectedAnalysis.rsi.toFixed(0)}
                          </Typography>
                        </Box>
                      </Box>

                      <Box sx={{ mb: 2 }}>
                        <Typography variant="subtitle2" sx={{ color: "text.secondary", mb: 1 }}>
                          Bollinger Position
                        </Typography>
                        <Box sx={{ display: "flex", alignItems: "center" }}>
                          <Box sx={{ width: "100%", mr: 1 }}>
                            <LinearProgress
                              variant="determinate"
                              value={selectedAnalysis.bollinger_position * 100}
                              sx={{
                                height: 8,
                                borderRadius: 4,
                                bgcolor: "rgba(255, 255, 255, 0.1)",
                                "& .MuiLinearProgress-bar": {
                                  bgcolor: theme.palette.info.main,
                                },
                              }}
                            />
                          </Box>
                          <Typography variant="body2" sx={{ minWidth: 35, fontWeight: 600 }}>
                            {(selectedAnalysis.bollinger_position * 100).toFixed(0)}%
                          </Typography>
                        </Box>
                      </Box>

                      <Box>
                        <Typography variant="subtitle2" sx={{ color: "text.secondary", mb: 1 }}>
                          Average True Range
                        </Typography>
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                          ${selectedAnalysis.average_true_range.toFixed(2)}
                        </Typography>
                      </Box>
                    </Grid>

                    <Grid size={{ xs: 12, md: 6 }}>
                      <Box sx={{ mb: 2 }}>
                        <Typography variant="subtitle2" sx={{ color: "text.secondary", mb: 1 }}>
                          Support Levels
                        </Typography>
                        {selectedAnalysis.support_levels.map((level, index) => (
                          <Typography
                            key={index}
                            variant="body2"
                            sx={{
                              fontWeight: 600,
                              color: theme.palette.success.main,
                            }}
                          >
                            S{index + 1}: ${level.toFixed(2)}
                          </Typography>
                        ))}
                      </Box>

                      <Box>
                        <Typography variant="subtitle2" sx={{ color: "text.secondary", mb: 1 }}>
                          Resistance Levels
                        </Typography>
                        {selectedAnalysis.resistance_levels.map((level, index) => (
                          <Typography
                            key={index}
                            variant="body2"
                            sx={{
                              fontWeight: 600,
                              color: theme.palette.error.main,
                            }}
                          >
                            R{index + 1}: ${level.toFixed(2)}
                          </Typography>
                        ))}
                      </Box>
                    </Grid>
                  </Grid>
                  <Divider sx={{ my: 3, borderColor: theme.palette.divider }} />
                  <AIAnalysis symbol={selectedSymbol} />
                </>
              ) : (
                <Box sx={{ textAlign: "center", py: 4 }}>
                  <Typography variant="body1" sx={{ color: "text.secondary" }}>
                    Market data unavailable
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Quick Actions */}
        <Grid size={{ xs: 12, lg: 4 }}>
          <Card>
            <CardContent>
              <Typography variant="h5" sx={{ fontWeight: 600, mb: 3 }}>
                Quick Actions
              </Typography>

              <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                <Button
                  variant="contained"
                  size="large"
                  startIcon={<ModelTraining />}
                  onClick={() => navigate("/training")}
                  sx={{ justifyContent: "flex-start", py: 2 }}
                >
                  Train New Model
                </Button>

                <Button
                  variant="outlined"
                  size="large"
                  startIcon={<Assessment />}
                  onClick={() => navigate("/backtesting")}
                  sx={{ justifyContent: "flex-start", py: 2 }}
                >
                  Run Backtest
                </Button>

                <Button
                  variant="outlined"
                  size="large"
                  startIcon={<Timeline />}
                  onClick={() => navigate("/data")}
                  sx={{ justifyContent: "flex-start", py: 2 }}
                >
                  Explore Data
                </Button>

                <Button
                  variant="outlined"
                  size="large"
                  startIcon={<TrendingUp />}
                  onClick={() => navigate("/live")}
                  sx={{ justifyContent: "flex-start", py: 2 }}
                >
                  Live Trading
                </Button>
              </Box>

              <Box
                sx={{
                  mt: 3,
                  p: 2,
                  bgcolor: "rgba(25, 118, 210, 0.1)",
                  borderRadius: 2,
                }}
              >
                <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                  Recent Models
                </Typography>
                {models.slice(0, 3).map((model) => (
                  <Box
                    key={model.model_type}
                    sx={{
                      display: "flex",
                      justifyContent: "space-between",
                      mb: 1,
                    }}
                  >
                    <Typography variant="body2">
                      {model.symbol} ({model.model_type})
                    </Typography>
                    <Typography
                      variant="body2"
                      sx={{
                        fontWeight: 600,
                        color: theme.palette.primary.main,
                      }}
                    >
                      {(model.accuracy * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                ))}
                {models.length === 0 && (
                  <Typography variant="body2" sx={{ color: "text.secondary" }}>
                    No models trained yet
                  </Typography>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
}

export default Dashboard;
