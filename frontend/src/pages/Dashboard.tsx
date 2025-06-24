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
} from "@mui/material";
import {
  TrendingUp,
  TrendingDown,
  Assessment,
  ModelTraining,
  Refresh,
  Psychology,
  AccountBalance,
  Timeline,
} from "@mui/icons-material";
import { useTheme } from "@mui/material/styles";
import { useNavigate } from "react-router-dom";
import { apiService } from "../services/api";
import type { ModelDetails, MarketAnalysis } from "../services/api";

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

const Dashboard: React.FC = () => {
  const theme = useTheme();
  const navigate = useNavigate();
  const [models, setModels] = useState<ModelDetails[]>([]);
  const [marketAnalysis, setMarketAnalysis] = useState<MarketAnalysis | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch models
      const modelsResponse = await apiService.getTrainedModels();
      setModels(modelsResponse.models);

      // Fetch market analysis for SPY (S&P 500)
      try {
        const analysisResponse = await apiService.getMarketAnalysis("SPY");
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
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <MetricCard
            title="Active Models"
            value={activeModels.length}
            icon={<Psychology sx={{ fontSize: 32 }} />}
            color="primary"
          />
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <MetricCard
            title="Avg Accuracy"
            value={`${(avgAccuracy * 100).toFixed(1)}%`}
            icon={<Assessment sx={{ fontSize: 32 }} />}
            color="success"
          />
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <MetricCard
            title="Market Price"
            value={marketAnalysis ? `$${marketAnalysis.current_price.toFixed(2)}` : "$0.00"}
            change={marketAnalysis ? marketAnalysis.price_change_pct : undefined}
            icon={<Timeline sx={{ fontSize: 32 }} />}
            color="info"
          />
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <MetricCard
            title="Portfolio Value"
            value="$10,000"
            change={2.34}
            icon={<AccountBalance sx={{ fontSize: 32 }} />}
            color="secondary"
          />
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* Market Overview */}
        <Grid size={{ xs: 12, lg: 8 }}>
          <Card sx={{ height: 400 }}>
            <CardContent>
              <Box
                sx={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  mb: 3,
                }}
              >
                <Typography variant="h5" sx={{ fontWeight: 600 }}>
                  Market Overview
                </Typography>
                {marketAnalysis && (
                  <Chip
                    label={getSignalText(marketAnalysis.combined_signal)}
                    sx={{
                      bgcolor: `${getSignalColor(marketAnalysis.combined_signal)}20`,
                      color: getSignalColor(marketAnalysis.combined_signal),
                      fontWeight: 600,
                    }}
                  />
                )}
              </Box>

              {marketAnalysis ? (
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
                            value={marketAnalysis.rsi}
                            sx={{
                              height: 8,
                              borderRadius: 4,
                              bgcolor: "rgba(255, 255, 255, 0.1)",
                              "& .MuiLinearProgress-bar": {
                                bgcolor:
                                  marketAnalysis.rsi > 70
                                    ? theme.palette.error.main
                                    : marketAnalysis.rsi < 30
                                      ? theme.palette.success.main
                                      : theme.palette.warning.main,
                              },
                            }}
                          />
                        </Box>
                        <Typography variant="body2" sx={{ minWidth: 35, fontWeight: 600 }}>
                          {marketAnalysis.rsi.toFixed(0)}
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
                            value={marketAnalysis.bollinger_position * 100}
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
                          {(marketAnalysis.bollinger_position * 100).toFixed(0)}%
                        </Typography>
                      </Box>
                    </Box>

                    <Box>
                      <Typography variant="subtitle2" sx={{ color: "text.secondary", mb: 1 }}>
                        Volatility
                      </Typography>
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        {(marketAnalysis.volatility * 100).toFixed(2)}%
                      </Typography>
                    </Box>
                  </Grid>

                  <Grid size={{ xs: 12, md: 6 }}>
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="subtitle2" sx={{ color: "text.secondary", mb: 1 }}>
                        Support Levels
                      </Typography>
                      {marketAnalysis.support_levels.map((level, index) => (
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
                      {marketAnalysis.resistance_levels.map((level, index) => (
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
          <Card sx={{ height: 400 }}>
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
                    key={model.model_id}
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
};

export default Dashboard;
