import React, { useState, useEffect } from "react";
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Box,
  Alert,
  LinearProgress,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from "@mui/material";
import {
  PlayArrow,
  TrendingUp,
  TrendingDown,
  ExpandMore,
} from "@mui/icons-material";
import { apiService } from "../services/api";
import type { Model, BacktestRequest, BacktestResults } from "../services/api";

const Backtesting: React.FC = () => {
  const [symbol, setSymbol] = useState("AAPL");
  const [selectedModel, setSelectedModel] = useState("");
  const [initialCapital, setInitialCapital] = useState(10000);
  const [startDate, setStartDate] = useState("2023-01-01");
  const [endDate, setEndDate] = useState("2024-01-01");
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState<BacktestResults | null>(null);
  const [models, setModels] = useState<Model[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await apiService.getModels();
      setModels(response.models || []);
    } catch (err) {
      setError("Failed to fetch models");
    }
  };

  const runBacktest = async () => {
    if (!selectedModel) {
      setError("Please select a model");
      return;
    }

    setIsRunning(true);
    setError(null);

    try {
      const request: BacktestRequest = {
        symbol,
        model_id: selectedModel,
        initial_capital: initialCapital,
        start_date: startDate,
        end_date: endDate,
      };

      const backtestResults = await apiService.runBacktest(request);
      setResults(backtestResults);
    } catch (err) {
      setError("Failed to run backtest");
    } finally {
      setIsRunning(false);
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  const renderConfiguration = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Backtest Configuration
        </Typography>

        <Grid container spacing={3}>
          <Grid size={{ xs: 12, md: 6 }}>
            <TextField
              fullWidth
              label="Symbol"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              placeholder="AAPL"
            />
          </Grid>

          <Grid size={{ xs: 12, md: 6 }}>
            <FormControl fullWidth>
              <InputLabel>Model</InputLabel>
              <Select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                label="Model"
              >
                {models.map((model) => (
                  <MenuItem key={model.model_id} value={model.model_id}>
                    {model.symbol} - {model.model_type} (Accuracy:{" "}
                    {formatPercentage(model.accuracy)})
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>

          <Grid size={{ xs: 12, md: 4 }}>
            <TextField
              fullWidth
              label="Initial Capital"
              type="number"
              value={initialCapital}
              onChange={(e) => setInitialCapital(Number(e.target.value))}
            />
          </Grid>

          <Grid size={{ xs: 12, md: 4 }}>
            <TextField
              fullWidth
              label="Start Date"
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              InputLabelProps={{ shrink: true }}
            />
          </Grid>

          <Grid size={{ xs: 12, md: 4 }}>
            <TextField
              fullWidth
              label="End Date"
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              InputLabelProps={{ shrink: true }}
            />
          </Grid>
        </Grid>

        <Box sx={{ mt: 3 }}>
          <Button
            variant="contained"
            startIcon={<PlayArrow />}
            onClick={runBacktest}
            disabled={isRunning || !selectedModel}
            fullWidth
          >
            {isRunning ? "Running Backtest..." : "Run Backtest"}
          </Button>
        </Box>

        {isRunning && (
          <Box sx={{ mt: 2 }}>
            <LinearProgress />
          </Box>
        )}
      </CardContent>
    </Card>
  );

  const renderResults = () => {
    if (!results) return null;

    const { buy_and_hold, ml_strategy } = results.results;

    return (
      <Box sx={{ mt: 3 }}>
        <Grid container spacing={3}>
          {/* Strategy Comparison */}
          <Grid size={{ xs: 12 }}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Strategy Comparison
                </Typography>

                <TableContainer component={Paper} variant="outlined">
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Metric</TableCell>
                        <TableCell align="right">Buy & Hold</TableCell>
                        <TableCell align="right">ML Strategy</TableCell>
                        <TableCell align="right">Difference</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      <TableRow>
                        <TableCell>Final Value</TableCell>
                        <TableCell align="right">
                          {formatCurrency(buy_and_hold.final_value)}
                        </TableCell>
                        <TableCell align="right">
                          {formatCurrency(ml_strategy.final_value)}
                        </TableCell>
                        <TableCell align="right">
                          <Box
                            display="flex"
                            alignItems="center"
                            justifyContent="flex-end"
                          >
                            {ml_strategy.final_value >
                            buy_and_hold.final_value ? (
                              <>
                                <TrendingUp color="success" />
                                <Typography color="success.main">
                                  {formatCurrency(
                                    ml_strategy.final_value -
                                      buy_and_hold.final_value
                                  )}
                                </Typography>
                              </>
                            ) : (
                              <>
                                <TrendingDown color="error" />
                                <Typography color="error.main">
                                  {formatCurrency(
                                    ml_strategy.final_value -
                                      buy_and_hold.final_value
                                  )}
                                </Typography>
                              </>
                            )}
                          </Box>
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Total Return</TableCell>
                        <TableCell align="right">
                          {formatPercentage(buy_and_hold.total_return)}
                        </TableCell>
                        <TableCell align="right">
                          {formatPercentage(ml_strategy.total_return)}
                        </TableCell>
                        <TableCell align="right">
                          <Chip
                            label={formatPercentage(
                              ml_strategy.total_return -
                                buy_and_hold.total_return
                            )}
                            color={
                              ml_strategy.total_return >
                              buy_and_hold.total_return
                                ? "success"
                                : "error"
                            }
                            size="small"
                          />
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Sharpe Ratio</TableCell>
                        <TableCell align="right">
                          {buy_and_hold.sharpe_ratio?.toFixed(3)}
                        </TableCell>
                        <TableCell align="right">
                          {ml_strategy.sharpe_ratio?.toFixed(3)}
                        </TableCell>
                        <TableCell align="right">
                          <Chip
                            label={(
                              ml_strategy.sharpe_ratio -
                              buy_and_hold.sharpe_ratio
                            ).toFixed(3)}
                            color={
                              ml_strategy.sharpe_ratio >
                              buy_and_hold.sharpe_ratio
                                ? "success"
                                : "error"
                            }
                            size="small"
                          />
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Max Drawdown</TableCell>
                        <TableCell align="right">
                          {formatPercentage(buy_and_hold.max_drawdown)}
                        </TableCell>
                        <TableCell align="right">
                          {formatPercentage(ml_strategy.max_drawdown)}
                        </TableCell>
                        <TableCell align="right">
                          <Chip
                            label={formatPercentage(
                              ml_strategy.max_drawdown -
                                buy_and_hold.max_drawdown
                            )}
                            color={
                              ml_strategy.max_drawdown <
                              buy_and_hold.max_drawdown
                                ? "success"
                                : "error"
                            }
                            size="small"
                          />
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </Grid>

          {/* Detailed Metrics */}
          <Grid size={{ xs: 12, md: 6 }}>
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography variant="h6">Buy & Hold Details</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Table size="small">
                  <TableBody>
                    <TableRow>
                      <TableCell>Total Trades</TableCell>
                      <TableCell align="right">
                        {buy_and_hold.total_trades}
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Win Rate</TableCell>
                      <TableCell align="right">
                        {formatPercentage(buy_and_hold.win_rate)}
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Volatility</TableCell>
                      <TableCell align="right">
                        {formatPercentage(buy_and_hold.volatility)}
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Calmar Ratio</TableCell>
                      <TableCell align="right">
                        {buy_and_hold.calmar_ratio?.toFixed(3)}
                      </TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </AccordionDetails>
            </Accordion>
          </Grid>

          <Grid size={{ xs: 12, md: 6 }}>
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography variant="h6">ML Strategy Details</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Table size="small">
                  <TableBody>
                    <TableRow>
                      <TableCell>Total Trades</TableCell>
                      <TableCell align="right">
                        {ml_strategy.total_trades}
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Win Rate</TableCell>
                      <TableCell align="right">
                        {formatPercentage(ml_strategy.win_rate)}
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Volatility</TableCell>
                      <TableCell align="right">
                        {formatPercentage(ml_strategy.volatility)}
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Calmar Ratio</TableCell>
                      <TableCell align="right">
                        {ml_strategy.calmar_ratio?.toFixed(3)}
                      </TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </AccordionDetails>
            </Accordion>
          </Grid>
        </Grid>
      </Box>
    );
  };

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Backtesting
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Test your ML models against historical data to evaluate performance
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {renderConfiguration()}
      {renderResults()}
    </Container>
  );
};

export default Backtesting;
