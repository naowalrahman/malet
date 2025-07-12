import { useState, useEffect } from "react";
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
  Checkbox,
  ListItemText,
  OutlinedInput,
  ToggleButton,
  ToggleButtonGroup,
} from "@mui/material";
import { PlayArrow, ExpandMore, Info } from "@mui/icons-material";
import { apiService } from "../services/api";
import ModernPlot from "../components/ModernPlot";
import ModelDetailsDialog from "../components/ModelDetailsDialog";
import type { TrainedModelDetails, BacktestRequest, BacktestResults } from "../services/api";

function getModelLabel(model: TrainedModelDetails): string {
  return `${model.symbol} - ${model.model_type.toUpperCase()} (${(model.accuracy * 100).toFixed(1)}%)`;
}

function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
  }).format(value);
}

function formatPercentage(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

function ModelChip({ model, onDelete }: { model: TrainedModelDetails; onDelete: () => void }) {
  return (
    <Chip
      label={getModelLabel(model)}
      size="small"
      onDelete={onDelete}
      sx={{
        "& .MuiChip-deleteIcon": {
          fontSize: "18px",
        },
      }}
    />
  );
}

function Backtesting() {
  const [symbol, setSymbol] = useState("SPY");
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [initialCapital, setInitialCapital] = useState(10000);
  const [startDate, setStartDate] = useState("2020-01-01");
  const [endDate, setEndDate] = useState("2025-01-01");
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState<BacktestResults | null>(null);
  const [models, setModels] = useState<TrainedModelDetails[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [detailsDialogOpen, setDetailsDialogOpen] = useState(false);
  const [selectedModelForDetails, setSelectedModelForDetails] = useState<TrainedModelDetails | null>(null);
  const [selectedModelIndex, setSelectedModelIndex] = useState(0);

  useEffect(() => {
    fetchModels();
  }, []);

  async function fetchModels() {
    try {
      const response = await apiService.getTrainedModels();
      setModels(response.models || []);
    } catch (err) {
      setError("Failed to fetch models");
    }
  }

  async function runBacktest() {
    if (selectedModels.length === 0) {
      setError("Please select at least one model");
      return;
    }

    setIsRunning(true);
    setError(null);

    try {
      const request: BacktestRequest = {
        symbol,
        model_ids: selectedModels,
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
  }

  function renderPlots() {
    if (!results?.plots) return null;

    const plots = results.plots;
    const { model_ids } = results.results;

    return (
      <Box sx={{ mt: 3 }}>
        <Typography variant="h5" gutterBottom>
          Performance Visualizations
        </Typography>

        <Grid container spacing={3}>
          {/* Portfolio Comparison Plot */}
          {plots.portfolio_comparison && (
            <Grid size={{ xs: 12 }}>
              <ModernPlot
                data={JSON.parse(plots.portfolio_comparison).data}
                layout={JSON.parse(plots.portfolio_comparison).layout}
                title="Portfolio Value Comparison"
                height={420}
                accentColor="#3b82f6"
              />
            </Grid>
          )}

          {/* Individual Model Analysis with Toggle Buttons */}
          {model_ids && model_ids.length > 0 && (
            <Grid size={{ xs: 12 }}>
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
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      Individual Model Analysis
                    </Typography>
                    <ToggleButtonGroup
                      value={selectedModelIndex}
                      exclusive
                      onChange={(_, newValue) => {
                        if (newValue !== null) {
                          setSelectedModelIndex(newValue);
                        }
                      }}
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
                      {model_ids.map((modelId, index) => {
                        const model = models.find((m) => m.model_id === modelId);
                        return (
                          <ToggleButton key={modelId} value={index}>
                            {model ? getModelLabel(model) : `Model ${modelId.substring(0, 8)}...`}
                          </ToggleButton>
                        );
                      })}
                    </ToggleButtonGroup>
                  </Box>

                  {model_ids.map((modelId, index) => (
                    <Box key={modelId} sx={{ display: selectedModelIndex === index ? "block" : "none" }}>
                      <Grid container spacing={3}>
                        {/* Returns Distribution Plot */}
                        {plots.returns_distribution?.[modelId] && (
                          <Grid size={{ xs: 12, md: 6 }}>
                            <ModernPlot
                              data={JSON.parse(plots.returns_distribution[modelId]).data}
                              layout={JSON.parse(plots.returns_distribution[modelId]).layout}
                              title="Returns Distribution"
                              height={370}
                              accentColor="#10b981"
                            />
                          </Grid>
                        )}

                        {/* Drawdown Analysis Plot */}
                        {plots.drawdown_analysis?.[modelId] && (
                          <Grid size={{ xs: 12, md: 6 }}>
                            <ModernPlot
                              data={JSON.parse(plots.drawdown_analysis[modelId]).data}
                              layout={JSON.parse(plots.drawdown_analysis[modelId]).layout}
                              title="Drawdown Analysis"
                              height={370}
                              accentColor="#ef4444"
                              legendPosition="bottom-left"
                            />
                          </Grid>
                        )}

                        {/* Trade Analysis Plot */}
                        {plots.trade_analysis?.[modelId] && (
                          <Grid size={{ xs: 12 }}>
                            <ModernPlot
                              data={JSON.parse(plots.trade_analysis[modelId]).data}
                              layout={JSON.parse(plots.trade_analysis[modelId]).layout}
                              title="Trading Signals & Price Action"
                              height={420}
                              accentColor="#f59e0b"
                            />
                          </Grid>
                        )}
                      </Grid>
                    </Box>
                  ))}
                </CardContent>
              </Card>
            </Grid>
          )}
        </Grid>
      </Box>
    );
  }

  function renderConfiguration() {
    return (
      <Card sx={{ minHeight: selectedModels.length > 0 ? 350 : 250 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ mb: 2 }}>
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
                <InputLabel>Select Models</InputLabel>
                <Select
                  multiple
                  value={selectedModels}
                  onChange={(event) => {
                    const value = event.target.value;
                    setSelectedModels(typeof value === "string" ? value.split(",") : value);
                  }}
                  input={<OutlinedInput label="Select Models" />}
                  renderValue={(selected) => (
                    <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5 }}>
                      {selected.map((modelId) => {
                        const model = models.find((m) => m.model_id === modelId);
                        return model ? (
                          <ModelChip
                            key={modelId}
                            model={model}
                            onDelete={() => {
                              setSelectedModels(selectedModels.filter((id) => id !== modelId));
                            }}
                          />
                        ) : null;
                      })}
                    </Box>
                  )}
                  MenuProps={{
                    PaperProps: {
                      style: {
                        maxHeight: 300,
                        width: 350,
                      },
                    },
                  }}
                >
                  {models.map((model) => (
                    <MenuItem key={model.model_id} value={model.model_id}>
                      <Checkbox checked={selectedModels.indexOf(model.model_id) > -1} />
                      <ListItemText primary={getModelLabel(model)} />
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            {/* Selected Models Details */}
            {selectedModels.length > 0 && (
              <Grid size={{ xs: 12 }} sx={{ mb: 2 }}>
                    <Typography variant="h6" gutterBottom>
                      Selected Models ({selectedModels.length})
                    </Typography>
                    <Grid container spacing={2}>
                      {selectedModels.map((modelId) => {
                        const model = models.find((m) => m.model_id === modelId);
                        if (!model) return null;

                        return (
                          <Grid size={{ xs: 12, md: 6 }} key={modelId}>
                            <Card variant="outlined" sx={{ p: 2, bgcolor: "rgba(25, 118, 210, 0.1)" }}>
                              <Box
                                sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 1 }}
                              >
                                <Typography variant="subtitle2">{getModelLabel(model)}</Typography>
                                <Button
                                  variant="outlined"
                                  startIcon={<Info />}
                                  onClick={() => {
                                    setSelectedModelForDetails(model);
                                    setDetailsDialogOpen(true);
                                  }}
                                  size="small"
                                >
                                  Details
                                </Button>
                              </Box>
                              <Box sx={{ display: "flex", justifyContent: "space-between" }}>
                                <Typography variant="body2" color="text.secondary">
                                  Created: {new Date(model.created_at).toLocaleDateString()}
                                </Typography>
                              </Box>
                            </Card>
                          </Grid>
                        );
                      })}
                    </Grid>
              </Grid>
            )}

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
                slotProps={{
                  inputLabel: {
                    shrink: true,
                  },
                }}
              />
            </Grid>

            <Grid size={{ xs: 12, md: 4 }}>
              <TextField
                fullWidth
                label="End Date"
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                slotProps={{
                  inputLabel: {
                    shrink: true,
                  },
                }}
              />
            </Grid>
          </Grid>

          <Box sx={{ mt: 3 }}>
            <Button
              variant="contained"
              startIcon={<PlayArrow />}
              onClick={runBacktest}
              disabled={isRunning || selectedModels.length === 0}
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
  }

  function renderResults() {
    if (!results) return null;

    const { buy_and_hold, ml_strategies, model_ids } = results.results;

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
                        {model_ids.map((modelId) => {
                          const model = models.find((m) => m.model_id === modelId);
                          return (
                            <TableCell key={modelId} align="right">
                              {model ? getModelLabel(model) : `Model ${modelId.substring(0, 8)}...`}
                            </TableCell>
                          );
                        })}
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      <TableRow>
                        <TableCell>Final Value</TableCell>
                        <TableCell align="right">{formatCurrency(buy_and_hold.final_value)}</TableCell>
                        {model_ids.map((modelId) => (
                          <TableCell key={modelId} align="right">
                            {formatCurrency(ml_strategies[modelId].final_value)}
                          </TableCell>
                        ))}
                      </TableRow>
                      <TableRow>
                        <TableCell>Total Return</TableCell>
                        <TableCell align="right">{formatPercentage(buy_and_hold.total_return)}</TableCell>
                        {model_ids.map((modelId) => (
                          <TableCell key={modelId} align="right">
                            <Chip
                              label={formatPercentage(ml_strategies[modelId].total_return)}
                              color={
                                ml_strategies[modelId].total_return > buy_and_hold.total_return ? "success" : "error"
                              }
                              size="small"
                            />
                          </TableCell>
                        ))}
                      </TableRow>
                      <TableRow>
                        <TableCell>Sharpe Ratio</TableCell>
                        <TableCell align="right">{buy_and_hold.sharpe_ratio?.toFixed(3)}</TableCell>
                        {model_ids.map((modelId) => (
                          <TableCell key={modelId} align="right">
                            <Chip
                              label={ml_strategies[modelId].sharpe_ratio?.toFixed(3)}
                              color={
                                ml_strategies[modelId].sharpe_ratio > buy_and_hold.sharpe_ratio ? "success" : "error"
                              }
                              size="small"
                            />
                          </TableCell>
                        ))}
                      </TableRow>
                      <TableRow>
                        <TableCell>Max Drawdown</TableCell>
                        <TableCell align="right">{formatPercentage(buy_and_hold.max_drawdown)}</TableCell>
                        {model_ids.map((modelId) => (
                          <TableCell key={modelId} align="right">
                            <Chip
                              label={formatPercentage(ml_strategies[modelId].max_drawdown)}
                              color={
                                ml_strategies[modelId].max_drawdown < buy_and_hold.max_drawdown ? "success" : "error"
                              }
                              size="small"
                            />
                          </TableCell>
                        ))}
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
                      <TableCell align="right">{buy_and_hold.total_trades}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Win Rate</TableCell>
                      <TableCell align="right">{formatPercentage(buy_and_hold.win_rate)}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Volatility</TableCell>
                      <TableCell align="right">{formatPercentage(buy_and_hold.volatility)}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Calmar Ratio</TableCell>
                      <TableCell align="right">{buy_and_hold.calmar_ratio?.toFixed(3)}</TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </AccordionDetails>
            </Accordion>
          </Grid>

          {/* ML Strategies Details */}
          <Grid size={{ xs: 12, md: 6 }}>
            {model_ids.map((modelId) => {
              const model = models.find((m) => m.model_id === modelId);
              return (
                <Accordion key={modelId} sx={{ mb: 1 }}>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Typography variant="h6">
                      {model ? getModelLabel(model) : `Model ${modelId.substring(0, 8)}...`} Details
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Table size="small">
                      <TableBody>
                        <TableRow>
                          <TableCell>Total Trades</TableCell>
                          <TableCell align="right">{ml_strategies[modelId].total_trades}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Win Rate</TableCell>
                          <TableCell align="right">{formatPercentage(ml_strategies[modelId].win_rate)}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Volatility</TableCell>
                          <TableCell align="right">{formatPercentage(ml_strategies[modelId].volatility)}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Calmar Ratio</TableCell>
                          <TableCell align="right">{ml_strategies[modelId].calmar_ratio?.toFixed(3)}</TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </AccordionDetails>
                </Accordion>
              );
            })}
          </Grid>
        </Grid>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        display: "flex",
        justifyContent: "center",
        minHeight: "100vh",
        py: 4,
        backgroundColor: "background.default",
      }}
    >
      <Container maxWidth="xl">
        <Box sx={{ mb: 4, textAlign: "center" }}>
          <Typography variant="h3" component="h1" gutterBottom fontWeight={700}>
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

        <Box sx={{ minWidth: 800 }}>
          {renderConfiguration()}
          {renderResults()}
          {renderPlots()}
        </Box>

        {/* Model Details Dialog */}
        <ModelDetailsDialog
          open={detailsDialogOpen}
          onClose={() => {
            setDetailsDialogOpen(false);
            setSelectedModelForDetails(null);
          }}
          modelDetails={selectedModelForDetails}
        />
      </Container>
    </Box>
  );
}

export default Backtesting;
