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
  Stepper,
  Step,
  StepLabel,
  Alert,
  CircularProgress,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Slider,
} from "@mui/material";
import { PlayArrow, Download, Delete, ExpandMore, Info } from "@mui/icons-material";
import { apiService } from "../services/api";
import type { ModelDetails, TrainedModelDetails, TrainingRequest } from "../services/api";
import ModelDetailsDialog from "../components/ModelDetailsDialog";

const steps = ["Configure Model", "Training", "Results"];

function ModelTraining() {
  const [activeStep, setActiveStep] = useState(0);
  const [symbol, setSymbol] = useState("SPY");
  const [modelType, setModelType] = useState("lstm");
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(32);
  const [learningRate, setLearningRate] = useState(0.004);
  const [sequenceLength, setSequenceLength] = useState(120);
  const [predictionHorizon] = useState(5);
  const [threshold] = useState(0.02);
  const [startDate, setStartDate] = useState("2010-01-01");
  const [endDate, setEndDate] = useState("2020-01-01");

  const [isTraining, setIsTraining] = useState(false);
  const [currentModelId, setCurrentModelId] = useState<string | null>(null);
  const [trainingResults, setTrainingResults] = useState<any>(null);
  const [trainedModels, setTrainedModels] = useState<TrainedModelDetails[]>([]);
  const [availableModels, setAvailableModels] = useState<ModelDetails[]>([]);
  const [pollInterval, setPollInterval] = useState<NodeJS.Timeout | null>(null);

  const [error, setError] = useState<string | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [modelToDelete, setModelToDelete] = useState<string | null>(null);
  const [detailsDialogOpen, setDetailsDialogOpen] = useState(false);
  const [selectedModelForDetails, setSelectedModelForDetails] = useState<TrainedModelDetails | null>(null);

  useEffect(() => {
    fetchModels();
  }, []);

  // cleanup
  useEffect(() => {
    return () => {
      if (pollInterval) {
        clearInterval(pollInterval);
      }
    };
  }, [pollInterval]);

  async function fetchModels() {
    try {
      const response = await apiService.getTrainedModels();
      setTrainedModels(response.models);
    } catch (err) {
      setError("Failed to fetch models");
    }
    try {
      const response = await apiService.getAvailableModels();
      setAvailableModels(response.models);
    } catch (err) {
      setError("Failed to fetch available models");
    }
  }

  async function handleStartTraining() {
    try {
      setError(null);
      setIsTraining(true);
      setActiveStep(1);

      const request: TrainingRequest = {
        symbol,
        model_type: modelType,
        epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        sequence_length: sequenceLength,
        prediction_horizon: predictionHorizon,
        threshold,
        start_date: startDate,
        end_date: endDate,
      };

      // Start training and get job ID
      const response = await apiService.trainModel(request);
      const jobId = response.job_id;

      // Poll for training completion every 2 seconds
      const interval = setInterval(async () => {
        try {
          const status = await apiService.getTrainingStatus(jobId);
          if (status.status === "completed") {
            clearInterval(interval);
            setPollInterval(null);
            setIsTraining(false);
            setActiveStep(2);
            setTrainingResults(status.results || status.result);
            if (status.result?.model_id) {
              setCurrentModelId(status.result.model_id);
            }
            fetchModels(); // Refresh models list
          } else if (status.status === "failed") {
            clearInterval(interval);
            setPollInterval(null);
            throw new Error(status.error || "Training failed");
          }
          // If status is still "running" or "pending", continue polling
        } catch (statusError: any) {
          // Handle network errors gracefully - don't stop polling immediately
          // Only stop if it's a persistent error (not timeout/network issues)

          // If it's an axios timeout or network error, continue polling
          if (
            statusError.code === "ECONNABORTED" ||
            statusError.message?.includes("timeout") ||
            statusError.message?.includes("Network Error")
          ) {
            console.warn(
              "Error checking training status... continuing polling because error was timeout or network error"
            );
            return; // Continue polling
          }

          // For other errors, stop polling
          clearInterval(interval);
          setPollInterval(null);
          console.warn(`Error checking training status: ${statusError}`);
          throw statusError;
        }
      }, 2000); // Poll every 2 seconds

      setPollInterval(interval);
    } catch (err) {
      setError(`Failed to train model: ${err}`);
      setIsTraining(false);
      setActiveStep(0);
      if (pollInterval) {
        clearInterval(pollInterval);
        setPollInterval(null);
      }
    }
  }

  const handleDeleteModel = async (modelId: string) => {
    try {
      await apiService.deleteModel(modelId);
      fetchModels();
      setDeleteDialogOpen(false);
      setModelToDelete(null);
    } catch (err) {
      setError("Failed to delete model");
    }
  };

  const handleTrainInBackground = () => {
    if (pollInterval) {
      clearInterval(pollInterval);
      setPollInterval(null);
    }
    setIsTraining(false);
    setActiveStep(0);
    setError("Training in background");
  };

  function renderConfigurationStep() {
    return (
      <Card>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            Model Configuration
          </Typography>

          <Grid container spacing={3}>
            <Grid size={{ xs: 12, md: 6 }}>
              <TextField
                fullWidth
                label="Stock Symbol"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                placeholder="AAPL"
                margin="normal"
              />
            </Grid>

            <Grid size={{ xs: 12, md: 6 }}>
              <FormControl fullWidth margin="normal">
                <InputLabel>Model</InputLabel>
                <Select value={modelType} onChange={(e) => setModelType(e.target.value)} label="Model">
                  {availableModels.map((model) => (
                    <MenuItem key={model.model_type} value={model.model_type}>
                      {model.model_name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            <Grid size={{ xs: 12, md: 6 }}>
              <TextField
                fullWidth
                label="Training Start Date"
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                slotProps={{
                  inputLabel: {
                    shrink: true,
                  },
                }}
                margin="normal"
              />
            </Grid>

            <Grid size={{ xs: 12, md: 6 }}>
              <TextField
                fullWidth
                label="Training End Date"
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                slotProps={{
                  inputLabel: {
                    shrink: true,
                  },
                }}
                margin="normal"
              />
            </Grid>

            <Grid size={{ xs: 12 }}>
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Typography>Advanced Settings</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={3}>
                    <Grid size={{ xs: 12, md: 6 }}>
                      <Typography gutterBottom>Epochs: {epochs}</Typography>
                      <Slider
                        value={epochs}
                        onChange={(_, value) => setEpochs(value as number)}
                        min={10}
                        max={500}
                        step={10}
                        marks={Array.from({ length: 500 / 50 }, (_, i) => ({
                          value: 50 + i * 50,
                          label: String(50 + i * 50),
                        }))}
                      />
                    </Grid>

                    <Grid size={{ xs: 12, md: 6 }}>
                      <Typography gutterBottom>Batch Size: {batchSize}</Typography>
                      <Slider
                        value={batchSize}
                        onChange={(_, value) => setBatchSize(value as number)}
                        min={8}
                        max={128}
                        step={8}
                        marks={Array.from({ length: 128 / 16 }, (_, i) => ({
                          value: 16 + i * 16,
                          label: String(16 + i * 16),
                        }))}
                      />
                    </Grid>

                    <Grid size={{ xs: 12, md: 6 }}>
                      <Typography gutterBottom>Learning Rate: {learningRate}</Typography>
                      <Slider
                        value={learningRate}
                        onChange={(_, value) => setLearningRate(value as number)}
                        min={0.001}
                        max={0.01}
                        step={0.001}
                        marks={Array.from({ length: 0.01 / 0.001 }, (_, i) => ({
                          value: 0.002 + i * 0.001,
                          label: String(Math.round((0.002 + i * 0.001) * 1000) / 1000),
                        }))}
                      />
                    </Grid>

                    <Grid size={{ xs: 12, md: 6 }}>
                      <Typography gutterBottom>Sequence Length: {sequenceLength}</Typography>
                      <Slider
                        value={sequenceLength}
                        onChange={(_, value) => setSequenceLength(value as number)}
                        min={10}
                        max={240}
                        step={5}
                        marks={Array.from({ length: 240 / 20 }, (_, i) => ({
                          value: 20 + i * 20,
                          label: String(20 + i * 20),
                        }))}
                      />
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>
            </Grid>
          </Grid>

          <Box sx={{ mt: 3, display: "flex", justifyContent: "flex-end" }}>
            <Button variant="contained" onClick={handleStartTraining} startIcon={<PlayArrow />}>
              Start Training
            </Button>
          </Box>
        </CardContent>
      </Card>
    );
  }

  function renderTrainingStep() {
    return (
      <Card>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            Training in Progress
          </Typography>

          <Box
            sx={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              py: 6,
              minHeight: 300,
            }}
          >
            <CircularProgress size={60} sx={{ mb: 3 }} />
            <Typography variant="body1" color="text.secondary" gutterBottom>
              Training Model: {modelType.toUpperCase()} for {symbol}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              This may take several minutes...
            </Typography>
          </Box>

          <Box sx={{ display: "flex", justifyContent: "center", gap: 2 }}>
            <Button variant="outlined" onClick={handleTrainInBackground} disabled={!isTraining}>
              Train in Background
            </Button>
          </Box>
        </CardContent>
      </Card>
    );
  }

  function renderResultsStep() {
    return (
      <Card>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            Training Results
          </Typography>

          {trainingResults && (
            <Grid container spacing={3}>
              <Grid size={{ xs: 12, md: 6 }}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Model Performance
                    </Typography>
                    <Box
                      sx={{
                        display: "flex",
                        justifyContent: "space-between",
                        mb: 1,
                      }}
                    >
                      <Typography>Accuracy:</Typography>
                      <Chip label={`${(trainingResults.final_metrics.accuracy * 100).toFixed(2)}%`} color="primary" />
                    </Box>
                    <Box
                      sx={{
                        display: "flex",
                        justifyContent: "space-between",
                        mb: 1,
                      }}
                    >
                      <Typography>Precision:</Typography>
                      <Chip
                        label={`${(trainingResults.final_metrics.precision * 100).toFixed(2)}%`}
                        color="secondary"
                      />
                    </Box>
                    <Box
                      sx={{
                        display: "flex",
                        justifyContent: "space-between",
                        mb: 1,
                      }}
                    >
                      <Typography>Recall:</Typography>
                      <Chip label={`${(trainingResults.final_metrics.recall * 100).toFixed(2)}%`} color="success" />
                    </Box>
                    <Box sx={{ display: "flex", justifyContent: "space-between" }}>
                      <Typography>F1-Score:</Typography>
                      <Chip label={`${(trainingResults.final_metrics.f1_score * 100).toFixed(2)}%`} color="info" />
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              <Grid size={{ xs: 12, md: 6 }}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Training History
                    </Typography>
                    {trainingResults.training_history && (
                      <Box sx={{ height: 200 }}>
                        <Typography variant="body2">
                          Training completed with {trainingResults.training_history.train_losses?.length || 0} epochs
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Final Training Loss:{" "}
                          {trainingResults.training_history.train_losses?.slice(-1)[0]?.toFixed(4) || "N/A"}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Final Validation Loss:{" "}
                          {trainingResults.training_history.val_losses?.slice(-1)[0]?.toFixed(4) || "N/A"}
                        </Typography>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          )}

          <Box sx={{ mt: 3, display: "flex", justifyContent: "space-between" }}>
            <Button variant="outlined" onClick={() => setActiveStep(0)}>
              Train Another Model
            </Button>
            <Button
              variant="contained"
              startIcon={<Download />}
              onClick={() => currentModelId && apiService.downloadModel(currentModelId)}
            >
              Download Model
            </Button>
          </Box>
        </CardContent>
      </Card>
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
        <Typography variant="h3" sx={{ mb: 4, fontWeight: 700, textAlign: "center" }}>
          Model Training
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        <Grid container spacing={4}>
          <Grid size={{ xs: 12, lg: 8 }}>
            <Box sx={{ mb: 4, minWidth: 600 }}>
              <Stepper activeStep={activeStep}>
                {steps.map((label) => (
                  <Step key={label}>
                    <StepLabel>{label}</StepLabel>
                  </Step>
                ))}
              </Stepper>
            </Box>

            <Box sx={{ minWidth: 600 }}>
              {activeStep === 0 && renderConfigurationStep()}
              {activeStep === 1 && renderTrainingStep()}
              {activeStep === 2 && renderResultsStep()}
            </Box>
          </Grid>

          <Grid size={{ xs: 12, lg: 4 }}>
            <Card sx={{ minWidth: 350, bgcolor: "rgba(25, 118, 210, 0.1)" }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Trained Models
                </Typography>

                <List>
                  {trainedModels.map((model) => (
                    <ListItem
                      key={model.model_id}
                      secondaryAction={
                        <Box sx={{ display: "flex", gap: 1 }}>
                          <IconButton
                            onClick={() => {
                              setSelectedModelForDetails(model);
                              setDetailsDialogOpen(true);
                            }}
                            title="View Details"
                          >
                            <Info />
                          </IconButton>
                          <IconButton
                            edge="end"
                            onClick={() => {
                              setModelToDelete(model.model_id);
                              setDeleteDialogOpen(true);
                            }}
                            title="Delete Model"
                          >
                            <Delete />
                          </IconButton>
                        </Box>
                      }
                    >
                      <ListItemText
                        primary={`${model.symbol} - ${model.model_type.toUpperCase()}`}
                        secondary={`Accuracy: ${model.accuracy ? (model.accuracy * 100).toFixed(1) : "N/A"}%`}
                        sx={{
                          cursor: "pointer",
                          "&:hover": {
                            backgroundColor: "action.hover",
                          },
                        }}
                        onClick={() => {
                          setSelectedModelForDetails(model);
                          setDetailsDialogOpen(true);
                        }}
                      />
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Model Details Dialog */}
        <ModelDetailsDialog
          open={detailsDialogOpen}
          onClose={() => {
            setDetailsDialogOpen(false);
            setSelectedModelForDetails(null);
          }}
          modelDetails={selectedModelForDetails}
        />

        {/* Delete Confirmation Dialog */}
        <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
          <DialogTitle>Delete Model</DialogTitle>
          <DialogContent>
            <Typography>Are you sure you want to delete this model? This action cannot be undone.</Typography>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
            <Button onClick={() => modelToDelete && handleDeleteModel(modelToDelete)} color="error">
              Delete
            </Button>
          </DialogActions>
        </Dialog>
      </Container>
    </Box>
  );
}

export default ModelTraining;
