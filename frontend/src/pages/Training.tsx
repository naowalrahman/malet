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
  Stepper,
  Step,
  StepLabel,
  Alert,
  LinearProgress,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Slider,
} from "@mui/material";
import {
  PlayArrow,
  Stop,
  Download,
  Delete,
  ExpandMore,
} from "@mui/icons-material";
import { apiService } from "../services/api";
import type { Model, TrainingRequest } from "../services/api";

const steps = ["Configure Model", "Training", "Results"];

const ModelTraining: React.FC = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [symbol, setSymbol] = useState("AAPL");
  const [modelType, setModelType] = useState("lstm");
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(32);
  const [learningRate, setLearningRate] = useState(0.001);
  const [sequenceLength, setSequenceLength] = useState(60);
  const [predictionHorizon] = useState(5);
  const [threshold] = useState(0.02);

  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);
  const [currentModelId, setCurrentModelId] = useState<string | null>(null);
  const [, setCurrentJobId] = useState<string | null>(null);
  const [trainingResults, setTrainingResults] = useState<any>(null);
  const [models, setModels] = useState<Model[]>([]);

  const [error, setError] = useState<string | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [modelToDelete, setModelToDelete] = useState<string | null>(null);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await apiService.getModels();
      setModels(response.models);
    } catch (err) {
      setError("Failed to fetch models");
    }
  };

  const handleStartTraining = async () => {
    try {
      setError(null);
      setIsTraining(true);
      setActiveStep(1);
      setTrainingProgress(0);
      setTrainingLogs([]);

      const request: TrainingRequest = {
        symbol,
        model_type: modelType,
        epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        sequence_length: sequenceLength,
        prediction_horizon: predictionHorizon,
        threshold,
      };

      const response = await apiService.trainModel(request);
      setCurrentJobId(response.job_id);

      // Poll for training status
      pollTrainingStatus(response.job_id);
    } catch (err) {
      setError("Failed to start training");
      setIsTraining(false);
    }
  };

  const pollTrainingStatus = async (modelId: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const status = await apiService.getTrainingStatus(modelId);

        setTrainingProgress(status.progress || 0);

        if (status.status === "completed") {
          clearInterval(pollInterval);
          setIsTraining(false);
          setActiveStep(2);
          setTrainingResults(status.results || status.result);
          if (status.result?.model_id) {
            setCurrentModelId(status.result.model_id);
          }
          fetchModels(); // Refresh models list
        } else if (status.status === "failed") {
          clearInterval(pollInterval);
          setIsTraining(false);
          setError(status.error || "Training failed");
        }

        // Add status updates to logs
        const timestamp = new Date().toLocaleTimeString();
        setTrainingLogs((prev) => [
          ...prev,
          `${timestamp}: ${status.status} (${status.progress}%)`,
        ]);
      } catch (err) {
        clearInterval(pollInterval);
        setIsTraining(false);
        setError("Failed to get training status");
      }
    }, 2000);
  };

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

  const renderConfigurationStep = () => (
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
              <InputLabel>Model Type</InputLabel>
              <Select
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
              >
                <MenuItem value="lstm">LSTM Neural Network</MenuItem>
                <MenuItem value="cnn_lstm">CNN-LSTM Hybrid</MenuItem>
                <MenuItem value="transformer">Transformer</MenuItem>
              </Select>
            </FormControl>
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
                      marks={[
                        { value: 50, label: "50" },
                        { value: 100, label: "100" },
                        { value: 200, label: "200" },
                      ]}
                    />
                  </Grid>

                  <Grid size={{ xs: 12, md: 6 }}>
                    <Typography gutterBottom>
                      Batch Size: {batchSize}
                    </Typography>
                    <Slider
                      value={batchSize}
                      onChange={(_, value) => setBatchSize(value as number)}
                      min={8}
                      max={128}
                      step={8}
                      marks={[
                        { value: 16, label: "16" },
                        { value: 32, label: "32" },
                        { value: 64, label: "64" },
                      ]}
                    />
                  </Grid>

                  <Grid size={{ xs: 12, md: 6 }}>
                    <Typography gutterBottom>
                      Learning Rate: {learningRate}
                    </Typography>
                    <Slider
                      value={learningRate}
                      onChange={(_, value) => setLearningRate(value as number)}
                      min={0.0001}
                      max={0.01}
                      step={0.0001}
                      marks={[
                        { value: 0.001, label: "0.001" },
                        { value: 0.005, label: "0.005" },
                      ]}
                    />
                  </Grid>

                  <Grid size={{ xs: 12, md: 6 }}>
                    <Typography gutterBottom>
                      Sequence Length: {sequenceLength}
                    </Typography>
                    <Slider
                      value={sequenceLength}
                      onChange={(_, value) =>
                        setSequenceLength(value as number)
                      }
                      min={10}
                      max={120}
                      step={5}
                      marks={[
                        { value: 30, label: "30" },
                        { value: 60, label: "60" },
                        { value: 90, label: "90" },
                      ]}
                    />
                  </Grid>
                </Grid>
              </AccordionDetails>
            </Accordion>
          </Grid>
        </Grid>

        <Box sx={{ mt: 3, display: "flex", justifyContent: "space-between" }}>
          <Button variant="outlined" onClick={() => setActiveStep(0)} disabled>
            Previous
          </Button>
          <Button
            variant="contained"
            onClick={handleStartTraining}
            startIcon={<PlayArrow />}
          >
            Start Training
          </Button>
        </Box>
      </CardContent>
    </Card>
  );

  const renderTrainingStep = () => (
    <Card>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          Training in Progress
        </Typography>

        <Box sx={{ mb: 3 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Training Model: {modelType.toUpperCase()} for {symbol}
          </Typography>
          <LinearProgress
            variant="determinate"
            value={trainingProgress}
            sx={{ height: 8, borderRadius: 4 }}
          />
          <Typography variant="body2" sx={{ mt: 1 }}>
            {trainingProgress}% Complete
          </Typography>
        </Box>

        <Card
          variant="outlined"
          sx={{ mb: 3, maxHeight: 300, overflow: "auto" }}
        >
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Training Logs
            </Typography>
            {trainingLogs.map((log, index) => (
              <Typography
                key={index}
                variant="body2"
                sx={{ fontFamily: "monospace" }}
              >
                {log}
              </Typography>
            ))}
          </CardContent>
        </Card>

        <Box sx={{ display: "flex", justifyContent: "space-between" }}>
          <Button
            variant="outlined"
            onClick={() => setActiveStep(0)}
            disabled={isTraining}
          >
            Back to Configuration
          </Button>
          <Button
            variant="contained"
            color="error"
            onClick={() => setIsTraining(false)}
            disabled={!isTraining}
            startIcon={<Stop />}
          >
            Stop Training
          </Button>
        </Box>
      </CardContent>
    </Card>
  );

  const renderResultsStep = () => (
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
                    <Chip
                      label={`${(
                        trainingResults.final_metrics.accuracy * 100
                      ).toFixed(2)}%`}
                      color="primary"
                    />
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
                      label={`${(
                        trainingResults.final_metrics.precision * 100
                      ).toFixed(2)}%`}
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
                    <Chip
                      label={`${(
                        trainingResults.final_metrics.recall * 100
                      ).toFixed(2)}%`}
                      color="success"
                    />
                  </Box>
                  <Box
                    sx={{ display: "flex", justifyContent: "space-between" }}
                  >
                    <Typography>F1-Score:</Typography>
                    <Chip
                      label={`${(
                        trainingResults.final_metrics.f1_score * 100
                      ).toFixed(2)}%`}
                      color="info"
                    />
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
                        Training completed with{" "}
                        {trainingResults.training_history.train_losses
                          ?.length || 0}{" "}
                        epochs
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Final Training Loss:{" "}
                        {trainingResults.training_history.train_losses
                          ?.slice(-1)[0]
                          ?.toFixed(4) || "N/A"}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Final Validation Loss:{" "}
                        {trainingResults.training_history.val_losses
                          ?.slice(-1)[0]
                          ?.toFixed(4) || "N/A"}
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
            onClick={() =>
              currentModelId && apiService.downloadModel(currentModelId)
            }
          >
            Download Model
          </Button>
        </Box>
      </CardContent>
    </Card>
  );

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Typography variant="h3" sx={{ mb: 4, fontWeight: 700 }}>
        Model Training
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={4}>
        <Grid size={{ xs: 12, lg: 8 }}>
          <Box sx={{ mb: 4 }}>
            <Stepper activeStep={activeStep}>
              {steps.map((label) => (
                <Step key={label}>
                  <StepLabel>{label}</StepLabel>
                </Step>
              ))}
            </Stepper>
          </Box>

          {activeStep === 0 && renderConfigurationStep()}
          {activeStep === 1 && renderTrainingStep()}
          {activeStep === 2 && renderResultsStep()}
        </Grid>

        <Grid size={{ xs: 12, lg: 4 }}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Trained Models
              </Typography>

              <List>
                {models.map((model) => (
                  <ListItem key={model.model_id}>
                    <ListItemText
                      primary={`${
                        model.symbol
                      } - ${model.model_type.toUpperCase()}`}
                      secondary={`Accuracy: ${
                        model.accuracy
                          ? (model.accuracy * 100).toFixed(1)
                          : "N/A"
                      }%`}
                    />
                    <ListItemSecondaryAction>
                      <IconButton
                        edge="end"
                        onClick={() => {
                          setModelToDelete(model.model_id);
                          setDeleteDialogOpen(true);
                        }}
                      >
                        <Delete />
                      </IconButton>
                    </ListItemSecondaryAction>
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={() => setDeleteDialogOpen(false)}
      >
        <DialogTitle>Delete Model</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete this model? This action cannot be
            undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={() => modelToDelete && handleDeleteModel(modelToDelete)}
            color="error"
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default ModelTraining;
