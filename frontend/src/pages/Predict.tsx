import { useState, useEffect } from "react";
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  CircularProgress,
  Chip,
  Grid,
  Paper,
} from "@mui/material";
import { DatePicker } from "@mui/x-date-pickers/DatePicker";
import { LocalizationProvider } from "@mui/x-date-pickers/LocalizationProvider";
import { AdapterDayjs } from "@mui/x-date-pickers/AdapterDayjs";
import { TrendingUp, TrendingDown, Psychology } from "@mui/icons-material";
import dayjs, { Dayjs } from "dayjs";
import timezone from "dayjs/plugin/timezone";
import utc from "dayjs/plugin/utc";

dayjs.extend(timezone);
dayjs.extend(utc);

import { apiService, type TrainedModelDetails, type Prediction } from "../services/api";
import CustomIndicatorDatePicker from "../components/CustomIndicatorDatePicker";

interface PredictFormData {
  symbol: string;
  modelId: string;
  date: Dayjs | null;
  useCustomIndicatorStart: boolean;
  indicatorStartDate: Dayjs | null;
}

function Predict() {
  const [formData, setFormData] = useState<PredictFormData>({
    symbol: "",
    modelId: "",
    date: dayjs(), // default to today
    useCustomIndicatorStart: false,
    indicatorStartDate: dayjs().subtract(1, 'year'), // default to 1 year ago
  });
  const [models, setModels] = useState<TrainedModelDetails[]>([]);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch available trained models
  useEffect(() => {
    async function fetchModels() {
      try {
        setModelsLoading(true);
        const response = await apiService.getTrainedModels();
        const data = response.models;
        setModels(data);
      } catch (err) {
        console.error("Failed to fetch models:", err);
        setError("Failed to load available models");
      } finally {
        setModelsLoading(false);
      }
    }

    fetchModels();
  }, []);

  function getMaxPredictDate(): Dayjs {
    const now = dayjs().tz("America/New_York");
    const currentDay = now.day();
    const marketCloseHour = 16; // 4:00 PM EST

    // If today is a weekend, next trading day is Monday
    if (currentDay === 6) {
      // Saturday
      return now.add(2, "day").startOf("day");
    }
    if (currentDay === 0) {
      // Sunday
      return now.add(1, "day").startOf("day");
    }

    // If today is a weekday but before market close, can't predict for tomorrow yet
    if (now.hour() < marketCloseHour) {
      return now.startOf("day");
    }

    // If after market close, next trading day is tomorrow (unless Friday)
    if (currentDay === 5) {
      // Friday
      return now.add(3, "day").startOf("day");
    }
    return now.add(1, "day").startOf("day");
  }

  function handleInputChange(field: keyof PredictFormData, value: any) {
    setFormData((prev) => ({ ...prev, [field]: value }));
    setError(null);
  }

  async function handlePredict() {
    if (!formData.symbol || !formData.modelId || !formData.date) {
      setError("Please fill in all fields");
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const result = await apiService.makePrediction(
        formData.symbol.toUpperCase(),
        formData.modelId,
        formData.date.format("YYYY-MM-DD"),
        formData.useCustomIndicatorStart ? formData.indicatorStartDate?.format("YYYY-MM-DD") : undefined
      );
      setPrediction(result);
    } catch (err: any) {
      console.error("Prediction failed:", err);
      setError(err.response?.data?.detail || "Failed to make prediction");
    } finally {
      setLoading(false);
    }
  }

  function renderPredictionResult() {
    if (!prediction) return null;

    const isUpPrediction = prediction.prediction === "UP";
    const icon = isUpPrediction ? <TrendingUp /> : <TrendingDown />;
    const color = isUpPrediction ? "success" : "error";
    const confidencePct = (prediction.confidence * 100).toFixed(1) + "%";

    function PredictionInfo(props: { children: React.ReactNode, name: string }) {
      return <Paper elevation={2} sx={{
        flex: "1 1 220px",
        minWidth: 180,
        p: 2,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        bgcolor: "background.default",
      }}>
        <Typography variant="caption" color="text.secondary" gutterBottom>
          {props.name}
        </Typography>
        {props.children}
      </Paper>;
    }

    return (
      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
            <Psychology sx={{ mr: 1, color: "primary.main" }} />
            <Typography variant="h6" component="h2">
              Prediction Result
            </Typography>
          </Box>

          <Box
            sx={{
              display: "flex",
              flexWrap: "wrap",
              gap: 2,
              justifyContent: { xs: "center", md: "flex-start" },
              alignItems: "stretch",
              mb: 2,
            }}
          >
            <PredictionInfo name="Symbol">
              <Typography variant="h6" fontWeight={700}>
                {prediction.symbol}
              </Typography>
            </PredictionInfo>

            <PredictionInfo name="Prediction">
              <Chip
                icon={icon}
                label={prediction.prediction}
                color={color}
                variant="filled"
                sx={{ fontSize: "1rem", fontWeight: "bold" }}
              />
            </PredictionInfo>

            <PredictionInfo name="Confidence">
              <Typography variant="h6" fontWeight={700}>
                {confidencePct}
              </Typography>
            </PredictionInfo>
          </Box>

          <Box sx={{ mt: 2 }}>
            <PredictionInfo name="Prediction Timestamp">
              <Typography variant="body2" color="text.primary">
                {new Date(prediction.timestamp).toLocaleString()}
              </Typography>
            </PredictionInfo>
          </Box>

          <Alert severity={isUpPrediction ? "success" : "warning"} sx={{ mt: 2 }}>
            <Typography variant="body2">
              <strong>Model Response:</strong> The AI model predicts that {prediction.symbol} will trend{" "}
              <strong>{prediction.prediction}</strong> based on the selected date and historical data patterns.
            </Typography>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <LocalizationProvider dateAdapter={AdapterDayjs}>
      <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
        <Box sx={{ mb: 4, textAlign: "center" }}>
          <Typography variant="h3" component="h1" gutterBottom fontWeight={700}>
            Predict
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Predict how a stock will move on a given date using a trained model
          </Typography>
        </Box>

        <Card>
          <CardContent sx={{ p: 4 }}>
            <Grid container spacing={3}>
              <Grid size={{ xs: 12, md: 6 }}>
                <TextField
                  fullWidth
                  label="Stock Symbol"
                  value={formData.symbol}
                  onChange={(e) => handleInputChange("symbol", e.target.value)}
                  placeholder="e.g., AAPL, GOOGL, TSLA"
                  helperText="Enter the stock symbol you want to predict"
                />
              </Grid>

              <Grid size={{ xs: 12, md: 6 }}>
                <DatePicker
                  label="Prediction Date"
                  value={formData.date}
                  onChange={(newDate) => handleInputChange("date", newDate)}
                  maxDate={getMaxPredictDate()}
                  shouldDisableDate={(date: Dayjs) => date.day() === 0 || date.day() === 6}
                  slotProps={{
                    textField: {
                      fullWidth: true,
                      helperText: "Select a weekday for prediction (up to next trading day)",
                    },
                  }}
                />
              </Grid>

              {/* Custom Indicator Start Date */}
              <CustomIndicatorDatePicker
                checked={formData.useCustomIndicatorStart}
                onCheckedChange={(checked) => handleInputChange("useCustomIndicatorStart", checked)}
                value={formData.indicatorStartDate}
                onChange={(date) => handleInputChange("indicatorStartDate", date)}
                mainDate={formData.date}
                mainDateLabel="prediction date"
              />

              <Grid size={{ xs: 12 }}>
                <FormControl fullWidth>
                  <InputLabel>Select Trained Model</InputLabel>
                  <Select
                    value={formData.modelId}
                    onChange={(e) => handleInputChange("modelId", e.target.value)}
                    label="Select Trained Model"
                    disabled={modelsLoading}
                  >
                    {models.map((model) => (
                      <MenuItem key={model.model_id} value={model.model_id}>
                        <Box>
                          <Typography variant="body1">
                            {model.symbol} - {model.model_type.toUpperCase()}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Trained: {new Date(model.created_at).toLocaleDateString()} • Accuracy:{" "}
                            {model.accuracy ? (model.accuracy * 100).toFixed(1) + "%" : "N/A"}
                          </Typography>
                        </Box>
                      </MenuItem>
                    ))}
                  </Select>
                  {modelsLoading && (
                    <Box sx={{ display: "flex", alignItems: "center", mt: 1 }}>
                      <CircularProgress size={16} sx={{ mr: 1 }} />
                      <Typography variant="caption">Loading models...</Typography>
                    </Box>
                  )}
                  {!modelsLoading && models.length === 0 && (
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                      No trained models available. Train a model first in the Model Training page.
                    </Typography>
                  )}
                </FormControl>
              </Grid>

              {error && (
                <Grid size={{ xs: 12 }}>
                  <Alert severity="error">{error}</Alert>
                </Grid>
              )}

              <Grid size={{ xs: 12 }}>
                <Button
                  variant="contained"
                  size="large"
                  onClick={handlePredict}
                  disabled={loading || !formData.symbol || !formData.modelId || !formData.date}
                  startIcon={loading ? <CircularProgress size={20} /> : <Psychology />}
                  sx={{
                    px: 4,
                    py: 1.5,
                    fontSize: "1rem",
                    fontWeight: 600,
                  }}
                >
                  {loading ? "Making Prediction..." : "Get Prediction"}
                </Button>
              </Grid>
            </Grid>
          </CardContent>
        </Card>

        {renderPredictionResult()}
      </Container>
    </LocalizationProvider>
  );
}

export default Predict;
