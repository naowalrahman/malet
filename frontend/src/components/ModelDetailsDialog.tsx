import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  Grid,
  Card,
  CardContent,
  Chip,
  Divider,
} from "@mui/material";
import type { TrainedModelDetails } from "../services/api";

interface ModelDetailsDialogProps {
  open: boolean;
  onClose: () => void;
  modelDetails: TrainedModelDetails | null;
}

export default function ModelDetailsDialog({ open, onClose, modelDetails }: ModelDetailsDialogProps) {
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  const handleClose = () => {
    onClose();
  };

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Typography variant="h5" component="div">
          Model Details
        </Typography>
      </DialogTitle>

      <DialogContent>
        {modelDetails && (
          <Box>
            {/* Basic Information */}
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Basic Information
                </Typography>
                <Grid container spacing={2}>
                  <Grid size={{ xs: 6 }}>
                    <Typography variant="body2" color="text.secondary">
                      Symbol
                    </Typography>
                    <Typography variant="body1">
                      {modelDetails.symbol}
                    </Typography>
                  </Grid>
                  <Grid size={{ xs: 6 }}>
                    <Typography variant="body2" color="text.secondary">
                      Model Type
                    </Typography>
                    <Typography variant="body1">
                      {modelDetails.model_type.toUpperCase()}
                    </Typography>
                  </Grid>
                  <Grid size={{ xs: 6 }}>
                    <Typography variant="body2" color="text.secondary">
                      Created At
                    </Typography>
                    <Typography variant="body1">{formatDate(modelDetails.created_at)}</Typography>
                  </Grid>
                  <Grid size={{ xs: 6 }}>
                    <Typography variant="body2" color="text.secondary">
                      Model ID
                    </Typography>
                    <Typography variant="body2" sx={{ fontFamily: "monospace", fontSize: "0.8rem" }}>
                      {modelDetails.model_id}
                    </Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>

            {/* Training Parameters */}
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Training Parameters
                </Typography>
                <Grid container spacing={2}>
                  <Grid size={{ xs: 6, sm: 4 }}>
                    <Typography variant="body2" color="text.secondary">
                      Epochs
                    </Typography>
                    <Typography variant="body1">
                      {modelDetails.training_params.epochs}
                    </Typography>
                  </Grid>
                  <Grid size={{ xs: 6, sm: 4 }}>
                    <Typography variant="body2" color="text.secondary">
                      Batch Size
                    </Typography>
                    <Typography variant="body1">
                      {modelDetails.training_params.batch_size}
                    </Typography>
                  </Grid>
                  <Grid size={{ xs: 6, sm: 4 }}>
                    <Typography variant="body2" color="text.secondary">
                      Learning Rate
                    </Typography>
                    <Typography variant="body1">
                      {modelDetails.training_params.learning_rate}
                    </Typography>
                  </Grid>
                  <Grid size={{ xs: 6, sm: 4 }}>
                    <Typography variant="body2" color="text.secondary">
                      Sequence Length
                    </Typography>
                    <Typography variant="body1">
                      {modelDetails.training_params.sequence_length}
                    </Typography>
                  </Grid>
                  <Grid size={{ xs: 6, sm: 4 }}>
                    <Typography variant="body2" color="text.secondary">
                      Prediction Horizon
                    </Typography>
                    <Typography variant="body1">
                      {modelDetails.training_params.prediction_horizon} days
                    </Typography>
                  </Grid>
                  <Grid size={{ xs: 6, sm: 4 }}>
                    <Typography variant="body2" color="text.secondary">
                      Threshold
                    </Typography>
                    <Typography variant="body1">
                      {formatPercentage(modelDetails.training_params.threshold)}
                    </Typography>
                  </Grid>
                </Grid>

                {(modelDetails.training_params.start_date || modelDetails.training_params.end_date) && (
                  <>
                    <Divider sx={{ my: 2 }} />
                    <Typography variant="subtitle2" gutterBottom>
                      Training Period
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid size={{ xs: 6 }}>
                        <Typography variant="body2" color="text.secondary">
                          Start Date
                        </Typography>
                        <Typography variant="body1">
                          {modelDetails.training_params.start_date || "Not specified"}
                        </Typography>
                      </Grid>
                      <Grid size={{ xs: 6 }}>
                        <Typography variant="body2" color="text.secondary">
                          End Date
                        </Typography>
                        <Typography variant="body1">
                          {modelDetails.training_params.end_date || "Not specified"}
                        </Typography>
                      </Grid>
                    </Grid>
                  </>
                )}
              </CardContent>
            </Card>

            {/* Performance Metrics */}
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Performance Metrics
                </Typography>
                <Grid container spacing={2}>
                  <Grid size={{ xs: 6, sm: 3 }}>
                    <Box textAlign="center">
                      <Typography variant="body2" color="text.secondary">
                        Accuracy
                      </Typography>
                      <Chip
                        label={formatPercentage(modelDetails.training_metrics.accuracy)}
                        color="primary"
                        size="medium"
                      />
                    </Box>
                  </Grid>
                  <Grid size={{ xs: 6, sm: 3 }}>
                    <Box textAlign="center">
                      <Typography variant="body2" color="text.secondary">
                        Precision
                      </Typography>
                      <Chip
                        label={formatPercentage(modelDetails.training_metrics.precision)}
                        color="secondary"
                        size="medium"
                      />
                    </Box>
                  </Grid>
                  <Grid size={{ xs: 6, sm: 3 }}>
                    <Box textAlign="center">
                      <Typography variant="body2" color="text.secondary">
                        Recall
                      </Typography>
                      <Chip
                        label={formatPercentage(modelDetails.training_metrics.recall)}
                        color="success"
                        size="medium"
                      />
                    </Box>
                  </Grid>
                  <Grid size={{ xs: 6, sm: 3 }}>
                    <Box textAlign="center">
                      <Typography variant="body2" color="text.secondary">
                        F1-Score
                      </Typography>
                      <Chip
                        label={formatPercentage(modelDetails.training_metrics.f1_score)}
                        color="info"
                        size="medium"
                      />
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>

            {/* Training History */}
            {(modelDetails.training_history.train_losses || modelDetails.training_history.val_losses) && (
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Training History
                  </Typography>
                  <Grid container spacing={2}>
                    {modelDetails.training_history.train_losses && (
                      <Grid size={{ xs: 6 }}>
                        <Typography variant="body2" color="text.secondary">
                          Final Training Loss
                        </Typography>
                        <Typography variant="body1">
                          {modelDetails.training_history.train_losses.slice(-1)[0]?.toFixed(4) || "N/A"}
                        </Typography>
                      </Grid>
                    )}
                    {modelDetails.training_history.val_losses && (
                      <Grid size={{ xs: 6 }}>
                        <Typography variant="body2" color="text.secondary">
                          Final Validation Loss
                        </Typography>
                        <Typography variant="body1">
                          {modelDetails.training_history.val_losses.slice(-1)[0]?.toFixed(4) || "N/A"}
                        </Typography>
                      </Grid>
                    )}
                    <Grid size={{ xs: 12 }}>
                      <Typography variant="body2" color="text.secondary">
                        Total Epochs Completed
                      </Typography>
                      <Typography variant="body1">
                        {modelDetails.training_history.train_losses?.length || 0}
                      </Typography>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            )}
          </Box>
        )}
      </DialogContent>

      <DialogActions>
        <Button onClick={handleClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}
