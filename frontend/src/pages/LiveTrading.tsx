import React from "react";
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  Alert,
} from "@mui/material";

const LiveTrading: React.FC = () => {
  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Live Trading
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Execute trades with real-time ML predictions
        </Typography>
      </Box>

      <Card>
        <CardContent>
          <Alert severity="warning">
            Live Trading page is under development. This will include:
            <ul>
              <li>Real-time market predictions</li>
              <li>Automated trading strategies</li>
              <li>Risk management controls</li>
              <li>Portfolio monitoring</li>
              <li>Trade execution interface</li>
              <li>Performance tracking</li>
            </ul>
            <br />
            <strong>Note:</strong> This is for demonstration purposes only. Do
            not connect to real trading accounts without proper testing and risk
            management.
          </Alert>
        </CardContent>
      </Card>
    </Container>
  );
};

export default LiveTrading;
