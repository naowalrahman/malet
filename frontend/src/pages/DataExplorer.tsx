import React from "react";
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  Alert,
} from "@mui/material";

const DataExplorer: React.FC = () => {
  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Data Explorer
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Explore and visualize financial data
        </Typography>
      </Box>

      <Card>
        <CardContent>
          <Alert severity="info">
            Data Explorer page is under development. This will include:
            <ul>
              <li>Interactive stock price charts</li>
              <li>Technical indicator overlays</li>
              <li>Market data comparison tools</li>
              <li>Economic calendar integration</li>
              <li>Real-time data feeds</li>
            </ul>
          </Alert>
        </CardContent>
      </Card>
    </Container>
  );
};

export default DataExplorer;
