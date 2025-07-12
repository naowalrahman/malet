import React, { use } from "react";
import { Box, Typography, CircularProgress } from "@mui/material";
import { Psychology } from "@mui/icons-material";
import { useTheme } from "@mui/material/styles";
import MuiMarkdown from "mui-markdown";
import { apiService } from "../services/api";
import type { MarketAIAnalysis } from "../services/api";

// We're caching the AI analysis for each symbol to avoid refetching between symbol changes.

const aiAnalysisCache = new Map<string, Promise<MarketAIAnalysis>>();

function useAIAnalysis(symbol: string): MarketAIAnalysis {
  if (!aiAnalysisCache.has(symbol)) {
    aiAnalysisCache.set(symbol, apiService.getMarketAIAnalysis(symbol));
  }

  const promise = aiAnalysisCache.get(symbol)!;
  return use(promise);
}

function AIAnalysisContent({ symbol }: { symbol: string }) {
  const theme = useTheme();
  const aiAnalysis = useAIAnalysis(symbol);

  return (
    <>
      <Box sx={{ display: "flex", alignItems: "center", mb: 1 }}>
        <Psychology sx={{ color: theme.palette.secondary.main, mr: 1 }} />
        <Typography
          variant="h6"
          sx={{
            fontWeight: 700,
            letterSpacing: 1,
            color: theme.palette.secondary.main,
            textShadow: "0 1px 4px rgba(0,0,0,0.08)",
            textTransform: "uppercase",
          }}
        >
          AI Analysis
        </Typography>
      </Box>

      <Box sx={{ mt: 2 }}>
        <MuiMarkdown
          options={{
            overrides: {
              p: {
                props: {
                  style: {
                    color: theme.palette.text.secondary,
                    lineHeight: 1.6,
                    marginBottom: theme.spacing(1),
                  },
                },
              },
              strong: {
                props: {
                  style: {
                    color: theme.palette.primary.main,
                    fontWeight: 600,
                  },
                },
              },
              em: {
                props: {
                  style: {
                    color: theme.palette.secondary.main,
                    fontStyle: "italic",
                  },
                },
              },
            },
          }}
        >
          {aiAnalysis.ai_analysis}
        </MuiMarkdown>
      </Box>
    </>
  );
}

function AIAnalysisLoading() {
  return (
    <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", py: 4 }}>
      <CircularProgress size={24} sx={{ mr: 2 }} />
      <Typography variant="body2" sx={{ color: "text.secondary" }}>
        Generating AI analysis...
      </Typography>
    </Box>
  );
}

function AIAnalysis({ symbol }: { symbol: string }) {
  return (
    <React.Suspense fallback={<AIAnalysisLoading />}>
      <AIAnalysisContent symbol={symbol} />
    </React.Suspense>
  );
}

export default AIAnalysis;
