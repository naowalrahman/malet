import React from "react";
import Plot from "react-plotly.js";
import { Card, CardContent } from "@mui/material";

interface ModernPlotProps {
  data: any[];
  layout: any;
  title?: string;
  height?: number;
  accentColor?: string;
  config?: Partial<Plotly.Config>;
  cardSx?: any;
}

const ModernPlot: React.FC<ModernPlotProps> = ({
  data,
  layout,
  title,
  height = 400,
  accentColor = "#1976d2",
  config = {},
  cardSx = {},
}) => {
  // Modern color palette for consistent theming
  const modernColors = [
    "#1976d2", // Primary blue
    "#dc004e", // Secondary pink/red
    "#2e7d32", // Success green
    "#ed6c02", // Warning orange
    "#0288d1", // Info blue
    "#7b1fa2", // Purple
    "#d32f2f", // Error red
    "#42a5f5", // Light blue
    "#ff5983", // Light pink
    "#66bb6a", // Light green
  ];

  // Apply modern styling to plot data
  const getModernPlotData = (originalData: any[]) => {
    return originalData.map((trace, index) => {
      const newTrace = { ...trace };

      // Only modify line properties if they exist
      if (trace.line) {
        newTrace.line = {
          ...trace.line,
          color: trace.line.color || modernColors[index % modernColors.length],
          width: trace.line.width || 2.5,
        };
      }

      // Only modify marker properties if they exist
      if (trace.marker) {
        newTrace.marker = {
          ...trace.marker,
          color: trace.marker.color || modernColors[index % modernColors.length],
          size: trace.type === "scatter" ? trace.marker.size || 4 : trace.marker.size,
          opacity: trace.type === "histogram" ? 0.8 : (trace.marker.opacity ?? 1),
        };
      }

      // Only modify fill color if it exists
      if (trace.fillcolor && typeof trace.fillcolor === "string") {
        const colorMatch = trace.fillcolor.match(/rgba?\([^)]+\)/);
        if (colorMatch) {
          const baseColor = modernColors[index % modernColors.length];
          newTrace.fillcolor = baseColor.replace("#", "rgba(").replace(")", ", 0.3)");
        }
      }

      return newTrace;
    });
  };

  // Modern dark theme configuration for Plotly charts
  const getModernPlotLayout = (originalLayout: any, chartTitle?: string) => {
    const modernLayout = {
      ...originalLayout,
      autosize: true,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: {
        family: '"Roboto", "Helvetica", "Arial", sans-serif',
        size: 12,
        color: "rgba(255, 255, 255, 0.87)",
      },
      title: chartTitle
        ? {
            text: chartTitle,
            font: {
              family: '"Roboto", "Helvetica", "Arial", sans-serif',
              size: 16,
              color: "rgba(255, 255, 255, 0.87)",
              weight: 600,
            },
            x: 0.02,
            xanchor: "left",
          }
        : originalLayout.title,
      xaxis: {
        ...originalLayout.xaxis,
        gridcolor: "rgba(255, 255, 255, 0.1)",
        zerolinecolor: "rgba(255, 255, 255, 0.2)",
        tickfont: {
          family: '"Roboto", "Helvetica", "Arial", sans-serif',
          size: 11,
          color: "rgba(255, 255, 255, 0.7)",
        },
        titlefont: {
          family: '"Roboto", "Helvetica", "Arial", sans-serif',
          size: 12,
          color: "rgba(255, 255, 255, 0.87)",
        },
      },
      yaxis: {
        ...originalLayout.yaxis,
        gridcolor: "rgba(255, 255, 255, 0.1)",
        zerolinecolor: "rgba(255, 255, 255, 0.2)",
        tickfont: {
          family: '"Roboto", "Helvetica", "Arial", sans-serif',
          size: 11,
          color: "rgba(255, 255, 255, 0.7)",
        },
        titlefont: {
          family: '"Roboto", "Helvetica", "Arial", sans-serif',
          size: 12,
          color: "rgba(255, 255, 255, 0.87)",
        },
      },
      legend: {
        ...originalLayout.legend,
        font: {
          family: '"Roboto", "Helvetica", "Arial", sans-serif',
          size: 11,
          color: "rgba(255, 255, 255, 0.87)",
        },
        bgcolor: "rgba(26, 29, 53, 0.8)",
        bordercolor: "rgba(255, 255, 255, 0.1)",
        borderwidth: 1,
      },
      margin: {
        l: 60,
        r: 40,
        t: 60,
        b: 60,
      },
    };

    return modernLayout;
  };

  // Default modern config
  const defaultConfig: Partial<Plotly.Config> = {
    displayModeBar: true,
    modeBarButtonsToRemove: ["pan2d", "lasso2d", "select2d"] as any,
    displaylogo: false,
    toImageButtonOptions: {
      format: "png",
      filename: title?.toLowerCase().replace(/\s+/g, "_") || "chart",
      height: Math.max(height + 100, 500),
      width: 1000,
      scale: 1,
    },
    ...config,
  };

  // Card styling with gradient and accent color
  const defaultCardSx = {
    background: "linear-gradient(135deg, #1a1d35 0%, #1e2242 100%)",
    borderRadius: 3,
    border: `1px solid ${accentColor}33`, // 20% opacity
    ...cardSx,
  };

  return (
    <Card sx={defaultCardSx}>
      <CardContent>
        <Plot
          data={getModernPlotData(data)}
          layout={getModernPlotLayout(layout, title)}
          useResizeHandler={true}
          style={{ width: "100%", height: `${height}px` }}
          config={defaultConfig}
        />
      </CardContent>
    </Card>
  );
};

export default ModernPlot;
