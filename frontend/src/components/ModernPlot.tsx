import { lazy } from "react";
const Plot = lazy(() => import("react-plotly.js"));
import { Card, CardContent } from "@mui/material";

interface ModernPlotProps {
  data: any[];
  layout: any;
  title?: string;
  height?: number;
  accentColor?: string;
  config?: Partial<Plotly.Config>;
  cardSx?: any;
  legendPosition?: "top-left" | "top-right" | "bottom-left" | "bottom-right" | "horizontal-top" | "horizontal-bottom";
}

function ModernPlot({
  data,
  layout,
  title,
  height = 400,
  accentColor = "#3b82f6",
  config = {},
  cardSx = {},
  legendPosition = "horizontal-top",
}: ModernPlotProps) {
  // Modern color palette with contemporary design aesthetics
  const modernColors = [
    "#3b82f6", // Primary blue
    "#f59e0b", // Warm amber
    "#10b981", // Success green
    "#ef4444", // Modern red
    "#8b5cf6", // Vibrant purple
    "#06b6d4", // Cyan
    "#f97316", // Orange
    "#ec4899", // Pink
    "#84cc16", // Lime
    "#6366f1", // Indigo
    "#14b8a6", // Teal
    "#f43f5e", // Rose
  ];

  // Enhanced gradient colors for fills
  const gradientColors = [
    "rgba(59, 130, 246, 0.1)", // Blue gradient
    "rgba(245, 158, 11, 0.1)", // Amber gradient
    "rgba(16, 185, 129, 0.1)", // Green gradient
    "rgba(239, 68, 68, 0.1)", // Red gradient
    "rgba(139, 92, 246, 0.1)", // Purple gradient
    "rgba(6, 182, 212, 0.1)", // Cyan gradient
    "rgba(249, 115, 22, 0.1)", // Orange gradient
    "rgba(236, 72, 153, 0.1)", // Pink gradient
    "rgba(132, 204, 22, 0.1)", // Lime gradient
    "rgba(99, 102, 241, 0.1)", // Indigo gradient
    "rgba(20, 184, 166, 0.1)", // Teal gradient
    "rgba(244, 63, 94, 0.1)", // Rose gradient
  ];

  // Apply modern styling with enhanced visual appeal
  function getModernPlotData(originalData: any[]) {
    return originalData.map((trace, index) => {
      const newTrace = { ...trace };
      const colorIndex = index % modernColors.length;

      // Enhanced line styling
      if (trace.line) {
        newTrace.line = {
          ...trace.line,
          color: trace.line.color || modernColors[colorIndex],
          width: trace.line.width || 3,
          shape: trace.line.shape || "spline", // Smooth curves
        };
      }

      // Enhanced marker styling
      if (trace.marker) {
        newTrace.marker = {
          ...trace.marker,
          color: trace.marker.color || modernColors[colorIndex],
          size: trace.type === "scatter" ? trace.marker.size || 6 : trace.marker.size,
          opacity: trace.type === "histogram" ? 0.85 : (trace.marker.opacity ?? 1),
          line: {
            width: 1,
            color: "rgba(255, 255, 255, 0.3)",
          },
        };
      }

      // Enhanced fill colors with gradients
      if (trace.fill && trace.fill !== "none") {
        newTrace.fillcolor = gradientColors[colorIndex];
      }

      // Enhanced bar chart styling
      if (trace.type === "bar") {
        newTrace.marker = {
          ...newTrace.marker,
          color: modernColors[colorIndex],
          opacity: 0.9,
          line: {
            color: modernColors[colorIndex],
            width: 1,
          },
        };
      }

      // Enhanced histogram styling
      if (trace.type === "histogram") {
        newTrace.marker = {
          ...newTrace.marker,
          color: modernColors[colorIndex],
          opacity: 0.8,
          line: {
            color: "rgba(255, 255, 255, 0.2)",
            width: 1,
          },
        };
      }

      return newTrace;
    });
  }

  // Helper function to get legend positioning based on position prop
  function getLegendPosition(position: string) {
    switch (position) {
      case "top-left":
        return { x: 0.02, xanchor: "left", y: 0.98, yanchor: "top", orientation: "v" };
      case "top-right":
        return { x: 0.98, xanchor: "right", y: 0.98, yanchor: "top", orientation: "v" };
      case "bottom-left":
        return { x: 0.02, xanchor: "left", y: 0.02, yanchor: "bottom", orientation: "v" };
      case "bottom-right":
        return { x: 0.98, xanchor: "right", y: 0.02, yanchor: "bottom", orientation: "v" };
      case "horizontal-top":
        return { x: 0.5, xanchor: "center", y: 1.05, yanchor: "bottom", orientation: "h" };
      case "horizontal-bottom":
        return { x: 0.5, xanchor: "center", y: -0.1, yanchor: "top", orientation: "h" };
      default:
        return { x: 0.5, xanchor: "center", y: 1.05, yanchor: "bottom", orientation: "h" };
    }
  }

  // Ultra-modern dark theme with contemporary design
  function getModernPlotLayout(originalLayout: any, chartTitle?: string) {
    const modernLayout = {
      ...originalLayout,
      autosize: true,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: {
        family: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        size: 13,
        color: "#f8fafc",
      },
      title: chartTitle
        ? {
            text: chartTitle,
            font: {
              family: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
              size: 18,
              color: "#f8fafc",
              weight: 700,
            },
            x: 0.02,
            xanchor: "left",
            pad: { t: 20, b: 20 },
          }
        : originalLayout.title,
      xaxis: {
        ...originalLayout.xaxis,
        gridcolor: "rgba(148, 163, 184, 0.08)",
        gridwidth: 1,
        zerolinecolor: "rgba(148, 163, 184, 0.15)",
        zerolinewidth: 1,
        linecolor: "rgba(148, 163, 184, 0.15)",
        tickfont: {
          family: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
          size: 12,
          color: "#cbd5e1",
        },
        titlefont: {
          family: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
          size: 14,
          color: "#f8fafc",
          weight: 600,
        },
        showspikes: true,
        spikecolor: accentColor,
        spikethickness: 1,
        spikedash: "dot",
        spikemode: "across",
      },
      yaxis: {
        ...originalLayout.yaxis,
        gridcolor: "rgba(148, 163, 184, 0.08)",
        gridwidth: 1,
        zerolinecolor: "rgba(148, 163, 184, 0.15)",
        zerolinewidth: 1,
        linecolor: "rgba(148, 163, 184, 0.15)",
        tickfont: {
          family: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
          size: 12,
          color: "#cbd5e1",
        },
        titlefont: {
          family: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
          size: 14,
          color: "#f8fafc",
          weight: 600,
        },
        showspikes: true,
        spikecolor: accentColor,
        spikethickness: 1,
        spikedash: "dot",
        spikemode: "across",
      },
      legend: {
        ...originalLayout.legend,
        font: {
          family: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
          size: 12,
          color: "#f8fafc",
        },
        bgcolor: "rgba(26, 26, 26, 0.95)",
        bordercolor: "rgba(148, 163, 184, 0.15)",
        borderwidth: 1,
        ...getLegendPosition(legendPosition),
      },
      margin: {
        l: 70,
        r: 50,
        t: 70,
        b: 60,
      },
      hovermode: "closest",
      hoverlabel: {
        bgcolor: "rgba(26, 26, 26, 0.95)",
        bordercolor: accentColor,
        font: {
          family: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
          size: 12,
          color: "#f8fafc",
        },
      },
      // Add smooth animations
      transition: {
        duration: 300,
        easing: "cubic-in-out",
      },
    };

    return modernLayout;
  }

  // Enhanced configuration with modern interactions
  const defaultConfig: Partial<Plotly.Config> = {
    displayModeBar: true,
    displaylogo: false,
    responsive: true,
    toImageButtonOptions: {
      format: "png",
      filename: title?.toLowerCase().replace(/\s+/g, "_") || "chart",
      height: Math.max(height + 120, 600),
      width: 1200,
      scale: 2,
    },
    modeBarButtons: [["zoom2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"], ["hoverClosestCartesian", "hoverCompareCartesian" ], ["toImage"]] as any,
    ...config,
  };

  // Enhanced card styling with modern aesthetics
  const defaultCardSx = {
    background: "linear-gradient(135deg, #1a1a1a 0%, #262626 100%)",
    borderRadius: 4,
    border: `1px solid ${accentColor}20`, // More subtle accent
    boxShadow: "0 4px 20px rgba(0, 0, 0, 0.3), 0 1px 3px rgba(0, 0, 0, 0.4)",
    transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
    overflow: "hidden",
    position: "relative",
    "&::before": {
      content: '""',
      position: "absolute",
      top: 0,
      left: 0,
      right: 0,
      height: "2px",
      background: `linear-gradient(90deg, ${accentColor} 0%, transparent 100%)`,
      opacity: 0.6,
    },
    "&:hover": {
      transform: "translateY(-2px)",
      boxShadow: `0 8px 32px rgba(0, 0, 0, 0.4), 0 4px 16px ${accentColor}15`,
      border: `1px solid ${accentColor}40`,
    },
    ...cardSx,
  };

  return (
    <Card sx={defaultCardSx}>
      <CardContent sx={{ padding: 3 }}>
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
}

export default ModernPlot;
