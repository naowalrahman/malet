import { Box, Checkbox, FormControlLabel, IconButton, Tooltip, Grid, Typography } from "@mui/material";
import { DatePicker } from "@mui/x-date-pickers/DatePicker";
import { Dayjs } from "dayjs";
import { Info } from "@mui/icons-material";

function CustomIndicatorDatePicker({
  checked,
  onCheckedChange,
  value,
  onChange,
  mainDate,
  mainDateLabel,
  disabled = false,
}: {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  value: Dayjs | null;
  onChange: (date: Dayjs | null) => void;
  mainDate: Dayjs | null;
  mainDateLabel: string;
  disabled?: boolean;
}) {
  return (
    <>
      <Grid size={{ xs: 12 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
          <FormControlLabel
            control={
              <Checkbox checked={checked} onChange={(e) => onCheckedChange(e.target.checked)} disabled={disabled} />
            }
            label={
              <Box sx={{ display: "flex", alignItems: "center" }}>
                Use custom indicator start date
                <Tooltip
                  title={
                    <Box sx={{ maxWidth: 320, p: 1 }}>
                      <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 0.5 }}>
                        Custom Indicator Start Date
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        When checked, allows you to specify a custom start date for fetching historical data used to
                        calculate technical indicators. If unchecked, the default is 52 weeks before the {mainDateLabel}
                        . The custom start date must be at least 1 year before the {mainDateLabel}.
                      </Typography>
                    </Box>
                  }
                  placement="right"
                  arrow
                >
                  <IconButton sx={{ p: 0.5 }}>
                    <Info sx={{ fontSize: 16 }} />
                  </IconButton>
                </Tooltip>
              </Box>
            }
          />
        </Box>
      </Grid>

      {checked && (
        <Grid size={{ xs: 12, md: 6 }}>
          <DatePicker
            label="Indicator Start Date"
            value={value}
            onChange={onChange}
            maxDate={mainDate?.subtract(1, "year") || undefined}
            disabled={disabled}
            slotProps={{
              textField: {
                fullWidth: true,
                helperText: `Must be at least 1 year before ${mainDateLabel}`,
              },
            }}
          />
        </Grid>
      )}
    </>
  );
}

export default CustomIndicatorDatePicker;
