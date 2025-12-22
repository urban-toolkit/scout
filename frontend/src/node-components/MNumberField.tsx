import * as React from "react";
import { NumberField as BaseNumberField } from "@base-ui-components/react/number-field";
import IconButton from "@mui/material/IconButton";
import FormControl from "@mui/material/FormControl";
import FormHelperText from "@mui/material/FormHelperText";
import OutlinedInput from "@mui/material/OutlinedInput";
import InputAdornment from "@mui/material/InputAdornment";
import InputLabel from "@mui/material/InputLabel";
import KeyboardArrowUpIcon from "@mui/icons-material/KeyboardArrowUp";
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown";

function SSRInitialFilled(_: BaseNumberField.Root.Props) {
  return null;
}
SSRInitialFilled.muiName = "Input";

type MNumberFieldProps = {
  id?: string;
  label?: React.ReactNode;
  helperText?: string;
  placeholder?: string;
  error?: boolean;
  size?: "small" | "medium";
  min?: number;
  max?: number;
  step?: number; // can be float
  value: number | null;
  onChange: (v: number | null) => void;
  showStepper?: boolean; // 👈 NEW (default true)
};

export default function MNumberField({
  id: idProp,
  label,
  helperText,
  error,
  size = "small",
  min,
  max,
  step = 1,
  value,
  onChange,
  placeholder,
  showStepper = true,
}: MNumberFieldProps) {
  let id = React.useId();
  if (idProp) id = idProp;

  return (
    <BaseNumberField.Root
      allowWheelScrub
      value={value}
      min={min}
      max={max}
      step={step}
      onValueChange={(next) => onChange(next)}
      render={(props, state) => (
        <FormControl
          size={size}
          ref={props.ref}
          disabled={state.disabled}
          required={state.required}
          error={error}
          variant="outlined"
          fullWidth
        >
          {props.children}
        </FormControl>
      )}
    >
      <SSRInitialFilled />
      <InputLabel htmlFor={id}>{label}</InputLabel>

      <BaseNumberField.Input
        id={id}
        render={(props, state) => (
          <OutlinedInput
            label={label}
            inputRef={props.ref}
            value={state.inputValue}
            placeholder={placeholder}
            onBlur={props.onBlur}
            onChange={props.onChange}
            onKeyUp={props.onKeyUp}
            onKeyDown={props.onKeyDown}
            onFocus={props.onFocus}
            slotProps={{ input: props }}
            endAdornment={
              showStepper ? ( // 👈 toggle the arrows here
                <InputAdornment
                  position="end"
                  sx={{
                    flexDirection: "column",
                    maxHeight: "unset",
                    alignSelf: "stretch",
                    borderLeft: "1px solid",
                    borderColor: "divider",
                    ml: 0,
                    "& button": {
                      py: 0,
                      flex: 1,
                      borderRadius: 0.5,
                    },
                  }}
                >
                  <BaseNumberField.Increment
                    render={<IconButton size={size} aria-label="Increase" />}
                  >
                    <KeyboardArrowUpIcon
                      fontSize={size}
                      sx={{ transform: "translateY(2px)" }}
                    />
                  </BaseNumberField.Increment>

                  <BaseNumberField.Decrement
                    render={<IconButton size={size} aria-label="Decrease" />}
                  >
                    <KeyboardArrowDownIcon
                      fontSize={size}
                      sx={{ transform: "translateY(-2px)" }}
                    />
                  </BaseNumberField.Decrement>
                </InputAdornment>
              ) : undefined
            }
            sx={{ pr: showStepper ? 0 : undefined }}
          />
        )}
      />

      {helperText && (
        <FormHelperText
          sx={{
            ml: 0,
            textAlign: "center",
            "&:empty": { m: 0 },
          }}
        >
          {helperText}
        </FormHelperText>
      )}
    </BaseNumberField.Root>
  );
}
