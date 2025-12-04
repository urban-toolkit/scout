import type { ReactNode } from "react";
import {
  FormControl,
  FormLabel,
  RadioGroup,
  FormControlLabel,
  FormHelperText,
  Radio,
  TextField,
  Slider,
  Autocomplete,
  Checkbox,
  FormGroup,
  Typography,
} from "@mui/material";
import type { WidgetDef } from "./types";
import { AddressAutofill } from "@mapbox/search-js-react";

import dayjs, { Dayjs } from "dayjs";
import { LocalizationProvider } from "@mui/x-date-pickers/LocalizationProvider";
import { AdapterDayjs } from "@mui/x-date-pickers/AdapterDayjs";
import { DateTimePicker } from "@mui/x-date-pickers/DateTimePicker";
import MNumberField from "../../components/MNumberField";

import CheckBoxIcon from "@mui/icons-material/CheckBox";
import CheckBoxOutlineBlankIcon from "@mui/icons-material/CheckBoxOutlineBlank";

const icon = <CheckBoxOutlineBlankIcon fontSize="small" />;
const checkedIcon = <CheckBoxIcon fontSize="small" />;

export function renderWidgetFromWidgetDef(
  widgetDef: WidgetDef | undefined,
  value: any,
  onValueChange?: (id: string, variable: string, value: any) => void
): ReactNode {
  if (!widgetDef) return null;

  switch (widgetDef.type) {
    case "radio-group":
      return renderRadioGroup(widgetDef, value, onValueChange);

    case "datetime-picker":
      return renderDateTimePickerWidget(widgetDef, value, onValueChange);

    case "slider":
      return renderSliderWidget(widgetDef, value, onValueChange);

    case "number-input":
      return renderNumberInputWidget(widgetDef, value, onValueChange);

    case "location-input":
      return renderLocationFieldWidget(widgetDef, value, onValueChange);

    case "dropdown":
      return renderDropdownWidget(widgetDef, value, onValueChange);

    case "checkbox":
      return renderCheckboxWidget(widgetDef, value, onValueChange);

    case "text":
      return renderTextWidget(widgetDef, value);

    case "text-input":
      return renderTextInputWidget(widgetDef, value, onValueChange);

    default:
      return (
        <div style={{ fontSize: 12 }}>
          Unsupported widget type: <code>{widgetDef.type}</code>
        </div>
      );
  }
}

export function renderTextInputWidget(
  widget: WidgetDef,
  value: any,
  onValueChange?: (widgetId: string, variable: string, value: any) => void
): React.ReactNode {
  const variable = widget.variable ?? widget.id;

  const currentVal: string =
    typeof value === "string"
      ? value
      : typeof widget["default-value"] === "string"
      ? widget["default-value"]
      : "";

  const inputKind =
    (widget["input-kind"] as
      | "text"
      | "email"
      | "password"
      | "search"
      | "url"
      | "tel"
      | undefined) ?? "text";

  const multiline = widget.multiline === true;
  const maxLength =
    typeof widget["max-length"] === "number" ? widget["max-length"] : undefined;

  const minRows =
    typeof (widget as any).minRows === "number" ? (widget as any).minRows : 2;
  const maxRows =
    typeof (widget as any).maxRows === "number"
      ? (widget as any).maxRows
      : undefined;

  return (
    <div style={{ width: "100%", marginTop: "4px" }}>
      <TextField
        label={widget.title ?? variable}
        fullWidth
        size="small"
        type={inputKind}
        value={currentVal}
        onChange={(e) => {
          onValueChange?.(widget.id, variable, e.target.value);
        }}
        placeholder={widget.placeholder}
        helperText={widget.description || undefined}
        multiline={multiline}
        minRows={multiline ? minRows : undefined}
        maxRows={multiline ? maxRows : undefined}
        inputProps={{
          maxLength,
        }}
      />
    </div>
  );
}

export function renderTextWidget(
  widget: WidgetDef,
  value: any
): React.ReactNode {
  const text = value ?? widget["default-value"] ?? "";

  const fontSize = widget["text-size"] ?? 12;
  const color = widget.color ?? "#000";
  const align = widget.align ?? "left";
  const underline = widget.underline === true;
  const italic = widget.italic === true;
  const bold = widget.bold === true;

  // Compute a single style string
  const style: React.CSSProperties = {
    fontSize: fontSize,
    color: color,
    textAlign: align as any,
    fontWeight: bold ? 600 : 400,
    fontStyle: italic ? "italic" : "normal",
    textDecoration: underline ? "underline" : "none",
    width: "100%",
  };

  return (
    <div className="nodrag" style={{ width: "100%", textAlign: "center" }}>
      <Typography style={{ ...style, whiteSpace: "pre-line" }}>
        {text}
      </Typography>

      {widget.description && (
        <div
          style={{
            marginTop: 4,
            fontSize: "0.75rem",
            color: "#666",
            textAlign: "center",
          }}
        >
          {widget.description}
        </div>
      )}
    </div>
  );
}

export function renderCheckboxWidget(
  widget: WidgetDef,
  value: any,
  onValueChange?: (widgetId: string, variable: string, value: any) => void
): React.ReactNode {
  const variable = widget.variable ?? widget.id;
  const mode = widget.mode === "group" ? "group" : "single";
  const orientation =
    widget.orientation === "horizontal" ? "horizontal" : "vertical";

  // ----- SINGLE MODE -----
  if (mode === "single") {
    const checked =
      typeof value === "boolean"
        ? value
        : typeof widget["default-value"] === "boolean"
        ? widget["default-value"]
        : false;

    return (
      <div className="nodrag" style={{ width: "100%" }}>
        <FormControl>
          <FormLabel>{widget.title ?? ""}</FormLabel>
          <FormGroup>
            <FormControlLabel
              control={
                <Checkbox
                  checked={checked}
                  onChange={(_, isChecked) => {
                    onValueChange?.(widget.id, variable, isChecked);
                  }}
                />
              }
              label={widget.title ?? variable}
            />
          </FormGroup>
          {widget.description && (
            <FormHelperText
              style={{
                textAlign: "center",
                paddingLeft: 0,
                paddingRight: 0,
              }}
            >
              {widget.description}
            </FormHelperText>
          )}
        </FormControl>
      </div>
    );
  }

  // ----- GROUP MODE -----
  const items: string[] = (widget as any).items ?? [];
  const selected: string[] = Array.isArray(value)
    ? value
    : Array.isArray(widget["default-value"])
    ? widget["default-value"]
    : [];

  const isHorizontal = orientation === "horizontal";

  return (
    <div className="nodrag" style={{ width: "100%" }}>
      <FormControl component="fieldset">
        <FormLabel>{widget.title ?? variable}</FormLabel>
        <FormGroup row={isHorizontal}>
          {items.map((item) => {
            const checked = selected.includes(item);
            return (
              <FormControlLabel
                key={item}
                control={
                  <Checkbox
                    checked={checked}
                    onChange={(_, isChecked) => {
                      let next: string[];
                      if (isChecked) {
                        // add item
                        next = checked ? selected : [...selected, item];
                      } else {
                        // remove item
                        next = selected.filter((v) => v !== item);
                      }
                      onValueChange?.(widget.id, variable, next);
                    }}
                  />
                }
                label={item}
              />
            );
          })}
        </FormGroup>
        {widget.description && (
          <FormHelperText
            style={{
              textAlign: "center",
              paddingLeft: 0,
              paddingRight: 0,
            }}
          >
            {widget.description}
          </FormHelperText>
        )}
      </FormControl>
    </div>
  );
}

export function renderDropdownWidget(
  widget: WidgetDef,
  value: any,
  onValueChange?: (id: string, variable: string, value: any) => void
) {
  const variable = widget.variable ?? widget.id;

  const items: string[] = Array.isArray(widget.items) ? widget.items : [];
  const multiple = widget["multi-select"] === true;
  const basePlaceholder = widget.placeholder;
  const hasDescription = !!widget.description;

  // current value(s)
  let currentVal: any = multiple ? [] : null;

  if (multiple) {
    currentVal = Array.isArray(value)
      ? value
      : Array.isArray(widget["default-value"])
      ? widget["default-value"]
      : [];
  } else {
    currentVal =
      typeof value === "string"
        ? value
        : typeof widget["default-value"] === "string"
        ? widget["default-value"]
        : null;
  }

  // dynamic placeholder: "N selected" for multi, normal placeholder otherwise
  let placeholder = basePlaceholder;
  if (multiple) {
    const count = Array.isArray(currentVal) ? currentVal.length : 0;
    if (count > 0) {
      placeholder = `${count} selected`;
    }
  }

  return (
    <div style={{ width: "100%", marginTop: "-6px" }}>
      <Autocomplete
        options={items}
        multiple={multiple}
        disableCloseOnSelect={multiple}
        value={currentVal}
        onChange={(_, newValue) => {
          onValueChange?.(widget.id, variable, newValue);
        }}
        renderOption={(props, option, state) => {
          if (!multiple) {
            // single-select: default rendering
            return (
              <li {...props} key={option}>
                {option}
              </li>
            );
          }

          const { selected } = state;
          const { key, ...optionProps } = props;
          return (
            <li key={key} {...optionProps}>
              <Checkbox
                icon={icon}
                checkedIcon={checkedIcon}
                style={{ marginRight: 8 }}
                checked={selected}
              />
              {option}
            </li>
          );
        }}
        // searchable by default
        filterSelectedOptions={false}
        renderInput={(params) => (
          <TextField
            {...params}
            label={widget.title ?? variable}
            placeholder={placeholder}
            size="small"
            helperText={widget.description || undefined}
            slotProps={{
              formHelperText: hasDescription
                ? {
                    sx: {
                      display: "flex",
                      justifyContent: "center",
                      m: 0,
                      mt: "2px",
                    },
                  }
                : undefined,
            }}
          />
        )}
        sx={{
          mt: 1,
          // 🔑 Hide the chips so the input stays a single line
          "& .MuiAutocomplete-tag": {
            display: "none",
          },
        }}
      />
    </div>
  );
}

export function renderLocationFieldWidget(
  widget: WidgetDef,
  value: any,
  onValueChange?: (widgetId: string, variable: string, value: any) => void
): React.ReactNode {
  const variable = widget.variable ?? widget.id;

  const initialVal: string =
    typeof widget["default-value"] === "string" ? widget["default-value"] : "";

  const placeholder = widget.placeholder;
  const hasDescription = !!widget.description;

  return (
    <div style={{ width: "100%", marginTop: "4px" }}>
      <form onSubmit={(e) => e.preventDefault()}>
        <AddressAutofill
          accessToken="pk.eyJ1IjoicXNoYWhydWtoNDEiLCJhIjoiY20yeXhpd2tkMDVtZjJsb29tcW13dHJjMiJ9.uLUf8R7TESQ97G55AbAifw"
          onRetrieve={(res) => {
            const feat = res.features?.[0];
            if (!feat) return;

            // Extract Mapbox-provided coordinates
            const [lon, lat] = feat.geometry.coordinates;

            // Commit ONLY coordinates to widget state
            onValueChange?.(widget.id, variable, { lat, lon });
          }}
        >
          <TextField
            label={widget.title ?? variable}
            fullWidth
            size="small"
            defaultValue={initialVal}
            placeholder={placeholder}
            sx={{ mt: 1 }}
            helperText={widget.description || undefined}
            slotProps={{
              htmlInput: {
                autoComplete: "street-address",
                maxLength:
                  typeof widget["max-length"] === "number"
                    ? widget["max-length"]
                    : undefined,
              },
              formHelperText: hasDescription
                ? {
                    sx: {
                      display: "flex",
                      justifyContent: "center",
                      m: 0,
                    },
                  }
                : undefined,
            }}
          />
        </AddressAutofill>
      </form>
    </div>
  );
}

export function renderNumberInputWidget(
  widget: WidgetDef,
  value: any,
  onValueChange?: (widgetId: string, variable: string, value: any) => void
) {
  const variable = widget.variable ?? widget.id;

  const min = widget.min;
  const max = widget.max;
  const step: number | undefined =
    typeof widget.step === "number" ? widget.step : undefined;

  const currentVal: number | null =
    typeof value === "number"
      ? value
      : typeof widget["default-value"] === "number"
      ? widget["default-value"]
      : null;

  const handleChange = (v: number | null) => {
    onValueChange?.(widget.id, variable, v);
  };

  return (
    <div style={{ width: "100%", marginTop: "4px" }}>
      <MNumberField
        label={widget.title}
        helperText={widget.description}
        min={min}
        max={max}
        step={step}
        value={currentVal}
        placeholder={widget.placeholder}
        onChange={handleChange}
        showStepper={!widget.hideStepper}
      />
    </div>
  );
}

export function renderSliderWidget(
  widget: WidgetDef,
  value: any,
  onValueChange?: (widgetId: string, variable: string, value: any) => void
): React.ReactNode {
  const variable = widget.variable ?? widget.id;

  const currentVal =
    typeof value === "number"
      ? value
      : typeof widget["default-value"] === "number"
      ? widget["default-value"]
      : widget.min;

  const min = widget.min;
  const max = widget.max;
  const step = widget.step ?? 1;

  const orientation =
    widget.orientation === "vertical" ? "vertical" : "horizontal";
  const isVertical = orientation === "vertical";

  return (
    <div
      className="nodrag"
      style={{
        width: "100%",
      }}
    >
      {/* Title */}
      <div
        style={{
          marginBottom: "4px",
          fontSize: "16px",
          // fontWeight: 500,
          textAlign: "center",
        }}
      >
        {widget.title ?? variable}
      </div>

      {isVertical ? (
        // 🔹 Vertical layout: labels top/bottom, slider with fixed height
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            // important for vertical slider
          }}
        >
          {/* max on top */}
          <div
            style={{
              fontSize: "16px",
              color: "#666",
              marginBottom: 10,
            }}
          >
            {max}
          </div>

          <Slider
            value={currentVal}
            min={min}
            max={max}
            step={step}
            orientation="vertical"
            valueLabelDisplay="on"
            onChange={(_, v) => {
              if (typeof v === "number") {
                onValueChange?.(widget.id, variable, v);
              }
            }}
            sx={{ height: 200 }}
          />

          {/* min at bottom */}
          <div
            style={{
              fontSize: "16px",
              color: "#666",
              marginTop: 10,
            }}
          >
            {min}
          </div>
        </div>
      ) : (
        //
        <div style={{ padding: "0 16px" }}>
          {" "}
          {/* 🔹 equal left/right space */}
          <Slider
            value={currentVal}
            min={min}
            max={max}
            step={step}
            orientation="horizontal"
            valueLabelDisplay="on"
            onChange={(_, v) => {
              if (typeof v === "number") {
                onValueChange?.(widget.id, variable, v);
              }
            }}
            sx={{ mt: 1 }}
          />
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              fontSize: "16px",
              color: "#666",
              marginTop: 0,
            }}
          >
            <span>{min}</span>
            <span>{max}</span>
          </div>
        </div>
      )}

      {widget.description && (
        <div
          style={{
            fontSize: "0.75rem",
            color: "#666",
            textAlign: "center",
            marginTop: 4,
          }}
        >
          {widget.description}
        </div>
      )}
    </div>
  );
}

function renderDateTimePickerWidget(
  widget: WidgetDef,
  value: any,
  onValueChange?: (widgetId: string, variable: string, value: any) => void
): React.ReactNode {
  const variable = widget.variable ?? widget.id;

  // parse current value or default-value
  let currentVal: Dayjs | null = null;
  if (typeof value === "string" && value) {
    const p = dayjs(value);
    currentVal = p.isValid() ? p : null;
  } else if (widget["default-value"]) {
    const p = dayjs(widget["default-value"]);
    currentVal = p.isValid() ? p : null;
  }

  const fmt = widget["display-format"] ?? "YYYY-MM-DD HH:mm";

  // Build textField props safely
  const textFieldProps: any = {
    fullWidth: true,
    size: "small",
    sx: { mt: 1 },
  };

  // Add helper text only if it exists
  if (widget.description) {
    textFieldProps.helperText = widget.description;
    textFieldProps.FormHelperTextProps = {
      sx: {
        display: "flex",
        justifyContent: "center",
        m: 0,
      },
    };
  }

  return (
    <LocalizationProvider dateAdapter={AdapterDayjs}>
      <DateTimePicker
        label={widget.title ?? variable}
        value={currentVal}
        onChange={(newVal: Dayjs | null) => {
          const out = newVal ? newVal.format("YYYY-MM-DDTHH:mm:ss") : null;
          onValueChange?.(widget.id, variable, out);
        }}
        format={fmt}
        slotProps={{
          textField: textFieldProps,
        }}
      />
    </LocalizationProvider>
  );
}

function renderRadioGroup(
  def: WidgetDef,
  value: any,
  onValueChange?: (id: string, variable: string, value: any) => void
): ReactNode {
  const anyDef = def as any;
  const labelId = `${def.id}-label`;
  const items: string[] = anyDef.items ?? [];
  const orientation: "horizontal" | "vertical" =
    anyDef.orientation ?? "vertical";
  const defaultValue = anyDef["default-value"];

  const currentValue = value ?? defaultValue;

  return (
    <FormControl style={{ alignItems: "center" }}>
      <FormLabel id={labelId}>{def.title}</FormLabel>

      <RadioGroup
        aria-labelledby={labelId}
        value={currentValue} // controlled
        onChange={(e) => {
          const v = (e.target as HTMLInputElement).value;
          onValueChange?.(def.id, def.variable, v);
        }}
        name={def.id}
        row={orientation === "horizontal"}
      >
        {items.map((item) => (
          <FormControlLabel
            key={item}
            value={item}
            control={<Radio />}
            label={item}
          />
        ))}
      </RadioGroup>

      {def.description && (
        <FormHelperText
          style={{
            textAlign: "center",
            width: "100%",
            paddingLeft: 0,
            paddingRight: 0,
          }}
        >
          {def.description}
        </FormHelperText>
      )}
    </FormControl>
  );
}
