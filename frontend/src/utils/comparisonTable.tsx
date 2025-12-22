// utils/comparisonTable.tsx
// import React from "react";

export function ComparisonTable({
  values,
  metric,
  props,
}: {
  values: Record<string, number>;
  metric: string;
  props?: Record<string, any>;
}) {
  const entries = Object.entries(values);

  const label = props?.unit ? `${metric} (${props.unit})` : metric; // final header label

  return (
    <div
      style={{
        width: "100%",
        overflowX: "auto",
        fontFamily: "Inter, sans-serif",
      }}
    >
      <table
        style={{
          width: "100%",
          borderCollapse: "separate",
          borderSpacing: 0,
          fontSize: 15,
          borderRadius: 8,
          overflow: "hidden",
          textAlign: "center",
          boxShadow: "0 1px 4px rgba(0,0,0,0.1)",
        }}
      >
        <thead>
          <tr style={{ backgroundColor: "#f0f0f0" }}>
            <th
              style={{
                padding: "10px 12px",
                borderBottom: "1px solid #e0e0e0",
                fontWeight: 600,
                textAlign: "center",
              }}
            >
              Scenario
            </th>
            <th
              style={{
                padding: "10px 12px",
                borderBottom: "1px solid #e0e0e0",
                fontWeight: 600,
                textAlign: "center",
              }}
            >
              {label}
            </th>
          </tr>
        </thead>

        <tbody>
          {entries.map(([key, value], i) => (
            <tr
              key={key}
              style={{ background: i % 2 === 0 ? "white" : "#fafafa" }}
            >
              <td
                style={{
                  padding: "10px 12px",
                  borderBottom: "1px solid #f0f0f0",
                }}
              >
                {key}
              </td>

              <td
                style={{
                  padding: "10px 12px",
                  borderBottom: "1px solid #f0f0f0",
                  fontVariantNumeric: "tabular-nums",
                  fontWeight: 500,
                }}
              >
                {value.toFixed(5)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
