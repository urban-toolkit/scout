// utils/renderComparison.ts
import type { ComparisonDef } from "./types";
import type { ReactNode } from "react";
import { ComparisonBarChart } from "./comparisonBarChart";
import { ComparisonPieChart } from "./comparisonPieChart";
import { ComparisonTable } from "./comparisonTable";

type ComparisonAPIResponse = {
  status: string;
  metric: string;
  chart: string;
  props: Record<string, any>;
  values: Record<string, number>;
};

export async function renderComparisonFromDef(
  def: ComparisonDef,
  signal?: AbortSignal,
): Promise<ReactNode> {
  console.log("Rendering comparison from definition:", def);

  const res = await fetch(
    "https://giavanna-stripiest-mustafa.ngrok-free.dev/api/comparison-view",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
      },
      body: JSON.stringify(def),
      signal,
    },
  );

  if (!res.ok) {
    throw new Error(`Comparison fetch failed: ${res.status}`);
  }

  const data: ComparisonAPIResponse = await res.json();

  if (data.chart === "bar") {
    return (
      <ComparisonBarChart
        values={data.values}
        metric={data.metric}
        props={data.props}
      />
    );
  }

  if (data.chart === "pie") {
    return (
      <ComparisonPieChart
        values={data.values}
        metric={data.metric}
        props={data.props}
      />
    );
  }

  if (data.chart === "table") {
    return (
      <ComparisonTable
        values={data.values}
        metric={data.metric}
        props={data.props}
      />
    );
  }

  return (
    <pre style={{ fontSize: "10px" }}>{JSON.stringify(data, null, 2)}</pre>
  );
}
