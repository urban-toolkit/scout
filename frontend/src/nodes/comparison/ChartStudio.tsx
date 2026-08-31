// ChartStudio.tsx
import { useCallback, useState, type CSSProperties } from "react";
import JsCodeEditor from "../../charts/JsCodeEditor";
import JsonCodeEditor from "../../node-components/JsonCodeEditor";
import { CustomD3Renderer } from "../../charts/renderers/customD3Renderer";
import { VegaLiteRenderer } from "../../charts/renderers/vegaLiteRenderer";
import { publishChartType, seedMetricData, type ChartEngine } from "../../charts/chartTypes";

const D3_STARTER_CODE = `// available: container (a <div>), data, params, d3, containerWidth
// Grouped bar chart: one group per scenario key, one bar per metric column -
// this uses every column at once, which none of the built-in bar/pie/table
// charts can do (they're locked to a single x/y field). Uses containerWidth
// so it redraws responsively when the node is resized.
container.innerHTML = "";

const keys = Object.keys(data);
const metrics = Object.keys(data[keys[0]] ?? {});

const margin = { top: 30, right: 20, bottom: 40, left: 60 };
const width = containerWidth - margin.left - margin.right;
const height = 260 - margin.top - margin.bottom;

const svg = d3.select(container)
  .append("svg")
  .attr("width", width + margin.left + margin.right)
  .attr("height", height + margin.top + margin.bottom)
  .append("g")
  .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

const x0 = d3.scaleBand().domain(keys).range([0, width]).padding(0.25);
const x1 = d3.scaleBand().domain(metrics).range([0, x0.bandwidth()]).padding(0.1);
const y = d3.scaleLinear()
  .domain([0, d3.max(keys, (k) => d3.max(metrics, (m) => data[k][m]))])
  .nice()
  .range([height, 0]);
const color = d3.scaleOrdinal().domain(metrics).range(d3.schemeTableau10);

svg.append("g")
  .selectAll("g")
  .data(keys)
  .enter()
  .append("g")
  .attr("transform", (k) => "translate(" + x0(k) + ",0)")
  .selectAll("rect")
  .data((k) => metrics.map((m) => ({ metric: m, value: data[k][m] })))
  .enter()
  .append("rect")
  .attr("x", (d) => x1(d.metric))
  .attr("y", (d) => y(d.value))
  .attr("width", x1.bandwidth())
  .attr("height", (d) => height - y(d.value))
  .attr("fill", (d) => color(d.metric))
  .attr("rx", 3);

svg.append("g").attr("transform", "translate(0," + height + ")").call(d3.axisBottom(x0));
svg.append("g").call(d3.axisLeft(y).ticks(4));

// legend
const legend = svg.append("g").attr("transform", "translate(0, -20)");
metrics.forEach((m, i) => {
  const g = legend.append("g").attr("transform", "translate(" + i * 140 + ", 0)");
  g.append("rect").attr("width", 10).attr("height", 10).attr("fill", color(m));
  g.append("text").attr("x", 14).attr("y", 9).style("font-size", "10px").text(m);
});
`;

// Grouped-bar equivalent, expressed purely declaratively - demonstrates the
// same "uses every metric column at once" capability as the D3 starter,
// via the {key, field, value} long-format records data.values is
// auto-populated with (see vegaLiteRuntime.ts).
const VEGA_STARTER_SPEC = {
  $schema: "https://vega.github.io/schema/vega-lite/v6.json",
  description:
    "Grouped bar chart - one group per scenario key, one bar per metric field. " +
    "data.values is auto-populated as {key, field, value} records.",
  mark: "bar",
  encoding: {
    x: { field: "key", type: "nominal", title: "Scenario" },
    xOffset: { field: "field" },
    y: { field: "value", type: "quantitative" },
    color: { field: "field", type: "nominal", title: "Metric" },
  },
};

const STARTER_DATA = {
  A: { "median flood depth": 7.16, "mean flood depth": 6.35 },
  B: { "median flood depth": 5.02, "mean flood depth": 4.71 },
};

const STARTER_PARAMS = {};

export default function ChartStudio({
  onPublished,
  onCancel,
}: {
  onPublished: (name: string) => void;
  onCancel: () => void;
}) {
  const [name, setName] = useState("");
  const [engine, setEngine] = useState<ChartEngine>("d3");
  // Kept separate (not one shared "code" field) so toggling the engine back
  // and forth never discards either draft.
  const [d3Code, setD3Code] = useState(D3_STARTER_CODE);
  const [vegaSpec, setVegaSpec] = useState<unknown>(VEGA_STARTER_SPEC);
  const [sampleData, setSampleData] = useState<unknown>(STARTER_DATA);
  const [sampleParams, setSampleParams] = useState<unknown>(STARTER_PARAMS);
  const [previewProps, setPreviewProps] = useState<{
    data: Record<string, Record<string, number>>;
    params: Record<string, any>;
  } | null>(null);
  const [publishError, setPublishError] = useState<string | null>(null);
  const [publishing, setPublishing] = useState(false);
  const [pushError, setPushError] = useState<string | null>(null);
  const [pushSuccess, setPushSuccess] = useState<string | null>(null);
  const [pushing, setPushing] = useState(false);
  // Tracks the last code/spec that actually compiled+ran without throwing,
  // so Publish can require a proven-good Preview first - publishing broken
  // code becomes a real source-file parse error, not an isolated runtime one.
  // Tracked per engine so switching engines doesn't wrongly gate/ungate.
  const [lastGoodPreviewCode, setLastGoodPreviewCode] = useState<string | null>(null);
  const [lastGoodPreviewSpecJSON, setLastGoodPreviewSpecJSON] = useState<string | null>(null);

  // sampleData/sampleParams are always valid JSON already - JsonCodeEditor
  // (same editor "def" mode uses) only propagates onChange for parseable
  // JSON, showing syntax errors inline via its own lint gutter instead of
  // a submit-time try/catch.
  const runPreview = () => {
    setPreviewProps({
      data: sampleData as Record<string, Record<string, number>>,
      params: sampleParams as Record<string, any>,
    });
  };

  // Writes Sample data as real backend/data/served/metric/<key>.csv files -
  // lets you test a chart against real /api/comparison-view data without
  // running the actual computation pipeline. Overwrites existing scenario
  // data for the keys involved, so confirm first.
  const handlePushSampleData = async () => {
    const data = sampleData as Record<string, Record<string, number>>;
    const keys = Object.keys(data);
    if (keys.length === 0) return;

    const ok = window.confirm(
      `This overwrites real backend metric data for: ${keys.join(", ")}.\n\n` +
        "Any comparison node using these keys will pick up these sample values. Continue?",
    );
    if (!ok) return;

    setPushing(true);
    setPushError(null);
    setPushSuccess(null);
    try {
      const result = await seedMetricData(data);
      setPushSuccess(`Wrote: ${result.written.join(", ")}`);
    } catch (e) {
      setPushError(e instanceof Error ? e.message : "Failed to push sample data");
    } finally {
      setPushing(false);
    }
  };

  const canPublish =
    previewProps !== null &&
    (engine === "d3"
      ? d3Code === lastGoodPreviewCode
      : JSON.stringify(vegaSpec) === lastGoodPreviewSpecJSON);

  // Stable per-`d3Code` identity so CustomD3Renderer's effect only re-runs
  // when code/data/params actually change, not on unrelated re-renders
  // (e.g. editing the name field) that would otherwise change this
  // callback's reference on every keystroke.
  const handleD3PreviewResult = useCallback(
    (ok: boolean) => setLastGoodPreviewCode(ok ? d3Code : null),
    [d3Code],
  );
  const handleVegaPreviewResult = useCallback(
    (ok: boolean) => setLastGoodPreviewSpecJSON(ok ? JSON.stringify(vegaSpec) : null),
    [vegaSpec],
  );

  const handlePublish = async () => {
    if (!canPublish) {
      setPublishError("Preview your code successfully before publishing.");
      return;
    }

    const trimmed = name.trim();
    if (!/^[a-zA-Z0-9_-]+$/.test(trimmed)) {
      setPublishError(
        "Name must be non-empty and contain only letters, numbers, underscores, and hyphens.",
      );
      return;
    }
    setPublishing(true);
    setPublishError(null);
    try {
      const code = engine === "d3" ? d3Code : JSON.stringify(vegaSpec, null, 2);
      await publishChartType({ name: trimmed, description: "", code, engine });
      onPublished(trimmed);
    } catch (e) {
      setPublishError(e instanceof Error ? e.message : "Failed to publish chart type");
    } finally {
      setPublishing(false);
    }
  };

  const engineButtonStyle = (active: boolean): CSSProperties => ({
    width: "auto",
    padding: "3px 12px",
    fontSize: 11,
    fontWeight: active ? 700 : 400,
    background: active ? "#1f78b4" : "#fff",
    color: active ? "#fff" : "#111827",
  });

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1.1fr 0.9fr",
        gap: 0,
        width: "100%",
        height: "100%",
        minHeight: 0,
        boxSizing: "border-box",
      }}
    >
      {/* Left: authoring - code stays visible here at all times, never pushed
          out of view by previewing, so edit -> preview -> edit needs no
          scrolling back and forth. */}
      <div
        className="nodrag nowheel"
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 8,
          height: "100%",
          minHeight: 0,
          overflow: "auto",
          padding: "8px 10px",
          boxSizing: "border-box",
          borderRight: "1px solid #eee",
        }}
      >
        <input
          type="text"
          placeholder="Chart type name (e.g. mySankey)"
          value={name}
          onChange={(e) => setName(e.target.value)}
          style={{
            padding: "7px 10px",
            fontSize: 13,
            border: "1px solid #e5e7eb",
            borderRadius: 8,
            outline: "none",
            boxShadow: "0 1px 2px rgba(0,0,0,0.06)",
            fontFamily: "inherit",
          }}
        />

        <div style={{ display: "flex", gap: 4 }}>
          <button
            type="button"
            onClick={() => setEngine("d3")}
            className="cvnode__actionBtn"
            style={engineButtonStyle(engine === "d3")}
          >
            D3 / JS
          </button>
          <button
            type="button"
            onClick={() => setEngine("vega-lite")}
            className="cvnode__actionBtn"
            style={engineButtonStyle(engine === "vega-lite")}
          >
            Vega-Lite
          </button>
        </div>

        <div style={{ fontSize: 11, color: "#666" }}>
          {engine === "d3"
            ? "Code runs against the sample data below on Preview — this does not call the backend, so it won't catch a mismatch against real scenario data (e.g. a missing column)."
            : "Spec runs against the sample data below on Preview. data.values is auto-populated as {key, field, value} records — one per scenario/metric pair."}
        </div>

        {engine === "d3" ? (
          <JsCodeEditor value={d3Code} onChange={setD3Code} height="220px" />
        ) : (
          <JsonCodeEditor value={vegaSpec} onChange={setVegaSpec} height="220px" />
        )}

        {/* Fills the remaining height down to the bottom of the column,
            each box scrolling internally rather than resizing by hand. */}
        <div style={{ display: "flex", gap: 8, flex: 1, minHeight: 0 }}>
          <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
            <div style={{ fontSize: 11, color: "#666", marginBottom: 2 }}>Sample data (JSON)</div>
            <div style={{ flex: 1, minHeight: 0 }}>
              <JsonCodeEditor
                value={sampleData}
                onChange={setSampleData}
                lineNumbers={false}
                compact
                height="100%"
              />
            </div>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "flex-end", gap: 8, marginTop: 4 }}>
              {(pushError || pushSuccess) && (
                <span style={{ fontSize: 10, color: pushError ? "#b91c1c" : "#15803d" }}>
                  {pushError ?? pushSuccess}
                </span>
              )}
              <button
                type="button"
                onClick={handlePushSampleData}
                disabled={pushing}
                title="Write this sample data as real backend/data/served/metric/<key>.csv files, so you can test this chart against real /api/comparison-view data"
                className="cvnode__actionBtn"
                style={{ width: "auto", padding: "1px 8px", fontSize: 10 }}
              >
                {pushing ? "Saving…" : "Save to backend"}
              </button>
            </div>
          </div>
          <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
            <div style={{ fontSize: 11, color: "#666", marginBottom: 2 }}>Example usage</div>
            <div style={{ flex: 1, minHeight: 0 }}>
              <JsonCodeEditor
                value={sampleParams}
                onChange={setSampleParams}
                lineNumbers={false}
                compact
                height="100%"
              />
            </div>
            <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 4 }}>
              <button
                type="button"
                disabled
                title="Not yet implemented"
                className="cvnode__actionBtn"
                style={{ width: "auto", padding: "1px 8px", fontSize: 10 }}
              >
                Generate
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Right: output - preview render and publish, always visible next to
          the code, never covering it. */}
      <div
        className="nodrag nowheel"
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 8,
          height: "100%",
          minHeight: 0,
          overflow: "auto",
          padding: "8px 10px",
          boxSizing: "border-box",
        }}
      >
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <button
            type="button"
            onClick={runPreview}
            className="cvnode__actionBtn"
            style={{ width: "auto", padding: "4px 12px" }}
          >
            Preview
          </button>
        </div>

        <div style={{ border: "1px solid #eee", borderRadius: 8, padding: 8, flex: 1 }}>
          {previewProps ? (
            engine === "d3" ? (
              <CustomD3Renderer
                code={d3Code}
                data={previewProps.data}
                params={previewProps.params}
                onResult={handleD3PreviewResult}
              />
            ) : (
              <VegaLiteRenderer
                spec={vegaSpec}
                data={previewProps.data}
                onResult={handleVegaPreviewResult}
              />
            )
          ) : (
            <div style={{ fontSize: 12, color: "#999" }}>
              Click Preview to render your {engine === "d3" ? "code" : "spec"} against the sample data.
            </div>
          )}
        </div>

        <div style={{ display: "flex", gap: 8, alignItems: "center", justifyContent: "flex-end" }}>
          {publishError && <span style={{ color: "#b91c1c", fontSize: 12, flex: 1 }}>{publishError}</span>}
          <button
            type="button"
            onClick={onCancel}
            className="cvnode__actionBtn"
            style={{ width: "auto", padding: "4px 12px" }}
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handlePublish}
            disabled={publishing || !canPublish}
            title={canPublish ? undefined : "Preview your code successfully before publishing"}
            className="cvnode__actionBtn"
            style={{ width: "auto", padding: "4px 12px" }}
          >
            {publishing ? "Publishing…" : "Publish"}
          </button>
        </div>
      </div>
    </div>
  );
}
