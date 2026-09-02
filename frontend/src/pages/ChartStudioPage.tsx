// pages/ChartStudioPage.tsx
import { useCallback, useEffect, useState } from "react";
import type { CSSProperties } from "react";
import Dialog from "@mui/material/Dialog";
import DialogTitle from "@mui/material/DialogTitle";
import DialogContent from "@mui/material/DialogContent";
import DialogActions from "@mui/material/DialogActions";
import Button from "@mui/material/Button";
import TextField from "@mui/material/TextField";
import MenuItem from "@mui/material/MenuItem";
import Alert from "@mui/material/Alert";
import Typography from "@mui/material/Typography";

import JsCodeEditor from "../charts/JsCodeEditor";
import JsonCodeEditor from "../node-components/JsonCodeEditor";
import { CustomD3Renderer } from "../charts/renderers/customD3Renderer";
import { VegaLiteRenderer } from "../charts/renderers/vegaLiteRenderer";
import {
  getChartType,
  listChartTypes,
  publishChartType,
  seedMetricData,
  type ChartEngine,
  type ChartTypeSummary,
} from "../charts/chartTypes";
import "./ChartStudioPage.css";

// No worked example chart is pre-filled here on purpose - a fresh chart
// starts from a blank slate; only the function-signature comment stays, so
// the editor isn't a totally blank box with no hint of what's available.
const D3_STARTER_CODE = `// available: container (a <div>), data, params, d3, containerWidth
`;

const VEGA_STARTER_SPEC = {};

const STARTER_DATA = {
  A: { "median flood depth": 7.16, "mean flood depth": 6.35 },
  B: { "median flood depth": 5.02, "mean flood depth": 4.71 },
};

// "Example usage" holds the same shape a Comparison node's grammar JSON
// uses (schemas/comparison.json: key, x/y, chart, props) - not a flat
// params blob - so it doubles as real documentation for whoever wires this
// chart into a Comparison node, copy-pasteable as-is. Preview derives the
// actual render params (axis/axisLabel/props) from it exactly the way the
// backend's /api/comparison-view does (see asFieldList/deriveRenderParams
// below), so what you see in Preview matches real usage.
const STARTER_PARAMS = {
  key: ["A", "B"],
  y: "median flood depth",
  props: {},
};

// Mirrors backend server.py's _as_field_list: a bare string is shorthand
// for a 1-item set.
function asFieldList(value: unknown): string[] {
  if (value == null) return [];
  if (typeof value === "string") return [value];
  if (Array.isArray(value)) return value as string[];
  return [];
}

// Mirrors backend server.py's comparison_view() derivation of axis/axisLabel
// from a ComparisonDef's x/y fields, and renderComparison.tsx's merge of
// props with axis/axisLabel - so Preview matches what a real Comparison
// node would actually pass to this chart's render function.
function deriveRenderParams(example: unknown): {
  axis: "x" | "y";
  axisLabel: string;
  props: Record<string, any>;
} {
  const def = (example ?? {}) as {
    key?: unknown;
    x?: string | string[];
    y?: string | string[];
    props?: Record<string, any>;
  };

  if (!Array.isArray(def.key) || def.key.length === 0) {
    throw new Error('Example usage must include a non-empty "key" array.');
  }

  const xFields = asFieldList(def.x);
  const yFields = asFieldList(def.y);
  const fields = [...xFields, ...yFields];
  if (fields.length === 0) {
    throw new Error('Example usage must declare at least one field via "x" or "y".');
  }

  return {
    axis: xFields.length > 0 ? "x" : "y",
    axisLabel: fields[0],
    props: def.props ?? {},
  };
}

export default function ChartStudioPage() {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
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
  const [previewError, setPreviewError] = useState<string | null>(null);
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

  // "Load existing chart" picker
  const [chartList, setChartList] = useState<ChartTypeSummary[]>([]);
  const [chartListLoading, setChartListLoading] = useState(false);
  const [chartListError, setChartListError] = useState<string | null>(null);
  const [selectedExisting, setSelectedExisting] = useState("");
  const [loadingExisting, setLoadingExisting] = useState(false);

  // Publish flow: overwrite-confirm + error dialogs, inline success indicator.
  const [overwriteDialogOpen, setOverwriteDialogOpen] = useState(false);
  const [errorDialogMessage, setErrorDialogMessage] = useState<string | null>(null);
  const [publishSuccessName, setPublishSuccessName] = useState<string | null>(null);

  const fetchChartList = useCallback(async (signal?: AbortSignal) => {
    setChartListLoading(true);
    setChartListError(null);
    try {
      const list = await listChartTypes(signal);
      setChartList(list);
    } catch (e) {
      if (!(e instanceof Error) || e.name !== "AbortError") {
        setChartListError(e instanceof Error ? e.message : "Failed to load chart list");
      }
    } finally {
      setChartListLoading(false);
    }
  }, []);

  useEffect(() => {
    const ctrl = new AbortController();
    fetchChartList(ctrl.signal);
    return () => ctrl.abort();
  }, [fetchChartList]);

  const resetToNewChart = () => {
    setSelectedExisting("");
    setName("");
    setDescription("");
    setEngine("d3");
    setD3Code(D3_STARTER_CODE);
    setVegaSpec(VEGA_STARTER_SPEC);
    setSampleData(STARTER_DATA);
    setSampleParams(STARTER_PARAMS);
    setPreviewProps(null);
    setPreviewError(null);
    setLastGoodPreviewCode(null);
    setLastGoodPreviewSpecJSON(null);
    setPublishSuccessName(null);
  };

  const handleSelectExisting = async (value: string) => {
    setSelectedExisting(value);
    setPublishSuccessName(null);

    if (value === "") {
      resetToNewChart();
      return;
    }

    setLoadingExisting(true);
    setChartListError(null);
    try {
      const record = await getChartType(value);
      setName(record.name);
      setDescription(record.description ?? "");
      setEngine(record.engine);
      // Also blank out the *other* engine's editor - this chart has no
      // code for it, so leaving whatever a previously-loaded chart left
      // behind there would misleadingly look like it belongs to this one.
      if (record.engine === "d3") {
        setD3Code(record.code);
        setVegaSpec(VEGA_STARTER_SPEC);
      } else {
        setVegaSpec(JSON.parse(record.code));
        setD3Code(D3_STARTER_CODE);
      }
      // The built-in single-metric charts (bar/pie/table/lollipop) read
      // params.axisLabel to pick which sample-data column to plot - a real
      // comparison node gets that from the backend at render time, but
      // Chart Studio's Preview derives it from "Example usage" (see
      // deriveRenderParams above). Loading an existing chart previously
      // left that blank, so those charts silently rendered nothing until
      // you filled it in by hand. Guess a working example from the current
      // Sample data's keys/first metric column - harmless for charts that
      // don't use axisLabel.
      const data = sampleData as Record<string, Record<string, number>>;
      const keys = Object.keys(data);
      const firstMetricKey = keys.length > 0 ? Object.keys(data[keys[0]])[0] : undefined;
      setSampleParams(
        firstMetricKey
          ? { key: keys, y: firstMetricKey, chart: record.name, props: {} }
          : { key: keys, chart: record.name, props: {} },
      );
      // Force a fresh Preview before Publish is allowed again on loaded code.
      setPreviewProps(null);
      setPreviewError(null);
      setLastGoodPreviewCode(null);
      setLastGoodPreviewSpecJSON(null);
    } catch (e) {
      setChartListError(e instanceof Error ? e.message : `Failed to load chart '${value}'`);
    } finally {
      setLoadingExisting(false);
    }
  };

  // sampleData/sampleParams are always valid JSON already - JsonCodeEditor
  // (same editor "def" mode uses) only propagates onChange for parseable
  // JSON, showing syntax errors inline via its own lint gutter instead of
  // a submit-time try/catch. Example usage still needs its own validation
  // here though - it's valid JSON but not necessarily a valid comparison
  // definition (missing key/x/y), which deriveRenderParams checks.
  const runPreview = () => {
    try {
      const { axis, axisLabel, props } = deriveRenderParams(sampleParams);
      setPreviewError(null);
      setPreviewProps({
        data: sampleData as Record<string, Record<string, number>>,
        params: { ...props, axis, axisLabel },
      });
    } catch (e) {
      setPreviewProps(null);
      setPreviewError(e instanceof Error ? e.message : "Invalid \"Example usage\".");
    }
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

  const doPublish = async (trimmedName: string) => {
    setPublishing(true);
    try {
      const code = engine === "d3" ? d3Code : JSON.stringify(vegaSpec, null, 2);
      await publishChartType({
        name: trimmedName,
        description: description.trim(),
        code,
        engine,
      });
      setPublishSuccessName(trimmedName);
      setSelectedExisting(trimmedName);
      await fetchChartList();
    } catch (e) {
      setErrorDialogMessage(e instanceof Error ? e.message : "Failed to publish chart type");
    } finally {
      setPublishing(false);
    }
  };

  const handlePublishClick = async () => {
    if (!canPublish) return;

    const trimmed = name.trim();
    if (!/^[a-zA-Z0-9_-]+$/.test(trimmed)) {
      setErrorDialogMessage(
        "Name must be non-empty and contain only letters, numbers, underscores, and hyphens.",
      );
      return;
    }

    setPublishSuccessName(null);
    setPublishing(true);
    try {
      let exists = true;
      try {
        await getChartType(trimmed);
      } catch {
        exists = false;
      }

      if (exists) {
        setPublishing(false);
        setOverwriteDialogOpen(true);
        return;
      }

      await doPublish(trimmed);
    } finally {
      setPublishing(false);
    }
  };

  const confirmOverwrite = async () => {
    setOverwriteDialogOpen(false);
    await doPublish(name.trim());
  };

  const engineButtonStyle = (active: boolean): CSSProperties => ({
    fontWeight: active ? 700 : 400,
    background: active ? "#1f78b4" : "#fff",
    color: active ? "#fff" : "#111827",
  });

  return (
    <div className="chart-studio-page">
      <div className="chart-studio-page__header">
        <input
          type="text"
          placeholder="Write chart name…"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="chart-studio-page__name-input"
        />
        <TextField
          select
          label="Load existing chart"
          value={selectedExisting}
          onChange={(e) => void handleSelectExisting(e.target.value)}
          size="small"
          sx={{ minWidth: 220 }}
          disabled={chartListLoading || loadingExisting}
        >
          {chartList.map((c) => (
            <MenuItem key={c.name} value={c.name}>
              {c.name}
            </MenuItem>
          ))}
        </TextField>
        <button
          type="button"
          className="chart-studio-page__btn"
          onClick={resetToNewChart}
        >
          New chart
        </button>
        {chartListError && (
          <Typography variant="caption" color="error">
            {chartListError}
          </Typography>
        )}
      </div>

      <div className="chart-studio-page__body">
        {/* Left: authoring - code stays visible here at all times, never
            pushed out of view by previewing, so edit -> preview -> edit
            needs no scrolling back and forth. */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 8,
            height: "100%",
            minHeight: 0,
            overflow: "auto",
            padding: "12px 20px",
            boxSizing: "border-box",
            borderRight: "1px solid #eee",
          }}
        >
          <div style={{ position: "relative" }}>
            <textarea
              placeholder="Description (optional - shown in the chart list)"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={2}
              className="chart-studio-page__input"
              style={{
                width: "100%",
                boxSizing: "border-box",
                resize: "vertical",
                fontFamily: "inherit",
                paddingRight: 70,
              }}
            />
            <button
              type="button"
              disabled
              title="Not yet implemented"
              className="chart-studio-page__textlink-btn chart-studio-page__textlink-btn--corner"
            >
              Generate
            </button>
          </div>

          <div style={{ display: "flex", gap: 4 }}>
            <button
              type="button"
              onClick={() => setEngine("d3")}
              className="chart-studio-page__btn"
              style={engineButtonStyle(engine === "d3")}
            >
              D3 / JS
            </button>
            <button
              type="button"
              onClick={() => setEngine("vega-lite")}
              className="chart-studio-page__btn"
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

          {/* 3:2 split against the sample data/example usage row below - the
              code editor is the primary work area and gets more room. */}
          <div style={{ flex: 3, minHeight: 0 }}>
            {engine === "d3" ? (
              <JsCodeEditor value={d3Code} onChange={setD3Code} height="100%" />
            ) : (
              <JsonCodeEditor value={vegaSpec} onChange={setVegaSpec} height="100%" />
            )}
          </div>

          {/* Fills the remaining height down to the bottom of the column,
              each box scrolling internally rather than resizing by hand. */}
          <div style={{ display: "flex", gap: 8, flex: 2, minHeight: 0 }}>
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
                  className="chart-studio-page__textlink-btn"
                >
                  {pushing ? "Saving…" : "Save to backend"}
                </button>
              </div>
            </div>
            <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
              <div style={{ fontSize: 11, color: "#666", marginBottom: 2 }}>
                Example usage — the comparison JSON a Comparison node would use (key, x/y, props)
              </div>
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
                  className="chart-studio-page__textlink-btn"
                >
                  Generate
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Right: output - preview render and publish, always visible next
            to the code, never covering it. */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 8,
            height: "100%",
            minHeight: 0,
            overflow: "auto",
            padding: "12px 20px",
            boxSizing: "border-box",
          }}
        >
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <button type="button" onClick={runPreview} className="chart-studio-page__btn">
              Preview
            </button>
          </div>

          <div style={{ border: "1px solid #eee", borderRadius: 8, padding: 8, flex: 1 }}>
            {previewError ? (
              <div
                style={{
                  color: "#b91c1c",
                  fontSize: 12,
                  fontFamily: "monospace",
                  whiteSpace: "pre-wrap",
                }}
              >
                {previewError}
              </div>
            ) : previewProps ? (
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

          {/* Centered rather than right-aligned - a right-aligned Publish
              button would sit right under the fixed chat FAB (bottom-right
              corner) and get hidden behind it. */}
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 6,
            }}
          >
            {publishSuccessName && (
              <Typography variant="caption" sx={{ color: "#15803d" }}>
                Published "{publishSuccessName}" ✓
              </Typography>
            )}
            <button
              type="button"
              onClick={() => void handlePublishClick()}
              disabled={publishing || !canPublish}
              title={canPublish ? undefined : "Preview your code successfully before publishing"}
              className="chart-studio-page__btn chart-studio-page__btn--publish"
            >
              {publishing ? "Publishing…" : "Publish"}
            </button>
          </div>
        </div>
      </div>

      <Dialog open={overwriteDialogOpen} onClose={() => setOverwriteDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Overwrite existing chart?</DialogTitle>
        <DialogContent>
          <Typography variant="body2">
            A chart type named "{name.trim()}" already exists. Publishing will overwrite
            it — any comparison node currently rendering "{name.trim()}" will pick up the
            new definition.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOverwriteDialogOpen(false)} disabled={publishing}>
            Cancel
          </Button>
          <Button onClick={() => void confirmOverwrite()} variant="contained" color="warning" disabled={publishing}>
            {publishing ? "Publishing…" : "Overwrite"}
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog open={errorDialogMessage !== null} onClose={() => setErrorDialogMessage(null)} maxWidth="sm" fullWidth>
        <DialogTitle>Publish failed</DialogTitle>
        <DialogContent>
          <Alert severity="error">{errorDialogMessage}</Alert>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setErrorDialogMessage(null)}>Close</Button>
        </DialogActions>
      </Dialog>
    </div>
  );
}
