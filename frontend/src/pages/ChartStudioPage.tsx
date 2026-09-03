// pages/ChartStudioPage.tsx
import { useCallback, useEffect, useState } from "react";
import Dialog from "@mui/material/Dialog";
import DialogTitle from "@mui/material/DialogTitle";
import DialogContent from "@mui/material/DialogContent";
import DialogActions from "@mui/material/DialogActions";
import Button from "@mui/material/Button";
import TextField from "@mui/material/TextField";
import MenuItem from "@mui/material/MenuItem";
import ListSubheader from "@mui/material/ListSubheader";
import Divider from "@mui/material/Divider";
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
import { deriveRenderParams } from "../charts/deriveRenderParams";
import { SCENARIO_DATA, getGalleryExample } from "../charts/galleryExamples";
import "./ChartStudioPage.css";

// No worked example chart is pre-filled here on purpose - a fresh chart
// starts from a blank slate; only the function-signature comment stays, so
// the editor isn't a totally blank box with no hint of what's available.
const D3_STARTER_CODE = `// available: container (a <div>), data, params, d3, containerWidth
`;

const VEGA_STARTER_SPEC = {};

// Same scenario data Chart Gallery ships with (see charts/galleryExamples.ts)
// rather than Studio's own separately-invented placeholder rows - so a fresh
// chart previews against the same shape of data a published one documents.
const STARTER_DATA = SCENARIO_DATA;

// "Example usage" holds the same shape a Comparison node's grammar JSON
// uses (schemas/comparison.json: key, x/y, chart, props) - not a flat
// params blob - so it doubles as real documentation for whoever wires this
// chart into a Comparison node, copy-pasteable as-is. Preview derives the
// actual render params (axis/axisLabel/props) from it exactly the way the
// backend's /api/comparison-view does (see asFieldList/deriveRenderParams
// below), so what you see in Preview matches real usage.
const STARTER_PARAMS = {
  key: Object.keys(STARTER_DATA),
  y: "median flood depth",
  props: {},
};

export default function ChartStudioPage({
  initialChartName,
}: {
  // Set when arriving via "Edit in Chart Studio" from a chart's own Gallery
  // page (see ChartExamplePage) - auto-selects that chart on mount, same as
  // if it had just been picked from "Load existing chart" below.
  initialChartName?: string;
}) {
  // The one human-readable name shown here, in "Load existing chart", and in
  // Chart Gallery - e.g. "Bar chart". The real technical identifier (manifest
  // key, generated filename) is never edited directly; it's either the
  // already-loaded chart's existing id (tracked by selectedExisting below) or,
  // for a brand-new chart, minted server-side from this name + engine on
  // publish (see server.py's publish_chart_type/_engine_tag).
  const [displayName, setDisplayName] = useState("");
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
    setDisplayName("");
    setDescription("");
    setEngine("d3");
    setD3Code(D3_STARTER_CODE);
    setVegaSpec(VEGA_STARTER_SPEC);
    setSampleData(STARTER_DATA);
    setSampleParams(STARTER_PARAMS);
    setPreviewProps(null);
    setPreviewError(null);
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
      setDisplayName(record.displayName ?? record.name);
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
      // Prefer whatever this chart was actually last published with (see
      // server.py's publish_chart_type) - that's the author's own edits,
      // and republishing must round-trip them rather than silently
      // reverting to a guess. Only a chart published before this was
      // persisted (or one that's never been published, e.g. a fresh
      // built-in) falls back further: a Chart Gallery built-in gets its
      // hand-authored example (charts/galleryExamples.ts), and anything
      // else falls back to guessing a working example from the current
      // Sample data's keys/first metric column, since the built-in
      // single-metric charts (bar/pie/table/lollipop) read params.axisLabel
      // to pick which sample-data column to plot, and leaving that blank
      // means they silently render nothing until filled in by hand.
      const galleryExample = getGalleryExample(record.name);
      if (record.sampleData !== undefined && record.exampleUsage !== undefined) {
        setSampleData(record.sampleData);
        setSampleParams(record.exampleUsage);
      } else if (galleryExample) {
        setSampleData(galleryExample.data);
        setSampleParams(galleryExample.exampleUsage);
      } else {
        const data = sampleData as Record<string, Record<string, number>>;
        const keys = Object.keys(data);
        const firstMetricKey = keys.length > 0 ? Object.keys(data[keys[0]])[0] : undefined;
        setSampleParams(
          firstMetricKey
            ? { key: keys, y: firstMetricKey, chart: record.name, props: {} }
            : { key: keys, chart: record.name, props: {} },
        );
      }
      // Clear any stale preview from whatever was previously loaded.
      setPreviewProps(null);
      setPreviewError(null);
    } catch (e) {
      setChartListError(e instanceof Error ? e.message : `Failed to load chart '${value}'`);
    } finally {
      setLoadingExisting(false);
    }
  };

  // Auto-select the chart passed in via "Edit in Chart Studio" (see
  // ChartExamplePage), same as picking it from "Load existing chart" below.
  // Mount-only: this page unmounts/remounts fresh on every navigation (see
  // App.tsx), so initialChartName never changes out from under an existing
  // selection.
  useEffect(() => {
    if (initialChartName) {
      void handleSelectExisting(initialChartName);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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

  // trimmedDisplayName is always sent; `name` is only included when
  // overwriting a chart already loaded (selectedExisting) - a brand-new
  // chart has no existing identifier yet, so the backend mints one from
  // displayName + engine instead (see server.py's publish_chart_type).
  const doPublish = async (trimmedDisplayName: string) => {
    setPublishing(true);
    try {
      const code = engine === "d3" ? d3Code : JSON.stringify(vegaSpec, null, 2);
      const published = await publishChartType({
        name: selectedExisting || undefined,
        displayName: trimmedDisplayName,
        description: description.trim(),
        code,
        engine,
        exampleUsage: sampleParams,
        sampleData,
      });
      setPublishSuccessName(published.displayName);
      setSelectedExisting(published.name);
      setDisplayName(published.displayName);
      await fetchChartList();
    } catch (e) {
      setErrorDialogMessage(e instanceof Error ? e.message : "Failed to publish chart type");
    } finally {
      setPublishing(false);
    }
  };

  const handlePublishClick = () => {
    const trimmed = displayName.trim();
    if (!trimmed) {
      setErrorDialogMessage("Chart name must be non-empty.");
      return;
    }

    setPublishSuccessName(null);

    // selectedExisting means this chart was loaded from "Load existing
    // chart"/"Edit in Chart Studio" - we already know it exists, so confirm
    // before overwriting it. A brand-new chart has no existing identifier
    // to collide with (the backend mints a fresh one), so it publishes
    // straight away.
    if (selectedExisting) {
      setOverwriteDialogOpen(true);
      return;
    }

    void doPublish(trimmed);
  };

  const confirmOverwrite = async () => {
    setOverwriteDialogOpen(false);
    await doPublish(displayName.trim());
  };

  const d3Charts = chartList.filter((c) => c.engine === "d3");
  const vegaCharts = chartList.filter((c) => c.engine === "vega-lite");

  return (
    <div className="chart-studio-page">
      <div className="chart-studio-page__header">
        <input
          type="text"
          placeholder="Write chart name…"
          value={displayName}
          onChange={(e) => setDisplayName(e.target.value)}
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
          // The currently-loaded chart doesn't need a persistent highlight
          // in its own list - keep hover feedback, drop MUI's default
          // "selected" background so the list reads as plain options.
          SelectProps={{
            MenuProps: {
              sx: {
                // MUI also auto-focuses the currently-selected item when the
                // menu opens, adding Mui-focusVisible on top of Mui-selected -
                // that combination has its own (equally opaque) background,
                // so it needs overriding right alongside plain Mui-selected.
                "& .MuiMenuItem-root.Mui-selected": {
                  backgroundColor: "transparent !important",
                },
                "& .MuiMenuItem-root.Mui-selected:hover": {
                  backgroundColor: "action.hover",
                },
              },
            },
          }}
        >
          {/* Grouped by engine (D3 first, then Vega-Lite) with a section
              header as the separator - two charts can share a display name
              (e.g. "Grouped bar chart" as both a D3 and a Vega-Lite chart;
              see charts/galleryExamples.ts) and this tells them apart by
              position instead of a per-row tag. */}
          {d3Charts.length > 0 && <ListSubheader>D3</ListSubheader>}
          {d3Charts.map((c) => (
            <MenuItem key={c.name} value={c.name}>
              {c.displayName}
            </MenuItem>
          ))}
          {d3Charts.length > 0 && vegaCharts.length > 0 && <Divider />}
          {vegaCharts.length > 0 && <ListSubheader>Vega-Lite</ListSubheader>}
          {vegaCharts.map((c) => (
            <MenuItem key={c.name} value={c.name}>
              {c.displayName}
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

          {/* A single either/or picker rather than two toggle buttons - only
              one engine is ever active, and a dropdown makes that explicit
              instead of looking like two independently-toggleable options. */}
          <TextField
            select
            label="Engine"
            value={engine}
            onChange={(e) => setEngine(e.target.value as ChartEngine)}
            size="small"
            sx={{ width: 160 }}
          >
            <MenuItem value="d3">D3 / JS</MenuItem>
            <MenuItem value="vega-lite">Vega-Lite</MenuItem>
          </TextField>

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
                <CustomD3Renderer code={d3Code} data={previewProps.data} params={previewProps.params} />
              ) : (
                <VegaLiteRenderer spec={vegaSpec} data={previewProps.data} />
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
              disabled={publishing}
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
            "{displayName.trim()}" already exists. Publishing will overwrite it — any
            comparison node currently rendering it will pick up the new definition.
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
