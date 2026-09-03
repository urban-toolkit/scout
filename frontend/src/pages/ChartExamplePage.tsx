// pages/ChartExamplePage.tsx
import { useEffect, useMemo, useState } from "react";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import JsonCodeEditor from "../node-components/JsonCodeEditor";
import { CustomD3Renderer } from "../charts/renderers/customD3Renderer";
import { VegaLiteRenderer } from "../charts/renderers/vegaLiteRenderer";
import {
  assertFieldExistsInData,
  deriveRenderParams,
  filterDataByExampleKeys,
} from "../charts/deriveRenderParams";
import { getGalleryExample } from "../charts/galleryExamples";
import { getChartType, type ChartTypeRecord } from "../charts/chartTypes";
import { EngineBadge } from "../charts/EngineBadge";
import { ReadOnlyTag } from "../charts/ReadOnlyTag";
import "../charts/readOnlyCode.css";
import "./ChartExamplePage.css";

export default function ChartExamplePage({
  name,
  onBack,
  onEditInStudio,
}: {
  name: string;
  onBack: () => void;
  onEditInStudio: (name: string) => void;
}) {
  // charts/galleryExamples.ts is only ever a same-tick fallback now, for the
  // handful of built-ins published before exampleUsage/sampleData started
  // being persisted - any chart published from Chart Studio (see
  // pages/ChartStudioPage.tsx and server.py's publish_chart_type) carries
  // its own, which is what this page actually shows once loaded. A chart
  // with no gallery entry at all (e.g. a brand-new custom chart) still
  // renders fine - it just has nothing to show before the fetch below
  // resolves.
  const example = getGalleryExample(name);
  const [record, setRecord] = useState<ChartTypeRecord | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [exampleUsage, setExampleUsage] = useState<unknown>(example?.exampleUsage ?? {});

  useEffect(() => {
    let cancelled = false;
    setRecord(null);
    setLoadError(null);
    setExampleUsage(getGalleryExample(name)?.exampleUsage ?? {});
    getChartType(name)
      .then((r) => {
        if (cancelled) return;
        setRecord(r);
        if (r.exampleUsage !== undefined) setExampleUsage(r.exampleUsage);
      })
      .catch((e) => {
        if (!cancelled) {
          setLoadError(e instanceof Error ? e.message : `Failed to load chart "${name}"`);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [name]);

  // The published sample data (record.sampleData) if this chart has any,
  // else the same hand-authored fallback exampleUsage above defaults to -
  // the two always come from the same source, never mixed, since a
  // published example's fields only make sense against its own data.
  const effectiveData = (record?.sampleData ?? example?.data) as
    | Record<string, Record<string, number>>
    | undefined;
  const defaultExampleUsage = record?.exampleUsage ?? example?.exampleUsage;
  const title = record?.displayName ?? example?.title ?? name;
  const description = record?.description || example?.description || "";

  const isModified = useMemo(
    () => defaultExampleUsage !== undefined && JSON.stringify(exampleUsage) !== JSON.stringify(defaultExampleUsage),
    [exampleUsage, defaultExampleUsage],
  );

  if (loadError) {
    return (
      <div className="chart-example-page">
        <BackLink onBack={onBack} />
        <div className="chart-example-page__error">{loadError}</div>
      </div>
    );
  }

  let renderError: string | null = null;
  let derivedParams: Record<string, any> | null = null;
  let filteredData: Record<string, Record<string, number>> | null = null;
  try {
    if (!effectiveData) throw new Error("No sample data available for this chart.");
    // Applied for both engines: a "key" that's missing from the sample data,
    // or an "x"/"y" field that isn't a column on those rows, should surface
    // as an error rather than being silently ignored - and removing/adding
    // a key should visibly change which rows the chart sees.
    filteredData = filterDataByExampleKeys(effectiveData, exampleUsage);
    const { axis, axisLabel, props } = deriveRenderParams(exampleUsage);
    assertFieldExistsInData(filteredData, axisLabel);
    if (record && record.engine === "d3") {
      derivedParams = { ...props, axis, axisLabel };
    }
  } catch (e) {
    renderError = e instanceof Error ? e.message : "Invalid example usage";
  }

  return (
    <div className="chart-example-page">
      <BackLink onBack={onBack} />

      <div className="chart-example-page__title-row">
        <div className="chart-example-page__title-group">
          <h1 className="chart-example-page__title">{title}</h1>
          {record && <EngineBadge engine={record.engine} />}
        </div>
        <button
          type="button"
          className="chart-example-page__edit-btn"
          onClick={() => onEditInStudio(name)}
        >
          Edit in Chart Studio
        </button>
      </div>
      {description && <p className="chart-example-page__description">{description}</p>}

      <div className="chart-example-page__source-grid">
        <div>
          <div className="chart-example-page__section-row">
            <div className="chart-example-page__section-label">Example usage</div>
            {isModified && (
              <button
                type="button"
                className="chart-example-page__reset-btn"
                onClick={() => setExampleUsage(defaultExampleUsage)}
              >
                Reset to default
              </button>
            )}
          </div>
          <div className="chart-example-page__code">
            <JsonCodeEditor value={exampleUsage} onChange={setExampleUsage} height={240} />
          </div>
          <p className="chart-example-page__hint">
            Edit the JSON to see the chart update live. "key" names which rows of the
            sample data to use; "x"/"y" name which field in each row to plot.
          </p>
        </div>

        <div>
          <div className="chart-example-page__section-row">
            <div className="chart-example-page__section-label">
              Sample data
              <ReadOnlyTag />
            </div>
          </div>
          <div className="chart-example-page__code readonly-json-box">
            <JsonCodeEditor value={effectiveData ?? {}} readOnly height={240} />
          </div>
          <p className="chart-example-page__hint">
            The data this preview renders against - shaped like a real Comparison node's
            data would be: one row per scenario key, one column per metric field.
          </p>
        </div>
      </div>

      <div className="chart-example-page__preview-wrap">
        <div className="chart-example-page__section-row">
          <div className="chart-example-page__section-label">Preview</div>
        </div>
        <div className="chart-example-page__preview">
          {renderError || !filteredData ? (
            <div className="chart-example-page__error">{renderError ?? "Invalid example usage"}</div>
          ) : !record ? (
            <div className="chart-example-page__loading">Loading chart…</div>
          ) : record.engine === "vega-lite" ? (
            <VegaLiteRenderer spec={JSON.parse(record.code)} data={filteredData} />
          ) : (
            <CustomD3Renderer code={record.code} data={filteredData} params={derivedParams ?? {}} />
          )}
        </div>
      </div>
    </div>
  );
}

function BackLink({ onBack }: { onBack: () => void }) {
  return (
    <button type="button" className="chart-example-page__back" onClick={onBack}>
      <ArrowBackIcon sx={{ fontSize: 18 }} />
      Back to gallery
    </button>
  );
}
