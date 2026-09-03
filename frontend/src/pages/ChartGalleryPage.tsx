// pages/ChartGalleryPage.tsx
import { useEffect, useState } from "react";
import { CustomD3Renderer } from "../charts/renderers/customD3Renderer";
import { VegaLiteRenderer } from "../charts/renderers/vegaLiteRenderer";
import {
  assertFieldExistsInData,
  deriveRenderParams,
  filterDataByExampleKeys,
} from "../charts/deriveRenderParams";
import { getGalleryExample } from "../charts/galleryExamples";
import {
  getChartType,
  listChartTypes,
  type ChartEngine,
  type ChartTypeRecord,
} from "../charts/chartTypes";
import "./ChartGalleryPage.css";

const SECTION_TITLE: Record<ChartEngine, string> = {
  d3: "D3",
  "vega-lite": "Vega-Lite",
};

export default function ChartGalleryPage({
  onSelectChart,
  onCreateNewChart,
}: {
  onSelectChart: (name: string) => void;
  onCreateNewChart: () => void;
}) {
  // Every published chart type, not a fixed set - listChartTypes() reads
  // the live backend registry (see server.py), so a chart published from
  // Chart Studio shows up here automatically, using its own persisted
  // exampleUsage/sampleData (charts/galleryExamples.ts is only a fallback
  // for the handful of built-ins published before those were persisted).
  const [charts, setCharts] = useState<ChartTypeRecord[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const summaries = await listChartTypes();
        const records = await Promise.all(summaries.map((s) => getChartType(s.name)));
        if (!cancelled) setCharts(records);
      } catch (e) {
        if (!cancelled) {
          setLoadError(e instanceof Error ? e.message : "Failed to load chart gallery");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const d3Charts = (charts ?? []).filter((c) => c.engine === "d3");
  const vegaLiteCharts = (charts ?? []).filter((c) => c.engine === "vega-lite");

  return (
    <div className="chart-gallery-page">
      <div className="chart-gallery-page__header">
        <div>
          <h1>Chart Gallery</h1>
          <p>Every published chart type, with example usage and a live preview.</p>
        </div>
        <button
          type="button"
          className="chart-gallery-page__new-btn"
          onClick={onCreateNewChart}
        >
          + Create new chart
        </button>
      </div>

      {loadError && <div className="chart-gallery-page__error">{loadError}</div>}

      {charts === null && !loadError ? (
        <div className="chart-gallery-page__loading">Loading chart gallery…</div>
      ) : (
        <>
          <ChartGallerySection engine="d3" charts={d3Charts} onSelectChart={onSelectChart} />
          <ChartGallerySection engine="vega-lite" charts={vegaLiteCharts} onSelectChart={onSelectChart} />
        </>
      )}
    </div>
  );
}

function ChartGallerySection({
  engine,
  charts,
  onSelectChart,
}: {
  engine: ChartEngine;
  charts: ChartTypeRecord[];
  onSelectChart: (name: string) => void;
}) {
  if (charts.length === 0) return null;

  return (
    <div className="chart-gallery-section">
      <h2 className="chart-gallery-section__title">{SECTION_TITLE[engine]}</h2>
      <div className="chart-gallery-page__grid">
        {charts.map((record) => (
          <button
            key={record.name}
            type="button"
            className="chart-gallery-card"
            onClick={() => onSelectChart(record.name)}
          >
            <div className="chart-gallery-card__thumb">
              <ChartThumbnail record={record} />
            </div>
            <div className="chart-gallery-card__label">{record.displayName}</div>
          </button>
        ))}
      </div>
    </div>
  );
}

function ChartThumbnail({ record }: { record: ChartTypeRecord }) {
  // A chart published before exampleUsage/sampleData were persisted (or one
  // republished without touching them) falls back to the hand-authored
  // gallery entry, if there is one - either way, nothing to render against
  // just renders no thumbnail rather than crashing the whole grid.
  const fallback = getGalleryExample(record.name);
  const exampleUsage = record.exampleUsage ?? fallback?.exampleUsage;
  const sampleData = (record.sampleData ?? fallback?.data) as
    | Record<string, Record<string, number>>
    | undefined;
  if (exampleUsage === undefined || !sampleData) return null;

  if (record.engine === "vega-lite") {
    try {
      const spec = JSON.parse(record.code);
      const data = filterDataByExampleKeys(sampleData, exampleUsage);
      const { axisLabel } = deriveRenderParams(exampleUsage);
      assertFieldExistsInData(data, axisLabel);
      return (
        <div className="chart-gallery-card__thumb-scale">
          <VegaLiteRenderer spec={spec} data={data} />
        </div>
      );
    } catch {
      return null;
    }
  }

  try {
    const data = filterDataByExampleKeys(sampleData, exampleUsage);
    const { axis, axisLabel, props } = deriveRenderParams(exampleUsage);
    assertFieldExistsInData(data, axisLabel);
    return (
      <div className="chart-gallery-card__thumb-scale">
        <CustomD3Renderer code={record.code} data={data} params={{ ...props, axis, axisLabel }} />
      </div>
    );
  } catch {
    return null;
  }
}
