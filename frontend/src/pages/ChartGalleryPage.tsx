// pages/ChartGalleryPage.tsx
import { useEffect, useState } from "react";
import { CustomD3Renderer } from "../charts/renderers/customD3Renderer";
import { VegaLiteRenderer } from "../charts/renderers/vegaLiteRenderer";
import {
  assertFieldExistsInData,
  deriveRenderParams,
  filterDataByExampleKeys,
} from "../charts/deriveRenderParams";
import { GALLERY_EXAMPLES, type GalleryExample } from "../charts/galleryExamples";
import { getChartType, type ChartEngine, type ChartTypeRecord } from "../charts/chartTypes";
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
  const [records, setRecords] = useState<Record<string, ChartTypeRecord>>({});
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const entries = await Promise.all(
          GALLERY_EXAMPLES.map(
            async (ex) => [ex.name, await getChartType(ex.name)] as const,
          ),
        );
        if (!cancelled) {
          setRecords(Object.fromEntries(entries));
        }
      } catch (e) {
        if (!cancelled) {
          setLoadError(
            e instanceof Error ? e.message : "Failed to load chart gallery",
          );
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Grouped by engine into their own subsections rather than calling out
  // each card's engine individually (see ChartExamplePage, which still
  // shows a per-chart badge next to the title on a chart's own page).
  // Defaults an as-yet-unloaded chart to "d3" (5 of the 6 built-ins are)
  // so the grid doesn't visibly reshuffle once records finish loading.
  const d3Examples = GALLERY_EXAMPLES.filter(
    (ex) => (records[ex.name]?.engine ?? "d3") === "d3",
  );
  const vegaLiteExamples = GALLERY_EXAMPLES.filter(
    (ex) => records[ex.name]?.engine === "vega-lite",
  );

  return (
    <div className="chart-gallery-page">
      <div className="chart-gallery-page__header">
        <div>
          <h1>Chart Gallery</h1>
          <p>Every built-in chart type, with example usage and a live preview.</p>
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

      <ChartGallerySection engine="d3" examples={d3Examples} records={records} onSelectChart={onSelectChart} />
      <ChartGallerySection
        engine="vega-lite"
        examples={vegaLiteExamples}
        records={records}
        onSelectChart={onSelectChart}
      />
    </div>
  );
}

function ChartGallerySection({
  engine,
  examples,
  records,
  onSelectChart,
}: {
  engine: ChartEngine;
  examples: GalleryExample[];
  records: Record<string, ChartTypeRecord>;
  onSelectChart: (name: string) => void;
}) {
  if (examples.length === 0) return null;

  return (
    <div className="chart-gallery-section">
      <h2 className="chart-gallery-section__title">{SECTION_TITLE[engine]}</h2>
      <div className="chart-gallery-page__grid">
        {examples.map((ex) => {
          const record = records[ex.name];
          return (
            <button
              key={ex.name}
              type="button"
              className="chart-gallery-card"
              onClick={() => onSelectChart(ex.name)}
            >
              <div className="chart-gallery-card__thumb">
                {record && <ChartThumbnail record={record} example={ex} />}
              </div>
              <div className="chart-gallery-card__label">{ex.title}</div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function ChartThumbnail({
  record,
  example,
}: {
  record: ChartTypeRecord;
  example: GalleryExample;
}) {
  if (record.engine === "vega-lite") {
    let spec: unknown;
    try {
      spec = JSON.parse(record.code);
      const data = filterDataByExampleKeys(example.data, example.exampleUsage);
      const { axisLabel } = deriveRenderParams(example.exampleUsage);
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
    const data = filterDataByExampleKeys(example.data, example.exampleUsage);
    const { axis, axisLabel, props } = deriveRenderParams(example.exampleUsage);
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
