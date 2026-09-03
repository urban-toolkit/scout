// charts/galleryExamples.ts
// Hand-authored sample data + "example usage" (the same comparison-node
// JSON shape Chart Studio's "Example usage" box uses - see
// pages/ChartStudioPage.tsx) for each built-in chart type. The backend
// chart manifest (backend/data/chart_manifest.json, served via
// /api/chart-types) only stores a chart's render code, not a worked
// example, so Chart Gallery supplies its own for thumbnails and example
// pages.
export type GalleryExample = {
  name: string; // must match a name in chart_manifest.json / GET /api/chart-types
  title: string;
  description: string;
  data: Record<string, Record<string, number>>;
  exampleUsage: {
    key: string[];
    x?: string;
    y?: string;
    chart: string;
    props?: Record<string, any>;
  };
};

// Exported so Chart Studio's blank-slate "New chart" starting point (and its
// "Load existing chart" fallback for a custom chart not in GALLERY_EXAMPLES)
// uses the exact same scenario names/columns as the gallery, instead of its
// own separately-invented placeholder data.
export const SCENARIO_DATA = {
  Baseline: { "median flood depth": 5.02, "peak flood depth": 7.16 },
  "Levee raised": { "median flood depth": 3.14, "peak flood depth": 4.71 },
  "Culvert widened": { "median flood depth": 2.4, "peak flood depth": 3.62 },
};

// Colors keyed by scenario so it's obvious "color" maps 1:1 with "key" -
// bar/pie also accept a single string (applied to every bar/slice) or a
// positional array, but the per-key object form is the clearest to read.
const SCENARIO_COLORS = {
  Baseline: "#4C72B0",
  "Levee raised": "#DD8452",
  "Culvert widened": "#C44E52",
};

export const GALLERY_EXAMPLES: GalleryExample[] = [
  {
    name: "bar",
    title: "Bar chart",
    description:
      "Compare a single metric across scenarios as vertical or horizontal bars. " +
      'Supports props: "unit", "color"/"colors".',
    data: SCENARIO_DATA,
    exampleUsage: {
      key: Object.keys(SCENARIO_DATA),
      y: "median flood depth",
      chart: "bar",
      props: { unit: "m", color: SCENARIO_COLORS },
    },
  },
  {
    name: "pie",
    title: "Pie chart",
    description:
      "Show a single metric's share across scenarios as a pie chart. " +
      'Supports props: "unit", "color"/"colors".',
    data: SCENARIO_DATA,
    exampleUsage: {
      key: Object.keys(SCENARIO_DATA),
      y: "median flood depth",
      chart: "pie",
      props: { unit: "m", color: SCENARIO_COLORS },
    },
  },
  {
    name: "table",
    title: "Table",
    description:
      'List a single metric per scenario in a plain two-column table. Supports props: "unit".',
    data: SCENARIO_DATA,
    exampleUsage: {
      key: Object.keys(SCENARIO_DATA),
      y: "median flood depth",
      chart: "table",
      props: { unit: "m" },
    },
  },
  {
    name: "lollipop",
    title: "Lollipop chart",
    description:
      "Compare a single metric across scenarios as a horizontal lollipop chart. " +
      'Supports props: "unit", "color"/"colors".',
    data: SCENARIO_DATA,
    exampleUsage: {
      key: Object.keys(SCENARIO_DATA),
      y: "median flood depth",
      chart: "lollipop",
      props: { unit: "m", color: SCENARIO_COLORS },
    },
  },
  {
    name: "groupedBar",
    title: "Grouped bar chart",
    description:
      "Compare every metric column across scenarios at once, grouped by scenario. " +
      'Doesn\'t read any props, and "y" here only satisfies the example schema - every ' +
      "column in the data is plotted regardless of its value.",
    data: SCENARIO_DATA,
    exampleUsage: {
      key: Object.keys(SCENARIO_DATA),
      y: "median flood depth",
      chart: "groupedBar",
      props: {},
    },
  },
  {
    name: "GroupedBarVega",
    title: "Grouped bar chart",
    description:
      "The same grouped-bar comparison, written as a declarative chart spec instead of by hand. " +
      'Doesn\'t read any props, and "y" here only satisfies the example schema - every ' +
      "column in the data is plotted regardless of its value.",
    data: SCENARIO_DATA,
    exampleUsage: {
      key: Object.keys(SCENARIO_DATA),
      y: "median flood depth",
      chart: "GroupedBarVega",
      props: {},
    },
  },
];

export function getGalleryExample(name: string): GalleryExample | undefined {
  return GALLERY_EXAMPLES.find((e) => e.name === name);
}
