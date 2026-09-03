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
  // Record<string, number> rows for every built-in chart - widened to `any`
  // only because a handful of charts (e.g. TraSculptorMatrixD3) read a
  // richer per-key shape (nested per-road objects, a "parent" pointer, ...)
  // that ignores the flat x/y-field grammar entirely, the same way
  // groupedBar ignores "y" and just plots every column.
  data: Record<string, any>;
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

// Sample data for TraSculptorMatrixD3, from the paper's Sioux Falls case
// study (Round II: building new roads/tunnels to relieve the two most
// congested roads). Baseline has "parent": null; P1/P2/P3 all branch
// directly off Baseline (three alternatives, not a single chain), matching
// the paper's Fig. 8 - "The expert performed three modifications based on
// the initial road network."
const TRASCULPTOR_DATA = {
  Baseline: {
    parent: null,
    "optimization improvement (%)": 0,
    "cumulative cost ($M)": 0,
    modification: "none",
    "modification type": "none",
    "Road 16": { capacity: 4800, "traffic flow": 6300, FFTT: 4.0, "travel time": 8.6 },
    "Road 19": { capacity: 4800, "traffic flow": 6200, FFTT: 3.8, "travel time": 8.2 },
    "Road 17": { capacity: 4000, "traffic flow": 3200, FFTT: 3.0, "travel time": 3.8 },
    "Road 20": { capacity: 3900, "traffic flow": 3150, FFTT: 3.2, "travel time": 4.0 },
    "Road 11": { capacity: 3600, "traffic flow": 2800, FFTT: 2.6, "travel time": 3.3 },
    "Road 9": { capacity: 3500, "traffic flow": 2750, FFTT: 2.8, "travel time": 3.5 },
    "Road 47": { capacity: 4200, "traffic flow": 3300, FFTT: 3.4, "travel time": 4.1 },
    "Road 22": { capacity: 4100, "traffic flow": 3250, FFTT: 3.3, "travel time": 4.0 },
  },
  P1: {
    parent: "Baseline",
    "optimization improvement (%)": 11.6,
    "cumulative cost ($M)": 6.55,
    modification: "Build a new road",
    "modification type": "two-way road",
    "Road 16": { capacity: 4800, "traffic flow": 4300, FFTT: 4.0, "travel time": 5.0 },
    "Road 19": { capacity: 4800, "traffic flow": 4200, FFTT: 3.8, "travel time": 4.8 },
    "Road 17": { capacity: 4000, "traffic flow": 4500, FFTT: 3.0, "travel time": 5.2 },
    "Road 20": { capacity: 3900, "traffic flow": 4400, FFTT: 3.2, "travel time": 5.5 },
    "Road 11": { capacity: 3600, "traffic flow": 3500, FFTT: 2.6, "travel time": 4.2 },
    "Road 9": { capacity: 3500, "traffic flow": 3400, FFTT: 2.8, "travel time": 4.4 },
    "Road 47": { capacity: 4200, "traffic flow": 4200, FFTT: 3.4, "travel time": 5.0 },
    "Road 22": { capacity: 4100, "traffic flow": 4050, FFTT: 3.3, "travel time": 4.9 },
  },
  P2: {
    parent: "Baseline",
    "optimization improvement (%)": 16.9,
    "cumulative cost ($M)": 15.07,
    modification: "Build a new road",
    "modification type": "two-way tunnel",
    "Road 16": { capacity: 4800, "traffic flow": 4700, FFTT: 4.0, "travel time": 5.6 },
    "Road 19": { capacity: 4800, "traffic flow": 4550, FFTT: 3.8, "travel time": 5.3 },
    "Road 17": { capacity: 4000, "traffic flow": 3600, FFTT: 3.0, "travel time": 4.1 },
    "Road 20": { capacity: 3900, "traffic flow": 3550, FFTT: 3.2, "travel time": 4.3 },
    "Road 11": { capacity: 3600, "traffic flow": 3900, FFTT: 2.6, "travel time": 4.6 },
    "Road 9": { capacity: 3500, "traffic flow": 3750, FFTT: 2.8, "travel time": 4.8 },
    "Road 47": { capacity: 4200, "traffic flow": 3200, FFTT: 3.4, "travel time": 3.9 },
    "Road 22": { capacity: 4100, "traffic flow": 3150, FFTT: 3.3, "travel time": 3.8 },
  },
  P3: {
    parent: "Baseline",
    "optimization improvement (%)": 16.9,
    "cumulative cost ($M)": 23.77,
    modification: "Build a new road",
    "modification type": "two-way tunnel",
    "Road 16": { capacity: 4800, "traffic flow": 5000, FFTT: 4.0, "travel time": 6.1 },
    "Road 19": { capacity: 4800, "traffic flow": 4900, FFTT: 3.8, "travel time": 5.8 },
    "Road 17": { capacity: 4000, "traffic flow": 3000, FFTT: 3.0, "travel time": 3.5 },
    "Road 20": { capacity: 3900, "traffic flow": 2950, FFTT: 3.2, "travel time": 3.6 },
    "Road 11": { capacity: 3600, "traffic flow": 2600, FFTT: 2.6, "travel time": 3.0 },
    "Road 9": { capacity: 3500, "traffic flow": 2550, FFTT: 2.8, "travel time": 3.1 },
    "Road 47": { capacity: 4200, "traffic flow": 2900, FFTT: 3.4, "travel time": 3.7 },
    "Road 22": { capacity: 4100, "traffic flow": 2850, FFTT: 3.3, "travel time": 3.6 },
  },
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
  {
    name: "TraSculptorMatrixD3",
    title: "TraSculptor Matrix",
    description:
      "A history tree of network states aligned with a per-road comparison matrix, adapted " +
      "from TraSculptor (Deng et al., TVCG 2025). Each column is a state - \"parent\" points " +
      "to the state it was modified from, so branches (multiple alternatives off the same " +
      'parent) render side by side. Each row is a road: cell width = traffic flow relative ' +
      "to capacity, cell height = closeness of travel time to free-flow time (taller is " +
      "faster), cell background = improved/worsened vs the baseline state (the row with " +
      '"parent": null). Doesn\'t read "x"/"y" - "y" only satisfies the example schema; every ' +
      'road and state field in the data is used directly. "key" must include every ancestor ' +
      "of a state you want shown, since a state's tree parent is looked up only within the " +
      'keys selected. Supports props: "roadOrder" (string[], which rows to show and in what ' +
      'order), "baselineKey", "rowHeight", "colWidth", "labelColWidth", "treeHeight", ' +
      '"capacityDomainMax" (default 1.5, how far past 100% capacity a bar can extend), ' +
      '"color"/"colors" ({improve, regress}), "currencySymbol", "costSuffix", "showCost", ' +
      '"showModificationType".',
    data: TRASCULPTOR_DATA,
    exampleUsage: {
      key: Object.keys(TRASCULPTOR_DATA),
      y: "cumulative cost ($M)",
      chart: "TraSculptorMatrixD3",
      props: {},
    },
  },
];

export function getGalleryExample(name: string): GalleryExample | undefined {
  return GALLERY_EXAMPLES.find((e) => e.name === name);
}
