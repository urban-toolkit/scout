// types.ts

// DataLayerDef
export type PhysicalLayerDef = {
  id: string;
  type: "raster" | "vector";
  datafile: string;
  region_of_interest: { type: "bbox" | "geojson"; value: number[] | string };
  layers: { tag: string; features: string[] }[];
};

// ViewDef only. No ParsedView should exist.
export type ViewDef = {
  ref?: string;
  ref_base?: string;
  ref_comp?: string;

  type: string;
  file_type?: string;
  geom_type?: string;

  style: Record<string, any>;
};

export type InteractionDef = {
  id: string;

  // "click" or "hover"
  type: string;

  // - "remove"
  // - "modify_feature"
  // - "highlight"
  // - "highlight+show"
  action: string;

  ref: string;

  // layer: {
  //   tag: string;
  feature?: string;
  // };
};

export type WidgetDef = {
  id: string;
  variable: string;
  type: string;
  title: string;
  description: string;
  "default-value": any;

  [key: string]: any;
};

export type ComparisonDef = {
  key: string[];
  metric: string;
  encoding: string;
};

export type WidgetOutput = {
  id: string;
  variable: string;
  value: any;
};
