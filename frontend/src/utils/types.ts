// types.ts

// DataLayerDef
export type DataLayerDef = {
  id: string;
  source: string;

  // type: "raster" | "vector";
  dtype: string;

  roi: { datafile: string; type: "bbox" | "geojson"; value: number[] | string };

  osm_features: { feature: string; attributes: string[] }[];
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

  ref: string;

  // "click" or "hover"
  itype: string;

  // - "remove"
  // - "modify_feature"
  // - "highlight"
  // - "highlight+show"
  action: string;

  attribute?: string;
};

export type WidgetDef = {
  // id: string;
  wtype: string;
  variable: string;
  choices: any[];
  default: any;

  props: Record<string, any>;

  // title: string;
  // description: string;
  // [key: string]: any;
};

export type ComparisonDef = {
  key: string[];
  metric: string;
  chart: string;
};

export type WidgetOutput = {
  // id: string;
  variable: string;
  value: any;
};
