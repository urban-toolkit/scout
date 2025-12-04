// types.ts
export type PhysicalLayerDef = {
  id: string;
  type: "raster" | "vector";
  datafile: string;
  region_of_interest: { type: "bbox" | "geojson"; value: number[] | string };
  layers: { tag: string; features: string[] }[];
};

export type ViewDef = {
  physical_layer?: { ref: string };
  thematic_layer?: { ref: string | string[] };
  type: string;
  file_type?: string;
  zoom_pan?: boolean;
  operation?: string;
  layers?: { tag: string; style: Record<string, any> }[];
  style?: Record<string, any>;
};

export type ParsedLayer = {
  tag: string;
  fill?: { attribute?: string; colormap?: string };
  stroke?: { color?: string; width?: number };
  opacity?: number;
  border?: { color?: string; width?: number };
};

export type ParsedView = {
  physicalLayerRef?: string;
  thematicLayerRef?: string | string[];
  type?: string;
  file_type?: string;
  opacity?: string;
  colormap?: string;
  operation?: string;
  style?: Record<string, any>;
  zoom_level?: number;
  layers?: ParsedLayer[];
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

  physicalLayerRef: string;

  layer: {
    tag: string;
    feature?: string;
  };
};

export type ParsedInteraction = {
  id: string;
  type: string;
  action: string;
  physicalLayerRef: string;
  layer: {
    tag: string;
    feature?: string;
  };
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
