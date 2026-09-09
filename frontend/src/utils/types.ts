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
//
// `ref` names a dataset the same way /api/data-catalog's "id" field does:
// a "folder/file" path into the shared data catalog (data/catalog/), or a
// "computed/name" path into this dataflow's own computed outputs
// (data/dataflows/{dataflow_id}_computed/) - see backend's
// _resolve_data_source. `ext` (the file extension, no leading dot - e.g.
// "geojson", "tif", "png") must be given alongside it since the backend no
// longer infers file type by probing data/served/; same for
// ref_base/ext_base and ref_comp/ext_comp on a comparison layer.
export type ViewDef = {
  ref?: string;
  ext?: string;
  ref_base?: string;
  ext_base?: string;
  ref_comp?: string;
  ext_comp?: string;

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
  x?: string | string[];
  y?: string | string[];
  chart: string;
  props?: Record<string, any>;
};

export type WidgetOutput = {
  // id: string;
  variable: string;
  value: any;
};
