export type TemplateKey =
  | "physical_layer"
  | "view"
  | "interaction"
  // | "transformation"
  | "widget_def"
  | "comparison_def";
// | "choice"
// | "join"

// for manhattan area
// "value": [
//   -73.995,
//   40.749,
//   -73.980,
//   40.757
// ]

export const physicalLayerTemplate = {
  physical_layer: {
    id: "baselayer-0",
    type: "vector",
    file_type: "feather",
    datafile: "chicago",
    region_of_interest: {
      type: "bbox",
      value: [-87.66, 41.86, -87.6, 41.9],
    },
    layers: [
      {
        tag: "buildings",
        "geom-type": "multipolygon",
        features: ["height"],
      },
      {
        tag: "roads",
        "geom-type": "linestring",
      },
    ],
  },
};

export const viewTemplate = {
  view: [
    {
      physical_layer: { ref: "baselayer-0" },
      type: "vector",
      file_type: "geojson",
      zoom_pan: true,
      layers: [
        {
          tag: "buildings",
          "geom-type": "multipolygon",
          style: {
            fill: { feature: "height", colormap: "greys" },
            "stroke-color": "#333333",
            opacity: 1,
            "z-index": 1,
          },
        },
        {
          tag: "roads",
          "geom-type": "linestring",
          style: { "z-index": 2, "stroke-color": "#444444" },
        },
      ],
    },
    // {
    //   thematic_layer: { ref: "S1" },
    //   type: "raster",
    //   style: { colormap: "reds", legend: true, opacity: 0.7 },
    // },
  ],
};

export const choiceTemplate = { choice: {} };
export const joinTemplate = { join: {} };

export const transformationTemplate = {
  transformation: {
    id: "rasters-baselayer-0",
    physical_layer: { ref: "baselayer-0" },
    operation: "rasterize",
    zoom: 16,
    layer: {
      tag: "buildings",
      feature: "height",
    },
  },
};

export const interactionTemplate = {
  interaction: {
    id: "interaction-0",
    physicalLayerRef: "baselayer-0",
    type: "click",
    action: "remove",
    layer: {
      tag: "buildings",
    },
  },
};

export const widgetDefTemplate = {
  widget: {
    id: "widget-0",
    variable: "season",
    title: "Season",
    type: "radio-group",
    description: "(select season for shadow analysis)",
    items: ["spring", "summer", "winter"],
    orientation: "horizontal",
    "default-value": "summer",
  },
};

export const comparisonDefTemplate = {
  comparison: {
    key: ["A_shadow", "B_shadow"],
    metric: "mean",
    encoding: "bar",
  },
};

export const TEMPLATES: Record<TemplateKey, any> = {
  physical_layer: physicalLayerTemplate,
  view: viewTemplate,
  // choice: choiceTemplate,
  // join: joinTemplate,
  // transformation: transformationTemplate,
  interaction: interactionTemplate,
  widget_def: widgetDefTemplate,
  comparison_def: comparisonDefTemplate,
};

export const TEMPLATE_LABELS: Record<TemplateKey, string> = {
  physical_layer: "data layer",
  view: "view",
  // choice: "choice",
  // join: "join",
  // transformation: "transformation",
  interaction: "interaction",
  widget_def: "widget",
  comparison_def: "comparison",
};

// -------------------------------------------
// Download data:
// -------------------------------------------

// from download_data import download_osm_data
// from download_data import extract_buildings
// from download_data import extract_roads

// input_filename = "north-america-latest"
// location = "Los Angeles, USA"
// output_filename = "la"

// download_osm_data(
//   input_filename, location, output_filename
// )

// extract_buildings("la")
// extract_roads("la")

// -------------------------------------------
// Conversion to raster:
// -------------------------------------------

// from convert_to_raster import convert_raster

// ref = "A"
// input = "%s"%(ref)
// output = "%s_rasters"%(ref)

// tag = "buildings"
// feature = "height"
// zoom = 16

// convert_raster(input, tag, feature, zoom, output)

// -------------------------------------------
// Run shadow model:
// -------------------------------------------

// from deep_umbra import run_shadow_model

// ref = "A"
// input = '%s_rasters'%(ref)
// output = '%s_shadow'%(ref)

// # season = 'winter'
// # colormap = 'Blues'

// run_shadow_model(input, season, colormap, output)
