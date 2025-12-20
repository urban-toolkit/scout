export type TemplateKey =
  | "data_layer"
  | "view"
  | "interaction"
  // | "transformation"
  | "widget"
  | "comparison";
// | "choice"
// | "join"

// for manhattan area
// "value": [
//   -73.995,
//   40.749,
//   -73.980,
//   40.757
// ]

export const dataLayerTemplate = {
  data_layer: {
    id: "A",
    source: "osm",
    dtype: "physical",

    roi: {
      datafile: "chicago",
      type: "bbox",
      value: [-87.66, 41.86, -87.64, 41.88],
    },

    osm_features: [
      {
        feature: "buildings",
        attributes: ["height"],
      },
      {
        feature: "roads",
      },
    ],
  },
};

export const viewTemplate = {
  view: [
    {
      ref: "A_buildings",
      type: "vector",
      file_type: "geojson",
      geom_type: "multipolygon",

      style: {
        fill: {
          feature: "height",
          range: [0, 550],
          colormap: "blues",
        },
        "stroke-color": "#333333",
        opacity: 1,
      },
    },
    // {
    //   ref: "A_roads",
    //   type: "vector",
    //   file_type: "geojson",
    //   geom_type: "linestring",
    //   style: {
    //     "stroke-color": "#333333",
    //     opacity: 1,
    //   },
    // },
  ],
};
// {
//   thematic_layer: { ref: "S1" },
//   type: "raster",
//   style: { colormap: "reds", legend: true, opacity: 0.7 },
// },

export const choiceTemplate = { choice: {} };
export const joinTemplate = { join: {} };

export const interactionTemplate = {
  interaction: {
    ref: "A_buildings",
    itype: "hover",
    action: "highlight+show",
    attribute: "height",
  },
};

export const widgetTemplate = {
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

export const comparisonTemplate = {
  comparison: {
    key: ["A_shadow", "B_shadow"],
    metric: "mean",
    chart: "bar",
  },
};

export const TEMPLATES: Record<TemplateKey, any> = {
  data_layer: dataLayerTemplate,
  view: viewTemplate,
  // choice: choiceTemplate,
  // join: joinTemplate,
  // transformation: transformationTemplate,
  interaction: interactionTemplate,
  widget: widgetTemplate,
  comparison: comparisonTemplate,
};

export const TEMPLATE_LABELS: Record<TemplateKey, string> = {
  data_layer: "Data layer",
  view: "View",
  // choice: "choice",
  // join: "join",
  // transformation: "transformation",
  interaction: "Interaction",
  widget: "Widget",
  comparison: "Comparison",
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
