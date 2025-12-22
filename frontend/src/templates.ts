export type TemplateKey =
  | "data_layer"
  | "view"
  | "interaction"
  // | "transformation"
  | "widget"
  | "comparison";
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
    wtype: "radio-group",
    variable: "season",
    choices: ["spring", "summer", "winter"],
    default: "summer",
    props: {
      title: "Season",
      description: "(select season for shadow analysis)",
      orientation: "horizontal",
    },
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
  // join: joinTemplate,
  // transformation: transformationTemplate,
  interaction: interactionTemplate,
  widget: widgetTemplate,
  comparison: comparisonTemplate,
};

export const TEMPLATE_LABELS: Record<TemplateKey, string> = {
  data_layer: "Data layer",
  view: "View",
  // join: "join",
  // transformation: "transformation",
  interaction: "Interaction",
  widget: "Widget",
  comparison: "Comparison",
};

// -------------------------------------------
// Conversion to raster:
// -------------------------------------------

// from transformations.raster_conversion.scripts.convert_to_raster import convert_raster

// input = "A_buildings"
// output = "A_rasters"

// attribute = "height"
// zoom = 16

// convert_raster(input, attribute, zoom, output

// -------------------------------------------
// Run shadow model:
// -------------------------------------------

// from models.shadow.scripts.deep_umbra import run_shadow_model

// input = 'A_rasters'
// output = 'A_shadow'

// run_shadow_model(input, season, output)
