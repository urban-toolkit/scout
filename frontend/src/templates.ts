export type TemplateKey =
  | "data_layer"
  | "view"
  | "interaction"
  | "widget"
  | "comparison"
  | "join";

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
      ref: "computed/A_buildings",
      ext: "geojson",
      style: {
        fill: {
          feature: "height",
          range: [0, 550],
          colormap: "blues",
        },
        stroke: {
          color: "#333333",
        },
        opacity: 1,
      },
    },
    {
      ref: "computed/A_roads",
      ext: "geojson",
      style: {
        stroke: {
          color: "#333333",
        },
        opacity: 1,
      },
    },
  ],
};
// {
//   "view": [
//     {
//       "ref_base": "computed/B",
//       "ext_base": "tif",
//       "ref_comp": "computed/A",
//       "ext_comp": "tif",

//       "style": {
//         "opacity": 1,
//         "colormap": "reds"
//       }
//     }
//   ]
// }

export const joinTemplate = {
  join: {
    id: "A_buildings_mean_flood_depth",
    ref_left: "A_buildings",
    ref_right: "flood_depth",
    op: "contains",
    aggr: "mean",
  },
};

// `ref` here must match a view layer's own `ref` exactly (see
// renderViewLayers.ts's buildInteractionSpecsForLayer) - it's how an
// interaction gets attached to the right layer, not a separate file lookup.
export const interactionTemplate = {
  interaction: {
    ref: "computed/A_buildings",
    itype: "click",
    action: "remove",
  },
};

export const widgetTemplate = {
  widget: {
    wtype: "checkbox",
    variable: "season",
    choices: ["spring", "summer", "winter"],
    default: ["spring", "winter"],
    props: {
      title: "Season",
      mode: "group",
      description: "(select season for shadow analysis)",
      orientation: "horizontal",
    },
  },
};

export const comparisonTemplate = {
  comparison: {
    key: ["A_shadow", "B_shadow"],
    y: "Mean Acc shadow",
    chart: "table",
    props: {
      unit: "minutes",
    },
  },
};

export const TEMPLATES: Record<TemplateKey, any> = {
  data_layer: dataLayerTemplate,
  view: viewTemplate,
  join: joinTemplate,
  interaction: interactionTemplate,
  widget: widgetTemplate,
  comparison: comparisonTemplate,
};

export const TEMPLATE_LABELS: Record<TemplateKey, string> = {
  data_layer: "Data layer",
  view: "View",
  join: "Join",
  interaction: "Interaction",
  widget: "Widget",
  comparison: "Comparison",
};

// The canvas node type (see ./nodes) each template creates - shared between
// App.tsx (adding via the NodeRail/canvas drop) and any agent that creates
// nodes itself (see agents/nodeAgent.ts), so both use the exact same
// mapping rather than two copies that can drift. Lives here rather than in
// App.tsx so a leaf module like nodeAgent.ts can import it without a
// circular dependency back on App.tsx.
export const TEMPLATE_NODE_TYPE: Record<TemplateKey, string> = {
  data_layer: "dataLayerNode",
  join: "joinNode",
  view: "viewNode",
  interaction: "interactionNode",
  widget: "widgetNode",
  comparison: "comparisonNode",
};

// -------------------------------------------
// Conversion to raster:
// -------------------------------------------

// from transformations.raster_conversion.scripts.convert_to_raster import convert_raster

// input = "A_buildings"
// output = "A_rasters"

// attribute = "height"
// zoom = 16

// convert_raster(input, attribute, zoom, output)

// -------------------------------------------
// Run shadow model:
// -------------------------------------------

// from models.shadow.scripts.deep_umbra import run_shadow_model

// input = 'A_rasters'
// output = 'A_shadow'

// run_shadow_model(input, season, output)
