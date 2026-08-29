import type { Edge, Node } from "@xyflow/react";

import { attachNodeBehaviors, type LoadWorkflowArgs } from "./workflowHelpers";

export function loadShadowComparisonExample({
  setNodes,
  setEdges,
  getNode,
  onRunInteraction,
  onRunWidget,
  setIdCounter,
}: LoadWorkflowArgs) {
  const rawNodes: Node<any>[] = [
    {
      id: "grammar-1",
      type: "dataLayerNode",
      position: { x: 88.07692833259216, y: 215.48447943743872 },
      data: {
        value: {
          data_layer: {
            id: "A",
            source: "osm",
            dtype: "physical",
            roi: {
              datafile: "chicago",
              type: "bbox",
              value: [-87.635, 41.882, -87.63, 41.887],
            },
            osm_features: [
              {
                feature: "buildings",
                attributes: ["height"],
              },
            ],
          },
        },
      },
    },
    {
      id: "grammar-2",
      type: "viewNode",
      position: { x: 487.0244755760218, y: 264.6331092032282 },
      width: 411,
      height: 457,
      data: {
        value: {
          view: [
            {
              ref: "A_buildings",
              style: {
                fill: {
                  feature: "height",
                  range: [0, 550],
                  colormap: "greys",
                },
                "stroke-color": "#333333",
                opacity: 1,
              },
            },
          ],
        },
        mode: "view",
        pushToken: "9376f3b2-0b32-4fa2-a312-4d014705ada0",
      },
    },
    {
      id: "grammar-3",
      type: "dataLayerNode",
      position: { x: 85.33886368925349, y: 962.3795339030556 },
      data: {
        value: {
          data_layer: {
            id: "B",
            source: "osm",
            dtype: "physical",
            roi: {
              datafile: "chicago",
              type: "bbox",
              value: [-87.635, 41.882, -87.63, 41.887],
            },
            osm_features: [
              {
                feature: "buildings",
                attributes: ["height"],
              },
            ],
          },
        },
      },
    },
    {
      id: "grammar-4",
      type: "viewNode",
      position: { x: 482.9904070638938, y: 1010.2301295861629 },
      width: 421,
      height: 462,
      data: {
        value: {
          view: [
            {
              ref: "B_buildings",
              style: {
                fill: {
                  feature: "height",
                  range: [0, 550],
                  colormap: "greys",
                },
                "stroke-color": "#333333",
                opacity: 1,
              },
            },
          ],
        },
        mode: "view",
        pushToken: "65abcbc7-0ce1-4af1-b090-569aba7f88f8",
        interactions: [
          {
            ref: "B_buildings",
            itype: "click",
            action: "remove",
          },
        ],
      },
    },
    {
      id: "grammar-5",
      type: "interactionNode",
      position: { x: 543.0160643907737, y: 746.2635080226084 },
      width: 300,
      height: 231,
      data: {
        value: {
          interaction: {
            ref: "B_buildings",
            itype: "click",
            action: "remove",
          },
        },
      },
    },
    {
      id: "pyCodeEditor-6",
      type: "pyCodeEditorNode",
      position: { x: 949.8285014413963, y: 274.04156937761365 },
      width: 400,
      data: {
        code: `from transformations.raster_conversion.scripts.convert_to_raster import convert_raster

input = "A_buildings"
output = "A_rasters"

attribute = "height"
zoom = 16

convert_raster(input, attribute, zoom, output)`,
      },
    },
    {
      id: "pyCodeEditor-7",
      type: "pyCodeEditorNode",
      position: { x: 945.8135059414925, y: 1022.9172526097107 },
      width: 400,
      data: {
        code: `from transformations.raster_conversion.scripts.convert_to_raster import convert_raster

input = "B_buildings"
output = "B_rasters"

attribute = "height"
zoom = 16

convert_raster(input, attribute, zoom, output)`,
      },
    },
    {
      id: "pyCodeEditor-8",
      type: "pyCodeEditorNode",
      position: { x: 1400.8868271271701, y: 282.22214055084834 },
      width: 460,
      height: 428,
      data: {
        code: `from models.shadow.scripts.deep_umbra import run_shadow_model

input = 'A_rasters'
output = 'A_shadow'

run_shadow_model(input, season, output)`,
        widgetOutputs: [
          {
            variable: "season",
            value: "summer",
          },
        ],
      },
    },
    {
      id: "pyCodeEditor-9",
      type: "pyCodeEditorNode",
      position: { x: 1398.3082290651691, y: 1022.4083694110318 },
      width: 465,
      height: 440,
      data: {
        code: `from models.shadow.scripts.deep_umbra import run_shadow_model

input = 'B_rasters'
output = 'B_shadow'

run_shadow_model(input, season, output)`,
        widgetOutputs: [
          {
            variable: "season",
            value: "summer",
          },
        ],
      },
    },
    {
      id: "grammar-10",
      type: "widgetNode",
      position: { x: 1454.932014237732, y: 780.3485007780372 },
      width: 352,
      height: 191,
      data: {
        value: {
          widget: {
            wtype: "radio-group",
            variable: "season",
            choices: ["spring", "summer", "winter"],
            default: "summer",
            props: {
              title: "Season",
              orientation: "horizontal",
            },
          },
        },
        mode: "view",
        pushToken: "1de3f941-3264-490e-8757-a1c67decc105",
        output: {
          variable: "season",
          value: "winter",
        },
      },
    },
    {
      id: "grammar-11",
      type: "viewNode",
      position: { x: 1926.8109292472998, y: 245.07341915047215 },
      width: 426,
      height: 451,
      data: {
        value: {
          view: [
            {
              ref: "A_shadow",
              style: {
                opacity: 1,
                colormap: "reds",
              },
            },
          ],
        },
        mode: "view",
        pushToken: "fc0957b8-cf66-4801-9f5d-4e77a266c181",
      },
    },
    {
      id: "grammar-12",
      type: "viewNode",
      position: { x: 1925.370335060804, y: 932.0598893654062 },
      width: 433,
      height: 467,
      data: {
        value: {
          view: [
            {
              ref: "B_shadow",
              style: {
                opacity: 1,
                colormap: "reds",
              },
            },
          ],
        },
        mode: "view",
        pushToken: "83a11ac1-4e18-4b75-bb5d-355bfdad833b",
      },
    },
    {
      id: "grammar-13",
      type: "viewNode",
      position: { x: 2404.619851154798, y: 304.4918591738118 },
      width: 416,
      height: 457,
      data: {
        value: {
          view: [
            {
              ref_base: "B_shadow",
              ref_comp: "A_shadow",
              style: {
                opacity: 1,
                colormap: "reds",
              },
            },
          ],
        },
        mode: "view",
        pushToken: "47a1496f-a17b-403a-9945-cd52ccae0bc6",
      },
    },
    {
      id: "grammar-14",
      type: "comparisonNode",
      position: { x: 2408.3437729457355, y: 1016.1420018997687 },
      width: 414,
      height: 416,
      data: {
        value: {
          comparison: {
            key: ["A_shadow", "B_shadow"],
            y: "Mean Acc shadow",
            chart: "pie",
            props: {
              unit: "minutes",
              colors: {
                A_shadow: "#E3882E",
                B_shadow: "#4D6D9C",
              },
            },
          },
        },
        mode: "view",
        previewToken: "023ef09f-d011-4f6b-b036-b89d7b35dc5a",
      },
    },
  ];

  const edges: Edge[] = [
    {
      source: "grammar-1",
      sourceHandle: "data-out",
      target: "grammar-2",
      targetHandle: "view-in-2",
      animated: true,
      id: "xy-edge__grammar-1data-out-grammar-2view-in-2",
    },
    {
      source: "grammar-5",
      sourceHandle: "interaction-out-4",
      target: "grammar-4",
      targetHandle: "view-in-1",
      animated: true,
      id: "xy-edge__grammar-5interaction-out-4-grammar-4view-in-1",
    },
    {
      source: "grammar-10",
      sourceHandle: "widget-out-1",
      target: "pyCodeEditor-8",
      targetHandle: "viewport-in-3",
      animated: true,
      id: "xy-edge__grammar-10widget-out-1-pyCodeEditor-8viewport-in-3",
    },
    {
      source: "grammar-10",
      sourceHandle: "widget-out-4",
      target: "pyCodeEditor-9",
      targetHandle: "viewport-in-1",
      animated: true,
      id: "xy-edge__grammar-10widget-out-4-pyCodeEditor-9viewport-in-1",
    },
    {
      source: "grammar-2",
      sourceHandle: "view-out",
      target: "pyCodeEditor-6",
      targetHandle: "viewport-in-2",
      animated: true,
      id: "xy-edge__grammar-2view-out-pyCodeEditor-6viewport-in-2",
    },
    {
      source: "grammar-3",
      sourceHandle: "data-out",
      target: "grammar-4",
      targetHandle: "view-in-2",
      animated: true,
      id: "xy-edge__grammar-3data-out-grammar-4view-in-2",
    },
    {
      source: "grammar-4",
      sourceHandle: "view-out",
      target: "pyCodeEditor-7",
      targetHandle: "viewport-in-2",
      animated: true,
      id: "xy-edge__grammar-4view-out-pyCodeEditor-7viewport-in-2",
    },
    {
      source: "pyCodeEditor-6",
      sourceHandle: "viewport-out",
      target: "pyCodeEditor-8",
      targetHandle: "viewport-in-2",
      animated: true,
      id: "xy-edge__pyCodeEditor-6viewport-out-pyCodeEditor-8viewport-in-2",
    },
    {
      source: "pyCodeEditor-7",
      sourceHandle: "viewport-out",
      target: "pyCodeEditor-9",
      targetHandle: "viewport-in-2",
      animated: true,
      id: "xy-edge__pyCodeEditor-7viewport-out-pyCodeEditor-9viewport-in-2",
    },
    {
      source: "pyCodeEditor-8",
      sourceHandle: "viewport-out",
      target: "grammar-11",
      targetHandle: "view-in-2",
      animated: true,
      id: "xy-edge__pyCodeEditor-8viewport-out-grammar-11view-in-2",
    },
    {
      source: "pyCodeEditor-9",
      sourceHandle: "viewport-out",
      target: "grammar-12",
      targetHandle: "view-in-2",
      animated: true,
      id: "xy-edge__pyCodeEditor-9viewport-out-grammar-12view-in-2",
    },
    {
      source: "pyCodeEditor-9",
      sourceHandle: "viewport-out",
      target: "grammar-13",
      targetHandle: "view-in-2",
      animated: true,
      id: "xy-edge__pyCodeEditor-9viewport-out-grammar-13view-in-2",
    },
    {
      source: "pyCodeEditor-8",
      sourceHandle: "viewport-out",
      target: "grammar-13",
      targetHandle: "view-in-2",
      animated: true,
      id: "xy-edge__pyCodeEditor-8viewport-out-grammar-13view-in-2",
    },
    {
      source: "pyCodeEditor-8",
      sourceHandle: "viewport-out",
      target: "grammar-14",
      targetHandle: "comparison-in-1",
      animated: true,
      id: "xy-edge__pyCodeEditor-8viewport-out-grammar-14comparison-in-1",
    },
    {
      source: "pyCodeEditor-9",
      sourceHandle: "viewport-out",
      target: "grammar-14",
      targetHandle: "comparison-in-1",
      animated: true,
      id: "xy-edge__pyCodeEditor-9viewport-out-grammar-14comparison-in-1",
    },
  ];

  const hydratedNodes = rawNodes.map((node) =>
    attachNodeBehaviors(node, setNodes, getNode, onRunInteraction, onRunWidget),
  );

  setNodes(hydratedNodes);
  setEdges(edges);

  // next created node should start after 14
  setIdCounter?.(15);
}

