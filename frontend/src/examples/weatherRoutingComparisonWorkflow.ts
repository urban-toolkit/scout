import type { Edge, Node } from "@xyflow/react";

import { attachNodeBehaviors, type LoadWorkflowArgs } from "./workflowHelpers";

export function loadWeatherRoutingComparisonExample({
  setNodes,
  setEdges,
  getNode,
  onRunInteraction,
  onRunWidget,
  setIdCounter,
}: LoadWorkflowArgs) {
  const rawNodes: Node<any>[] = [
    {
      id: "grammar-12",
      type: "widgetNode",
      position: {
        x: 624.3588969569157,
        y: 240.16794324359154,
      },
      width: 336,
      height: 178,
      data: {
        value: {
          widget: {
            wtype: "slider",
            variable: "rain",
            default: 0.85834,
            props: {
              title: "Rain weight",
              min: 0,
              max: 1,
              step: 0.00001,
              orientation: "horizontal",
            },
          },
        },
        mode: "view",
        pushToken: "7f5ea8d5-2f20-4acb-914a-a9ef354e9085",
        output: {
          variable: "rain",
          value: 0.85834,
        },
        title: "Widget",
      },
    },
    {
      id: "grammar-13",
      type: "widgetNode",
      position: {
        x: 258.2933844920683,
        y: 240.44011308900429,
      },
      width: 343,
      height: 180,
      data: {
        value: {
          widget: {
            wtype: "slider",
            variable: "wind",
            default: 0.01657,
            props: {
              title: "Wind weight",
              description: "(choose a wind weight)",
              min: 0,
              max: 1,
              step: 0.00001,
              orientation: "horizontal",
            },
          },
        },
        mode: "view",
        pushToken: "1c7d5efc-39f3-4134-b209-708dd3a12710",
        output: {
          variable: "wind",
          value: 0.01657,
        },
      },
    },
    {
      id: "grammar-14",
      type: "widgetNode",
      position: {
        x: 258.40810413161466,
        y: 47.114819498937834,
      },
      width: 346,
      height: 179,
      data: {
        value: {
          widget: {
            wtype: "location-input",
            variable: "origin",
            default: "1256 West Chicago Avenue",
            props: {
              title: "Origin",
              multiline: false,
              placeholder: "e.g., Chicago, IL",
              "min-length": 0,
              "max-length": 200,
            },
          },
        },
        mode: "view",
        pushToken: "783ddc63-1533-43c0-a060-5c811d8914fa",
        output: {
          variable: "origin",
          value: "1256 West Chicago Avenue",
        },
      },
    },
    {
      id: "grammar-15",
      type: "widgetNode",
      position: {
        x: 624.7629644136255,
        y: 47.105818514119875,
      },
      width: 339,
      height: 180,
      data: {
        value: {
          widget: {
            wtype: "location-input",
            variable: "destination",
            default: "1410 South Special Olympics Drive",
            props: {
              title: "Destination",
              multiline: false,
              placeholder: "e.g., Chicago, IL",
              "min-length": 0,
              "max-length": 200,
            },
          },
        },
        mode: "view",
        pushToken: "f5ed9f74-18e7-4ac0-b4e0-06d71f3d979d",
        output: {
          variable: "destination",
          value: "1410 South Special Olympics Drive",
        },
      },
    },
    {
      id: "grammar-16",
      type: "widgetNode",
      position: {
        x: 623.3534522125582,
        y: 433.89150592708484,
      },
      width: 340,
      height: 170,
      data: {
        value: {
          widget: {
            wtype: "dropdown",
            variable: "mode",
            choices: [
              "Default weights",
              "Custom weights",
              "Single-factor weights",
            ],
            default: "Default weights",
            props: {
              title: "Modes",
              "multi-select": false,
              searchable: true,
            },
          },
        },
        mode: "view",
        pushToken: "8ed50208-b259-4a69-95b7-3282fe37229a",
        output: {
          variable: "mode",
          value: "Default weights",
        },
      },
    },
    {
      id: "grammar-17",
      type: "widgetNode",
      position: {
        x: 256.88507237983384,
        y: 618.8061243296587,
      },
      width: 343,
      height: 180,
      data: {
        value: {
          widget: {
            wtype: "number-input",
            variable: "k",
            default: 1,
            props: {
              title: "K",
              "input-kind": "number",
              min: 1,
              max: 3,
            },
          },
        },
        mode: "view",
        pushToken: "9925ced6-e458-4243-9e02-e7f0ab750dd6",
        output: {
          variable: "k",
          value: 1,
        },
      },
    },
    {
      id: "grammar-18",
      type: "widgetNode",
      position: {
        x: 257.1440376532605,
        y: 432.23005433247727,
      },
      width: 344,
      height: 172,
      data: {
        value: {
          widget: {
            wtype: "datetime-picker",
            variable: "time",
            default: "2025-07-06T00:00:00",
            props: {
              title: "Start time",
              mode: "datetime",
              "display-format": "YYYY-MM-DD HH:mm",
            },
          },
        },
        mode: "view",
        pushToken: "8db070ab-345f-42a1-a3e4-7abb781a3836",
        output: {
          variable: "time",
          value: "2025-07-06T00:00:00",
        },
      },
    },
    {
      id: "pyCodeEditor-20",
      type: "pyCodeEditorNode",
      position: {
        x: 1110.1573107720985,
        y: 672.2394683238832,
      },
      width: 300,
      height: 172,
      data: {
        code: `from models.routing.scripts.weather_routing import calculate_weather_route

datafile = "chicago"
input = "baselayer"
outputs = ["C", "D"]

calculate_weather_route(
    datafile,
    input,
    outputs,
    origin,
    destination,
    mode=mode,
    K=k,
    time_=time,
    rain=rain,
    wind=wind,
)`,
        widgetOutputs: [
          {
            variable: "origin",
            value: {
              lat: 41.896438,
              lon: -87.659758,
            },
          },
          {
            variable: "destination",
            value: {
              lat: 41.861649,
              lon: -87.614034,
            },
          },
          {
            variable: "mode",
            value: "Default weights",
          },
          {
            variable: "k",
            value: 1,
          },
          {
            variable: "time",
            value: "2025-07-06T00:00:00",
          },
          {
            variable: "rain",
            value: 0.85834,
          },
          {
            variable: "wind",
            value: 0.01657,
          },
        ],
        title: "Generate alternate routes",
      },
    },
    {
      id: "grammar-21",
      type: "dataLayerNode",
      position: {
        x: 639.0512662009569,
        y: 625.8535821731028,
      },
      width: 300,
      height: 176,
      data: {
        value: {
          data_layer: {
            id: "baselayer",
            source: "osm",
            dtype: "physical",
            roi: {
              datafile: "chicago",
              type: "bbox",
              value: [-87.662, 41.859, -87.613, 41.898],
            },
            osm_features: [
              {
                feature: "roads",
              },
            ],
          },
        },
        title: "Fetch network data",
      },
    },
    {
      id: "grammar-22",
      type: "comparisonNode",
      position: {
        x: 1558.8877117516072,
        y: 30.926194926872867,
      },
      width: 293,
      height: 336,
      data: {
        value: {
          comparison: {
            key: ["C", "D"],
            metric: "duration",
            chart: "bar",
            props: {
              labelY: "Travel time (minutes)",
              colors: {
                C: "#42A5F5",
                D: "#00838F",
              },
            },
          },
        },
        mode: "view",
        previewToken: "513e0e3a-a7f6-452c-943f-0b599f7adf17",
      },
    },
    {
      id: "grammar-23",
      type: "comparisonNode",
      position: {
        x: 1880.3653080927227,
        y: 384.47547169673567,
      },
      width: 290,
      height: 334,
      data: {
        value: {
          comparison: {
            key: ["C", "D"],
            metric: "rain_exposure",
            chart: "bar",
            props: {
              labelY: "Rain exposure",
              colors: {
                C: "#42A5F5",
                D: "#00838F",
              },
            },
          },
        },
        mode: "view",
        previewToken: "0a6d8510-6be5-4c10-a476-a57eb184fa62",
      },
    },
    {
      id: "grammar-24",
      type: "comparisonNode",
      position: {
        x: 1880.5821846515726,
        y: 30.709077480859833,
      },
      width: 288,
      height: 338,
      data: {
        value: {
          comparison: {
            key: ["C", "D"],
            metric: "distance",
            chart: "bar",
            props: {
              labelY: "Distance (km)",
              colors: {
                C: "#42A5F5",
                D: "#00838F",
              },
            },
          },
        },
        mode: "view",
        previewToken: "3a8b2cbb-cb54-4eb9-b6c3-3eeeb04463ab",
      },
    },
    {
      id: "grammar-25",
      type: "comparisonNode",
      position: {
        x: 1559.167538706423,
        y: 381.4669839926429,
      },
      width: 292,
      height: 336,
      data: {
        value: {
          comparison: {
            key: ["C", "D"],
            metric: "wind_exposure",
            chart: "bar",
            props: {
              labelY: "Wind exposure",
              colors: {
                C: "#42A5F5",
                D: "#00838F",
              },
            },
          },
        },
        mode: "view",
        previewToken: "d7858bb0-a7bb-400a-a8ae-a5f7a496cdef",
      },
    },
    {
      id: "grammar-26",
      type: "viewNode",
      position: {
        x: 985.8268774270337,
        y: 69.95975482283484,
      },
      width: 547,
      height: 600,
      data: {
        value: {
          view: [
            {
              ref: "route_C",
              style: {
                stroke: {
                  color: "#90CAF9",
                  width: 3,
                },
                "border-color": "#42A5F5",
                "border-width": 6,
              },
            },
            {
              ref: "route_D",
              style: {
                stroke: {
                  color: "#00ACC1",
                  width: 3,
                },
                "border-color": "#00838F",
                "border-width": 6,
              },
            },
            {
              ref: "route_origin",
              style: {
                fill: "#1A73E8",
                stroke: {
                  color: "white",
                  width: 1,
                },
                size: 8,
              },
            },
            {
              ref: "route_destination",
              style: {
                fill: "#fc4e2a",
                stroke: {
                  color: "white",
                  width: 1,
                },
                size: 8,
                opacity: 1,
              },
            },
          ],
        },
        mode: "view",
        pushToken: "383e79df-3854-4a9d-be2e-9a15ad02bb24",
      },
    },
  ];

  const edges: Edge[] = [
    {
      source: "grammar-14",
      sourceHandle: "widget-out-3",
      target: "pyCodeEditor-20",
      targetHandle: "viewport-in-2",
      animated: true,
      id: "xy-edge__grammar-14widget-out-3-pyCodeEditor-20viewport-in-2",
      hidden: true,
    },
    {
      source: "grammar-15",
      sourceHandle: "widget-out-3",
      target: "pyCodeEditor-20",
      targetHandle: "viewport-in-2",
      animated: true,
      id: "xy-edge__grammar-15widget-out-3-pyCodeEditor-20viewport-in-2",
      hidden: true,
    },
    {
      source: "grammar-16",
      sourceHandle: "widget-out-3",
      target: "pyCodeEditor-20",
      targetHandle: "viewport-in-2",
      animated: true,
      id: "xy-edge__grammar-16widget-out-3-pyCodeEditor-20viewport-in-2",
      hidden: true,
    },
    {
      source: "grammar-17",
      sourceHandle: "widget-out-3",
      target: "pyCodeEditor-20",
      targetHandle: "viewport-in-2",
      animated: true,
      id: "xy-edge__grammar-17widget-out-3-pyCodeEditor-20viewport-in-2",
      hidden: true,
    },
    {
      source: "grammar-18",
      sourceHandle: "widget-out-3",
      target: "pyCodeEditor-20",
      targetHandle: "viewport-in-2",
      animated: true,
      id: "xy-edge__grammar-18widget-out-3-pyCodeEditor-20viewport-in-2",
      hidden: true,
    },
    {
      source: "grammar-12",
      sourceHandle: "widget-out-3",
      target: "pyCodeEditor-20",
      targetHandle: "viewport-in-2",
      animated: true,
      id: "xy-edge__grammar-12widget-out-3-pyCodeEditor-20viewport-in-2",
      hidden: true,
    },
    {
      source: "grammar-13",
      sourceHandle: "widget-out-3",
      target: "pyCodeEditor-20",
      targetHandle: "viewport-in-2",
      animated: true,
      id: "xy-edge__grammar-13widget-out-3-pyCodeEditor-20viewport-in-2",
      hidden: true,
    },
    {
      source: "grammar-21",
      sourceHandle: "data-out",
      target: "pyCodeEditor-20",
      targetHandle: "viewport-in-2",
      animated: true,
      id: "xy-edge__grammar-21data-out-pyCodeEditor-20viewport-in-2",
      hidden: true,
    },
    {
      source: "pyCodeEditor-20",
      sourceHandle: "viewport-out",
      target: "grammar-22",
      targetHandle: "comparison-in-1",
      animated: true,
      id: "xy-edge__pyCodeEditor-20viewport-out-grammar-22comparison-in-1",
      hidden: true,
    },
    {
      source: "pyCodeEditor-20",
      sourceHandle: "viewport-out",
      target: "grammar-23",
      targetHandle: "comparison-in-1",
      animated: true,
      id: "xy-edge__pyCodeEditor-20viewport-out-grammar-23comparison-in-1",
      hidden: true,
    },
    {
      source: "pyCodeEditor-20",
      sourceHandle: "viewport-out",
      target: "grammar-24",
      targetHandle: "comparison-in-1",
      animated: true,
      id: "xy-edge__pyCodeEditor-20viewport-out-grammar-24comparison-in-1",
      hidden: true,
    },
    {
      source: "pyCodeEditor-20",
      sourceHandle: "viewport-out",
      target: "grammar-25",
      targetHandle: "comparison-in-1",
      animated: true,
      id: "xy-edge__pyCodeEditor-20viewport-out-grammar-25comparison-in-1",
      hidden: true,
    },
    {
      source: "pyCodeEditor-20",
      sourceHandle: "viewport-out",
      target: "grammar-26",
      targetHandle: "view-in-2",
      animated: true,
      id: "xy-edge__pyCodeEditor-20viewport-out-grammar-26view-in-2",
      hidden: true,
    },
  ];

  const hydratedNodes = rawNodes.map((node) =>
    attachNodeBehaviors(node, setNodes, getNode, onRunInteraction, onRunWidget),
  );

  setNodes(hydratedNodes);
  setEdges(edges);
  setIdCounter?.(27);
}
