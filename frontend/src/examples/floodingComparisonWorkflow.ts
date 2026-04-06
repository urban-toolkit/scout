import type { Edge, Node } from "@xyflow/react";

import { attachNodeBehaviors, type LoadWorkflowArgs } from "./workflowHelpers";

export function loadFloodingComparisonExample({
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
      type: "widgetNode",
      position: { x: 69.5716792872519, y: 266.08132305726883 },
      width: 317,
      height: 399,
      data: {
        value: {
          widget: {
            wtype: "checkbox",
            variable: "nbs_s1",
            choices: [
              "Bioswales/Infiltration trenches",
              "Permeable pavements",
              "Retention ponds",
              "Infiltration trench",
              "Bioswales",
              "Constructed wetlands",
            ],
            default: [
              "Bioswales/Infiltration trenches",
              "Permeable pavements",
              "Retention ponds",
              "Infiltration trench",
              "Bioswales",
              "Constructed wetlands",
            ],
            props: {
              title: "NbS (Scenario-1)",
              mode: "group",
              orientation: "horizontal",
            },
          },
        },
        mode: "view",
        pushToken: "fea262e8-9776-467b-ac63-b305bb9f75bd",
        output: {
          variable: "nbs_s1",
          value: [],
        },
      },
    },
    {
      id: "grammar-2",
      type: "widgetNode",
      position: { x: 63.01872306597616, y: 743.9617531086445 },
      width: 323,
      height: 411,
      data: {
        value: {
          widget: {
            wtype: "checkbox",
            variable: "nbs_s2",
            choices: [
              "Bioswales/Infiltration trenches",
              "Permeable pavements",
              "Retention ponds",
              "Infiltration trench",
              "Bioswales",
              "Constructed wetlands",
            ],
            default: [],
            props: {
              title: "NbS (Scenario-2)",
              mode: "group",
              orientation: "horizontal",
            },
          },
        },
        mode: "view",
        pushToken: "752ff0b4-826f-40ad-ae27-be2f3c615e3b",
        output: {
          variable: "nbs_s2",
          value: [
            "Bioswales/Infiltration trenches",
            "Permeable pavements",
            "Retention ponds",
            "Infiltration trench",
            "Bioswales",
            "Constructed wetlands",
          ],
        },
      },
    },
    {
      id: "grammar-3",
      type: "widgetNode",
      position: { x: 420.2957912736662, y: 492.4364570905455 },
      width: 358,
      height: 177,
      data: {
        value: {
          widget: {
            wtype: "text-input",
            variable: "topleft",
            default: "-90.6879323, 41.6242105",
            props: {
              title: "Top-left",
              "input-kind": "text",
            },
          },
        },
        mode: "view",
        pushToken: "3968586b-d24b-4d0b-8a70-f37ecefa7b5f",
        output: {
          variable: "topleft",
          value: "-90.6879323, 41.6242105",
        },
      },
    },
    {
      id: "grammar-4",
      type: "widgetNode",
      position: { x: 417.757081664261, y: 745.7798043626369 },
      width: 364,
      height: 178,
      data: {
        value: {
          widget: {
            wtype: "text-input",
            variable: "bottomright",
            default: "-90.4252158, 41.4150156",
            props: {
              title: "Bottom-right",
              "input-kind": "text",
            },
          },
        },
        mode: "view",
        pushToken: "8bf48c3c-a4ac-4c24-917e-ff3122ad189a",
        output: {
          variable: "bottomright",
          value: "-90.4252158, 41.4150156",
        },
      },
    },
    {
      id: "pyCodeEditor-5",
      type: "pyCodeEditorNode",
      position: { x: 874.079809256554, y: 203.84064586073134 },
      width: 415,
      height: 366,
      data: {
        code: `from models.flooding.scripts.flood_simulation import simulate_flood_projection

output = 'A'

simulate_flood_projection(
    topleft,
    bottomright,
    output,
    year=timeline,
    use_NBS_classes= nbs_s1
)`,
        widgetOutputs: [
          {
            variable: "nbs_s1",
            value: [],
          },
          {
            variable: "topleft",
            value: "-90.6879323, 41.6242105",
          },
          {
            variable: "timeline",
            value: "2020 - 2040",
          },
          {
            variable: "bottomright",
            value: "-90.4252158, 41.4150156",
          },
        ],
      },
    },
    {
      id: "pyCodeEditor-6",
      type: "pyCodeEditorNode",
      position: { x: 873.7981393819651, y: 876.1839822996333 },
      width: 415,
      height: 335,
      data: {
        code: `from models.flooding.scripts.flood_simulation import simulate_flood_projection

output = 'B'

simulate_flood_projection(
    topleft,
    bottomright,
    output,
    year=timeline,
    use_NBS_classes= nbs_s2
)`,
        widgetOutputs: [
          {
            variable: "nbs_s2",
            value: [
              "Bioswales/Infiltration trenches",
              "Permeable pavements",
              "Retention ponds",
              "Infiltration trench",
              "Bioswales",
              "Constructed wetlands",
            ],
          },
          {
            variable: "bottomright",
            value: "-90.4252158, 41.4150156",
          },
          {
            variable: "timeline",
            value: "2020 - 2040",
          },
          {
            variable: "topleft",
            value: "-90.6879323, 41.6242105",
          },
        ],
      },
    },
    {
      id: "grammar-7",
      type: "viewNode",
      position: { x: 1339.1774720001263, y: 820.8941291784362 },
      width: 441,
      height: 466,
      data: {
        value: {
          view: [
            {
              ref: "B",
              style: {
                opacity: 1,
                colormap: "blues",
              },
            },
          ],
        },
        mode: "view",
        pushToken: "57153287-2182-463f-b824-150ec4c51df9",
      },
    },
    {
      id: "grammar-8",
      type: "viewNode",
      position: { x: 1337.6070458880688, y: 130.24176774240385 },
      width: 439,
      height: 469,
      data: {
        value: {
          view: [
            {
              ref: "A",
              style: {
                opacity: 1,
                colormap: "blues",
              },
            },
          ],
        },
        mode: "view",
        pushToken: "085d4642-8ff9-4a62-8612-9c249811f22e",
      },
    },
    {
      id: "grammar-9",
      type: "viewNode",
      position: { x: 1809.60101337549, y: 224.2828308932178 },
      width: 432,
      height: 472,
      data: {
        value: {
          view: [
            {
              ref_base: "B",
              ref_comp: "A",
              style: {
                opacity: 1,
                colormap: "blues",
              },
            },
          ],
        },
        mode: "view",
        pushToken: "344706d8-9a77-4778-a599-5b7430ff41b6",
      },
    },
    {
      id: "grammar-10",
      type: "comparisonNode",
      position: { x: 1804.4002631512146, y: 895.0508530885373 },
      width: 446,
      height: 249,
      data: {
        value: {
          comparison: {
            key: ["A", "B"],
            metric: "median flood depth",
            chart: "table",
            props: {
              unit: "meter",
            },
          },
        },
        mode: "view",
        previewToken: "207f8050-140f-4e71-95c9-884a30f47b1f",
      },
    },
    {
      id: "grammar-11",
      type: "widgetNode",
      position: { x: 853.9280011959954, y: 622.2818210451762 },
      width: 455,
      height: 197,
      data: {
        value: {
          widget: {
            wtype: "radio-group",
            variable: "timeline",
            choices: ["2020 - 2040", "2050 - 2080", "2080 - 2100"],
            default: "2020 - 2040",
            props: {
              title: "Projection timeline",
              orientation: "horizontal",
            },
          },
        },
        mode: "view",
        pushToken: "9d38b549-6032-401f-bc13-437d7962e8bc",
        output: {
          variable: "timeline",
          value: "2020 - 2040",
        },
      },
    },
  ];

  const edges: Edge[] = [
    {
      source: "grammar-1",
      sourceHandle: "widget-out-3",
      target: "pyCodeEditor-5",
      targetHandle: "viewport-in-2",
      animated: true,
      id: "xy-edge__grammar-1widget-out-3-pyCodeEditor-5viewport-in-2",
    },
    {
      source: "grammar-3",
      sourceHandle: "widget-out-3",
      target: "pyCodeEditor-5",
      targetHandle: "viewport-in-2",
      animated: true,
      id: "xy-edge__grammar-3widget-out-3-pyCodeEditor-5viewport-in-2",
    },
    {
      source: "grammar-2",
      sourceHandle: "widget-out-3",
      target: "pyCodeEditor-6",
      targetHandle: "viewport-in-2",
      animated: true,
      id: "xy-edge__grammar-2widget-out-3-pyCodeEditor-6viewport-in-2",
    },
    {
      source: "grammar-4",
      sourceHandle: "widget-out-3",
      target: "pyCodeEditor-6",
      targetHandle: "viewport-in-2",
      animated: true,
      id: "xy-edge__grammar-4widget-out-3-pyCodeEditor-6viewport-in-2",
    },
    {
      source: "grammar-11",
      sourceHandle: "widget-out-4",
      target: "pyCodeEditor-6",
      targetHandle: "viewport-in-1",
      animated: true,
      id: "xy-edge__grammar-11widget-out-4-pyCodeEditor-6viewport-in-1",
    },
    {
      source: "grammar-11",
      sourceHandle: "widget-out-1",
      target: "pyCodeEditor-5",
      targetHandle: "viewport-in-3",
      animated: true,
      id: "xy-edge__grammar-11widget-out-1-pyCodeEditor-5viewport-in-3",
    },
    {
      source: "grammar-3",
      sourceHandle: "widget-out-3",
      target: "pyCodeEditor-6",
      targetHandle: "viewport-in-2",
      animated: true,
      id: "xy-edge__grammar-3widget-out-3-pyCodeEditor-6viewport-in-2",
    },
    {
      source: "grammar-4",
      sourceHandle: "widget-out-3",
      target: "pyCodeEditor-5",
      targetHandle: "viewport-in-2",
      animated: true,
      id: "xy-edge__grammar-4widget-out-3-pyCodeEditor-5viewport-in-2",
    },
    {
      source: "pyCodeEditor-5",
      sourceHandle: "viewport-out",
      target: "grammar-8",
      targetHandle: "view-in-2",
      animated: true,
      id: "xy-edge__pyCodeEditor-5viewport-out-grammar-8view-in-2",
    },
    {
      source: "pyCodeEditor-6",
      sourceHandle: "viewport-out",
      target: "grammar-7",
      targetHandle: "view-in-2",
      animated: true,
      id: "xy-edge__pyCodeEditor-6viewport-out-grammar-7view-in-2",
    },
    {
      source: "pyCodeEditor-5",
      sourceHandle: "viewport-out",
      target: "grammar-10",
      targetHandle: "comparison-in-1",
      animated: true,
      id: "xy-edge__pyCodeEditor-5viewport-out-grammar-10comparison-in-1",
    },
    {
      source: "pyCodeEditor-6",
      sourceHandle: "viewport-out",
      target: "grammar-10",
      targetHandle: "comparison-in-1",
      animated: true,
      id: "xy-edge__pyCodeEditor-6viewport-out-grammar-10comparison-in-1",
    },
    {
      source: "pyCodeEditor-6",
      sourceHandle: "viewport-out",
      target: "grammar-9",
      targetHandle: "view-in-2",
      animated: true,
      id: "xy-edge__pyCodeEditor-6viewport-out-grammar-9view-in-2",
    },
    {
      source: "pyCodeEditor-5",
      sourceHandle: "viewport-out",
      target: "grammar-9",
      targetHandle: "view-in-2",
      animated: true,
      id: "xy-edge__pyCodeEditor-5viewport-out-grammar-9view-in-2",
    },
  ];

  const hydratedNodes = rawNodes.map((node) =>
    attachNodeBehaviors(node, setNodes, getNode, onRunInteraction, onRunWidget),
  );

  setNodes(hydratedNodes);
  setEdges(edges);
  setIdCounter?.(12);
}

