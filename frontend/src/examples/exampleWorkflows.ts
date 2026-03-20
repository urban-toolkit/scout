import type { Dispatch, SetStateAction } from "react";
import type { Edge, Node } from "@xyflow/react";

import type { BaseNodeData } from "../node-components/BaseGrammar";
import type { PyCodeEditorNodeData } from "../nodes/computation/PyCodeEditorNode";

type AppNodeData = BaseNodeData | PyCodeEditorNodeData;

type LoadWorkflowArgs = {
  setNodes: Dispatch<SetStateAction<Node<AppNodeData>[]>>;
  setEdges: Dispatch<SetStateAction<Edge[]>>;
  getNode: (id: string) => Node | undefined;
  onRunInteraction: (srcId: string) => void;
  onRunWidget: (srcId: string) => void;
  setIdCounter?: (next: number) => void;
};

function attachNodeBehaviors(
  node: Node<any>,
  setNodes: Dispatch<SetStateAction<Node<AppNodeData>[]>>,
  getNode: (id: string) => Node | undefined,
  onRunInteraction: (srcId: string) => void,
  onRunWidget: (srcId: string) => void,
): Node<AppNodeData> {
  const grammarTypes = new Set([
    "dataLayerNode",
    "viewNode",
    "interactionNode",
    "widgetNode",
    "comparisonNode",
  ]);

  if (!grammarTypes.has(node.type ?? "")) {
    return node as Node<AppNodeData>;
  }

  return {
    ...node,
    data: {
      ...node.data,
      onChange: (val: any, targetId: string) => {
        setNodes((nds) =>
          nds.map((n) =>
            n.id === targetId
              ? {
                  ...n,
                  data: {
                    ...n.data,
                    value: val,
                  },
                }
              : n,
          ),
        );
      },
      onRun: (nodeId: string) => {
        const current = getNode(nodeId);
        if (!current) return;

        if (current.type === "interactionNode") {
          onRunInteraction(nodeId);
        } else if (current.type === "widgetNode") {
          onRunWidget(nodeId);
        }
      },
    },
  } as Node<AppNodeData>;
}

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
                colormap: "blues",
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
                colormap: "blues",
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
                colormap: "blues",
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
            metric: "Mean Acc shadow",
            chart: "pie",
            props: {
              unit: "minutes",
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
      position: { x: -30.46559045483024, y: 190.95995416050903 },
      width: 218,
      height: 421,
      data: {
        value: {
          widget: {
            wtype: "slider",
            variable: "rain",
            default: 0.85834,
            props: {
              title: "Rain",
              description: "(choose a rain weight)",
              min: 0,
              max: 1,
              step: 0.00001,
              orientation: "vertical",
            },
          },
        },
        mode: "view",
        pushToken: "ffca0aa3-7000-4816-8eea-465acaa9986e",
        output: {
          variable: "rain",
          value: 0.85834,
        },
      },
    },
    {
      id: "grammar-13",
      type: "widgetNode",
      position: { x: -41.43107223839236, y: 874.5445406436346 },
      width: 223,
      height: 419,
      data: {
        value: {
          widget: {
            wtype: "slider",
            variable: "wind",
            default: 0.01657,
            props: {
              title: "Wind",
              description: "(choose a wind weight)",
              min: 0,
              max: 1,
              step: 0.00001,
              orientation: "vertical",
            },
          },
        },
        mode: "view",
        pushToken: "2648432b-6b0c-4c06-be02-c928c481454d",
        output: {
          variable: "wind",
          value: 0.01657,
        },
      },
    },
    {
      id: "grammar-14",
      type: "widgetNode",
      position: { x: 290.67768179564655, y: 189.38439716296983 },
      width: 355,
      height: 190,
      data: {
        value: {
          widget: {
            wtype: "location-input",
            variable: "origin",
            default: "1256 West Chicago Avenue",
            props: {
              description: "Enter a place or address",
              title: "Origin",
              multiline: false,
              placeholder: "e.g., Chicago, IL",
              "min-length": 0,
              "max-length": 200,
            },
          },
        },
        mode: "view",
        pushToken: "41baace1-5c40-4eb3-9a29-aa054ccda5e0",
        output: {
          variable: "origin",
          value: {
            lat: 41.896438,
            lon: -87.659758,
          },
        },
      },
    },
    {
      id: "grammar-15",
      type: "widgetNode",
      position: { x: 290.78042197717434, y: 417.5049586949921 },
      width: 356,
      height: 191,
      data: {
        value: {
          widget: {
            wtype: "location-input",
            variable: "destination",
            default: "1410 South Special Olympics Drive",
            props: {
              description: "Enter a place or address",
              title: "Destination",
              multiline: false,
              placeholder: "e.g., Chicago, IL",
              "min-length": 0,
              "max-length": 200,
            },
          },
        },
        mode: "view",
        pushToken: "a770d1f9-5582-4feb-bd23-1ab366d8020b",
        output: {
          variable: "destination",
          value: {
            lat: 41.861649,
            lon: -87.614034,
          },
        },
      },
    },
    {
      id: "grammar-16",
      type: "widgetNode",
      position: { x: 288.245749071263, y: 670.2310361122444 },
      width: 356,
      height: 171,
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
      position: { x: 286.97841261830746, y: 882.1375676543713 },
      width: 356,
      height: 171,
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
      position: { x: 288.2457490712631, y: 1086.8422409758386 },
      width: 352,
      height: 195,
      data: {
        value: {
          widget: {
            wtype: "datetime-picker",
            variable: "time",
            default: "2025-07-06T00:00:00",
            props: {
              description: "Select the start date and time",
              title: "Start time",
              mode: "datetime",
              "display-format": "YYYY-MM-DD HH:mm",
            },
          },
        },
        mode: "view",
        pushToken: "c854d786-7fc9-4e1c-a4d5-48ed4ca67230",
        output: {
          variable: "time",
          value: "2025-07-06T00:00:00",
        },
      },
    },
    {
      id: "pyCodeEditor-20",
      type: "pyCodeEditorNode",
      position: { x: 786.1395946155501, y: 400.38570778763955 },
      width: 562,
      height: 462,
      data: {
        code: `from models.routing.scripts.weather_routing import calculate_weather_route

datafile = "chicago"
input = "baselayer"
outputs = ["A", "B"]

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
      },
    },
    {
      id: "grammar-21",
      type: "dataLayerNode",
      position: { x: -394.849079949325, y: 380.9287846275125 },
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
      },
    },
    {
      id: "grammar-22",
      type: "comparisonNode",
      position: { x: 1427.6635025757732, y: 250.41855129787422 },
      width: 300,
      height: 367,
      data: {
        value: {
          comparison: {
            key: ["A", "B"],
            metric: "Travel time",
            chart: "bar",
            props: {
              unit: "minutes",
            },
          },
        },
      },
    },
    {
      id: "grammar-23",
      type: "comparisonNode",
      position: { x: 1427.8999411228542, y: 680.5361020199223 },
      width: 300,
      height: 356,
      data: {
        value: {
          comparison: {
            key: ["A", "B"],
            metric: "rain exposure",
            chart: "bar",
          },
        },
      },
    },
    {
      id: "grammar-24",
      type: "comparisonNode",
      position: { x: 1804.3558032104686, y: 144.9226504426457 },
      width: 298,
      height: 367,
      data: {
        value: {
          comparison: {
            key: ["A", "B"],
            metric: "Distance",
            chart: "bar",
            props: {
              unit: "km",
            },
          },
        },
      },
    },
    {
      id: "grammar-25",
      type: "comparisonNode",
      position: { x: 1806.071929868022, y: 829.9749887809029 },
      width: 300,
      height: 354,
      data: {
        value: {
          comparison: {
            key: ["A", "B"],
            metric: "wind exposure",
            chart: "bar",
          },
        },
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
    },
    {
      source: "grammar-15",
      sourceHandle: "widget-out-3",
      target: "pyCodeEditor-20",
      targetHandle: "viewport-in-2",
      animated: true,
      id: "xy-edge__grammar-15widget-out-3-pyCodeEditor-20viewport-in-2",
    },
    {
      source: "grammar-16",
      sourceHandle: "widget-out-3",
      target: "pyCodeEditor-20",
      targetHandle: "viewport-in-2",
      animated: true,
      id: "xy-edge__grammar-16widget-out-3-pyCodeEditor-20viewport-in-2",
    },
    {
      source: "grammar-17",
      sourceHandle: "widget-out-3",
      target: "pyCodeEditor-20",
      targetHandle: "viewport-in-2",
      animated: true,
      id: "xy-edge__grammar-17widget-out-3-pyCodeEditor-20viewport-in-2",
    },
    {
      source: "grammar-18",
      sourceHandle: "widget-out-3",
      target: "pyCodeEditor-20",
      targetHandle: "viewport-in-2",
      animated: true,
      id: "xy-edge__grammar-18widget-out-3-pyCodeEditor-20viewport-in-2",
    },
    {
      source: "grammar-12",
      sourceHandle: "widget-out-3",
      target: "pyCodeEditor-20",
      targetHandle: "viewport-in-2",
      animated: true,
      id: "xy-edge__grammar-12widget-out-3-pyCodeEditor-20viewport-in-2",
    },
    {
      source: "grammar-13",
      sourceHandle: "widget-out-3",
      target: "pyCodeEditor-20",
      targetHandle: "viewport-in-2",
      animated: true,
      id: "xy-edge__grammar-13widget-out-3-pyCodeEditor-20viewport-in-2",
    },
    {
      source: "grammar-21",
      sourceHandle: "data-out",
      target: "pyCodeEditor-20",
      targetHandle: "viewport-in-2",
      animated: true,
      id: "xy-edge__grammar-21data-out-pyCodeEditor-20viewport-in-2",
    },
    {
      source: "pyCodeEditor-20",
      sourceHandle: "viewport-out",
      target: "grammar-22",
      targetHandle: "comparison-in-1",
      animated: true,
      id: "xy-edge__pyCodeEditor-20viewport-out-grammar-22comparison-in-1",
    },
    {
      source: "pyCodeEditor-20",
      sourceHandle: "viewport-out",
      target: "grammar-23",
      targetHandle: "comparison-in-1",
      animated: true,
      id: "xy-edge__pyCodeEditor-20viewport-out-grammar-23comparison-in-1",
    },
    {
      source: "pyCodeEditor-20",
      sourceHandle: "viewport-out",
      target: "grammar-24",
      targetHandle: "comparison-in-1",
      animated: true,
      id: "xy-edge__pyCodeEditor-20viewport-out-grammar-24comparison-in-1",
    },
    {
      source: "pyCodeEditor-20",
      sourceHandle: "viewport-out",
      target: "grammar-25",
      targetHandle: "comparison-in-1",
      animated: true,
      id: "xy-edge__pyCodeEditor-20viewport-out-grammar-25comparison-in-1",
    },
  ];

  const hydratedNodes = rawNodes.map((node) =>
    attachNodeBehaviors(node, setNodes, getNode, onRunInteraction, onRunWidget),
  );

  setNodes(hydratedNodes);
  setEdges(edges);

  // next created node should start after 25
  setIdCounter?.(26);
}
