// src/App.tsx
import {
  ReactFlow,
  ReactFlowProvider,
  useNodesState,
  useEdgesState,
  useReactFlow,
  addEdge,
  MarkerType,
  type DefaultEdgeOptions,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useCallback, useEffect, useRef } from "react";

import { nodeTypes } from "./nodes"; // <-- { dataLayerNode, viewNode, ... }
import type { Node, Connection, Edge } from "@xyflow/react";
import type { BaseNodeData } from "./node-components/BaseGrammar";

import { TEMPLATES, TemplateKey } from "./templates";
import "./App.css";
import type { WidgetNodeData } from "./nodes/widget/WidgetNode";
import type { PyCodeEditorNodeData } from "./nodes/computation/PyCodeEditorNode";

import {
  loadShadowComparisonExample,
  loadFloodingComparisonExample,
  loadWeatherRoutingComparisonExample,
} from "./examples/exampleWorkflows";
import ChatWidget from "./components/ai/ChatWidget";
import NodeRail from "./components/NodeRail";

const defaultEdgeOptions: DefaultEdgeOptions = {
  style: {
    stroke: "#888",
    strokeWidth: 2, // optional but improves visibility
  },
  markerEnd: {
    type: MarkerType.ArrowClosed, // or MarkerType.ArrowClosed
    width: 20, // default is 20
    height: 20, // default is 20
    color: "#888", // optional
  },
};

type WorkflowRoute = "shadow" | "flooding" | "routing";

function getAppBasePath() {
  const base = import.meta.env.BASE_URL ?? "/";
  return base === "/" ? "/" : base.endsWith("/") ? base : `${base}/`;
}

function getWorkflowRouteFromPath(
  pathname = window.location.pathname,
): WorkflowRoute | null {
  const base = getAppBasePath();
  const relativePath = pathname.startsWith(base)
    ? pathname.slice(base.length)
    : pathname.startsWith("/")
      ? pathname.slice(1)
      : pathname;
  const route = relativePath.replace(/\/+$/, "");

  if (route === "shadow" || route === "flooding" || route === "routing") {
    return route;
  }

  return null;
}

function getWorkflowPath(route: WorkflowRoute) {
  return `${getAppBasePath()}${route}`;
}

export default function App() {
  return (
    <ReactFlowProvider>
      <Canvas />
    </ReactFlowProvider>
  );
}

function Canvas() {
  const idCounter = useRef(1);
  const [nodes, setNodes, onNodesChange] = useNodesState<
    Node<BaseNodeData | PyCodeEditorNodeData>
  >([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const { getNode, getEdges, fitView } = useReactFlow();

  // const dumpWorkflow = useCallback(() => {
  //   const nodes = getNodes();
  //   const edges = getEdges();

  //   console.log("NODES");
  //   console.log(JSON.stringify(nodes, null, 2));

  //   console.log("EDGES");
  //   console.log(JSON.stringify(edges, null, 2));
  // }, [getNodes, getEdges]);

  const pushInteractionToView = useCallback(
    (srcId: string, trgId?: string) => {
      const src = getNode(srcId);
      if (!src || src.type !== "interactionNode") return;

      const val: any = (src.data as BaseNodeData).value;
      const i = val?.interaction;
      if (!i) return;

      const targetIds = trgId
        ? [trgId]
        : getEdges()
            .filter((e) => e.source === srcId)
            .map((e) => e.target!)
            .filter(Boolean);

      setNodes((nds) =>
        nds.map((n) => {
          if (!targetIds.includes(n.id) || n.type !== "viewNode") return n;

          const existing = ((n.data as any).interactions ?? []) as any[];

          const already = existing.some((e) => e?.itype === i?.itype);

          const nextInteractions = already
            ? existing.map((e) => (e?.itype === i?.itype ? i : e))
            : [...existing, i];

          return {
            ...n,
            data: {
              ...n.data,
              interactions: nextInteractions,
            },
          };
        }),
      );
    },
    [getNode, getEdges, setNodes],
  );

  const pushWidgetToPyCodeEditorNode = useCallback(
    (srcId: string, trgId?: string) => {
      const src = getNode(srcId);
      if (!src || src.type !== "widgetNode") return;
      const val: WidgetNodeData = src.data as WidgetNodeData;

      console.log(val);

      const targetIds = trgId
        ? [trgId]
        : getEdges()
            .filter((e) => e.source === srcId)
            .map((e) => e.target!)
            .filter(Boolean);

      setNodes((nds) =>
        nds.map((n) => {
          if (!targetIds.includes(n.id) || n.type !== "pyCodeEditorNode")
            return n;
          const existing = (n.data as PyCodeEditorNodeData).widgetOutputs ?? [];
          const already = existing.some(
            (e) => e.variable === val.output?.variable,
          );
          const nextWidgetOutputs = already
            ? existing.map((e) =>
                e.variable === val.output?.variable
                  ? {
                      variable: val.output.variable,
                      value: val.output.value,
                    }
                  : e,
              )
            : [
                ...existing,
                {
                  variable: val.output?.variable,
                  value: val.output?.value,
                },
              ];
          return {
            ...n,
            data: {
              ...n.data,
              widgetOutputs: nextWidgetOutputs,
            } as PyCodeEditorNodeData,
          };
        }),
      );
    },
    [getNode, getEdges, setNodes],
  );

  // Then remove the oncloseNode from createGrammarNode calls and declarations
  const addNode = useCallback(
    (tpl: TemplateKey) => {
      const nextId = `grammar-${idCounter.current++}`;
      createGrammarNode({
        id: nextId,
        setNodes,
        template: tpl,
        getNode,
        onRunInteraction: pushInteractionToView,
        onRunWidget: pushWidgetToPyCodeEditorNode,
      });
    },
    [setNodes, getNode, pushInteractionToView, pushWidgetToPyCodeEditorNode],
  );

  const addPyCodeEditorNode = useCallback(() => {
    const nextId = `pyCodeEditor-${idCounter.current++}`;
    createPyCodeEditorNode({
      id: nextId,
      setNodes,
      // onRunViewport: pushViewportToTransformation,
    });
  }, [setNodes]);

  const allow = useCallback(
    (conn: Connection | Edge) => {
      if (!conn.source || !conn.target) return false;
      const src = getNode(conn.source);
      const trg = getNode(conn.target);
      if (!src || !trg) return false;
      const dataLayerToView =
        src.type === "dataLayerNode" && trg.type === "viewNode";

      const dataLayerToPyCodeEditor =
        src.type === "dataLayerNode" && trg.type === "pyCodeEditorNode";

      const interactionToView =
        src.type === "interactionNode" && trg.type === "viewNode";

      const viewToView = src.type === "viewNode" && trg.type === "viewNode";

      const viewToPyCodeEditor =
        src.type === "viewNode" && trg.type === "pyCodeEditorNode";

      const pyCodeEditorToView =
        src.type === "pyCodeEditorNode" && trg.type === "viewNode";

      const pyCodeEditorToPyCodeEditor =
        src.type === "pyCodeEditorNode" && trg.type === "pyCodeEditorNode";

      const pyCodeEditorToComparison =
        src.type === "pyCodeEditorNode" && trg.type === "comparisonNode";

      const widgetToPyCodeEditor =
        src.type === "widgetNode" && trg.type === "pyCodeEditorNode";

      return (
        dataLayerToView ||
        dataLayerToPyCodeEditor ||
        interactionToView ||
        viewToPyCodeEditor ||
        pyCodeEditorToView ||
        pyCodeEditorToPyCodeEditor ||
        widgetToPyCodeEditor ||
        viewToView ||
        pyCodeEditorToComparison
      );
    },
    [getNode],
  );

  // onConnect is fine. Should be there.. Here we handle connections and onConnections between nodes
  const onConnect = useCallback(
    (conn: Connection) => {
      if (!allow(conn)) return;

      setEdges((eds) => addEdge({ ...conn, animated: true }, eds));

      const srcId = conn.source!;
      const src = getNode(conn.source!);
      const trg = getNode(conn.target!);
      const trgId = conn.target!;
      if (!src || !trg) return;

      if (src.type === "interactionNode" && trg.type === "viewNode") {
        pushInteractionToView(srcId, trgId);
        return;
      }

      if (src.type === "widgetNode" && trg.type === "pyCodeEditorNode") {
        pushWidgetToPyCodeEditorNode(srcId, trgId);
        return;
      }
    },
    [
      allow,
      getNode,
      setEdges,
      pushInteractionToView,
      pushWidgetToPyCodeEditorNode,
    ],
  );

  const loadShadowWorkflow = useCallback(() => {
    loadShadowComparisonExample({
      setNodes,
      setEdges,
      getNode,
      onRunInteraction: pushInteractionToView,
      onRunWidget: pushWidgetToPyCodeEditorNode,
      setIdCounter: (next) => {
        idCounter.current = next;
      },
    });

    requestAnimationFrame(() => {
      fitView({ padding: 0.15 });
    });
  }, [
    setNodes,
    setEdges,
    getNode,
    pushInteractionToView,
    pushWidgetToPyCodeEditorNode,
    fitView,
  ]);

  const loadFloodingWorkflow = useCallback(() => {
    loadFloodingComparisonExample({
      setNodes,
      setEdges,
      getNode,
      onRunInteraction: pushInteractionToView,
      onRunWidget: pushWidgetToPyCodeEditorNode,
      setIdCounter: (next) => {
        idCounter.current = next;
      },
    });

    requestAnimationFrame(() => {
      fitView({ padding: 0.15 });
    });
  }, [
    setNodes,
    setEdges,
    getNode,
    pushInteractionToView,
    pushWidgetToPyCodeEditorNode,
    fitView,
  ]);

  const loadWeatherRoutingWorkflow = useCallback(() => {
    loadWeatherRoutingComparisonExample({
      setNodes,
      setEdges,
      getNode,
      onRunInteraction: pushInteractionToView,
      onRunWidget: pushWidgetToPyCodeEditorNode,
      setIdCounter: (next) => {
        idCounter.current = next;
      },
    });

    requestAnimationFrame(() => {
      fitView({ padding: 0.15 });
    });
  }, [
    setNodes,
    setEdges,
    getNode,
    pushInteractionToView,
    pushWidgetToPyCodeEditorNode,
    fitView,
  ]);

  const loadWorkflowRoute = useCallback(
    (route: WorkflowRoute) => {
      if (route === "shadow") {
        loadShadowWorkflow();
      } else if (route === "flooding") {
        loadFloodingWorkflow();
      } else {
        loadWeatherRoutingWorkflow();
      }
    },
    [loadShadowWorkflow, loadFloodingWorkflow, loadWeatherRoutingWorkflow],
  );

  const navigateToWorkflowRoute = useCallback(
    (route: WorkflowRoute) => {
      const nextPath = getWorkflowPath(route);
      if (window.location.pathname !== nextPath) {
        window.history.pushState(null, "", nextPath);
      }
      loadWorkflowRoute(route);
    },
    [loadWorkflowRoute],
  );

  const clearCanvasAndNavigateHome = useCallback(() => {
    const homePath = getAppBasePath();
    if (window.location.pathname !== homePath) {
      window.history.pushState(null, "", homePath);
    }
    setNodes([]);
    setEdges([]);
    idCounter.current = 1;
  }, [setEdges, setNodes]);

  useEffect(() => {
    const route = getWorkflowRouteFromPath();
    if (route) {
      loadWorkflowRoute(route);
    }

    const handlePopState = () => {
      const nextRoute = getWorkflowRouteFromPath();
      if (nextRoute) {
        loadWorkflowRoute(nextRoute);
      }
    };

    window.addEventListener("popstate", handlePopState);
    return () => {
      window.removeEventListener("popstate", handlePopState);
    };
  }, [loadWorkflowRoute]);

  return (
    <div className="app">
      <ReactFlow
        className="canvas"
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        isValidConnection={allow}
        fitView
        minZoom={0.005}
        maxZoom={2}
        defaultEdgeOptions={defaultEdgeOptions}
        proOptions={{ hideAttribution: true }}
      >
        {/* <Background /> */}
        <NodeRail
          onAdd={addNode}
          onAddPyCodeEditor={addPyCodeEditorNode}
          onClear={clearCanvasAndNavigateHome}
          onLoadShadowWorkflow={() => navigateToWorkflowRoute("shadow")}
          onLoadFloodingWorkflow={() => navigateToWorkflowRoute("flooding")}
          onLoadWeatherRoutingWorkflow={() =>
            navigateToWorkflowRoute("routing")
          }
        />
      </ReactFlow>
      {/* <button onClick={dumpWorkflow} className="toolbar__btn__dump">
        Dump
      </button> */}
      <ChatWidget />
    </div>
  );
}

// Map template key -> node type key from ./nodes
const kindToType: Record<TemplateKey, keyof typeof nodeTypes> = {
  data_layer: "dataLayerNode",
  join: "joinNode",
  view: "viewNode",
  interaction: "interactionNode",
  widget: "widgetNode",
  comparison: "comparisonNode",
};

function createGrammarNode({
  id,
  setNodes,
  template,
  getNode,
  onRunInteraction,
  onRunWidget,
}: // onRunWidgetView
{
  id: string;
  setNodes: React.Dispatch<
    React.SetStateAction<Node<BaseNodeData | PyCodeEditorNodeData>[]>
  >;
  template: TemplateKey;
  getNode: (id: string) => Node | undefined;
  onRunInteraction: (srcId: string) => void;
  onRunWidget: (srcId: string) => void;
}) {
  const pos = (window as any)._desiredGrammarPos ?? { x: 100, y: 100 };
  const type = kindToType[template];

  const newNode: Node<BaseNodeData> = {
    id,
    type,
    position: pos,
    data: {
      value: TEMPLATES[template] ?? {},
      onChange: (val, targetId) => {
        setNodes((nds) =>
          nds.map((n) =>
            n.id === targetId ? { ...n, data: { ...n.data, value: val } } : n,
          ),
        );
      },
      // Each node type decides how to "run" itself
      onRun: (nodeId) => {
        const node = getNode(nodeId);
        if (!node) return;
        else if (node.type === "interactionNode") {
          onRunInteraction(nodeId);
        } else if (node.type === "widgetNode") {
          onRunWidget(nodeId);
        }
      },
    },
  };

  setNodes((nds) => nds.concat(newNode));
}

function createPyCodeEditorNode({
  id,
  setNodes,
}: // onRunViewport,
{
  id: string;
  setNodes: React.Dispatch<
    React.SetStateAction<Node<BaseNodeData | PyCodeEditorNodeData>[]>
  >;
  // onRunViewport?: (srcId: string) => void;
}) {
  const pos = { x: 150, y: 150 };

  const newNode: Node<PyCodeEditorNodeData> = {
    id,
    type: "pyCodeEditorNode",
    position: pos,
    width: 400,
    // height: 300,
    data: {},
  };

  setNodes((nds) => nds.concat(newNode));
}
