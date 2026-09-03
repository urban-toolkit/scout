// src/App.tsx
import {
  ReactFlow,
  ReactFlowProvider,
  useNodesState,
  useEdgesState,
  useReactFlow,
  addEdge,
  type DefaultEdgeOptions,
  type EdgeTypes,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useCallback, useEffect, useRef, useState } from "react";

import { nodeTypes } from "./nodes"; // <-- { dataLayerNode, viewNode, ... }
import type { Node, Connection, Edge } from "@xyflow/react";
import type { BaseNodeData } from "./node-components/BaseGrammar";

import { TEMPLATES, TemplateKey } from "./templates";
import "./App.css";
import type { PyCodeEditorNodeData } from "./nodes/computation/PyCodeEditorNode";

import {
  loadShadowComparisonExample,
  loadFloodingComparisonExample,
  loadWeatherRoutingComparisonExample,
} from "./examples/exampleWorkflows";
import ChatWidget from "./components/ai/ChatWidget";
import NodeRail, { NODE_DRAG_MIME, PY_CODE_DRAG_VALUE } from "./components/NodeRail";
import Toolbar from "./components/Toolbar";
import ChartStudioPage from "./pages/ChartStudioPage";
import ChartGalleryPage from "./pages/ChartGalleryPage";
import ChartExamplePage from "./pages/ChartExamplePage";
import { pushWidgetOutputToConnectedCode } from "./utils/widgetPropagation";
import ArrowAboveEdge from "./edges/ArrowAboveEdge";

// Custom edge: draws the line behind nodes but the arrowhead above them -
// see ArrowAboveEdge.tsx for why the default marker-based arrow can't do both.
const edgeTypes: EdgeTypes = { default: ArrowAboveEdge };

const defaultEdgeOptions: DefaultEdgeOptions = {
  style: {
    stroke: "#888",
    strokeWidth: 2, // optional but improves visibility
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

type AppRoute =
  | { kind: "workflow"; route: WorkflowRoute }
  | { kind: "chart-studio"; name?: string }
  | { kind: "chart-gallery" }
  | { kind: "chart-example"; name: string }
  | null;

function getAppRouteFromPath(pathname = window.location.pathname): AppRoute {
  const workflowRoute = getWorkflowRouteFromPath(pathname);
  if (workflowRoute) return { kind: "workflow", route: workflowRoute };

  const base = getAppBasePath();
  const relativePath = pathname.startsWith(base)
    ? pathname.slice(base.length)
    : pathname.startsWith("/")
      ? pathname.slice(1)
      : pathname;
  const route = relativePath.replace(/\/+$/, "");

  if (route === "chart-studio") return { kind: "chart-studio" };
  if (route.startsWith("chart-studio/")) {
    const name = decodeURIComponent(route.slice("chart-studio/".length));
    if (name) return { kind: "chart-studio", name };
  }
  if (route === "chart-gallery") return { kind: "chart-gallery" };
  if (route.startsWith("chart-gallery/")) {
    const name = decodeURIComponent(route.slice("chart-gallery/".length));
    if (name) return { kind: "chart-example", name };
  }

  return null;
}

function getChartStudioPath(name?: string) {
  const base = `${getAppBasePath()}chart-studio`;
  return name ? `${base}/${encodeURIComponent(name)}` : base;
}

function getChartGalleryPath() {
  return `${getAppBasePath()}chart-gallery`;
}

function getChartExamplePath(name: string) {
  return `${getAppBasePath()}chart-gallery/${encodeURIComponent(name)}`;
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
  const { getNode, getNodes, getEdges, fitView, screenToFlowPosition } = useReactFlow();
  const [page, setPage] = useState<
    "canvas" | "chart-studio" | "chart-gallery" | "chart-example"
  >("canvas");
  const [chartExampleName, setChartExampleName] = useState<string | null>(null);
  const [chartStudioInitialName, setChartStudioInitialName] = useState<string | null>(null);

  // const dumpWorkflow = useCallback(() => {
  //   const nodes = getNodes();
  //   const edges = getEdges();

  //   console.log("NODES");
  //   console.log(JSON.stringify(nodes, null, 2));

  //   console.log("EDGES");
  //   console.log(JSON.stringify(edges, null, 2));
  // }, [getNodes, getEdges]);

  const pushInteractionToView = useCallback(
    (srcId: string, trgId?: string): boolean => {
      const src = getNode(srcId);
      if (!src || src.type !== "interactionNode") return false;

      const val: any = (src.data as BaseNodeData).value;
      const i = val?.interaction;
      if (!i) return false;

      const targetIds = trgId
        ? [trgId]
        : getEdges()
            .filter((e) => e.source === srcId)
            .map((e) => e.target!)
            .filter(Boolean);

      const viewTargetIds = targetIds.filter(
        (tid) => getNode(tid)?.type === "viewNode",
      );
      if (!viewTargetIds.length) return false;

      setNodes((nds) =>
        nds.map((n) => {
          if (!viewTargetIds.includes(n.id)) return n;

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

      return true;
    },
    [getNode, getEdges, setNodes],
  );

  const pushWidgetToPyCodeEditorNode = useCallback(
    (srcId: string, trgId?: string): boolean => {
      const updated = pushWidgetOutputToConnectedCode(
        srcId,
        getNodes(),
        getEdges(),
        setNodes,
        trgId,
      );
      return updated.length > 0;
    },
    [getNodes, getEdges, setNodes],
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

  // Dragging a NodeRail icon onto the canvas: the drop position (rather than
  // NodeRail's own viewport-center fallback) becomes _desiredGrammarPos, so
  // the same createGrammarNode/createPyCodeEditorNode handoff used by
  // click-to-add places the node exactly where it was dropped.
  const handleCanvasDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    if (!e.dataTransfer.types.includes(NODE_DRAG_MIME)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  }, []);

  const handleCanvasDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      const dragValue = e.dataTransfer.getData(NODE_DRAG_MIME);
      if (!dragValue) return;
      e.preventDefault();

      (window as any)._desiredGrammarPos = screenToFlowPosition({
        x: e.clientX,
        y: e.clientY,
      });

      if (dragValue === PY_CODE_DRAG_VALUE) {
        addPyCodeEditorNode();
      } else {
        addNode(dragValue as TemplateKey);
      }
    },
    [screenToFlowPosition, addNode, addPyCodeEditorNode],
  );

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
      setPage("canvas");
      loadWorkflowRoute(route);
    },
    [loadWorkflowRoute],
  );

  const navigateToChartStudio = useCallback((name?: string) => {
    const nextPath = getChartStudioPath(name);
    if (window.location.pathname !== nextPath) {
      window.history.pushState(null, "", nextPath);
    }
    setChartStudioInitialName(name ?? null);
    setPage("chart-studio");
  }, []);

  const navigateToChartGallery = useCallback(() => {
    const nextPath = getChartGalleryPath();
    if (window.location.pathname !== nextPath) {
      window.history.pushState(null, "", nextPath);
    }
    setPage("chart-gallery");
  }, []);

  const navigateToChartExample = useCallback((name: string) => {
    const nextPath = getChartExamplePath(name);
    if (window.location.pathname !== nextPath) {
      window.history.pushState(null, "", nextPath);
    }
    setChartExampleName(name);
    setPage("chart-example");
  }, []);

  const clearCanvasAndNavigateHome = useCallback(() => {
    const homePath = getAppBasePath();
    if (window.location.pathname !== homePath) {
      window.history.pushState(null, "", homePath);
    }
    setPage("canvas");
    setNodes([]);
    setEdges([]);
    idCounter.current = 1;
  }, [setEdges, setNodes]);

  useEffect(() => {
    const applyRoute = (appRoute: AppRoute) => {
      if (appRoute?.kind === "workflow") {
        setPage("canvas");
        loadWorkflowRoute(appRoute.route);
      } else if (appRoute?.kind === "chart-studio") {
        setChartStudioInitialName(appRoute.name ?? null);
        setPage("chart-studio");
      } else if (appRoute?.kind === "chart-gallery") {
        setPage("chart-gallery");
      } else if (appRoute?.kind === "chart-example") {
        setChartExampleName(appRoute.name);
        setPage("chart-example");
      } else {
        // No recognized route (e.g. bare "/") - always show the canvas.
        // Node/edge state is untouched here (only clearCanvasAndNavigateHome
        // resets that), so this just makes sure Chart Studio doesn't stay
        // on screen after navigating back past it.
        setPage("canvas");
      }
    };

    applyRoute(getAppRouteFromPath());

    const handlePopState = () => applyRoute(getAppRouteFromPath());

    window.addEventListener("popstate", handlePopState);
    return () => {
      window.removeEventListener("popstate", handlePopState);
    };
  }, [loadWorkflowRoute]);

  // A display:none -> visible round-trip on a ResizeObserver-driven library
  // like @xyflow/react can leave the viewport stale until something forces a
  // recompute - re-fitView the same way example-loading already does.
  useEffect(() => {
    if (page === "canvas") {
      requestAnimationFrame(() => fitView({ padding: 0.15 }));
    }
  }, [page, fitView]);

  return (
    <div className="app">
      <Toolbar
        onNavigateHome={clearCanvasAndNavigateHome}
        onLoadShadowWorkflow={() => navigateToWorkflowRoute("shadow")}
        onLoadFloodingWorkflow={() => navigateToWorkflowRoute("flooding")}
        onLoadWeatherRoutingWorkflow={() =>
          navigateToWorkflowRoute("routing")
        }
        onOpenChartGallery={navigateToChartGallery}
        onOpenChartStudio={() => navigateToChartStudio()}
      />
      <div
        className="canvas-wrap"
        style={page !== "canvas" ? { display: "none" } : undefined}
        onDragOver={handleCanvasDragOver}
        onDrop={handleCanvasDrop}
      >
        <ReactFlow
          className="canvas"
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
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
          />
        </ReactFlow>
        {/* <button onClick={dumpWorkflow} className="toolbar__btn__dump">
          Dump
        </button> */}
      </div>
      {page === "chart-studio" && (
        <div className="page-wrap">
          <ChartStudioPage initialChartName={chartStudioInitialName ?? undefined} />
        </div>
      )}
      {page === "chart-gallery" && (
        <div className="page-wrap">
          <ChartGalleryPage
            onSelectChart={navigateToChartExample}
            onCreateNewChart={() => navigateToChartStudio()}
          />
        </div>
      )}
      {page === "chart-example" && chartExampleName && (
        <div className="page-wrap">
          <ChartExamplePage
            name={chartExampleName}
            onBack={navigateToChartGallery}
            onEditInStudio={navigateToChartStudio}
          />
        </div>
      )}
      {/* Rendered outside canvas-wrap (which hides via display:none on the
          Chart Studio page) so the chat widget stays available on every
          page, not just the canvas. */}
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
  onRunInteraction: (srcId: string) => boolean;
  onRunWidget: (srcId: string) => boolean;
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
          return onRunInteraction(nodeId);
        } else if (node.type === "widgetNode") {
          return onRunWidget(nodeId);
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
  const pos = (window as any)._desiredGrammarPos ?? { x: 150, y: 150 };

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
