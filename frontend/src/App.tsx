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

import { TEMPLATES, TEMPLATE_NODE_TYPE, TemplateKey } from "./templates";
import "./App.css";
import type { PyCodeEditorNodeData } from "./nodes/computation/PyCodeEditorNode";

import { attachNodeBehaviors } from "./examples/workflowHelpers";
import ChatWidget from "./components/ai/ChatWidget";
import NodeRail, { NODE_DRAG_MIME, PY_CODE_DRAG_VALUE } from "./components/NodeRail";
import Toolbar from "./components/Toolbar";
import DataflowNameLabel from "./components/DataflowNameLabel";
import DataCatalogPanel from "./components/DataCatalogPanel";
import ChartStudioPage from "./pages/ChartStudioPage";
import ChartGalleryPage from "./pages/ChartGalleryPage";
import ChartExamplePage from "./pages/ChartExamplePage";
import DataflowsHomePage from "./pages/DataflowsHomePage";
import { getDataflow, renameDataflow, saveDataflow, deleteComputedDataset } from "./utils/dataflows";
import type { CatalogDataset } from "./utils/dataCatalog";
import type { ProjectDataset } from "./utils/projectDatasets";
import { decodeDatasetDrag, type DatasetDragPayload } from "./utils/datasetDrag";
import { pushWidgetOutputToConnectedCode } from "./utils/widgetPropagation";
import { pushInteractionToView as pushInteractionToViewShared } from "./utils/interactionPropagation";
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

function getAppBasePath() {
  const base = import.meta.env.BASE_URL ?? "/";
  return base === "/" ? "/" : base.endsWith("/") ? base : `${base}/`;
}

function getRelativeRoute(pathname: string): string {
  const base = getAppBasePath();
  const relativePath = pathname.startsWith(base)
    ? pathname.slice(base.length)
    : pathname.startsWith("/")
      ? pathname.slice(1)
      : pathname;
  return relativePath.replace(/\/+$/, "");
}

type AppRoute =
  | { kind: "home" }
  | { kind: "dataflow"; id: string }
  | { kind: "chart-studio"; name?: string }
  | { kind: "chart-gallery" }
  | { kind: "chart-example"; name: string };

function getAppRouteFromPath(pathname = window.location.pathname): AppRoute {
  const route = getRelativeRoute(pathname);

  if (route === "") return { kind: "home" };

  if (route.startsWith("dataflow/")) {
    const id = decodeURIComponent(route.slice("dataflow/".length));
    if (id) return { kind: "dataflow", id };
  }
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

  // Unrecognized path (or a bare "/") - the home page (Projects) is the
  // only safe fallback now that it's a distinct page rather than a blank
  // canvas.
  return { kind: "home" };
}

function getDataflowPath(id: string) {
  return `${getAppBasePath()}dataflow/${encodeURIComponent(id)}`;
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

// Ids are minted as `grammar-<n>` / `pyCodeEditor-<n>` off one shared
// counter (see addNode/addPyCodeEditorNode below) - resuming a saved
// dataflow needs that counter picked up from the highest suffix already in
// use, or a newly added node could collide with one that was loaded in.
const NODE_ID_RE = /^(?:grammar|pyCodeEditor)-(\d+)$/;

function nextIdCounterFromNodes(nodes: { id: string }[]): number {
  let max = 0;
  for (const n of nodes) {
    const match = NODE_ID_RE.exec(n.id);
    if (match) max = Math.max(max, parseInt(match[1], 10));
  }
  return max + 1;
}

export default function App() {
  return (
    <ReactFlowProvider>
      <AppShell />
    </ReactFlowProvider>
  );
}

function AppShell() {
  const idCounter = useRef(1);
  const [nodes, setNodes, onNodesChange] = useNodesState<
    Node<BaseNodeData | PyCodeEditorNodeData>
  >([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const { getNode, getNodes, getEdges, fitView, screenToFlowPosition } = useReactFlow();
  const [page, setPage] = useState<
    "home" | "canvas" | "chart-studio" | "chart-gallery" | "chart-example"
  >("home");
  const [chartExampleName, setChartExampleName] = useState<string | null>(null);
  const [chartStudioInitialName, setChartStudioInitialName] = useState<string | null>(null);
  // Data Catalog and the AI chat both dock in the same right-side slot, so
  // only one of them can be open at once.
  const [activeSidebar, setActiveSidebar] = useState<"data" | "chat" | null>(null);
  // The dataflow currently open on the canvas (from the /dataflow/:id URL),
  // and whether its saved nodes/edges have finished loading - autosave must
  // stay off until that's true, or it would immediately overwrite the
  // dataflow being loaded with the still-empty initial nodes/edges state.
  const [dataflowId, setDataflowId] = useState<string | null>(null);
  const [dataflowLoaded, setDataflowLoaded] = useState(false);
  const [dataflowName, setDataflowName] = useState<string | null>(null);
  const saveTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Datasets added from the Data Catalog sidebar - in-memory only for now
  // (not yet persisted with the dataflow), so it resets on navigation/reload.
  const [projectDatasets, setProjectDatasets] = useState<ProjectDataset[]>([]);

  const handleAddDatasetsToProject = useCallback((toAdd: CatalogDataset[]) => {
    setProjectDatasets((prev) => {
      const byId = new Map(prev.map((d) => [d.id, d]));
      for (const d of toAdd) byId.set(d.id, d);
      return [...byId.values()];
    });
  }, []);

  const handleRemoveDatasetsFromProject = useCallback(
    (ids: string[]) => {
      const idSet = new Set(ids);
      setProjectDatasets((prev) => prev.filter((d) => !idSet.has(d.id)));

      // A "computed" dataset is this dataflow's own derived output, not
      // shared catalog source data - removing it from the project should
      // actually delete its file (see deleteComputedDataset), so a
      // View/Interaction node still referencing it fails to resolve rather
      // than silently keeping the "removed" data alive.
      if (!dataflowId) return;
      for (const id of ids) {
        if (!id.startsWith("computed/")) continue;
        deleteComputedDataset(dataflowId, id).catch((err) =>
          console.error(`Failed to delete computed dataset "${id}"`, err),
        );
      }
    },
    [dataflowId],
  );

  // const dumpWorkflow = useCallback(() => {
  //   const nodes = getNodes();
  //   const edges = getEdges();

  //   console.log("NODES");
  //   console.log(JSON.stringify(nodes, null, 2));

  //   console.log("EDGES");
  //   console.log(JSON.stringify(edges, null, 2));
  // }, [getNodes, getEdges]);

  const pushInteractionToView = useCallback(
    (srcId: string, trgId?: string): boolean =>
      pushInteractionToViewShared(srcId, getNodes(), getEdges(), setNodes, trgId),
    [getNodes, getEdges, setNodes],
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
        onAddComputedDatasets: handleAddDatasetsToProject,
      });
    },
    [
      setNodes,
      getNode,
      pushInteractionToView,
      pushWidgetToPyCodeEditorNode,
      handleAddDatasetsToProject,
    ],
  );

  // Dragging a dataset (or a whole added folder) from NodeRail's "Datasets
  // in project" list onto the canvas - only OSM catalog data maps to a
  // data_layer definition right now, so anything else is a silent no-op.
  const addDataLayerNodeFromDrag = useCallback(
    (payload: DatasetDragPayload) => {
      const files = payload.kind === "file" ? [payload.dataset] : payload.files;
      const osmFiles = files.filter((f) => f.group === "osm");
      if (osmFiles.length === 0) return;

      const nextId = `grammar-${idCounter.current++}`;
      const dataLayerId = nextDataLayerId(getNodes());
      createGrammarNode({
        id: nextId,
        setNodes,
        template: "data_layer",
        getNode,
        onRunInteraction: pushInteractionToView,
        onRunWidget: pushWidgetToPyCodeEditorNode,
        onAddComputedDatasets: handleAddDatasetsToProject,
        valueOverride: buildOsmDataLayerValue(osmFiles, dataLayerId),
      });
    },
    [
      setNodes,
      getNode,
      getNodes,
      pushInteractionToView,
      pushWidgetToPyCodeEditorNode,
      handleAddDatasetsToProject,
    ],
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

      const datasetPayload = decodeDatasetDrag(dragValue);
      if (datasetPayload) {
        addDataLayerNodeFromDrag(datasetPayload);
        return;
      }

      if (dragValue === PY_CODE_DRAG_VALUE) {
        addPyCodeEditorNode();
      } else {
        addNode(dragValue as TemplateKey);
      }
    },
    [screenToFlowPosition, addNode, addPyCodeEditorNode, addDataLayerNodeFromDrag],
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

  const navigateToDataflow = useCallback((id: string) => {
    const nextPath = getDataflowPath(id);
    if (window.location.pathname !== nextPath) {
      window.history.pushState(null, "", nextPath);
    }
    setPage("canvas");
    setDataflowId(id);
  }, []);

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

  // Brand-logo click: leaves the current dataflow entirely and returns to
  // the Projects home page. Distinct from the NodeRail trash icon (see
  // clearCurrentCanvas below), which only wipes the dataflow you're still on.
  const navigateHome = useCallback(() => {
    const homePath = getAppBasePath();
    if (window.location.pathname !== homePath) {
      window.history.pushState(null, "", homePath);
    }
    setPage("home");
    setDataflowId(null);
    setDataflowLoaded(false);
    setDataflowName(null);
    setProjectDatasets([]);
    setNodes([]);
    setEdges([]);
    idCounter.current = 1;
  }, [setEdges, setNodes]);

  // NodeRail's "Clear canvas" trash icon - wipes the nodes/edges of the
  // dataflow currently open, without navigating away from it (the cleared
  // state autosaves like any other edit).
  const clearCurrentCanvas = useCallback(() => {
    setNodes([]);
    setEdges([]);
    idCounter.current = 1;
  }, [setEdges, setNodes]);

  // Toolbar's dataflow-name field - optimistic update with rollback if the
  // backend rename call fails, so a flaky request doesn't leave the
  // displayed name out of sync with what's actually saved.
  const handleRenameDataflow = useCallback(
    (name: string) => {
      if (!dataflowId) return;
      const previous = dataflowName;
      setDataflowName(name);
      renameDataflow(dataflowId, name).catch((e) => {
        console.error("Failed to rename dataflow", e);
        setDataflowName(previous);
      });
    },
    [dataflowId, dataflowName],
  );

  useEffect(() => {
    const applyRoute = (appRoute: AppRoute) => {
      if (appRoute.kind === "dataflow") {
        setPage("canvas");
        setDataflowId(appRoute.id);
      } else if (appRoute.kind === "chart-studio") {
        setChartStudioInitialName(appRoute.name ?? null);
        setPage("chart-studio");
      } else if (appRoute.kind === "chart-gallery") {
        setPage("chart-gallery");
      } else if (appRoute.kind === "chart-example") {
        setChartExampleName(appRoute.name);
        setPage("chart-example");
      } else {
        setPage("home");
      }
    };

    applyRoute(getAppRouteFromPath());

    const handlePopState = () => applyRoute(getAppRouteFromPath());

    window.addEventListener("popstate", handlePopState);
    return () => {
      window.removeEventListener("popstate", handlePopState);
    };
  }, []);

  // Loads the dataflow named by the URL whenever it changes - both on first
  // navigation to it and when the user picks a different one without a full
  // page reload (e.g. via the browser back/forward buttons).
  useEffect(() => {
    if (dataflowId === null) return;
    let cancelled = false;
    setDataflowLoaded(false);
    // Cleared immediately so a different dataflow's "added" datasets don't
    // flash on screen while this one's own (persisted) list is still
    // loading - the real value comes back from the record just below.
    setProjectDatasets([]);

    getDataflow(dataflowId)
      .then((record) => {
        if (cancelled) return;
        const hydrated = record.nodes.map((node) =>
          attachNodeBehaviors(
            node,
            setNodes,
            getNode,
            pushInteractionToView,
            pushWidgetToPyCodeEditorNode,
            handleAddDatasetsToProject,
          ),
        );
        setNodes(hydrated);
        setEdges(record.edges);
        setDataflowName(record.name);
        setProjectDatasets(record.projectDatasets ?? []);
        idCounter.current = nextIdCounterFromNodes(record.nodes);
        setDataflowLoaded(true);
        requestAnimationFrame(() => fitView({ padding: 0.15 }));
      })
      .catch((e) => {
        // Most likely a stale/invalid id in the URL (e.g. a bad link) -
        // bounce back to the Projects list rather than getting stuck.
        console.error("Failed to load dataflow", dataflowId, e);
        if (!cancelled) navigateHome();
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataflowId]);

  // Autosaves the open dataflow a short moment after nodes/edges stop
  // changing - there's no explicit Save action, so this is the only thing
  // that persists edits. Disabled until the initial load above completes,
  // so it can't race that load and overwrite the saved dataflow with the
  // still-empty state the canvas starts in.
  useEffect(() => {
    if (!dataflowId || !dataflowLoaded) return;

    if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
    saveTimeoutRef.current = setTimeout(() => {
      let payload: { nodes: typeof nodes; edges: typeof edges; projectDatasets: typeof projectDatasets };
      try {
        // Round-trips through JSON so the onChange/onRun functions
        // attachNodeBehaviors puts on each node's data are dropped before
        // sending (they can't be serialized, and shouldn't be persisted),
        // and so a genuinely non-serializable value fails here rather than
        // inside the fetch call.
        payload = JSON.parse(JSON.stringify({ nodes, edges, projectDatasets }));
      } catch (e) {
        console.error("Dataflow autosave: nodes/edges are not serializable", e);
        return;
      }
      saveDataflow(dataflowId, payload).catch((e) => {
        console.error("Dataflow autosave failed", e);
      });
    }, 800);

    return () => {
      if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
    };
  }, [nodes, edges, projectDatasets, dataflowId, dataflowLoaded]);

  // A display:none -> visible round-trip on a ResizeObserver-driven library
  // like @xyflow/react can leave the viewport stale until something forces a
  // recompute - re-fitView the same way example-loading already does.
  useEffect(() => {
    if (page === "canvas") {
      requestAnimationFrame(() => fitView({ padding: 0.15 }));
    }
  }, [page, fitView]);

  if (page === "home") {
    return (
      <div className="app">
        <Toolbar
          onNavigateHome={navigateHome}
          onOpenDataflows={navigateHome}
          onOpenChartGallery={navigateToChartGallery}
        />
        <div className="page-wrap">
          <DataflowsHomePage onOpenDataflow={navigateToDataflow} />
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <Toolbar
        onNavigateHome={navigateHome}
        onOpenDataflows={navigateHome}
        onOpenChartGallery={navigateToChartGallery}
        inDataflow
        onOpenChartStudio={() => navigateToChartStudio()}
        onOpenDataCatalog={() => setActiveSidebar("data")}
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
          {dataflowName != null && (
            <DataflowNameLabel name={dataflowName} onRename={handleRenameDataflow} />
          )}
          <NodeRail
            onAdd={addNode}
            onAddPyCodeEditor={addPyCodeEditorNode}
            onClear={clearCurrentCanvas}
            onOpenDataCatalog={() => setActiveSidebar("data")}
            projectDatasets={projectDatasets}
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
      <ChatWidget
        open={activeSidebar === "chat"}
        onOpenChange={(next) => setActiveSidebar(next ? "chat" : null)}
      />
      <DataCatalogPanel
        open={activeSidebar === "data"}
        onClose={() => setActiveSidebar(null)}
        projectDatasets={projectDatasets}
        onAddToProject={handleAddDatasetsToProject}
        onRemoveFromProject={handleRemoveDatasetsFromProject}
      />
    </div>
  );
}

// Known default bounding boxes for OSM regions in the catalog, used to
// prefill roi.value when dragging a dataset onto the canvas - matches the
// bbox already hand-authored as an example in templates.ts's
// dataLayerTemplate. Regions with no known default are left for the user
// to fill in themselves.
const OSM_REGION_DEFAULT_BBOX: Record<string, number[]> = {
  chicago: [-87.66, 41.86, -87.64, 41.88],
};

// Picks a data_layer id not already used by another data_layer node on the
// canvas - single letters first (matching the app's own convention, e.g.
// "A"), falling back to A2, A3, ... if every letter is taken.
function nextDataLayerId(nodes: Node[]): string {
  const used = new Set<string>();
  for (const n of nodes) {
    if (n.type !== "dataLayerNode") continue;
    const id = (n.data as any)?.value?.data_layer?.id;
    if (typeof id === "string") used.add(id);
  }
  for (let i = 0; i < 26; i++) {
    const letter = String.fromCharCode(65 + i);
    if (!used.has(letter)) return letter;
  }
  let n = 2;
  while (used.has(`A${n}`)) n++;
  return `A${n}`;
}

// Builds a data_layer value from one or more OSM catalog files dropped onto
// the canvas - dragging just "buildings" makes a single-feature layer,
// dragging a whole region folder (e.g. "chicago") combines every distinct
// feature found in it. Attribute selection is left for the user to define.
function buildOsmDataLayerValue(files: CatalogDataset[], id: string) {
  // Every file here shares one catalog path shape - "osm/<datafile>/<name>"
  // - so any of them tells us which region (datafile) this layer is for.
  const pathParts = files[0].id.split("/");
  const datafile = pathParts.length >= 2 ? pathParts[1] : pathParts[0];

  const seenFeatures = new Set<string>();
  const osm_features: Record<string, unknown>[] = [];
  for (const f of files) {
    const feature = f.name;
    if (seenFeatures.has(feature)) continue;
    seenFeatures.add(feature);
    osm_features.push({ feature, attributes: [] });
  }

  return {
    data_layer: {
      id,
      source: "osm",
      dtype: "physical",
      roi: {
        datafile,
        type: "bbox",
        value: OSM_REGION_DEFAULT_BBOX[datafile] ?? [],
      },
      osm_features,
    },
  };
}

function createGrammarNode({
  id,
  setNodes,
  template,
  getNode,
  onRunInteraction,
  onRunWidget,
  onAddComputedDatasets,
  valueOverride,
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
  // DataLayerNode only - see BaseNodeData.onAddComputedDatasets.
  onAddComputedDatasets?: (datasets: CatalogDataset[]) => void;
  // Used when a node is seeded from something other than the blank
  // template - e.g. dragging a dataset from the Data Catalog onto the
  // canvas (see addDataLayerNodeFromDrag) prefills a real data_layer value
  // instead of the generic placeholder.
  valueOverride?: any;
}) {
  const pos = (window as any)._desiredGrammarPos ?? { x: 100, y: 100 };
  const type = TEMPLATE_NODE_TYPE[template];

  const newNode: Node<BaseNodeData> = {
    id,
    type,
    position: pos,
    data: {
      value: valueOverride ?? TEMPLATES[template] ?? {},
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
      onAddComputedDatasets,
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
