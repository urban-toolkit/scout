// src/App.tsx
import {
  ReactFlow,
  ReactFlowProvider,
  useNodesState,
  useEdgesState,
  Controls,
  useReactFlow,
  addEdge,
  MarkerType,
  type DefaultEdgeOptions,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useCallback, useRef, useState } from "react";

import { nodeTypes } from "./nodes"; // <-- { physicalLayerNode, viewNode, ... }
import type { Node, Connection, Edge } from "@xyflow/react";
import type { BaseNodeData } from "./nodes/BaseGrammarNode";

import { TEMPLATES, TEMPLATE_LABELS, TemplateKey } from "./templates";
import "./App.css";
// import { ViewNodeData } from "./nodes/ViewNode";
import type { ViewportNodeData } from "./nodes/ViewportNode";
import type { WidgetViewNodeData } from "./nodes/WidgetViewNode";
import type { PyCodeEditorNodeData } from "./nodes/PyCodeEditorNode";
import { TransformationNodeData } from "./nodes/TransformationNode";
import { ComparisonViewNodeData } from "./nodes/ComparisonViewNode";

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
    Node<
      | BaseNodeData
      | ViewportNodeData
      | PyCodeEditorNodeData
      | WidgetViewNodeData
      | ComparisonViewNodeData
    >
  >([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const { getNode, getEdges } = useReactFlow();

  const pushPhysicalToViews = useCallback(
    (srcId: string, trgId?: string) => {
      // Don't think we need to pass anything from physical layer to view for now!!
      // Later might need to pass some metadata? or will remove this function!
      const src = getNode(srcId);
      if (!src || src.type !== "physicalLayerNode") return;

      const val: any = (src.data as BaseNodeData).value;
      const pl_def = val?.physical_layer;
      if (!pl_def) return;

      const targetIds = trgId
        ? [trgId]
        : getEdges()
            .filter((e) => e.source === srcId)
            .map((e) => e.target!)
            .filter(Boolean);
    },
    [
      getNode,
      getEdges,
      // setNodes
    ]
  );

  const pushViewToViewports = useCallback(
    (srcId: string, trgId?: string) => {
      // console.log("Pushing view to viewports", srcId, trgId);
      const src = getNode(srcId);
      if (!src || src.type !== "viewNode") return;

      const value: any = (src.data as BaseNodeData).value;
      const viewSpec = value?.view;
      if (!Array.isArray(viewSpec)) return;

      const targetIds = trgId
        ? [trgId]
        : getEdges()
            .filter((e) => e.source === srcId)
            .map((e) => e.target!)
            .filter(Boolean);

      setNodes((nds) =>
        nds.map((n) => {
          if (!targetIds.includes(n.id)) return n;
          if (n.type !== "viewportNode") return n;
          return { ...n, data: { ...n.data, view: viewSpec } };
        })
      );
    },
    [getNode, getEdges, setNodes]
  );

  const pushInteractionToViewport = useCallback(
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
          if (!targetIds.includes(n.id) || n.type !== "viewportNode") return n;

          const existing = (n.data as ViewportNodeData).interactions ?? [];
          const already = existing.some((e) => e.id === i.id);
          const nextInteractions = already
            ? existing.map((e) => (e.id === i.id ? i : e))
            : [...existing, i];

          return {
            ...n,
            data: {
              ...n.data,
              interactions: nextInteractions,
            } as ViewportNodeData,
          };
        })
      );
    },
    [getNode, getEdges, setNodes]
  );

  const pushWidgetDefToWidgetView = useCallback(
    (srcId: string, trgId?: string) => {
      const src = getNode(srcId);
      if (!src || src.type !== "widgetDefNode") return;

      const val: any = (src.data as BaseNodeData).value;
      const wDef = val?.widget;

      if (!wDef) return;

      const targetIds = trgId
        ? [trgId]
        : getEdges()
            .filter((e) => e.source === srcId)
            .map((e) => e.target!)
            .filter(Boolean);

      const uuid = crypto.randomUUID();

      setNodes((nds) =>
        nds.map((n) => {
          if (!targetIds.includes(n.id) || n.type !== "widgetViewNode")
            return n;
          return {
            ...n,
            data: {
              ...n.data,
              widget: wDef,
              pushToken: uuid, // 👈 always changes on push
            } as WidgetViewNodeData,
          };
        })
      );
    },
    [getNode, getEdges, setNodes]
  );

  const pushComparisonDefToComparisonView = useCallback(
    (srcId: string, trgId?: string) => {
      const src = getNode(srcId);
      if (!src || src.type !== "comparisonDefNode") return;

      const val: any = (src.data as BaseNodeData).value;
      const cDef = val?.comparison;

      if (!cDef) return;

      const targetIds = trgId
        ? [trgId]
        : getEdges()
            .filter((e) => e.source === srcId)
            .map((e) => e.target!)
            .filter(Boolean);

      // After defining comparisonViewNode

      setNodes((nds) =>
        nds.map((n) => {
          if (!targetIds.includes(n.id) || n.type !== "comparisonViewNode")
            return n;
          return {
            ...n,
            data: {
              ...n.data,
              comparison: cDef,
            } as ComparisonViewNodeData,
          };
        })
      );
    },
    [getNode, getEdges, setNodes]
  );

  const pushWidgetViewToPyCodeEditorNode = useCallback(
    (srcId: string, trgId?: string) => {
      const src = getNode(srcId);
      if (!src || src.type !== "widgetViewNode") return;
      const val: WidgetViewNodeData = src.data;

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
          const already = existing.some((e) => e.id === val.output?.id);
          const nextWidgetOutputs = already
            ? existing.map((e) =>
                e.id === val.output?.id
                  ? {
                      id: val.output?.id,
                      variable: val.output.variable,
                      value: val.output.value,
                    }
                  : e
              )
            : [
                ...existing,
                {
                  id: val.output?.id,
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
        })
      );
    },
    [getNode, getEdges, setNodes]
  );

  const pushViewportToTransformation = useCallback(
    (srcId: string, trgId?: string) => {
      // Don't think we need to pass anything from viewport to transformation for now!!
      // Later might need to pass some metadata? or will remove this function!
      const src = getNode(srcId);
      if (!src || src.type !== "viewportNode") return;

      const val: any = src.data as ViewportNodeData;

      const targetIds = trgId
        ? [trgId]
        : getEdges()
            .filter((e) => e.source === srcId)
            .map((e) => e.target!)
            .filter(Boolean);
    },
    [
      getNode,
      getEdges,
      // setNodes
    ]
  );

  const handleCloseWidgetView = useCallback(
    (nodeId: string) => {
      const n = getNode(nodeId);
      if (!n || n.type !== "widgetViewNode") return;

      const widgetOutputId = (n.data as WidgetViewNodeData).output?.id;
      const curEdges = getEdges();

      // All targets currently connected FROM this view node
      const targetIds = curEdges
        .filter((e) => e.source === nodeId)
        .map((e) => e.target);

      setNodes((nds) =>
        nds
          .map((nn) => {
            if (nn.type !== "pyCodeEditorNode" || !targetIds.includes(nn.id))
              return nn;

            const pyd = nn.data as PyCodeEditorNodeData;
            const existing = pyd.widgetOutputs ?? [];

            const nextOutputs = widgetOutputId
              ? existing.filter((w) => w.id !== widgetOutputId)
              : existing;

            const nextData: PyCodeEditorNodeData = {
              ...pyd,
              widgetOutputs: nextOutputs.length ? nextOutputs : undefined,
            };

            return { ...nn, data: nextData };
          })
          .filter((nn) => nn.id !== nodeId)
      );

      setEdges((eds) =>
        eds.filter((e) => e.source !== nodeId && e.target !== nodeId)
      );
    },
    [getNode, getEdges, setNodes, setEdges]
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
        onRunPhysical: pushPhysicalToViews,
        onRunView: pushViewToViewports,
        onRunInteraction: pushInteractionToViewport,
        onRunWidgetDef: pushWidgetDefToWidgetView,
        onRunComparisonDef: pushComparisonDefToComparisonView,
      });
    },
    [
      setNodes,
      getNode,
      pushPhysicalToViews,
      pushViewToViewports,
      pushInteractionToViewport,
      pushWidgetDefToWidgetView,
      pushComparisonDefToComparisonView,
    ]
  );

  const addViewport = useCallback(() => {
    const nextId = `viewport-${idCounter.current++}`;
    createViewportNode({
      id: nextId,
      setNodes,
      onRunViewport: pushViewportToTransformation,
    });
  }, [setNodes, pushViewportToTransformation]);

  const addPyCodeEditorNode = useCallback(() => {
    const nextId = `pyCodeEditor-${idCounter.current++}`;
    createPyCodeEditorNode({
      id: nextId,
      setNodes,
      // onRunViewport: pushViewportToTransformation,
    });
  }, [setNodes]);

  const addWidgetViewNode = useCallback(() => {
    const nextId = `widgetView-${idCounter.current++}`;
    createWidgetViewNode({
      id: nextId,
      setNodes,
      onRunWidgetView: pushWidgetViewToPyCodeEditorNode,
      onCloseWidgetView: handleCloseWidgetView,
    });
  }, [setNodes, pushWidgetViewToPyCodeEditorNode, handleCloseWidgetView]);

  const addComparisonViewNode = useCallback(() => {
    const nextId = `comparisonView-${idCounter.current++}`;
    createComparisonViewNode({
      id: nextId,
      setNodes,
      // onRunComparisonView: ...
    });
  }, [setNodes]);

  // --- allow only physicalLayerNode -> viewNode
  const allow = useCallback(
    (conn: Connection) => {
      if (!conn.source || !conn.target) return false;
      const src = getNode(conn.source);
      const trg = getNode(conn.target);
      if (!src || !trg) return false;
      const physToView =
        src.type === "physicalLayerNode" && trg.type === "viewNode";

      const physToViewPort =
        src.type === "physicalLayerNode" && trg.type === "viewportNode";

      const viewToViewport =
        src.type === "viewNode" && trg.type === "viewportNode";
      const interactionToViewport =
        src.type === "interactionNode" && trg.type === "viewportNode";

      // This is not required, as instead of using transformation node, we are using code instead.
      // Maybe later will remove this condition block
      const viewportToTransformation =
        src.type === "viewportNode" && trg.type === "transformationNode";

      // Instead lets connect viewport to pyCodeEditorNode
      const viewportToPyCodeEditor =
        src.type === "viewportNode" && trg.type === "pyCodeEditorNode";

      const viewportToViewport =
        src.type === "viewportNode" && trg.type === "viewportNode";

      const transformationToPyCodeEditor =
        src.type === "transformationNode" && trg.type === "pyCodeEditorNode";
      const PyCodeEditorToView =
        src.type === "pyCodeEditorNode" && trg.type === "viewNode";

      const pyCodeEditorToPyCodeEditor =
        src.type === "pyCodeEditorNode" && trg.type === "pyCodeEditorNode";

      const pyCodeEditorToViewport =
        src.type === "pyCodeEditorNode" && trg.type === "viewportNode";

      const pyCodeEditorToComparisonDef =
        src.type === "pyCodeEditorNode" && trg.type === "comparisonDefNode";

      const widgetDefToWidgetView =
        src.type === "widgetDefNode" && trg.type === "widgetViewNode";

      const widgetViewToPyCodeEditor =
        src.type === "widgetViewNode" && trg.type === "pyCodeEditorNode";

      // After defining comparisonViewNode
      const comparisonDefToComparisonView =
        src.type === "comparisonDefNode" && trg.type === "comparisonViewNode";

      const pyCodeToComparisonView =
        src.type === "pyCodeEditorNode" && trg.type === "comparisonViewNode";

      return (
        physToView ||
        physToViewPort ||
        viewToViewport ||
        interactionToViewport ||
        viewportToPyCodeEditor ||
        viewportToTransformation ||
        transformationToPyCodeEditor ||
        PyCodeEditorToView ||
        pyCodeEditorToPyCodeEditor ||
        widgetDefToWidgetView ||
        widgetViewToPyCodeEditor ||
        comparisonDefToComparisonView ||
        pyCodeEditorToViewport ||
        pyCodeEditorToComparisonDef ||
        viewportToViewport ||
        pyCodeToComparisonView
      );
    },
    [getNode]
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

      if (src.type === "physicalLayerNode" && trg.type === "viewNode") {
        pushPhysicalToViews(srcId, trgId);
        return;
      }

      if (src.type === "viewNode" && trg.type === "viewportNode") {
        pushViewToViewports(srcId, trgId);
        return;
      }

      if (src.type === "interactionNode" && trg.type === "viewportNode") {
        pushInteractionToViewport(srcId, trgId);
        return;
      }

      // This is not required, as instead of using transformation node, we are using code instead.
      // Maybe later will remove this if block
      if (src.type === "viewportNode" && trg.type === "transformationNode") {
        pushViewportToTransformation(srcId, trgId);
        return;
      }

      if (
        src.type === "transformationNode" &&
        trg.type === "pyCodeEditorNode"
      ) {
        // pushTransformationToPyCodeEditor(srcId, trgId);
        return;
      }

      if (src.type === "widgetDefNode" && trg.type === "widgetViewNode") {
        pushWidgetDefToWidgetView(srcId, trgId);
        return;
      }

      if (src.type === "widgetViewNode" && trg.type === "pyCodeEditorNode") {
        pushWidgetViewToPyCodeEditorNode(srcId, trgId);
        return;
      }

      if (
        src.type === "comparisonDefNode" &&
        trg.type === "comparisonViewNode"
      ) {
        pushComparisonDefToComparisonView(srcId, trgId);
        return;
      }
    },
    [
      allow,
      getNode,
      setEdges,
      pushPhysicalToViews,
      pushViewToViewports,
      pushInteractionToViewport,
      pushViewportToTransformation,
      pushWidgetDefToWidgetView,
      pushWidgetViewToPyCodeEditorNode,
      pushComparisonDefToComparisonView,
    ]
  );

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
      >
        {/* <Background /> */}
        <Controls position="bottom-right" />
        <Toolbar
          onAdd={addNode}
          onAddViewport={addViewport}
          onAddPyCodeEditor={addPyCodeEditorNode}
          onAddWidgetView={addWidgetViewNode}
          onAddComparisonView={addComparisonViewNode}
        />
      </ReactFlow>
    </div>
  );
}

function Toolbar({
  onAdd,
  onAddViewport,
  onAddPyCodeEditor,
  onAddWidgetView,
  onAddComparisonView,
}: {
  onAdd: (tpl: TemplateKey) => void;
  onAddViewport: () => void;
  onAddPyCodeEditor: () => void;
  onAddWidgetView: () => void;
  onAddComparisonView: () => void;
}) {
  const { screenToFlowPosition } = useReactFlow();
  const [open, setOpen] = useState(false);

  const getDropPosition = useCallback(() => {
    return screenToFlowPosition({
      x: window.innerWidth / 2,
      y: window.innerHeight / 2,
    });
  }, [screenToFlowPosition]);

  const handleChoose = useCallback(
    (tpl: TemplateKey) => {
      (window as any)._desiredGrammarPos = getDropPosition();
      onAdd(tpl);
      setOpen(false);
    },
    [getDropPosition, onAdd]
  );

  const handleAddViewport = useCallback(() => {
    (window as any)._desiredGrammarPos = getDropPosition();
    onAddViewport();
  }, [getDropPosition, onAddViewport]);

  const handleAddPyCodeEditor = useCallback(() => {
    (window as any)._desiredGrammarPos = getDropPosition();
    onAddPyCodeEditor();
  }, [getDropPosition, onAddPyCodeEditor]);

  const handleAddWidgetView = useCallback(() => {
    (window as any)._desiredGrammarPos = getDropPosition();
    onAddWidgetView();
  }, [getDropPosition, onAddWidgetView]);

  const handleAddComparisonView = useCallback(() => {
    (window as any)._desiredGrammarPos = getDropPosition();
    onAddComparisonView();
  }, [getDropPosition, onAddComparisonView]);

  return (
    <div className="toolbar">
      <div className="toolbar__dropdown">
        <button
          onClick={() => setOpen((v) => !v)}
          className="toolbar__btn"
          aria-haspopup="menu"
          aria-expanded={open}
        >
          ➕ Grammar
        </button>

        {open && (
          <div role="menu" className="menu">
            <div className="menu__title">Select template</div>
            <div className="menu__divider" />
            {(Object.keys(TEMPLATES) as TemplateKey[]).map((key) => (
              <button
                key={key}
                role="menuitem"
                onClick={() => handleChoose(key)}
                className="menu__item"
              >
                {TEMPLATE_LABELS[key]}
              </button>
            ))}
          </div>
        )}
      </div>

      <button onClick={handleAddPyCodeEditor} className="toolbar__btn">
        ➕ Code
      </button>

      <button onClick={handleAddViewport} className="toolbar__btn">
        ➕ Viewport
      </button>

      <button onClick={handleAddWidgetView} className="toolbar__btn">
        ➕ Widget
      </button>

      <button onClick={handleAddComparisonView} className="toolbar__btn">
        ➕ Comparison
      </button>
    </div>
  );
}

// Map template key -> node type key from ./nodes
const kindToType: Record<TemplateKey, keyof typeof nodeTypes> = {
  physical_layer: "physicalLayerNode",
  view: "viewNode",
  interaction: "interactionNode",
  // transformation: "transformationNode",
  widget_def: "widgetDefNode",
  comparison_def: "comparisonDefNode",
};

function createGrammarNode({
  id,
  setNodes,
  template,
  getNode,
  onRunPhysical,
  onRunView,
  onRunInteraction,
  onRunWidgetDef,
  onRunComparisonDef,
}: // onRunWidgetView
{
  id: string;
  setNodes: React.Dispatch<
    React.SetStateAction<
      Node<
        | BaseNodeData
        | ViewportNodeData
        | PyCodeEditorNodeData
        | WidgetViewNodeData
      >[]
    >
  >;
  template: TemplateKey;
  getNode: (id: string) => Node | undefined;
  onRunPhysical: (srcId: string) => void;
  onRunView: (srcId: string) => void;
  onRunInteraction: (srcId: string) => void;
  onRunWidgetDef: (srcId: string) => void;
  onRunComparisonDef: (srcId: string) => void;
}) {
  const pos = (window as any)._desiredGrammarPos ?? { x: 100, y: 100 };
  const type = kindToType[template];

  const newNode: Node<BaseNodeData> = {
    id,
    type,
    position: pos,
    data: {
      // title: "Grammar",
      value: TEMPLATES[template] ?? {},
      onChange: (val, targetId) => {
        setNodes((nds) =>
          nds.map((n) =>
            n.id === targetId ? { ...n, data: { ...n.data, value: val } } : n
          )
        );
      },
      // Each node type decides how to "run" itself
      onRun: (nodeId) => {
        const node = getNode(nodeId);
        if (!node) return;
        if (node.type === "physicalLayerNode") {
          onRunPhysical(nodeId);
        } else if (node.type === "viewNode") {
          onRunView(nodeId);
        } else if (node.type === "interactionNode") {
          onRunInteraction(nodeId);
        } else if (node.type === "transformationNode") {
          const data = node.data as TransformationNodeData;
          console.log("Transformation node data:", data);
        } else if (node.type === "widgetDefNode") {
          onRunWidgetDef(nodeId);
        } else if (node.type === "comparisonDefNode") {
          onRunComparisonDef(nodeId);
        }
      },
    },
  };

  setNodes((nds) => nds.concat(newNode));
}

function createViewportNode({
  id,
  setNodes,
  data,
  onRunViewport,
}: {
  id: string;
  setNodes: React.Dispatch<
    React.SetStateAction<
      Node<
        | BaseNodeData
        | ViewportNodeData
        | PyCodeEditorNodeData
        | WidgetViewNodeData
      >[]
    >
  >;
  data?: { center?: [number, number]; zoom?: number };
  onRunViewport?: (srcId: string) => void;
}) {
  const pos = { x: 100, y: 100 };

  const newNode: Node<ViewportNodeData> = {
    id,
    type: "viewportNode",
    position: pos,
    width: 400,
    height: 400,
    data: {
      center: data?.center ?? [41.881, -87.63],
      onRun: onRunViewport
        ? (srcId: string) => onRunViewport(srcId)
        : undefined,
    },
  };

  setNodes((nds) => nds.concat(newNode));
}

function createWidgetViewNode({
  id,
  setNodes,
  onRunWidgetView,
  onCloseWidgetView,
}: // onRunWidgetView,
{
  id: string;
  setNodes: React.Dispatch<
    React.SetStateAction<
      Node<
        | BaseNodeData
        | ViewportNodeData
        | PyCodeEditorNodeData
        | WidgetViewNodeData
      >[]
    >
  >;
  onRunWidgetView?: (srcId: string) => void;
  onCloseWidgetView?: (nodeId: string) => void;
}) {
  const pos = { x: 150, y: 150 };
  const newNode: Node<WidgetViewNodeData> = {
    id,
    type: "widgetViewNode",
    position: pos,
    width: 400,
    height: 300,
    data: {
      // onClose: onCloseWidgetView
      // onRun: onRunWidgetView
      onRun: onRunWidgetView
        ? (srcId: string) => onRunWidgetView(srcId)
        : undefined,
      onClose: onCloseWidgetView
        ? (nodeId: string) => onCloseWidgetView(nodeId)
        : undefined,
    },
  };
  setNodes((nds) => nds.concat(newNode));
}

function createComparisonViewNode({
  id,
  setNodes,
}: {
  id: string;
  setNodes: React.Dispatch<
    React.SetStateAction<
      Node<
        | BaseNodeData
        | ViewportNodeData
        | PyCodeEditorNodeData
        | WidgetViewNodeData
        | ComparisonViewNodeData
      >[]
    >
  >;
}) {
  const pos = { x: 150, y: 150 };

  const newNode: Node<ComparisonViewNodeData> = {
    id,
    type: "comparisonViewNode",
    position: pos,
    width: 400,
    height: 300,
    data: {
      // Add any necessary data properties here
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
    React.SetStateAction<
      Node<BaseNodeData | PyCodeEditorNodeData | WidgetViewNodeData>[]
    >
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
