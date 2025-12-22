import { memo, useCallback, useState } from "react";
import type { NodeProps, Node } from "@xyflow/react";
import { useReactFlow, Handle, Position, NodeResizer } from "@xyflow/react";
import BaseGrammarNode, {
  BaseNodeData,
} from "../../node-components/BaseGrammar";
import schema from "../../schemas/view.json";
import type { ViewportNodeData } from "./ViewportNode";

import "../../node-components/BaseGrammar.css";

import expandPng from "../../assets/expand.png";
import restartPng from "../../assets/restart.png";

export type ViewNodeData = BaseNodeData;

export type ViewNode = Node<ViewNodeData, "viewNode">;

const NODE_MIN_WIDTH = 300;
const NODE_MIN_HEIGHT = 180;
const NODE_MINIMIZED_WIDTH = 150;
const NODE_MINIMIZED_HEIGHT = 48;

const ViewNode = memo(function ViewNode(props: NodeProps<ViewNode>) {
  const { id, data, selected } = props;
  const { getNode, getEdges, setNodes, setEdges } = useReactFlow();
  const rf = useReactFlow();
  const [minimized, setMinimized] = useState(false);

  const onCloseViewNode = useCallback(
    (nodeId: string) => {
      const n = getNode(nodeId);
      if (!n || n.type !== "viewNode") return;

      const curEdges = getEdges();

      // All targets currently connected FROM this view node
      const targetIds = curEdges
        .filter((e) => e.source === nodeId)
        .map((e) => e.target);

      // 1) Clear view on connected viewport nodes
      setNodes((nds) =>
        nds
          .map((nn) => {
            if (nn.type !== "viewportNode" || !targetIds.includes(nn.id))
              return nn;

            const vd = nn.data as ViewportNodeData;
            const nextData: ViewportNodeData = {
              ...vd,
              view: undefined,
            };

            return { ...nn, data: nextData };
          })
          // 2) Remove this view node itself
          .filter((nn) => nn.id !== nodeId)
      );

      // 3) Remove all edges touching the closed view node
      setEdges((eds) =>
        eds.filter((e) => e.source !== nodeId && e.target !== nodeId)
      );
    },
    [getNode, getEdges, setNodes, setEdges]
  );

  const handleToggleMinimize = useCallback(() => {
    setMinimized((prev) => {
      const next = !prev;

      // Resize node
      rf.setNodes((nodes) =>
        nodes.map((n) => {
          if (n.id !== id) return n;

          if (next) {
            // going to minimized
            return {
              ...n,
              width: NODE_MINIMIZED_WIDTH,
              height: NODE_MINIMIZED_HEIGHT,
            };
          } else {
            // restoring
            const nextWidth =
              n.width && n.width > NODE_MIN_WIDTH ? n.width : NODE_MIN_WIDTH;
            const nextHeight =
              n.height && n.height > NODE_MIN_HEIGHT
                ? n.height
                : NODE_MIN_HEIGHT;

            return {
              ...n,
              width: nextWidth,
              height: nextHeight,
            };
          }
        })
      );

      // Hide/show edges
      setEdges((eds) =>
        eds.map((e) =>
          e.source === id || e.target === id ? { ...e, hidden: next } : e
        )
      );

      return next;
    });
  }, [id, rf, setEdges]);

  const handleRun = useCallback(() => {
    if (data?.onRun) {
      return data.onRun(id);
    }
  }, [data, id]);

  return (
    <>
      {!minimized ? (
        <BaseGrammarNode
          id={id}
          selected={selected}
          data={{
            ...data,
            title: data.title ?? "View",
            schema,
            pickInner: (v) => (v as any)?.view,
            onClose: onCloseViewNode,
            onToggleMinimize: handleToggleMinimize,
          }}
        />
      ) : (
        <div className="gnode gnode--minimized">
          <NodeResizer
            minWidth={minimized ? NODE_MINIMIZED_WIDTH : NODE_MIN_WIDTH}
            maxWidth={Infinity}
            minHeight={minimized ? NODE_MINIMIZED_HEIGHT : NODE_MIN_HEIGHT}
            maxHeight={minimized ? NODE_MINIMIZED_HEIGHT : Infinity}
          />
          <div className="gnode__minimized">
            {/* Big fetch button */}
            <button type="button" className="gnode__minimizedNodeTtitleBtn">
              {data.title ?? "View"}
            </button>
            {/* Floating restore (top-left) */}
            <button
              type="button"
              className="gnode__minimizedRestoreCircle_1 gnode__minimizedRestoreCircle--topLeft"
              onClick={handleToggleMinimize}
            >
              <img src={expandPng} alt="Restore" />
            </button>

            {/* Floating fetch/update (bottom-right) */}
            <button
              type="button"
              className="gnode__minimizedRestoreCircle_2 gnode__minimizedRestoreCircle--bottomRight"
              onClick={handleRun}
            >
              <img src={restartPng} alt="Fetch / update" />
            </button>
          </div>
        </div>
      )}
      <Handle
        type="target"
        position={Position.Left}
        id="view-in"
        className={`gnode__handle gnode__handle--left ${
          minimized ? "gnode__handle--hidden" : ""
        }`}
      />

      {/* Source: to ViewportNode */}
      <Handle
        type="source"
        position={Position.Right}
        id="view-out"
        className={`gnode__handle gnode__handle--right ${
          minimized ? "gnode__handle--hidden" : ""
        }`}
      />
    </>
  );
});

export default ViewNode;
