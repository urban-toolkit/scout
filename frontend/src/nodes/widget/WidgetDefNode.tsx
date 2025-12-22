import { memo, useCallback, useState } from "react";
import type { NodeProps, Node } from "@xyflow/react";
import { Handle, Position, useReactFlow, NodeResizer } from "@xyflow/react";
import BaseGrammarNode, {
  BaseNodeData,
} from "../../node-components/BaseGrammar";
import schema from "../../schemas/widget.json";

import "../../node-components/BaseGrammar.css";

import expandPng from "../../assets/expand.png";
import restartPng from "../../assets/restart.png";

export type WidgetDefNodeData = BaseNodeData;

export type WidgetDefNode = Node<WidgetDefNodeData, "widgetDefNode">;

const NODE_MIN_WIDTH = 300;
const NODE_MIN_HEIGHT = 180;
const NODE_MINIMIZED_WIDTH = 150;
const NODE_MINIMIZED_HEIGHT = 48;

const WidgetDefNode = memo(function WidgetDefNode(
  props: NodeProps<WidgetDefNode>
) {
  const { id, data, selected } = props;
  const { getNode, getEdges, setNodes, setEdges } = useReactFlow();
  const rf = useReactFlow();
  const [minimized, setMinimized] = useState(false);

  const onCloseWidgetDefNode = useCallback(
    (nodeId: string) => {
      const n = getNode(nodeId);
      if (!n || n.type !== "widgetDefNode") return;

      const curEdges = getEdges();

      const wdValue = (n.data as BaseNodeData)?.value as any;
      const wdId: string | undefined = wdValue?.widget?.id;

      const targetIds = curEdges
        .filter((e) => e.source === nodeId)
        .map((e) => e.target);

      setNodes((nds) =>
        nds
          //   .map((nn) => {
          //     if (nn.type !== "viewportNode" || !targetIds.includes(nn.id)) {
          //       return nn;
          //     }

          //     const vpData = nn.data as ViewportNodeData;
          //     const existing = vpData.interactions ?? [];

          //     const next = iId ? existing.filter((d) => d.id !== iId) : existing;

          //     return {
          //       ...nn,
          //       data: {
          //         ...nn.data,
          //         interactions: next.length ? next : undefined,
          //       } as ViewportNodeData,
          //     };
          //   })
          .filter((nn) => nn.id !== nodeId)
      );

      // 3) Remove all edges touching this node
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
            title: data.title ?? "Widget",
            schema,
            pickInner: (v) => (v as any)?.widget,
            onClose: onCloseWidgetDefNode,
            onToggleMinimize: handleToggleMinimize,
            // no custom onClose; BaseGrammarNode default remove is fine
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
              {data.title ?? "Widget"}
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

      {/* <Handle
        type="source"
        position={Position.Left}
        id="widgetDef-out-1"
        className="gnode__handle__source"
      />

      <Handle
        type="source"
        position={Position.Right}
        id="widgetDef-out-2"
        className="gnode__handle__source"
      /> */}

      <Handle
        type="source"
        position={Position.Top}
        id="widgetDef-out-3"
        className={`gnode__handle__source ${
          minimized ? "gnode__handle--hidden" : ""
        }`}
      />

      <Handle
        type="source"
        position={Position.Bottom}
        id="widgetDef-out-4"
        className={`gnode__handle__source ${
          minimized ? "gnode__handle--hidden" : ""
        }`}
      />
    </>
  );
});

export default WidgetDefNode;
