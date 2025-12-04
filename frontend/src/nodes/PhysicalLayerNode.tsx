import { memo, useCallback, useState } from "react";
import type { NodeProps, Node } from "@xyflow/react";
import { useReactFlow, Handle, Position, NodeResizer } from "@xyflow/react";

import BaseGrammarNode, { BaseNodeData } from "./BaseGrammarNode";
import schema from "../schemas/physical_layer.json";

import fetchPng from "../assets/fetch.png";
import fetchPng2 from "../assets/fetch_2.png";

import checkPng from "../assets/check-mark.png";

import expandPng from "../assets/expand.png";
import restartPng from "../assets/restart.png";

import "./PhysicalLayerNode.css";
import "./BaseGrammarNode.css";

export type PhysicalLayerNode = Node<BaseNodeData, "physicalLayerNode">;

const NODE_MIN_WIDTH = 300;
const NODE_MIN_HEIGHT = 180;

const NODE_MINIMIZED_WIDTH = 150;
const NODE_MINIMIZED_HEIGHT = 48;

const PhysicalLayerNode = memo(function PhysicalLayerNode(
  props: NodeProps<PhysicalLayerNode>
) {
  const { id, data, selected } = props;
  const rf = useReactFlow();
  const { setEdges } = useReactFlow();
  const [loading, setLoading] = useState(false);
  const [loadingSuccess, setLoadingSuccess] = useState(false);

  const [minimized, setMinimized] = useState(false);

  const onFetch = useCallback(async () => {
    // console.log(data.title, "fetching physical layer data...");
    const val: any = (data.value as any)?.physical_layer;

    if (!val) {
      console.warn("No physical_layer data found for node", id);
      return;
    }

    try {
      setLoading(true);
      setLoadingSuccess(false);
      const response = await fetch(
        "http://127.0.0.1:5000/api/ingest-physical-layer",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(val),
        }
      );

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      // const result = await response.json();
      // console.log("Flask response:", result);
      setLoadingSuccess(true);
      setTimeout(() => setLoadingSuccess(false), 2000);
    } catch (err) {
      console.error("Error sending data to Flask:", err);
    } finally {
      setLoading(false);
    }
  }, [data, id]);

  const onClosePhysicalNode = useCallback(
    (nodeId: string) => {
      rf.setNodes((nds) => nds.filter((n) => n.id !== nodeId));

      // 3) Remove all edges touching this node
      setEdges((eds) =>
        eds.filter((e) => e.source !== nodeId && e.target !== nodeId)
      );
    },
    [rf, setEdges]
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
      {minimized ? (
        <div className="gnode gnode--minimized">
          <NodeResizer
            minWidth={minimized ? NODE_MINIMIZED_WIDTH : NODE_MIN_WIDTH}
            maxWidth={Infinity}
            minHeight={minimized ? NODE_MINIMIZED_HEIGHT : NODE_MIN_HEIGHT}
            maxHeight={minimized ? NODE_MINIMIZED_HEIGHT : Infinity}
          />
          <div className="gnode__minimized">
            {/* Big fetch button */}
            <button
              type="button"
              className="gnode__minimizedFetchBtn"
              style={{
                backgroundColor: "#f5d1d2",
                borderColor: "#cb181d",
                color: "#000",
              }}
              onClick={onFetch}
              disabled={loading}
              aria-busy={loading}
              title={loading ? "Fetching..." : "Fetch data"}
            >
              {loading ? (
                <span className="gnode__spinner" aria-hidden="true" />
              ) : loadingSuccess ? (
                <img
                  src={checkPng}
                  alt="Success"
                  className="gnode__minimizedIcon"
                />
              ) : (
                <img
                  src={fetchPng2}
                  alt="Fetch data"
                  className="gnode__minimizedIcon"
                />
              )}

              <span className="gnode__minimizedText">
                {loading
                  ? "Fetching..."
                  : data.title ?? "Grammar • physical_layer"}
              </span>
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
              disabled={loading}
            >
              <img src={restartPng} alt="Fetch / update" />
            </button>
          </div>
        </div>
      ) : (
        <BaseGrammarNode
          id={id}
          selected={selected}
          data={{
            ...data,
            title: data.title ?? "Grammar • physical_layer",
            schema,
            pickInner: (v) => (v as any)?.physical_layer,
            onClose: onClosePhysicalNode,
            onToggleMinimize: handleToggleMinimize,
            footerActions: (
              <button
                type="button"
                onClick={onFetch}
                title={loading ? "Fetching..." : "Fetch data"}
                aria-label="Fetch data"
                className="gnode__actionBtn"
                disabled={loading}
              >
                {loading ? (
                  <span className="gnode__spinner" aria-hidden="true" />
                ) : loadingSuccess ? (
                  <img
                    src={checkPng}
                    alt="Success"
                    className="gnode__actionIcon"
                  />
                ) : (
                  <img
                    src={fetchPng2}
                    alt="Fetch data"
                    className="gnode__actionIcon"
                  />
                )}
              </button>
            ),
          }}
        />
      )}

      {/* Handle is ALWAYS rendered, just hidden when minimized */}
      <Handle
        type="source"
        position={Position.Right}
        id="physical-out"
        className={`gnode__handle gnode__handle--right ${
          minimized ? "gnode__handle--hidden" : ""
        }`}
      />
    </>
  );
});

export default PhysicalLayerNode;
