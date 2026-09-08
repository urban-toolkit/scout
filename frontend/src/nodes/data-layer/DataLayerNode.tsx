import { memo, useCallback, useEffect, useState } from "react";
import type { NodeProps, Node } from "@xyflow/react";
import { useReactFlow, Handle, Position, NodeResizer } from "@xyflow/react";
import Snackbar from "@mui/material/Snackbar";
import Alert from "@mui/material/Alert";

import BaseGrammarNode, {
  BaseNodeData,
} from "../../node-components/BaseGrammar";
import schema from "../../schemas/data_layer.json";

import fetchPng2 from "../../assets/fetch_2.png";
import checkPng from "../../assets/check-mark.png";
import expandPng from "../../assets/expand.png";
import { appUrl } from "../../utils/runtimePaths";
import { registerNodeAction } from "../../utils/nodeActionRegistry";
import { getCurrentDataflowId } from "../../utils/dataflows";
// import restartPng from "../../assets/restart.png";

import "./DataLayerNode.css";
import "../../node-components/BaseGrammar.css";

export type DataLayerNode = Node<BaseNodeData, "dataLayerNode">;

const NODE_MIN_WIDTH = 300;
const NODE_MIN_HEIGHT = 180;

const NODE_MINIMIZED_WIDTH = 150;
const NODE_MINIMIZED_HEIGHT = 48;

const DataLayerNode = memo(function DataLayerNode(
  props: NodeProps<DataLayerNode>,
) {
  const { id, data, selected } = props;
  const rf = useReactFlow();
  const { setEdges } = useReactFlow();
  const [loading, setLoading] = useState(false);
  const [loadingSuccess, setLoadingSuccess] = useState(false);

  // Seeded from the persisted node data (rather than always false) so a
  // dataflow saved with this node minimized reopens minimized too.
  const [minimized, setMinimized] = useState(() => Boolean(data.minimized));
  // Surfaced as a popup (see the Snackbar below) rather than just a console
  // log - the most common cause is a feature that hasn't been added to
  // this dataflow's project yet, which is otherwise a silent no-op.
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const onFetch = useCallback(async (): Promise<boolean> => {
    const val: any = (data.value as any)?.data_layer;

    if (!val) {
      console.warn("No data_layer data found for node", id);
      return false;
    }

    try {
      setLoading(true);
      setLoadingSuccess(false);
      const response = await fetch(appUrl("/api/extract-data-layer"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        // dataflow_id tells the backend which project's dataset list to
        // check a feature against - fetching only succeeds for a feature
        // that's actually been added to this dataflow's project.
        body: JSON.stringify({ ...val, dataflow_id: getCurrentDataflowId() }),
      });

      const body = await response.json().catch(() => ({}));

      if (!response.ok && response.status !== 207) {
        throw new Error(body?.error ?? `Server returned ${response.status}`);
      }

      const problems: { feature?: string; error: string }[] = body?.problems ?? [];
      const computed = body?.computed ?? [];

      // Whatever was actually fetched still shows up as available, even if
      // other features in the same request were skipped for not being in
      // the project - so it's added to the project (and the Computed tab)
      // regardless of whether there were also problems.
      if (computed.length > 0) {
        data.onAddComputedDatasets?.(computed);
      }

      if (problems.length > 0) {
        setErrorMessage(problems.map((p) => p.error).join(" "));
        return false;
      }

      setErrorMessage(null);
      setLoadingSuccess(true);
      setTimeout(() => setLoadingSuccess(false), 2000);
      return true;
    } catch (err) {
      console.error("Error sending data to Flask:", err);
      setErrorMessage(
        err instanceof Error ? err.message : "Failed to fetch this data layer.",
      );
      return false;
    } finally {
      setLoading(false);
    }
  }, [data, id]);

  // Lets the "Run Dataflow" orchestrator fetch this node directly and await
  // the result - see utils/nodeActionRegistry.ts.
  useEffect(() => registerNodeAction(id, onFetch), [id, onFetch]);

  const onCloseDataNode = useCallback(
    (nodeId: string) => {
      rf.setNodes((nds) => nds.filter((n) => n.id !== nodeId));

      // 3) Remove all edges touching this node
      setEdges((eds) =>
        eds.filter((e) => e.source !== nodeId && e.target !== nodeId),
      );
    },
    [rf, setEdges],
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
              data: { ...n.data, minimized: next },
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
              data: { ...n.data, minimized: next },
            };
          }
        }),
      );

      // Hide/show edges
      setEdges((eds) =>
        eds.map((e) =>
          e.source === id || e.target === id ? { ...e, hidden: next } : e,
        ),
      );

      return next;
    });
  }, [id, rf, setEdges]);

  // const handleRun = useCallback(() => {
  //   if (data?.onRun) {
  //     return data.onRun(id);
  //   }
  // }, [data, id]);
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
                {loading ? "Fetching..." : (data.title ?? "Data layer")}
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
            {/* <button
              type="button"
              className="gnode__minimizedRestoreCircle_2 gnode__minimizedRestoreCircle--bottomRight"
              onClick={handleRun}
              disabled={loading}
            >
              <img src={restartPng} alt="Fetch / update" />
            </button> */}
          </div>
        </div>
      ) : (
        <BaseGrammarNode
          id={id}
          selected={selected}
          data={{
            ...data,
            title: data.title ?? "Data layer",
            schema,
            pickInner: (v) => (v as any)?.data_layer,
            onClose: onCloseDataNode,
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
        id="data-out"
        className={`gnode__handle gnode__handle--right ${
          minimized ? "gnode__handle--hidden" : ""
        }`}
      />

      <Snackbar
        open={errorMessage !== null}
        autoHideDuration={6000}
        onClose={() => setErrorMessage(null)}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      >
        <Alert
          onClose={() => setErrorMessage(null)}
          severity="error"
          variant="filled"
          sx={{ maxWidth: 420 }}
        >
          {errorMessage}
        </Alert>
      </Snackbar>
    </>
  );
});

export default DataLayerNode;
