import { memo, useCallback, useEffect, useState } from "react";
import type { NodeProps, Node } from "@xyflow/react";
import { useReactFlow, Handle, Position, NodeResizer } from "@xyflow/react";
import Snackbar from "@mui/material/Snackbar";
import Alert from "@mui/material/Alert";
import Dialog from "@mui/material/Dialog";
import DialogTitle from "@mui/material/DialogTitle";
import DialogContent from "@mui/material/DialogContent";
import DialogContentText from "@mui/material/DialogContentText";
import DialogActions from "@mui/material/DialogActions";
import Button from "@mui/material/Button";

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
  // Set when the backend reports that fetching again would overwrite a
  // previous fetch's output (see runFetch) - drives the confirm Dialog
  // below. null means "no confirmation pending", not "no conflicts exist".
  const [pendingOverwrite, setPendingOverwrite] = useState<
    { feature: string; name: string }[] | null
  >(null);

  // The actual network call, parameterized by whether overwriting is
  // already agreed to. Returns false (without setting errorMessage) when
  // the backend instead asks for confirmation - the caller just stops
  // there, nothing failed.
  const runFetch = useCallback(
    async (confirmOverwrite: boolean): Promise<boolean> => {
      const val: any = (data.value as any)?.data_layer;

      if (!val) {
        console.warn("No data_layer data found for node", id);
        return false;
      }

      const response = await fetch(appUrl("/api/extract-data-layer"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        // dataflow_id tells the backend which project's dataset list to
        // check a feature against - fetching only succeeds for a feature
        // that's actually been added to this dataflow's project.
        body: JSON.stringify({
          ...val,
          dataflow_id: getCurrentDataflowId(),
          confirmOverwrite,
        }),
      });

      const body = await response.json().catch(() => ({}));

      if (body?.status === "confirm_overwrite") {
        setPendingOverwrite(body.existing ?? []);
        return false;
      }

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
      return true;
    },
    [data, id],
  );

  const runWithFeedback = useCallback(
    async (confirmOverwrite: boolean): Promise<boolean> => {
      try {
        setLoading(true);
        setLoadingSuccess(false);
        const ok = await runFetch(confirmOverwrite);
        if (ok) {
          setLoadingSuccess(true);
          setTimeout(() => setLoadingSuccess(false), 2000);
        }
        return ok;
      } catch (err) {
        console.error("Error sending data to Flask:", err);
        setErrorMessage(
          err instanceof Error ? err.message : "Failed to fetch this data layer.",
        );
        return false;
      } finally {
        setLoading(false);
      }
    },
    [runFetch],
  );

  const onFetch = useCallback(
    async (opts?: { interactive?: boolean }): Promise<boolean> => {
      // A direct click checks first and pauses for confirmation if this
      // would overwrite a previous fetch's output (see the Dialog below).
      // An orchestrated "Run Dataflow" run (registerNodeAction calls this
      // with no args, so interactive defaults to false) auto-overwrites
      // instead - there's no one present mid-run to answer a popup, and
      // that's this node's existing behavior for that path today.
      const interactive = opts?.interactive ?? false;
      return runWithFeedback(!interactive);
    },
    [runWithFeedback],
  );

  const handleConfirmOverwrite = useCallback(() => {
    setPendingOverwrite(null);
    void runWithFeedback(true);
  }, [runWithFeedback]);

  const handleCancelOverwrite = useCallback(() => {
    setPendingOverwrite(null);
  }, []);

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
              onClick={() => onFetch({ interactive: true })}
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
                onClick={() => onFetch({ interactive: true })}
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

      <Dialog open={pendingOverwrite !== null} onClose={handleCancelOverwrite}>
        <DialogTitle sx={{ fontWeight: 700 }}>Overwrite existing data</DialogTitle>
        <DialogContent>
          <DialogContentText component="ul" sx={{ mt: 0, mb: 0, pl: 2.5 }}>
            {pendingOverwrite?.map((c) => <li key={c.name}>{c.name}</li>)}
          </DialogContentText>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={handleCancelOverwrite} sx={{ textTransform: "none" }}>
            Cancel
          </Button>
          <Button
            onClick={handleConfirmOverwrite}
            color="error"
            variant="contained"
            disableElevation
            sx={{ textTransform: "none", fontWeight: 600 }}
          >
            Overwrite
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
});

export default DataLayerNode;
