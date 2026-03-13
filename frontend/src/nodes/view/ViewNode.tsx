import { memo, useCallback, useMemo, useRef, useState } from "react";
import type { NodeProps, Node } from "@xyflow/react";
import { useReactFlow, Handle, Position, NodeResizer } from "@xyflow/react";
import BaseGrammarNode, {
  type BaseNodeData,
} from "../../node-components/BaseGrammar";
import schema from "../../schemas/view.json";
import type { ViewportNodeData } from "./ViewportNode";
import ViewportCanvas from "./ViewportCanvas";

import "../../node-components/BaseGrammar.css";
import "./ViewportNode.css";

import expandPng from "../../assets/expand.png";
import restartPng from "../../assets/restart.png";
import flipPng from "../../assets/restart-2.png";
import persistPng from "../../assets/update-data.png";
import mapPng from "../../assets/map.png";
import checkPng from "../../assets/check-mark.png";

export type ViewNodeData = BaseNodeData & {
  mode?: "def" | "view";
  pushToken?: string;
};

export type ViewNode = Node<ViewNodeData, "viewNode">;

const NODE_MIN_WIDTH = 300;
const NODE_MIN_HEIGHT = 180;
const NODE_MINIMIZED_WIDTH = 150;
const NODE_MINIMIZED_HEIGHT = 48;

const VIS_MIN_WIDTH = 300;
const VIS_MIN_HEIGHT = 260;

const ViewNode = memo(function ViewNode(props: NodeProps<ViewNode>) {
  const { id, data, selected } = props;
  const { getNode, getEdges, setNodes, setEdges } = useReactFlow();
  const rf = useReactFlow();

  const [minimized, setMinimized] = useState(false);
  const [persisting, setPersisting] = useState(false);
  const [persistSuccess, setPersistSuccess] = useState(false);
  const [showBasemap, setShowBasemap] = useState(false);

  const pendingRef = useRef<Record<string, any>>({});

  const mode = data.mode ?? "def";

  const viewSpec = useMemo(() => {
    const v: any = (data as BaseNodeData)?.value;
    return v?.view;
  }, [data]);

  const onCloseViewNode = useCallback(
    (nodeId: string) => {
      const n = getNode(nodeId);
      if (!n || n.type !== "viewNode") return;

      const curEdges = getEdges();

      const targetIds = curEdges
        .filter((e) => e.source === nodeId)
        .map((e) => e.target);

      setNodes((nds) =>
        nds
          .map((nn) => {
            if (nn.type !== "viewportNode" || !targetIds.includes(nn.id)) {
              return nn;
            }

            const vd = nn.data as ViewportNodeData;
            const nextData: ViewportNodeData = {
              ...vd,
              view: undefined,
            };

            return { ...nn, data: nextData };
          })
          .filter((nn) => nn.id !== nodeId),
      );

      setEdges((eds) =>
        eds.filter((e) => e.source !== nodeId && e.target !== nodeId),
      );
    },
    [getNode, getEdges, setNodes, setEdges],
  );

  const handleToggleMinimize = useCallback(() => {
    setMinimized((prev) => {
      const next = !prev;

      rf.setNodes((nodes) =>
        nodes.map((n) => {
          if (n.id !== id) return n;

          if (next) {
            return {
              ...n,
              width: NODE_MINIMIZED_WIDTH,
              height: NODE_MINIMIZED_HEIGHT,
            };
          }

          const nextWidth =
            n.width &&
            n.width > (mode === "view" ? VIS_MIN_WIDTH : NODE_MIN_WIDTH)
              ? n.width
              : mode === "view"
                ? VIS_MIN_WIDTH
                : NODE_MIN_WIDTH;

          const nextHeight =
            n.height &&
            n.height > (mode === "view" ? VIS_MIN_HEIGHT : NODE_MIN_HEIGHT)
              ? n.height
              : mode === "view"
                ? VIS_MIN_HEIGHT
                : NODE_MIN_HEIGHT;

          return {
            ...n,
            width: nextWidth,
            height: nextHeight,
          };
        }),
      );

      setEdges((eds) =>
        eds.map((e) =>
          e.source === id || e.target === id ? { ...e, hidden: next } : e,
        ),
      );

      return next;
    });
  }, [id, mode, rf, setEdges]);

  const handleRun = useCallback(() => {
    if (data?.onRun) {
      return data.onRun(id);
    }
  }, [data, id]);

  const onPersist = useCallback(async () => {
    const entries = Object.values(pendingRef.current) as {
      ref: string;
      geojson: any;
    }[];

    if (!entries.length) return;

    setPersisting(true);
    setPersistSuccess(false);

    try {
      const tasks = entries.map(({ ref, geojson }) =>
        fetch("http://127.0.0.1:5000/api/update-data-layer", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ref,
            geojson,
          }),
        }),
      );

      await Promise.allSettled(tasks);
      pendingRef.current = {};

      setPersistSuccess(true);
      setTimeout(() => setPersistSuccess(false), 2000);
    } finally {
      setPersisting(false);
    }

    pendingRef.current = {};
  }, []);

  const goToView = useCallback(() => {
    const token = crypto.randomUUID();

    setNodes((nds) =>
      nds.map((n) =>
        n.id === id
          ? {
              ...n,
              width: n.width ?? 420,
              height: n.height ?? 320,
              data: {
                ...n.data,
                mode: "view",
                pushToken: token,
              } as ViewNodeData,
            }
          : n,
      ),
    );
  }, [id, setNodes]);

  const goToDef = useCallback(() => {
    setNodes((nds) =>
      nds.map((n) =>
        n.id === id
          ? {
              ...n,
              data: {
                ...n.data,
                mode: "def",
              } as ViewNodeData,
            }
          : n,
      ),
    );
  }, [id, setNodes]);

  const handleFlip = useCallback(() => {
    if (mode === "def") {
      goToView();
    } else {
      goToDef();
    }
  }, [mode, goToView, goToDef]);

  if (mode === "def") {
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
              onRun: handleRun,
              footerActions: (
                <button
                  type="button"
                  onClick={handleFlip}
                  title="Flip to view"
                  aria-label="Flip to view"
                  className="gnode__actionBtn"
                >
                  <img
                    src={flipPng}
                    alt="Flip to view"
                    className="gnode__actionIcon"
                  />
                </button>
              ),
            }}
          />
        ) : (
          <div className="gnode gnode--minimized">
            <NodeResizer
              minWidth={NODE_MINIMIZED_WIDTH}
              maxWidth={Infinity}
              minHeight={NODE_MINIMIZED_HEIGHT}
              maxHeight={NODE_MINIMIZED_HEIGHT}
            />
            <div className="gnode__minimized">
              <button type="button" className="gnode__minimizedNodeTtitleBtn">
                {data.title ?? "View"}
              </button>

              <button
                type="button"
                className="gnode__minimizedRestoreCircle_1 gnode__minimizedRestoreCircle--topLeft"
                onClick={handleToggleMinimize}
              >
                <img src={expandPng} alt="Restore" />
              </button>

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
  }

  return (
    <>
      <div className="vpnode">
        <NodeResizer minWidth={VIS_MIN_WIDTH} minHeight={VIS_MIN_HEIGHT} />

        {!minimized && (
          <div className="vpnode__header">
            <div className="vpnode__titleWrapper">
              <input
                type="text"
                className="vpnode__titleInput"
                value={data.title ?? "View"}
                onChange={(e) => {
                  const nextTitle = e.target.value;
                  rf.setNodes((nodes) =>
                    nodes.map((n) =>
                      n.id === id
                        ? { ...n, data: { ...n.data, title: nextTitle } }
                        : n,
                    ),
                  );
                }}
              />
            </div>

            <button
              type="button"
              className="vpnode__iconBtn vpnode__iconBtn--close"
              onClick={() => onCloseViewNode(id)}
            >
              ✕
            </button>
          </div>
        )}

        <div className="vpnode__body">
          <ViewportCanvas
            id={id}
            center={[41.881, -87.63]}
            view={Array.isArray(viewSpec) ? viewSpec : []}
            interactions={[]}
            showBasemap={showBasemap}
            className="vpnode__map nodrag nowheel"
            onDirty={({ ref, featureCollection }) => {
              pendingRef.current[ref] = {
                ref,
                geojson: featureCollection,
              };
            }}
          />

          {!minimized && (
            <div className="vpnode__footer">
              <button
                type="button"
                onClick={handleRun}
                title="update"
                aria-label="update"
                className="vpnode__actionBtn"
              >
                <img
                  src={restartPng}
                  alt="update"
                  className="vpnode__actionIcon"
                />
              </button>

              <button
                type="button"
                onClick={onPersist}
                title="Save edits"
                aria-label="Save edits"
                className="vpnode__actionBtn"
                disabled={persisting}
              >
                {persisting ? (
                  <span className="vpnode__spinner" />
                ) : persistSuccess ? (
                  <img
                    src={checkPng}
                    alt="Success"
                    className="vpnode__actionIcon"
                  />
                ) : (
                  <img
                    src={persistPng}
                    alt="Save edits"
                    className="vpnode__actionIcon"
                  />
                )}
              </button>

              <button
                type="button"
                onClick={() => setShowBasemap((b) => !b)}
                title="toggle map"
                aria-label="toggle map"
                className="vpnode__actionBtn"
              >
                <img
                  src={mapPng}
                  alt="toggle map"
                  className="vpnode__actionIcon"
                />
              </button>

              <button
                type="button"
                onClick={handleFlip}
                title="Flip to grammar"
                aria-label="Flip to grammar"
                className="vpnode__actionBtn"
              >
                <img
                  src={flipPng}
                  alt="Flip to grammar"
                  className="vpnode__actionIcon"
                />
              </button>
            </div>
          )}
        </div>

        <Handle
          type="target"
          position={Position.Left}
          id="view-in"
          className={`vpnode__handle vpnode__handle--left ${
            minimized ? "gnode__handle--hidden" : ""
          }`}
        />

        <Handle
          type="source"
          position={Position.Right}
          id="view-out"
          className={`vpnode__handle vpnode__handle--right ${
            minimized ? "gnode__handle--hidden" : ""
          }`}
        />
      </div>
    </>
  );
});

export default ViewNode;
