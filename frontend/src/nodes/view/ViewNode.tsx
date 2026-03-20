import { memo, useCallback, useMemo, useRef, useState, useEffect } from "react";
import type { NodeProps, Node } from "@xyflow/react";
import {
  useReactFlow,
  Handle,
  Position,
  NodeResizer,
  useUpdateNodeInternals,
} from "@xyflow/react";
import BaseGrammarNode, {
  type BaseNodeData,
} from "../../node-components/BaseGrammar";
import schema from "../../schemas/view.json";
// import type { ViewportNodeData } from "./ViewportNode";
import ViewportCanvas from "./ViewportCanvas";

import "../../node-components/BaseGrammar.css";
import "./ViewNode.css";

import flipPng from "../../assets/restart-2.png";
import persistPng from "../../assets/update-data.png";
import mapPng from "../../assets/map.png";
import checkPng from "../../assets/check-mark.png";

export type ViewNodeData = BaseNodeData & {
  mode?: "def" | "view";
  pushToken?: string;
};

export type ViewNode = Node<ViewNodeData, "viewNode">;

const VIS_MIN_WIDTH = 300;
const VIS_MIN_HEIGHT = 260;

const ViewNode = memo(function ViewNode(props: NodeProps<ViewNode>) {
  const { id, data, selected } = props;
  const { setNodes, setEdges } = useReactFlow();
  const rf = useReactFlow();

  // const [minimized, setMinimized] = useState(false);
  const [persisting, setPersisting] = useState(false);
  const [persistSuccess, setPersistSuccess] = useState(false);
  const [showBasemap, setShowBasemap] = useState(false);

  const pendingRef = useRef<Record<string, any>>({});

  const updateNodeInternals = useUpdateNodeInternals();

  const mode = data.mode ?? "def";

  useEffect(() => {
    requestAnimationFrame(() => {
      updateNodeInternals(id);
    });
  }, [id, mode, updateNodeInternals]);

  const viewSpec = useMemo(() => {
    const v: any = (data as BaseNodeData)?.value;
    return v?.view;
  }, [data]);

  const onCloseViewNode = useCallback(
    (nodeId: string) => {
      rf.setNodes((nds) => nds.filter((n) => n.id !== nodeId));

      setEdges((eds) =>
        eds.filter((e) => e.source !== nodeId && e.target !== nodeId),
      );
    },
    [rf, setEdges],
  );

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

  const handleDirty = useCallback(
    ({ ref, featureCollection }: { ref: string; featureCollection: any }) => {
      pendingRef.current[ref] = {
        ref,
        geojson: featureCollection,
      };
    },
    [],
  );

  const stableView = useMemo(() => {
    return Array.isArray(viewSpec) ? viewSpec : [];
  }, [viewSpec]);

  const stableInteractions = useMemo(() => {
    const interactions = (data as any)?.interactions;
    return Array.isArray(interactions) ? interactions : [];
  }, [data]);

  const stableCenter = useMemo<[number, number]>(() => [41.881, -87.63], []);

  if (mode === "def") {
    return (
      <>
        <BaseGrammarNode
          id={id}
          selected={selected}
          data={{
            ...data,
            title: data.title ?? "View",
            schema,
            pickInner: (v) => (v as any)?.view,
            onClose: onCloseViewNode,
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

        <Handle
          type="target"
          position={Position.Top}
          id="view-in-1"
          className={`gnode__handle`}
        />

        <Handle
          type="target"
          position={Position.Left}
          id="view-in-2"
          className={`gnode__handle gnode__handle--left`}
        />

        <Handle
          type="target"
          position={Position.Bottom}
          id="view-in-3"
          className={`gnode__handle`}
        />

        <Handle
          type="source"
          position={Position.Right}
          id="view-out"
          className={`gnode__handle gnode__handle--right`}
        />
      </>
    );
  }

  return (
    <>
      <div className="vpnode">
        <NodeResizer minWidth={VIS_MIN_WIDTH} minHeight={VIS_MIN_HEIGHT} />

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

        <div className="vpnode__body">
          <ViewportCanvas
            id={id}
            center={stableCenter}
            view={stableView}
            interactions={stableInteractions}
            showBasemap={showBasemap}
            className="vpnode__map nodrag nowheel"
            onDirty={handleDirty}
          />

          <div className="vpnode__footer">
            {/* <button
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
              </button> */}

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
        </div>

        <Handle
          type="target"
          position={Position.Top}
          id="view-in-1"
          className={`vpnode__handle`}
        />

        <Handle
          type="target"
          position={Position.Left}
          id="view-in-2"
          className={`vpnode__handle vpnode__handle--left`}
        />

        <Handle
          type="target"
          position={Position.Bottom}
          id="view-in-3"
          className={`vpnode__handle`}
        />

        <Handle
          type="source"
          position={Position.Right}
          id="view-out"
          className={`vpnode__handle vpnode__handle--right`}
        />
      </div>
    </>
  );
});

export default ViewNode;
