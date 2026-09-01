import { memo, useCallback, useEffect, useMemo, useState } from "react";
import type { NodeProps, Node } from "@xyflow/react";
import {
  Handle,
  Position,
  NodeResizer,
  useReactFlow,
  useUpdateNodeInternals,
} from "@xyflow/react";
import type { ReactNode } from "react";

import BaseGrammarNode, {
  BaseNodeData,
} from "../../node-components/BaseGrammar";
import schema from "../../schemas/comparison.json";

import type { ComparisonDef } from "../../utils/types";
import { renderComparisonFromDef } from "../../charts/renderers/renderComparison";
import ChartStudio from "./ChartStudio";

import "./ComparisonNode.css"; // reuse if you want, or make a new css

import flipPng from "../../assets/restart-2.png";
import { registerNodeAction } from "../../utils/nodeActionRegistry";

export type ComparisonNodeData = BaseNodeData & {
  mode?: "def" | "view" | "studio";
  previewToken?: string;
};

export type ComparisonNode = Node<ComparisonNodeData, "comparisonNode">;

// Minimum size for Studio mode's two-column layout - also used as the
// NodeResizer floor so dragging smaller can't reintroduce the overflow
// this was set up to avoid.
const STUDIO_MIN_WIDTH = 1020;
const STUDIO_MIN_HEIGHT = 500;

const ComparisonNode = memo(function ComparisonNode(
  props: NodeProps<ComparisonNode>,
) {
  const { id, data, selected } = props;
  const { setNodes, setEdges } = useReactFlow();
  const updateNodeInternals = useUpdateNodeInternals();

  const mode = data.mode ?? "def";

  const [bodyContent, setBodyContent] = useState<ReactNode | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [draftTitle, setDraftTitle] = useState(data.title ?? "Comparison");

  useEffect(() => {
    setDraftTitle(data.title ?? "Comparison");
  }, [data.title]);

  // close removes node + edges (same as your view node close)
  const handleClose = useCallback(() => {
    setNodes((nds) => nds.filter((n) => n.id !== id));
    setEdges((eds) => eds.filter((e) => e.source !== id && e.target !== id));
  }, [id, setNodes, setEdges]);

  useEffect(() => {
    requestAnimationFrame(() => {
      updateNodeInternals(id);
    });
  }, [id, mode, updateNodeInternals]);

  // read comparison def from grammar value
  const comparison: ComparisonDef | undefined = useMemo(() => {
    const v: any = data.value;
    return v?.comparison;
  }, [data.value]);

  const commitTitle = useCallback(() => {
    const nextTitle = draftTitle.trim() || "Comparison";
    if (nextTitle === (data.title ?? "Comparison")) return;

    setNodes((nodes) =>
      nodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, title: nextTitle } } : n,
      ),
    );
  }, [data.title, draftTitle, id, setNodes]);

  const goToView = useCallback(() => {
    const token = crypto.randomUUID();
    setNodes((nds) =>
      nds.map((n) =>
        n.id === id
          ? {
              ...n,
              width: n.width ?? 420,
              height: n.height ?? 320,
              data: { ...n.data, mode: "view", previewToken: token },
            }
          : n,
      ),
    );
  }, [id, setNodes]);

  const goToDef = useCallback(() => {
    setNodes((nds) =>
      nds.map((n) =>
        n.id === id
          ? { ...n, data: { ...n.data, mode: "def" } as ComparisonNodeData }
          : n,
      ),
    );
  }, [id, setNodes]);

  // "Run Dataflow" registers this so a Comparison node picks up freshly
  // written metric data after its upstream Code nodes finish. Unlike View,
  // the fetch effect below already watches previewToken directly, so
  // bumping it is enough - no unmount/remount round-trip needed. A node
  // still showing its grammar gets flipped into view mode (same as clicking
  // "Generate comparison view") rather than skipped, so the run actually
  // shows results - Studio mode (active chart design) is left alone though,
  // since forcing the user out of that mid-edit would be wrong. Comparison
  // is always a terminal node in this app's connection rules (nothing reads
  // its output), so this is fire-and-forget rather than awaiting the fetch
  // below to resolve.
  const runRefresh = useCallback(async (): Promise<boolean> => {
    if (mode === "studio") return true;

    if (mode !== "view") {
      goToView();
      return true;
    }

    const token = crypto.randomUUID();
    setNodes((nds) =>
      nds.map((n) =>
        n.id === id
          ? {
              ...n,
              data: { ...n.data, mode: "view", previewToken: token },
            }
          : n,
      ),
    );

    return true;
  }, [id, mode, setNodes, goToView]);

  useEffect(() => registerNodeAction(id, runRefresh), [id, runRefresh]);

  const goToStudio = useCallback(() => {
    setNodes((nds) =>
      nds.map((n) =>
        n.id === id
          ? {
              ...n,
              // Studio's two-column layout needs real room - guarantee at
              // least STUDIO_MIN_WIDTH/HEIGHT on entry (regardless of
              // whatever size the node had in def/view mode) rather than
              // only falling back when width/height were never set at all,
              // but keep a larger size if the node was already bigger.
              width: Math.max(n.width ?? 0, STUDIO_MIN_WIDTH),
              height: Math.max(n.height ?? 0, STUDIO_MIN_HEIGHT),
              data: { ...n.data, mode: "studio" } as ComparisonNodeData,
            }
          : n,
      ),
    );
  }, [id, setNodes]);

  const handleChartPublished = useCallback(
    (name: string) => {
      setNodes((nds) =>
        nds.map((n) => {
          if (n.id !== id) return n;
          const value: any = n.data.value ?? {};
          return {
            ...n,
            data: {
              ...n.data,
              mode: "def",
              value: {
                ...value,
                comparison: { ...(value.comparison ?? {}), chart: name },
              },
            } as ComparisonNodeData,
          };
        }),
      );
    },
    [id, setNodes],
  );

  // render when in view mode AND comparison changes OR previewToken changes
  useEffect(() => {
    if (mode !== "view") return;

    if (!comparison) {
      setBodyContent(null);
      setError("No comparison definition found.");
      setLoading(false);
      return;
    }

    const ctrl = new AbortController();
    setLoading(true);
    setError(null);

    (async () => {
      try {
        const content = await renderComparisonFromDef(comparison, ctrl.signal);
        if (!ctrl.signal.aborted) setBodyContent(content);
      } catch (e: any) {
        if (e?.name === "AbortError") return;
        console.error(e);
        setError(e?.message ?? "Failed to load comparison");
      } finally {
        if (!ctrl.signal.aborted) setLoading(false);
      }
    })();

    return () => ctrl.abort();
    // include previewToken so "Run" re-renders even if definition object reference is same
  }, [mode, comparison, data.previewToken]);

  if (mode === "def") {
    return (
      <>
        <BaseGrammarNode
          id={id}
          selected={selected}
          data={{
            ...data,
            title: data.title ?? "Comparison",
            schema,
            pickInner: (v) => (v as any)?.comparison,
            onClose: handleClose,
            // Add a "Run" hook: your BaseGrammarNode likely already has a run button that calls data.onRun(id)
            // If you want Run to switch to view mode, do it here by overriding onRun:
            // onRun: () => goToView(),
            footerActions: (
              <>
                <button
                  type="button"
                  onClick={goToStudio}
                  title="Design custom chart"
                  aria-label="Design custom chart"
                  className="gnode__actionBtn"
                  style={{ width: "auto", padding: "0 10px", fontSize: 12 }}
                >
                  Design chart
                </button>
                <button
                  type="button"
                  onClick={goToView}
                  title="Generate comparison view"
                  aria-label="Generate comparison view"
                  className="gnode__actionBtn"
                >
                  <img
                    src={flipPng}
                    alt="Generate comparison view"
                    className="gnode__actionIcon"
                  />
                </button>
              </>
            ),
          }}
        />

        <Handle
          type="target"
          position={Position.Left}
          id="comparison-in-1"
          className="cvnode__handle__target"
        />

        <Handle
          type="target"
          position={Position.Bottom}
          id="comparison-in-2"
          className="cvnode__handle__target"
        />

        <Handle
          type="target"
          position={Position.Right}
          id="comparison-in-3"
          className="cvnode__handle__target"
        />

        <Handle
          type="target"
          position={Position.Top}
          id="comparison-in-4"
          className="cvnode__handle__target"
        />
      </>
    );
  }

  if (mode === "studio") {
    return (
      <div className="cvnode">
        <NodeResizer minWidth={STUDIO_MIN_WIDTH} minHeight={STUDIO_MIN_HEIGHT} />

        <div className="cvnode__header">
          <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
            <span className="cvnode__title">{data.title ?? "Comparison"}</span>
            <span
              style={{
                fontSize: 11,
                fontWeight: 600,
                color: "#1f78b4",
                background: "rgba(255,255,255,0.6)",
                border: "1px solid #1f78b4",
                borderRadius: 999,
                padding: "1px 8px",
                whiteSpace: "nowrap",
              }}
            >
              Design mode
            </span>
          </div>
          <div className="cvnode__headerBtns">
            <button
              type="button"
              className="cvnode__iconBtn cvnode__iconBtn--close"
              onClick={handleClose}
            >
              ✕
            </button>
          </div>
        </div>

        <ChartStudio onPublished={handleChartPublished} onCancel={goToDef} />

        <Handle
          type="target"
          position={Position.Left}
          id="comparison-in-1"
          className="cvnode__handle__target"
        />
        <Handle
          type="target"
          position={Position.Bottom}
          id="comparison-in-2"
          className="cvnode__handle__target"
        />
        <Handle
          type="target"
          position={Position.Right}
          id="comparison-in-3"
          className="cvnode__handle__target"
        />
        <Handle
          type="target"
          position={Position.Top}
          id="comparison-in-4"
          className="cvnode__handle__target"
        />
      </div>
    );
  }

  // mode === "view"
  return (
    <div className="cvnode">
      <NodeResizer />

      <div className="cvnode__header">
        <input
          type="text"
          className="cvnode__titleInput"
          value={draftTitle}
          onChange={(e) => setDraftTitle(e.target.value)}
          onBlur={commitTitle}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.currentTarget.blur();
            }
          }}
        />

        <div className="cvnode__headerBtns">
          {/* <button type="button" className="cvnode__iconBtn" onClick={goToDef}>
            ←
          </button> */}
          <button
            type="button"
            className="cvnode__iconBtn cvnode__iconBtn--close"
            onClick={handleClose}
          >
            ✕
          </button>
        </div>
      </div>

      <div className="cvnode__body">
        {loading && <div>Loading…</div>}
        {error && <div className="cvnode__error">{error}</div>}
        {!loading && !error && bodyContent}
      </div>

      <div className="wvnode__footer">
        <button
          type="button"
          onClick={goToDef}
          title="Flip to grammar"
          aria-label="Flip to grammar"
          className="wvnode__actionBtn"
        >
          <img
            src={flipPng}
            alt="Flip to grammar"
            className="wvnode__actionIcon"
          />
        </button>
      </div>

      <Handle
        type="target"
        position={Position.Left}
        id="comparison-in-1"
        className="cvnode__handle__target"
      />

      <Handle
        type="target"
        position={Position.Bottom}
        id="comparison-in-2"
        className="cvnode__handle__target"
      />

      <Handle
        type="target"
        position={Position.Right}
        id="comparison-in-3"
        className="cvnode__handle__target"
      />

      <Handle
        type="target"
        position={Position.Top}
        id="comparison-in-4"
        className="cvnode__handle__target"
      />
    </div>
  );
});

export default ComparisonNode;
