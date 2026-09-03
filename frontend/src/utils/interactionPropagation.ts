import type { Edge, Node } from "@xyflow/react";
import type { BaseNodeData } from "../node-components/BaseGrammar";

type SetNodes = (updater: (nodes: Node[]) => Node[]) => void;

/**
 * Pushes an interaction node's `data.value.interaction` onto every connected
 * viewNode's `data.interactions` - the same effect running the interaction
 * node produces (App.tsx's onConnect handler, or a grammar node's own
 * "update" button via BaseGrammarNode's onRun). Shared so the manual UI
 * path and any agent that creates/runs nodes itself (see
 * agents/nodeAgent.ts) apply the exact same logic rather than two copies
 * that can drift.
 *
 * Returns whether anything actually changed (a real interaction node,
 * connected to at least one viewNode).
 */
export function pushInteractionToView(
  srcId: string,
  nodes: Node[],
  edges: Edge[],
  setNodes: SetNodes,
  targetNodeId?: string,
): boolean {
  const src = nodes.find((n) => n.id === srcId);
  if (!src || src.type !== "interactionNode") return false;

  const val: any = (src.data as BaseNodeData).value;
  const i = val?.interaction;
  if (!i) return false;

  const targetIds = targetNodeId
    ? [targetNodeId]
    : (edges
        .filter((e) => e.source === srcId)
        .map((e) => e.target)
        .filter(Boolean) as string[]);

  const viewTargetIds = targetIds.filter(
    (tid) => nodes.find((n) => n.id === tid)?.type === "viewNode",
  );
  if (!viewTargetIds.length) return false;

  setNodes((nds) =>
    nds.map((n) => {
      if (!viewTargetIds.includes(n.id)) return n;

      const existing = ((n.data as any).interactions ?? []) as any[];
      const already = existing.some((e) => e?.itype === i?.itype);
      const nextInteractions = already
        ? existing.map((e) => (e?.itype === i?.itype ? i : e))
        : [...existing, i];

      return {
        ...n,
        data: {
          ...n.data,
          interactions: nextInteractions,
        },
      };
    }),
  );

  return true;
}
