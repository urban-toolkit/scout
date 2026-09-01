import type { Edge, Node } from "@xyflow/react";
import type { WidgetNodeData } from "../nodes/widget/WidgetNode";
import type { PyCodeEditorNodeData } from "../nodes/computation/PyCodeEditorNode";

type SetNodes = (updater: (nodes: Node[]) => Node[]) => void;

/**
 * Pushes a widget node's current `data.output` onto every connected
 * pyCodeEditorNode's `widgetOutputs` - the same effect the widget's own
 * "update" button (App.tsx's onConnect handler) already produces. Shared so
 * the manual UI path and the Widget Agent apply the exact same logic rather
 * than two copies that can drift.
 *
 * Returns the ids of the pyCodeEditorNode targets that were updated.
 */
export function pushWidgetOutputToConnectedCode(
  widgetNodeId: string,
  nodes: Node[],
  edges: Edge[],
  setNodes: SetNodes,
  targetNodeId?: string,
): string[] {
  const src = nodes.find((n) => n.id === widgetNodeId);
  if (!src || src.type !== "widgetNode") return [];
  const output = (src.data as WidgetNodeData).output;
  if (!output) return [];

  const targetIds = targetNodeId
    ? [targetNodeId]
    : edges
        .filter((e) => e.source === widgetNodeId)
        .map((e) => e.target)
        .filter(Boolean);

  const codeTargetIds = targetIds.filter(
    (tid) => nodes.find((n) => n.id === tid)?.type === "pyCodeEditorNode",
  );
  if (!codeTargetIds.length) return [];

  setNodes((nds) =>
    nds.map((n) => {
      if (!codeTargetIds.includes(n.id)) return n;

      const existing = (n.data as PyCodeEditorNodeData).widgetOutputs ?? [];
      const already = existing.some((e) => e.variable === output.variable);
      const nextWidgetOutputs = already
        ? existing.map((e) =>
            e.variable === output.variable
              ? { variable: output.variable, value: output.value }
              : e,
          )
        : [...existing, { variable: output.variable, value: output.value }];

      return {
        ...n,
        data: {
          ...n.data,
          widgetOutputs: nextWidgetOutputs,
        } as PyCodeEditorNodeData,
      };
    }),
  );

  return codeTargetIds;
}

/**
 * Triggers a pyCodeEditorNode's Run action from outside the node component,
 * by bumping `data.runToken` - the node's own effect (see
 * PyCodeEditorNode.tsx) watches that field and calls the same handleRun()
 * its Run button uses.
 */
export function triggerCodeNodeRun(nodeIds: string[], setNodes: SetNodes): void {
  if (!nodeIds.length) return;
  const tokens = new Map(nodeIds.map((id) => [id, crypto.randomUUID()]));

  setNodes((nds) =>
    nds.map((n) =>
      tokens.has(n.id)
        ? { ...n, data: { ...n.data, runToken: tokens.get(n.id) } }
        : n,
    ),
  );
}
