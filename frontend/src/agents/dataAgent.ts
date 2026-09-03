import type { Edge, Node, ReactFlowInstance } from "@xyflow/react";

import type { PyCodeEditorNodeData } from "../nodes/computation/PyCodeEditorNode";
import { triggerCodeNodeRun } from "../utils/widgetPropagation";

/** One Python Code node on the canvas, as described to the LLM. */
export interface PyCodeNodeSummary {
  id: string;
  title: string;
  code: string;
}

/** What the model is asked to return for one turn. */
export interface AgentAction {
  nodeId: string;
  code: string;
}

export interface AgentReply {
  message: string;
  action: AgentAction | null;
}

/**
 * Code nodes on the canvas, with their current contents - the only node
 * type this agent is allowed to act on (see dataAgent's scope: replace one
 * Code node's code, never add nodes, widgets, or anything else).
 */
export function collectPyCodeNodeSummaries(nodes: Node[]): PyCodeNodeSummary[] {
  return nodes
    .filter((n) => n.type === "pyCodeEditorNode")
    .map((n) => {
      const data = n.data as PyCodeEditorNodeData;
      return {
        id: n.id,
        title: data.title ?? "Code",
        code: data.code ?? "",
      } satisfies PyCodeNodeSummary;
    });
}

export function buildSystemPrompt(pyNodes: PyCodeNodeSummary[]): string {
  const catalog = pyNodes.map((n) => ({
    nodeId: n.id,
    title: n.title,
    currentCode: n.code,
  }));

  return [
    "You are the Data Agent inside SCOUT, a visual dataflow tool.",
    "Your only capability is writing or rewriting the Python code inside one of the Code nodes listed below - nothing else. You cannot add/remove nodes, edit widgets, connect edges, or do anything outside replacing one Code node's code.",
    "",
    "Code nodes currently on the canvas:",
    JSON.stringify(catalog, null, 2),
    "",
    "Reply with ONLY a JSON object (no markdown fences, no other text) matching:",
    '{"message": string, "action": {"nodeId": string, "code": string} | null}',
    "",
    '"code" must be the COMPLETE new contents of that node, not a diff or a snippet to insert - it fully replaces whatever code is there now.',
    "",
    "Set action to null and explain in message if the request is unclear, refers to a node that isn't listed, or asks for anything beyond writing code into one Code node.",
  ].join("\n");
}

/** Pulls the first {...} JSON object out of a reply, tolerating stray prose
 * or markdown fences some models wrap structured output in. */
export function parseAgentReply(raw: string): AgentReply {
  const match = raw.match(/\{[\s\S]*\}/);
  if (match) {
    try {
      const parsed = JSON.parse(match[0]);
      if (typeof parsed?.message === "string") {
        const action =
          parsed.action &&
          typeof parsed.action.nodeId === "string" &&
          typeof parsed.action.code === "string"
            ? { nodeId: parsed.action.nodeId, code: parsed.action.code }
            : null;
        return { message: parsed.message, action };
      }
    } catch {
      // fall through to the raw-text fallback below
    }
  }
  return { message: raw.trim(), action: null };
}

/**
 * The real safety boundary: an action only applies if it names a Code node
 * that's actually on the canvas - regardless of what the model claims.
 */
export function validateAction(
  action: AgentAction | null,
  pyNodes: PyCodeNodeSummary[],
): AgentAction | null {
  if (!action) return null;
  return pyNodes.some((n) => n.id === action.nodeId) ? action : null;
}

/**
 * Applies a validated action: overwrites the Code node's code, then re-runs
 * it - mirroring the node's own Run button - via the same runToken hook the
 * Widget Agent uses to re-run nodes it just updated (see
 * utils/widgetPropagation.ts).
 */
export function applyAgentAction(
  action: AgentAction,
  rf: ReactFlowInstance<Node, Edge>,
): void {
  rf.setNodes((nds) =>
    nds.map((n) =>
      n.id === action.nodeId ? { ...n, data: { ...n.data, code: action.code } } : n,
    ),
  );

  triggerCodeNodeRun([action.nodeId], rf.setNodes);
}
