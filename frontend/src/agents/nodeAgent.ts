import type { Edge, Node, ReactFlowInstance } from "@xyflow/react";

import {
  TEMPLATES,
  TEMPLATE_LABELS,
  TEMPLATE_NODE_TYPE,
  type TemplateKey,
} from "../templates";
import { pushInteractionToView } from "../utils/interactionPropagation";
import { pushWidgetOutputToConnectedCode } from "../utils/widgetPropagation";

/** "code" isn't a TemplateKey (it maps to pyCodeEditorNode, which has no
 * grammar template) - every other value is a real TemplateKey. */
export type NodeKind = TemplateKey | "code";

const NODE_KIND_LABELS: Record<NodeKind, string> = {
  ...TEMPLATE_LABELS,
  code: "Code",
};

/** One node already on the canvas, as described to the LLM. */
export interface NodeSummary {
  id: string;
  kind: NodeKind | null; // null for a node type this agent doesn't recognize
  title: string;
}

export type AgentAction =
  | { kind: "add"; nodeKind: NodeKind; title: string | null }
  | { kind: "rename"; nodeId: string; title: string };

export interface AgentReply {
  message: string;
  action: AgentAction | null;
}

const TYPE_TO_KIND: Record<string, NodeKind> = Object.fromEntries(
  Object.entries(TEMPLATE_NODE_TYPE).map(([tpl, type]) => [type, tpl as TemplateKey]),
);
TYPE_TO_KIND["pyCodeEditorNode"] = "code";

/** Every node on the canvas, with the id/title the agent needs to target
 * one for a rename - the only two actions this agent is allowed to take
 * (add a new node, rename an existing one). */
export function collectNodeSummaries(nodes: Node[]): NodeSummary[] {
  return nodes.map((n) => {
    const kind: NodeKind | null = n.type ? (TYPE_TO_KIND[n.type] ?? null) : null;
    return {
      id: n.id,
      kind,
      title: (n.data as { title?: string })?.title ?? (kind ? NODE_KIND_LABELS[kind] : "Untitled"),
    };
  });
}

export function buildSystemPrompt(nodes: NodeSummary[]): string {
  const kindCatalog = Object.entries(NODE_KIND_LABELS).map(([kind, label]) => ({ kind, label }));
  const nodeCatalog = nodes.map((n) => ({ nodeId: n.id, kind: n.kind, title: n.title }));

  return [
    "You are the Node Agent inside SCOUT, a visual dataflow tool.",
    "You have exactly two capabilities: (1) add a new, blank node of one of the kinds listed below onto the canvas, optionally giving it a title, and (2) rename an existing node (change its header text) by its nodeId. Nothing else - you cannot fill in a node's grammar/content, connect edges, delete nodes, or run anything.",
    "",
    "Node kinds you can add:",
    JSON.stringify(kindCatalog, null, 2),
    "",
    "Nodes currently on the canvas (for renaming - \"kind\" is null if it's some other node type you can't act on):",
    JSON.stringify(nodeCatalog, null, 2),
    "",
    "Reply with ONLY a JSON object (no markdown fences, no other text) matching ONE of:",
    '{"message": string, "action": {"type": "add", "nodeKind": string, "title": string | null} | null}',
    '{"message": string, "action": {"type": "rename", "nodeId": string, "title": string} | null}',
    "",
    '"nodeKind" must be exactly one of the kind values listed above. For "rename", "nodeId" must be exactly one of the nodeId values listed above.',
    "",
    "Set action to null and explain in message if the request is unclear, names a node kind or nodeId that isn't listed, or asks for anything beyond adding or renaming one node.",
  ].join("\n");
}

/** Pulls the first {...} JSON object out of a reply, tolerating stray prose
 * or markdown fences some models wrap structured output in. */
export function parseAgentReply(raw: string): AgentReply {
  const match = raw.match(/\{[\s\S]*\}/);
  if (match) {
    try {
      const parsed = JSON.parse(match[0]);
      if (typeof parsed?.message !== "string") {
        return { message: raw.trim(), action: null };
      }

      const a = parsed.action;
      let action: AgentAction | null = null;
      if (a && a.type === "add" && typeof a.nodeKind === "string") {
        action = {
          kind: "add",
          nodeKind: a.nodeKind,
          title: typeof a.title === "string" && a.title.trim() ? a.title.trim() : null,
        };
      } else if (a && a.type === "rename" && typeof a.nodeId === "string" && typeof a.title === "string") {
        action = { kind: "rename", nodeId: a.nodeId, title: a.title };
      }

      return { message: parsed.message, action };
    } catch {
      // fall through to the raw-text fallback below
    }
  }
  return { message: raw.trim(), action: null };
}

/**
 * The real safety boundary: an "add" only applies for a node kind that's
 * actually offered, and a "rename" only applies to a node that's actually
 * on the canvas - regardless of what the model claims.
 */
export function validateAction(
  action: AgentAction | null,
  nodes: NodeSummary[],
): AgentAction | null {
  if (!action) return null;

  if (action.kind === "add") {
    return action.nodeKind in NODE_KIND_LABELS ? action : null;
  }

  return nodes.some((n) => n.id === action.nodeId) ? action : null;
}

let nextAgentNodeId = 1;

/**
 * Applies a validated action:
 * - "add" creates a new node exactly the way NodeRail's click/drag path
 *   does (App.tsx's createGrammarNode/createPyCodeEditorNode) - same
 *   template data, same onChange/onRun wiring (via the same shared
 *   interaction/widget propagation utilities App.tsx itself now delegates
 *   to), just at a staggered default position since there's no drop point
 *   from a mouse event to work from here.
 * - "rename" sets data.title - the same field both BaseGrammarNode's and
 *   PyCodeEditorNode's own header-text inputs write to.
 */
export function applyAgentAction(
  action: AgentAction,
  rf: ReactFlowInstance<Node, Edge>,
): void {
  if (action.kind === "rename") {
    rf.setNodes((nds) =>
      nds.map((n) => (n.id === action.nodeId ? { ...n, data: { ...n.data, title: action.title } } : n)),
    );
    return;
  }

  const count = rf.getNodes().length;
  const position = { x: 120 + (count % 5) * 260, y: 120 + Math.floor(count / 5) * 220 };
  const id = `agent-node-${nextAgentNodeId++}`;

  if (action.nodeKind === "code") {
    rf.setNodes((nds) =>
      nds.concat({
        id,
        type: "pyCodeEditorNode",
        position,
        width: 400,
        data: { title: action.title ?? undefined },
      }),
    );
    return;
  }

  const template = action.nodeKind;
  rf.setNodes((nds) =>
    nds.concat({
      id,
      type: TEMPLATE_NODE_TYPE[template],
      position,
      data: {
        value: TEMPLATES[template] ?? {},
        title: action.title ?? undefined,
        onChange: (val: unknown, targetId: string) => {
          rf.setNodes((n) => n.map((node) => (node.id === targetId ? { ...node, data: { ...node.data, value: val } } : node)));
        },
        onRun: (nodeId: string) => {
          const node = rf.getNode(nodeId);
          if (!node) return;
          if (node.type === "interactionNode") {
            return pushInteractionToView(nodeId, rf.getNodes(), rf.getEdges(), rf.setNodes);
          } else if (node.type === "widgetNode") {
            return pushWidgetOutputToConnectedCode(nodeId, rf.getNodes(), rf.getEdges(), rf.setNodes).length > 0;
          }
        },
      },
    }),
  );
}
