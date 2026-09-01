import type { Edge, Node, ReactFlowInstance } from "@xyflow/react";

import type { WidgetNodeData } from "../nodes/widget/WidgetNode";
import {
  pushWidgetOutputToConnectedCode,
  triggerCodeNodeRun,
} from "../utils/widgetPropagation";

/** One choice-based widget on the canvas, as described to the LLM. */
export interface WidgetSummary {
  id: string;
  title: string;
  variable: string;
  wtype: string;
  /** "checkbox" widgets hold an array subset of `choices`; every other
   * wtype (radio-group, dropdown, ...) holds exactly one choice. */
  multiSelect: boolean;
  choices: unknown[];
  currentValue: unknown;
}

/** What the model is asked to return for one turn. */
export interface AgentAction {
  widgetId: string;
  value: unknown;
}

export interface AgentReply {
  message: string;
  action: AgentAction | null;
}

/**
 * Widget nodes whose grammar declares a `choices` array - the only shape
 * this agent is allowed to act on (see widgetAgent's scope: pick a value a
 * widget already offers, never write arbitrary code).
 */
export function collectWidgetSummaries(nodes: Node[]): WidgetSummary[] {
  return nodes
    .filter((n) => n.type === "widgetNode")
    .map((n) => {
      const data = n.data as WidgetNodeData;
      const widget = (data.value as any)?.widget;
      if (!widget || !Array.isArray(widget.choices)) return null;
      return {
        id: n.id,
        title: data.title ?? widget.props?.title ?? widget.variable,
        variable: widget.variable,
        wtype: widget.wtype,
        multiSelect: widget.wtype === "checkbox",
        choices: widget.choices,
        currentValue: data.output?.value ?? widget.default,
      } satisfies WidgetSummary;
    })
    .filter((w): w is WidgetSummary => w !== null);
}

export function buildSystemPrompt(widgets: WidgetSummary[]): string {
  const catalog = widgets.map((w) => ({
    widgetId: w.id,
    title: w.title,
    variable: w.variable,
    selectionType: w.multiSelect ? "multiple" : "single",
    choices: w.choices,
    currentValue: w.currentValue,
  }));

  return [
    "You are the Widget Agent inside SCOUT, a visual dataflow tool.",
    "Your only capability is picking new value(s) for one of the widgets listed below, from that widget's own `choices` - nothing else. You cannot write or edit code, add nodes, or do anything outside this one action.",
    "",
    "Widgets currently on the canvas:",
    JSON.stringify(catalog, null, 2),
    "",
    "Reply with ONLY a JSON object (no markdown fences, no other text) matching:",
    '{"message": string, "action": {"widgetId": string, "value": <see below>} | null}',
    "",
    "For a widget with selectionType \"single\", value must be exactly one item from that widget's choices (not an array).",
    'For a widget with selectionType "multiple", value must be an array containing only items from that widget\'s choices - the full set of choices that should end up selected, not just the ones being added or removed.',
    "",
    "Set action to null and explain in message if the request is unclear, refers to a widget/value that isn't listed, or asks for anything beyond picking value(s).",
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
          typeof parsed.action.widgetId === "string" &&
          "value" in parsed.action
            ? { widgetId: parsed.action.widgetId, value: parsed.action.value }
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
 * The real safety boundary: an action only applies if it names a widget
 * that's actually on the canvas and a value that widget's own `choices`
 * actually contains - regardless of what the model claims.
 */
export function validateAction(
  action: AgentAction | null,
  widgets: WidgetSummary[],
): AgentAction | null {
  if (!action) return null;
  const widget = widgets.find((w) => w.id === action.widgetId);
  if (!widget) return null;

  const isChoice = (v: unknown) =>
    widget.choices.some((c) => JSON.stringify(c) === JSON.stringify(v));

  if (widget.multiSelect) {
    const isAllowed =
      Array.isArray(action.value) && action.value.every(isChoice);
    return isAllowed ? action : null;
  }

  return isChoice(action.value) ? action : null;
}

/**
 * Applies a validated action: writes the widget's new output, pushes it to
 * connected Code nodes (mirrors the widget's own "update" button), and runs
 * those Code nodes (mirrors pressing their Run button).
 */
export function applyAgentAction(
  action: AgentAction,
  rf: ReactFlowInstance<Node, Edge>,
): { ranNodeIds: string[] } {
  const { widgetId, value } = action;
  const nodes = rf.getNodes();
  const widgetNode = nodes.find((n) => n.id === widgetId);
  const grammar = (widgetNode?.data as WidgetNodeData | undefined)?.value as
    | { widget?: { variable?: string } }
    | undefined;
  const variable = grammar?.widget?.variable;
  if (!widgetNode || !variable) return { ranNodeIds: [] };

  rf.setNodes((nds) =>
    nds.map((n) =>
      n.id === widgetId
        ? { ...n, data: { ...n.data, output: { variable, value } } }
        : n,
    ),
  );

  // pushWidgetOutputToConnectedCode reads the widget node's data.output, so
  // it needs the write above applied first - rf.getNodes() would still see
  // the stale value here, hence passing the updated list explicitly.
  const updatedNodes = nodes.map((n) =>
    n.id === widgetId
      ? { ...n, data: { ...n.data, output: { variable, value } } }
      : n,
  );
  const ranNodeIds = pushWidgetOutputToConnectedCode(
    widgetId,
    updatedNodes,
    rf.getEdges(),
    rf.setNodes,
  );
  triggerCodeNodeRun(ranNodeIds, rf.setNodes);

  return { ranNodeIds };
}
