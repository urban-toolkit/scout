// Parses the "# Widget: 'name'" markers computeCodeGen.ts emits (see
// variableWireComment in computeCodeGen.ts) back out of a code node's
// source. This is the single source of truth for "which variables does this
// node need from a widget" - no separate metadata field to keep in sync,
// and the marker is deliberately short enough to type by hand: anyone can
// add "# Widget: 'x'" to a blank code node and get the same nudge, not just
// read it off a generated one.
export type RequiredVariable = {
  name: string;
  type: string | null;
};

// Drives PythonCodeEditor's inline highlighting - one entry per required
// variable, colored the same as the RequiredVariablesBadge popover row for
// that variable so the code and the badge read as one system.
export type VariableHighlight = {
  name: string;
  status: "unmet" | "resolved";
};

// "# Widget: 'name'" or "# Widget: 'name' (type)" - the current, short form.
const WIDGET_MARKER_RE =
  /#\s*Widget:\s*'([A-Za-z_][A-Za-z0-9_]*)'(?:\s*\(([^)]+)\))?/g;

// The original, more verbose marker this replaced - still parsed so code
// already saved with it (existing dataflows) keeps showing the nudge
// without needing to be regenerated.
const LEGACY_WIRE_COMMENT_RE =
  /#\s*Wire a widget with variable name '([A-Za-z_][A-Za-z0-9_]*)'(?:\s*\(type:\s*([^)]+)\))?\s*to set this parameter\./g;

export function parseRequiredVariables(code: string): RequiredVariable[] {
  const seen = new Map<string, RequiredVariable>();
  const addAll = (re: RegExp) => {
    for (const match of code.matchAll(re)) {
      const name = match[1];
      const type = match[2]?.replace(/^type:\s*/, "").trim() || null;
      // A variable can be requested more than once (e.g. reused across a
      // constructor and a method call) - keep the first sighting's type.
      if (!seen.has(name)) seen.set(name, { name, type });
    }
  };
  addAll(WIDGET_MARKER_RE);
  addAll(LEGACY_WIRE_COMMENT_RE);
  return [...seen.values()];
}
