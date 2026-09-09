// Reduces a param's raw annotation text (from the backend's AST unparse,
// e.g. "int", "Optional[str]", "str | None") down to the handful of kinds
// the config dialog gives special input controls/validation to. Anything
// else - a custom class, a generic we don't recognize, or no annotation at
// all - resolves to null, meaning "treat it like an unannotated param": a
// freeform text field, nothing blocked. We only ever narrow behavior for
// types we're confident about; an unrecognized type never becomes a wall.
export type SimpleParamType = "int" | "float" | "bool" | "str" | null;

export function simplifyParamType(raw: string | null | undefined): SimpleParamType {
  if (!raw) return null;
  const cleaned = raw.replace(/\s+/g, "");
  const optionalMatch = cleaned.match(/^Optional\[(.+)\]$/);
  const inner = optionalMatch
    ? optionalMatch[1]
    : cleaned.replace(/\|None$/, "").replace(/^None\|/, "");
  switch (inner) {
    case "int":
      return "int";
    case "float":
      return "float";
    case "bool":
      return "bool";
    case "str":
      return "str";
    default:
      return null;
  }
}

// Validates a splice-ready literal (what actually ends up in the generated
// code, e.g. "16" or "True") against the simplified type. Returns an error
// message, or null if it's valid - or if the type isn't one we check (str
// accepts any text, since pythonStringLiteral below can always quote it;
// null/unrecognized types are never checked, same as today's behavior).
export function validateFixedValue(type: SimpleParamType, literal: string): string | null {
  const trimmed = literal.trim();
  if (type === "int") {
    return /^[-+]?\d+$/.test(trimmed) ? null : "Expected a whole number";
  }
  if (type === "float") {
    return /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(trimmed) ? null : "Expected a number";
  }
  if (type === "bool") {
    return trimmed === "True" || trimmed === "False" ? null : "Expected True or False";
  }
  return null;
}

// str-typed params are edited as raw content ("chicago"), not Python literal
// syntax ('"chicago"') - these two convert between the two representations
// so the rest of the app (ParamChoice.fixedRepr, codegen) only ever sees
// real Python source text, same as every other type.
export function pythonStringLiteral(content: string): string {
  return JSON.stringify(content);
}

// Best-effort unquote for displaying an existing literal (e.g. a default
// value's source text) as raw content. Only strips a single matching pair of
// quote characters - doesn't attempt full Python escape-sequence decoding,
// which is unnecessary for the simple values these fields realistically hold.
export function contentFromPythonStringLiteral(source: string): string {
  const trimmed = source.trim();
  if (trimmed.length >= 2) {
    const first = trimmed[0];
    const last = trimmed[trimmed.length - 1];
    if ((first === '"' || first === "'") && first === last) {
      return trimmed.slice(1, -1);
    }
  }
  return trimmed;
}
