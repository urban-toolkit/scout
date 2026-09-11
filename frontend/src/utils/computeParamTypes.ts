// Reduces a param's raw annotation text (from the backend's AST unparse,
// e.g. "int", "Optional[str]", "str | None") down to the handful of kinds
// the config dialog shows an example placeholder for. Anything else - a
// custom class, a generic we don't recognize, or no annotation at all -
// resolves to null, meaning "no example to offer" - never a wall: the
// fixed-value field always accepts whatever raw Python literal the user
// types, this only decides what hint text to show alongside it.
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

// A one-line example of a real Python literal for this type - e.g. '"value"'
// (quotes included, since the field holds splice-ready source text, not raw
// content) - shown as the field's placeholder purely as a hint. The user is
// free to type anything else; nothing here is validated or reformatted.
export function exampleLiteralForType(type: SimpleParamType): string {
  switch (type) {
    case "str":
      return '"value"';
    case "int":
      return "10";
    case "float":
      return "1.5";
    case "bool":
      return "False";
    default:
      return "value";
  }
}
