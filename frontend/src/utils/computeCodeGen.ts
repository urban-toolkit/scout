import type { ComputeCatalogEntry } from "./computeCatalog";

export type ParamChoice = {
  name: string;
  mode: "variable" | "fixed";
  // Raw Python source the user typed for a "fixed" param (e.g. `"chicago"`,
  // `18`, `[41.8, 41.9]`) - spliced verbatim into the call, not re-quoted,
  // so the user has full control over the literal's type.
  fixedRepr?: string;
  // The param's type annotation, if any (e.g. "int", "str") - carried
  // through from ComputeParam so a "variable" param's comment (see
  // variableWireComment) can mention it. Unused for "fixed" params.
  type?: string | null;
};

export type ComputeSelection =
  | { kind: "function"; callableName: string; args: ParamChoice[] }
  | {
      kind: "class";
      callableName: string;
      ctorArgs: ParamChoice[];
      methodName: string;
      methodArgs: ParamChoice[];
    };

function moduleOf(importPath: string): string {
  return importPath.split(".").slice(0, -1).join(".");
}

function lowerFirst(name: string): string {
  return name.length ? name.charAt(0).toLowerCase() + name.slice(1) : name;
}

// A "variable" param becomes a bare identifier the widgetOutputs splice
// mechanism defines by prepending "<name> = <value>" ahead of this node's
// own code (see PyCodeEditorNode.tsx's handleRun) - both pieces run in the
// same exec() call, in that order. This must stay a comment, not an actual
// assignment: an assignment here would run *after* the widget's own and
// silently overwrite it back to whatever this line set, on every single
// run - which is exactly what a `= None` placeholder used to do, clobbering
// an otherwise-correctly-wired widget every time. If no widget is wired
// yet, the call below simply raises a NameError - a clear, honest signal
// rather than a silently-wrong None.
function variableWireComment(choice: ParamChoice): string {
  const typeNote = choice.type ? ` (type: ${choice.type})` : "";
  return `# Wire a widget with variable name '${choice.name}'${typeNote} to set this parameter.`;
}

function callArg(choice: ParamChoice): string {
  if (choice.mode === "variable") return `${choice.name}=${choice.name}`;
  const literal = (choice.fixedRepr ?? "").trim();
  return `${choice.name}=${literal || "None"}`;
}

export function generateComputeNodeCode(
  entry: ComputeCatalogEntry,
  selection: ComputeSelection,
): { code: string; title: string } {
  const callable = entry.callables.find((c) => c.name === selection.callableName);
  if (!callable) throw new Error(`Callable '${selection.callableName}' not found on '${entry.id}'`);

  const lines: string[] = [];

  if (selection.kind === "function") {
    if (callable.kind !== "function") {
      throw new Error(`'${selection.callableName}' is not a function`);
    }
    lines.push(`from ${moduleOf(callable.importPath)} import ${callable.name}`, "");
    for (const arg of selection.args) {
      if (arg.mode === "variable") lines.push(variableWireComment(arg));
    }
    const argsCode = selection.args.map(callArg).join(", ");
    lines.push(`${callable.name}(${argsCode})`);

    return { code: lines.join("\n") + "\n", title: entry.displayName };
  }

  if (callable.kind !== "class") {
    throw new Error(`'${selection.callableName}' is not a class`);
  }
  const method = callable.methods.find((m) => m.name === selection.methodName);
  if (!method) {
    throw new Error(`Method '${selection.methodName}' not found on '${selection.callableName}'`);
  }

  const instanceVar = lowerFirst(callable.name);
  lines.push(`from ${moduleOf(callable.importPath)} import ${callable.name}`, "");
  for (const arg of selection.ctorArgs) {
    if (arg.mode === "variable") lines.push(variableWireComment(arg));
  }
  const ctorArgsCode = selection.ctorArgs.map(callArg).join(", ");
  lines.push(`${instanceVar} = ${callable.name}(${ctorArgsCode})`);

  for (const arg of selection.methodArgs) {
    if (arg.mode === "variable") lines.push(variableWireComment(arg));
  }
  const methodArgsCode = selection.methodArgs.map(callArg).join(", ");
  lines.push(`${instanceVar}.${method.name}(${methodArgsCode})`);

  return {
    code: lines.join("\n") + "\n",
    title: entry.displayName,
  };
}
