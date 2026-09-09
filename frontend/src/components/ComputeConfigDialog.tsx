import { useEffect, useMemo, useState } from "react";
import Dialog from "@mui/material/Dialog";
import DialogTitle from "@mui/material/DialogTitle";
import DialogContent from "@mui/material/DialogContent";
import DialogActions from "@mui/material/DialogActions";
import Button from "@mui/material/Button";
import TextField from "@mui/material/TextField";
import Chip from "@mui/material/Chip";
import ToggleButton from "@mui/material/ToggleButton";
import ToggleButtonGroup from "@mui/material/ToggleButtonGroup";
import Alert from "@mui/material/Alert";
import Stack from "@mui/material/Stack";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";

import type { ComputeCallable, ComputeParam } from "../utils/computeCatalog";
import {
  generateComputeNodeCode,
  type ParamChoice,
  type ComputeSelection,
} from "../utils/computeCodeGen";
import {
  simplifyParamType,
  validateFixedValue,
  pythonStringLiteral,
  contentFromPythonStringLiteral,
} from "../utils/computeParamTypes";
import type { ProjectComputeItem } from "../utils/dataflows";

interface Props {
  open: boolean;
  onClose: () => void;
  entry: ProjectComputeItem | null;
  onCreate: (result: { code: string; title: string; selection: ComputeSelection }) => void;
}

// Matches the monospace look of the in-app Python editor (CodeMirror) -
// used anywhere a function/param/value name is shown, so the picker reads
// as "code" rather than ordinary UI text.
const CODE_FONT =
  '"Fira Code", "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';

const SECTION_LABEL_SX = {
  fontSize: 11,
  fontWeight: 700,
  color: "#94a3b8",
  textTransform: "uppercase" as const,
  letterSpacing: "0.06em",
};

// A required param (no default) most likely needs a real value supplied at
// run time, so it defaults to "variable" (wireable via a widget later); an
// optional one defaults to "fixed", prefilled with its own default so the
// generated call is runnable immediately without any further editing.
function initialChoice(p: ComputeParam): ParamChoice {
  return p.hasDefault
    ? { name: p.name, mode: "fixed", fixedRepr: p.defaultRepr ?? "", type: p.type }
    : { name: p.name, mode: "variable", type: p.type };
}

// null when the choice isn't a checkable "fixed" value (either it's wired
// as a variable, or its type isn't one we validate - see simplifyParamType).
function paramValidationError(param: ComputeParam, choice: ParamChoice): string | null {
  if (choice.mode !== "fixed") return null;
  return validateFixedValue(simplifyParamType(param.type), choice.fixedRepr ?? "");
}

function hasValidationErrors(params: ComputeParam[], choices: Record<string, ParamChoice>): boolean {
  return params.some((p) => paramValidationError(p, choices[p.name] ?? initialChoice(p)) !== null);
}

// Seeds a param-choices map from a remembered selection's args (see
// ProjectComputeItem.lastSelection) where possible, falling back to
// initialChoice per-param otherwise - so a param the underlying script no
// longer has (or never had, if remembered is undefined) never breaks the
// rest of the form, it just doesn't get a remembered value.
function seedChoices(
  params: ComputeParam[],
  remembered: ParamChoice[] | undefined,
): Record<string, ParamChoice> {
  return Object.fromEntries(
    params.map((p) => [p.name, remembered?.find((a) => a.name === p.name) ?? initialChoice(p)]),
  );
}

// Just the filename, not the full path within the package (e.g.
// "scripts/flood_simulation.py" -> "flood_simulation.py") - shown in the
// Script pills; the full path is still what's actually used to match a
// callable to its script (see sourceFile), only the label is shortened.
function scriptBasename(sourceFile: string): string {
  const parts = sourceFile.split("/");
  return parts[parts.length - 1] || sourceFile;
}

function CallablePills({
  items,
  selected,
  onSelect,
}: {
  items: { name: string; label?: string }[];
  selected: string;
  onSelect: (name: string) => void;
}) {
  return (
    <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.75 }}>
      {items.map((it) => (
        <Chip
          key={it.name}
          label={it.label ?? it.name}
          clickable
          size="small"
          onClick={() => onSelect(it.name)}
          variant={selected === it.name ? "filled" : "outlined"}
          color={selected === it.name ? "primary" : "default"}
          sx={{
            fontFamily: CODE_FONT,
            fontSize: 11,
            fontWeight: 400,
            borderRadius: "999px",
          }}
        />
      ))}
    </Box>
  );
}

function ParamCard({
  param,
  choice,
  onChange,
}: {
  param: ComputeParam;
  choice: ParamChoice;
  onChange: (next: ParamChoice) => void;
}) {
  const simpleType = simplifyParamType(param.type);
  const errorText = paramValidationError(param, choice);
  const setFixed = (fixedRepr: string) =>
    onChange({ name: param.name, mode: "fixed", fixedRepr, type: param.type });

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        gap: 0.5,
        border: "1px solid #e2e8f0",
        borderRadius: 1.5,
        p: 1,
        minWidth: 150,
      }}
    >
      <Box sx={{ display: "flex", alignItems: "baseline", gap: 0.6 }}>
        <Typography sx={{ fontFamily: CODE_FONT, fontSize: 12.5, fontWeight: 600, color: "#0f172a" }}>
          {param.name}
        </Typography>
        {param.type && (
          <Typography sx={{ fontFamily: CODE_FONT, fontSize: 10, color: "#94a3b8" }}>
            {param.type}
          </Typography>
        )}
      </Box>
      <ToggleButtonGroup
        value={choice.mode}
        exclusive
        size="small"
        sx={{ alignSelf: "flex-start" }}
        onChange={(_, next) => {
          if (!next) return;
          onChange(
            next === "variable"
              ? { name: param.name, mode: "variable", type: param.type }
              : { name: param.name, mode: "fixed", fixedRepr: param.defaultRepr ?? "", type: param.type },
          );
        }}
      >
        <ToggleButton value="variable" sx={{ fontSize: 10, textTransform: "none", py: 0.15, px: 0.9 }}>
          var
        </ToggleButton>
        <ToggleButton value="fixed" sx={{ fontSize: 10, textTransform: "none", py: 0.15, px: 0.9 }}>
          fixed
        </ToggleButton>
      </ToggleButtonGroup>
      {choice.mode === "fixed" &&
        (simpleType === "bool" ? (
          <ToggleButtonGroup
            value={choice.fixedRepr === "True" || choice.fixedRepr === "False" ? choice.fixedRepr : null}
            exclusive
            size="small"
            onChange={(_, next) => {
              if (next) setFixed(next);
            }}
          >
            <ToggleButton value="True" sx={{ fontSize: 10, textTransform: "none", py: 0.15, px: 0.9 }}>
              True
            </ToggleButton>
            <ToggleButton value="False" sx={{ fontSize: 10, textTransform: "none", py: 0.15, px: 0.9 }}>
              False
            </ToggleButton>
          </ToggleButtonGroup>
        ) : (
          <TextField
            size="small"
            type={simpleType === "int" || simpleType === "float" ? "number" : "text"}
            placeholder={simpleType === "str" ? "text" : "value"}
            error={Boolean(errorText)}
            value={
              simpleType === "str"
                ? contentFromPythonStringLiteral(choice.fixedRepr ?? "")
                : (choice.fixedRepr ?? "")
            }
            onChange={(e) =>
              setFixed(simpleType === "str" ? pythonStringLiteral(e.target.value) : e.target.value)
            }
            sx={{
              width: 130,
              "& .MuiInputBase-input": { fontFamily: CODE_FONT, fontSize: 12 },
            }}
          />
        ))}
      {errorText && (
        <Typography sx={{ fontSize: 10, color: "#dc2626" }}>{errorText}</Typography>
      )}
    </Box>
  );
}

function ParamGrid({
  label,
  params,
  choices,
  onChange,
}: {
  label: string;
  params: ComputeParam[];
  choices: Record<string, ParamChoice>;
  onChange: (name: string, next: ParamChoice) => void;
}) {
  if (params.length === 0) return null;
  return (
    <Stack spacing={0.75}>
      <Typography sx={SECTION_LABEL_SX}>{label}</Typography>
      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
        {params.map((p) => (
          <ParamCard
            key={p.name}
            param={p}
            choice={choices[p.name] ?? initialChoice(p)}
            onChange={(next) => onChange(p.name, next)}
          />
        ))}
      </Box>
    </Stack>
  );
}

export default function ComputeConfigDialog({ open, onClose, entry, onCreate }: Props) {
  const [sourceFile, setSourceFile] = useState("");
  const [callableName, setCallableName] = useState("");
  const [methodName, setMethodName] = useState("");
  const [ctorChoices, setCtorChoices] = useState<Record<string, ParamChoice>>({});
  const [argChoices, setArgChoices] = useState<Record<string, ParamChoice>>({});
  const [methodChoices, setMethodChoices] = useState<Record<string, ParamChoice>>({});
  const [error, setError] = useState<string | null>(null);

  // Every distinct script (source file) this entry has at least one
  // callable in - a package uploaded as a zip/folder can span several, e.g.
  // "scripts/convert_to_raster.py" and "scripts/load_static.py". Shown as
  // its own picker step so a multi-script package doesn't dump every
  // function from every file into one flat list.
  const scriptFiles = useMemo(() => {
    if (!entry) return [];
    return Array.from(new Set(entry.callables.map((c) => c.sourceFile))).sort();
  }, [entry]);

  const callablesInScript = useMemo(
    () => entry?.callables.filter((c) => c.sourceFile === sourceFile) ?? [],
    [entry, sourceFile],
  );

  const callable: ComputeCallable | undefined = useMemo(
    () => callablesInScript.find((c) => c.name === callableName),
    [callablesInScript, callableName],
  );

  // Picks the script whenever a different entry is opened - seeded from
  // lastSelection's own script when that callable still exists, or the
  // first script otherwise. Deliberately doesn't touch callableName here;
  // the effect below (keyed on sourceFile) handles that, so it runs the
  // same way whether the script came from opening the dialog or a manual
  // pill click.
  useEffect(() => {
    if (!open || !entry) return;
    setError(null);
    const last = entry.lastSelection;
    const lastCallable = last && entry.callables.find((c) => c.name === last.callableName);
    setSourceFile(lastCallable?.sourceFile ?? entry.callables[0]?.sourceFile ?? "");
  }, [open, entry]);

  // Picks the callable within the currently-selected script - from the
  // remembered lastSelection if it happens to belong to this script,
  // otherwise the script's first callable. Runs after the effect above sets
  // sourceFile, and also runs standalone on a manual script-pill click.
  useEffect(() => {
    if (!entry) {
      setCallableName("");
      return;
    }
    const last = entry.lastSelection;
    const lastValidHere = last && callablesInScript.some((c) => c.name === last.callableName);
    setCallableName(lastValidHere ? last.callableName : (callablesInScript[0]?.name ?? ""));
  }, [sourceFile, entry, callablesInScript]);

  // Re-derive param choices whenever the selected callable (or, for a
  // class, its selected method) changes - from the remembered lastSelection
  // when it matches this callable, otherwise fresh defaults.
  useEffect(() => {
    if (!callable) {
      setCtorChoices({});
      setArgChoices({});
      setMethodChoices({});
      setMethodName("");
      return;
    }
    const last = entry?.lastSelection;
    const lastMatches = last?.callableName === callable.name;

    if (callable.kind === "function") {
      setArgChoices(
        seedChoices(callable.params, lastMatches && last?.kind === "function" ? last.args : undefined),
      );
      setCtorChoices({});
      setMethodChoices({});
      setMethodName("");
    } else {
      const lastClassSelection = lastMatches && last?.kind === "class" ? last : undefined;
      setCtorChoices(seedChoices(callable.ctorParams, lastClassSelection?.ctorArgs));
      setArgChoices({});
      const rememberedMethodStillExists =
        lastClassSelection &&
        callable.methods.some((m) => m.name === lastClassSelection.methodName);
      setMethodName(
        rememberedMethodStillExists
          ? lastClassSelection.methodName
          : (callable.methods[0]?.name ?? ""),
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [callable?.name]);

  // For a class, method choices depend on which method is selected, so this
  // is a separate effect keyed on methodName rather than callable.name.
  useEffect(() => {
    if (!callable || callable.kind !== "class") return;
    const method = callable.methods.find((m) => m.name === methodName);
    const last = entry?.lastSelection;
    const lastMatches =
      last?.kind === "class" && last.callableName === callable.name && last.methodName === methodName;
    setMethodChoices(seedChoices(method?.params ?? [], lastMatches ? last.methodArgs : undefined));
  }, [callable, methodName, entry]);

  if (!entry) return null;

  const handleCreate = () => {
    if (!callable) return;
    try {
      const selection: ComputeSelection =
        callable.kind === "function"
          ? {
              kind: "function",
              callableName: callable.name,
              args: callable.params.map((p) => argChoices[p.name] ?? initialChoice(p)),
            }
          : {
              kind: "class",
              callableName: callable.name,
              ctorArgs: callable.ctorParams.map((p) => ctorChoices[p.name] ?? initialChoice(p)),
              methodName,
              methodArgs: (callable.methods.find((m) => m.name === methodName)?.params ?? []).map(
                (p) => methodChoices[p.name] ?? initialChoice(p),
              ),
            };
      const result = generateComputeNodeCode(entry, selection);
      onCreate({ ...result, selection });
    } catch (e: any) {
      setError(e?.message || "Failed to generate code for this selection.");
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle sx={{ pb: 0.5 }}>
        Configure
        <Typography variant="body2" sx={{ color: "#64748b", fontWeight: 400, mt: 0.25 }}>
          {entry.displayName}
        </Typography>
      </DialogTitle>
      <DialogContent>
        <Stack spacing={2.5} sx={{ mt: 0.5 }}>
          {entry.callables.length === 0 ? (
            <Alert severity="warning">
              No importable function or class was found in this upload - it can still be added to
              the canvas as a blank code node and wired up by hand.
            </Alert>
          ) : (
            <>
              <Stack spacing={0.75}>
                <Typography sx={SECTION_LABEL_SX}>Script</Typography>
                <CallablePills
                  items={scriptFiles.map((s) => ({ name: s, label: scriptBasename(s) }))}
                  selected={sourceFile}
                  onSelect={setSourceFile}
                />
              </Stack>

              <Stack spacing={0.75}>
                <Typography sx={SECTION_LABEL_SX}>Function</Typography>
                <CallablePills
                  items={callablesInScript.map((c) => ({ name: c.name }))}
                  selected={callableName}
                  onSelect={setCallableName}
                />
              </Stack>

              {callable?.kind === "class" &&
                (callable.methods.length > 0 ? (
                  <Stack spacing={0.75}>
                    <Typography sx={SECTION_LABEL_SX}>Method</Typography>
                    <CallablePills
                      items={callable.methods.map((m) => ({ name: m.name }))}
                      selected={methodName}
                      onSelect={setMethodName}
                    />
                  </Stack>
                ) : (
                  <Typography variant="caption" color="text.secondary">
                    This class has no public methods besides its constructor.
                  </Typography>
                ))}

              {callable?.kind === "function" && (
                <ParamGrid
                  label="Parameters"
                  params={callable.params}
                  choices={argChoices}
                  onChange={(name, next) => setArgChoices((prev) => ({ ...prev, [name]: next }))}
                />
              )}

              {callable?.kind === "class" && (
                <ParamGrid
                  label="Constructor parameters"
                  params={callable.ctorParams}
                  choices={ctorChoices}
                  onChange={(name, next) => setCtorChoices((prev) => ({ ...prev, [name]: next }))}
                />
              )}

              {callable?.kind === "class" && methodName && (
                <ParamGrid
                  label={`"${methodName}" parameters`}
                  params={callable.methods.find((m) => m.name === methodName)?.params ?? []}
                  choices={methodChoices}
                  onChange={(name, next) => setMethodChoices((prev) => ({ ...prev, [name]: next }))}
                />
              )}
            </>
          )}

          {error && <Alert severity="error">{error}</Alert>}
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          onClick={handleCreate}
          variant="contained"
          disabled={
            !callable ||
            (callable.kind === "class" && !methodName && callable.methods.length > 0) ||
            (callable?.kind === "function" && hasValidationErrors(callable.params, argChoices)) ||
            (callable?.kind === "class" &&
              (hasValidationErrors(callable.ctorParams, ctorChoices) ||
                hasValidationErrors(
                  callable.methods.find((m) => m.name === methodName)?.params ?? [],
                  methodChoices,
                )))
          }
        >
          Create
        </Button>
      </DialogActions>
    </Dialog>
  );
}
