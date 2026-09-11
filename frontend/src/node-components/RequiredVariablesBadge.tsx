import { memo, useEffect, useRef, useState } from "react";
import type { RequiredVariable } from "../utils/requiredVariables";
import type { WidgetOutput } from "../utils/types";
import "./RequiredVariablesBadge.css";

type Props = {
  requiredVars: RequiredVariable[];
  // Keyed by variable name - present means a widget currently supplies it
  // (see PyCodeEditorNode.tsx: built from data.widgetOutputs, the same
  // list handleRun already splices into the code before exec).
  resolvedByName: Map<string, WidgetOutput>;
};

function formatValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "string") return v.length > 22 ? `${v.slice(0, 19)}…` : v;
  if (Array.isArray(v)) return v.length ? `${v.length} selected` : "—";
  if (typeof v === "object") return "object";
  return String(v);
}

function WarnIcon({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 20 20" fill="none">
      <path d="M10 3.5L18 17H2L10 3.5Z" stroke="#b45309" strokeWidth="1.7" strokeLinejoin="round" />
      <line x1="10" y1="8.5" x2="10" y2="12" stroke="#b45309" strokeWidth="1.7" strokeLinecap="round" />
      <circle cx="10" cy="14.5" r="1" fill="#b45309" />
    </svg>
  );
}

function CheckIcon({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 20 20" fill="none">
      <polyline points="4,10.5 8,14.5 16,5.5" stroke="#059669" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function ChecklistIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
      <rect x="3" y="4" width="14" height="2" rx="1" fill="#92400e" />
      <rect x="3" y="9" width="14" height="2" rx="1" fill="#92400e" />
      <rect x="3" y="14" width="9" height="2" rx="1" fill="#92400e" />
    </svg>
  );
}

const RequiredVariablesBadge = memo(function RequiredVariablesBadge({
  requiredVars,
  resolvedByName,
}: Props) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as globalThis.Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  if (requiredVars.length === 0) return null;

  const unresolvedCount = requiredVars.filter((v) => !resolvedByName.has(v.name)).length;
  const allResolved = unresolvedCount === 0;

  return (
    <div className="reqvars nodrag" ref={rootRef}>
      <button
        type="button"
        className={`reqvars-badge ${allResolved ? "reqvars-badge--ok" : "reqvars-badge--warn"}`}
        onClick={() => setOpen((o) => !o)}
        title={allResolved ? "All required variables are wired" : `${unresolvedCount} variable${unresolvedCount === 1 ? "" : "s"} need a widget`}
      >
        {allResolved ? <CheckIcon /> : <ChecklistIcon />}
        {!allResolved && <span className="reqvars-badge__count">{unresolvedCount}</span>}
      </button>

      {open && (
        <div className="reqvars-popover nowheel">
          <div className="reqvars-popover__caret" />
          <div className="reqvars-popover__header">
            <div className="reqvars-popover__title">Required variables</div>
            <div className="reqvars-popover__subtitle">
              {unresolvedCount} unresolved of {requiredVars.length}
            </div>
          </div>
          <div className="reqvars-popover__divider" />
          <div className="reqvars-popover__list">
            {requiredVars.map((v) => {
              const resolved = resolvedByName.get(v.name);
              return (
                <div className="reqvars-row" key={v.name}>
                  <div className={`reqvars-row__icon ${resolved ? "reqvars-row__icon--ok" : "reqvars-row__icon--warn"}`}>
                    {resolved ? <CheckIcon size={12} /> : <WarnIcon size={12} />}
                  </div>
                  <div className="reqvars-row__main">
                    <div className="reqvars-row__name">
                      {v.name}
                      {v.type && <span className="reqvars-row__type"> · {v.type}</span>}
                    </div>
                    <div className={`reqvars-row__status ${resolved ? "reqvars-row__status--ok" : "reqvars-row__status--warn"}`}>
                      {resolved ? `Connected · ${formatValue(resolved.value)}` : "Needs widget"}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
});

export default RequiredVariablesBadge;
