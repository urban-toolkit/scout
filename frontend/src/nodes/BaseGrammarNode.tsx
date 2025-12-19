import {
  memo,
  useCallback,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
} from "react";
import { NodeResizer, useReactFlow } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import Ajv, { ErrorObject } from "ajv";
import addFormats from "ajv-formats";
import JsonCodeEditor from "../components/JsonCodeEditor";
import "./BaseGrammarNode.css";
import restartPng from "../assets/restart.png";

export type GrammarValue = unknown;

const nodeColor: Record<string, string> = {
  physical_layer: "#f5d1d2",
  data_layer: "#f5d1d2",
  join: "#f5d1d2",
  interaction: "#D2E4F0",
  view: "#D3E8DA",
  widget: "#D2E4F0",
  comparison: "#D2E4F0",
};

const nodeBorderColor: Record<string, string> = {
  physical_layer: "#cb181d",
  data_layer: "#cb181d",
  join: "#cb181d",
  interaction: "#1f78b4",
  view: "#238b45",
  widget: "#1f78b4",
  comparison: "#1f78b4",
};

export type BaseNodeData = {
  value: GrammarValue;
  title?: string;
  onChange?: (val: GrammarValue, id: string) => void;
  onClose?: (id: string) => void;
  onRun?: (id: string) => void;

  schema?: object;
  pickInner?: (v: GrammarValue) => unknown;

  footerActions?: React.ReactNode;
  onFetch?: (id: string) => void;

  onToggleMinimize?: (id: string) => void;
};

export type BaseNode = Node<BaseNodeData, string>;

function fmt(errs: ErrorObject[] | null | undefined, max = 4): string[] {
  if (!errs || !errs.length) return [];
  return errs
    .slice(0, max)
    .map((e) => `${e.instancePath || ""} ${e.message ?? ""}`.trim());
}

const BaseGrammarNode = memo(function BaseGrammarNode({
  id,
  data,
}: NodeProps<BaseNode>) {
  const [errors, setErrors] = useState<string[]>([]);
  const [isValid, setIsValid] = useState<boolean>(true);
  const [hasSyntaxError, setHasSyntaxError] = useState(false);
  const rf = useReactFlow();

  const ajvRef = useRef<Ajv | null>(null);
  if (!ajvRef.current) {
    const ajv = new Ajv({ allErrors: true, strict: false });
    addFormats(ajv);
    ajvRef.current = ajv;
  }

  const validate = useMemo(() => {
    if (!data.schema) return null;
    return ajvRef.current!.compile(data.schema as any);
  }, [data.schema]);

  const innerValue = useMemo(
    () => (data.pickInner ? data.pickInner(data.value) : data.value),
    [data]
  );

  const runValidation = useCallback(
    (val: unknown) => {
      if (!validate) {
        setIsValid(true);
        setErrors([]);
        return;
      }
      try {
        const ok = validate(val);
        setIsValid(!!ok);
        setErrors(fmt(validate.errors));
      } catch (e) {
        setIsValid(false);
        setErrors([String(e)]);
      }
    },
    [validate]
  );

  const handleChange = useCallback(
    (val: GrammarValue) => {
      data?.onChange?.(val, id);
      runValidation(data.pickInner ? data.pickInner(val) : val);
    },
    [data, id, runValidation]
  );

  const handleTitleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      // console.log(e.target.value);
      // console.log(id);
      const nextTitle = e.target.value;
      rf.setNodes((nodes) =>
        nodes.map((n) =>
          n.id === id ? { ...n, data: { ...n.data, title: nextTitle } } : n
        )
      );
    },
    [id, rf]
  );

  useMemo(() => {
    runValidation(innerValue);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [validate]); // update when schema (and thus validate) changes

  const title = data?.title ?? "Grammar";
  const overallValid = isValid && !hasSyntaxError;

  const onClose = useCallback(() => {
    if (data?.onClose) return data.onClose(id);
    rf.setNodes((nds) => nds.filter((n) => n.id !== id));
  }, [data, id, rf]);

  const onRun = useCallback(() => {
    // console.log("[run]", id);
    if (data?.onRun) return data.onRun(id);
    // console.log("[run]", id);
  }, [data, id]);

  const obj = data.value as Record<string, any>;
  const key = Object.keys(obj)[0];

  // console.log(key);
  return (
    <div className="gnode">
      <NodeResizer minWidth={300} minHeight={180} />

      {/* Header */}
      <div
        className="gnode__header"
        // style={{ backgroundColor: nodeColor[key] ?? "#444" }}
        style={{
          backgroundColor: nodeColor[key] ?? "#444",
          borderColor: nodeBorderColor[key] ?? "#000",
        }}
      >
        <div className="gnode__title">
          <input
            type="text"
            className="gnode__titleInput"
            value={title}
            onChange={handleTitleChange}
          />
        </div>
        <div className="gnode__headerActions">
          <span
            className={`gnode__badge ${
              overallValid ? "is-valid" : "is-invalid"
            }`}
          >
            {overallValid ? "VALID" : "INVALID"}
          </span>
          <button
            type="button"
            className="gnode__iconBtn"
            onClick={() => data?.onToggleMinimize?.(id)}
          >
            &#8211;
          </button>
          <button
            type="button"
            className="gnode__iconBtn gnode__iconBtn--close"
            onClick={onClose}
          >
            ✕
          </button>
        </div>
      </div>

      {/* Editor */}
      <div className="gnode__editor">
        <div className="gnode__editorInner">
          <JsonCodeEditor
            value={data.value}
            onChange={handleChange as (v: unknown) => void}
            onDiagnostics={(diags) => setHasSyntaxError(diags.length > 0)}
            height="100%"
          />
        </div>
        {/* {!isValid && errors.length > 0 && (
          <div className="gnode__errors">
            <div className="gnode__errorsTitle">Schema errors:</div>
            {errors.map((e, i) => (
              <div key={i}>• {e}</div>
            ))}
          </div>
        )} */}
      </div>

      {/* Footer action bar */}
      <div className="gnode__footer">
        <button
          type="button"
          onClick={onRun}
          title="update"
          aria-label="update"
          className="gnode__actionBtn"
        >
          <img src={restartPng} alt="update" className="gnode__actionIcon" />
        </button>

        {data.footerActions}
      </div>
    </div>
  );
});

export default BaseGrammarNode;
