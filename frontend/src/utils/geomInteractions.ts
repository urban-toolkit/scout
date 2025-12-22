// src/nodes/utils/geomInteractions.ts
import * as d3 from "d3";
import type { Selection } from "d3-selection";

export type GeometryDatum = any;

export type InteractionType =
  | "hover-highlight"
  | "hover-tooltip"
  | "click"
  | "click+user_input";

export type ActionType =
  | "highlight"
  | "highlight+show"
  | "remove"
  | "modify_feature";

export interface InteractionSpec {
  interaction: InteractionType;
  action?: ActionType;
  tooltipAccessor?: (d: GeometryDatum) => string;
  modifyPrompt?: (d: GeometryDatum) => { prop: string; message: string };
}

export interface InteractionContext {
  featureCollection?: any;

  onCollectionChange?: (args: { featureCollection: any }) => void;
  shouldHandleClick?: () => boolean;
  showTooltip?: (content: string, evt: MouseEvent, d: GeometryDatum) => void;
  hideTooltip?: () => void;
}

function getDefaultTooltipEl(): HTMLDivElement {
  let el = document.querySelector<HTMLDivElement>(".geom-tooltip");
  if (!el) {
    el = document.createElement("div");
    el.className = "geom-tooltip";
    Object.assign(el.style, {
      position: "fixed",
      zIndex: "9999",
      pointerEvents: "none",
      padding: "2px 4px",
      fontSize: "10px",
      background: "rgba(0,0,0,0.8)",
      color: "#fff",
      borderRadius: "3px",
      opacity: "0",
      transition: "opacity 0.05s linear",
      whiteSpace: "nowrap",
    } as Partial<CSSStyleDeclaration>);
    document.body.appendChild(el);
  }
  return el;
}

function defaultShowTooltip(content: string, evt: MouseEvent) {
  const el = getDefaultTooltipEl();
  el.textContent = content;
  el.style.left = `${evt.clientX + 8}px`;
  el.style.top = `${evt.clientY - 8}px`;
  el.style.opacity = "1";
}

function defaultHideTooltip() {
  const el = document.querySelector<HTMLDivElement>(".geom-tooltip");
  if (el) el.style.opacity = "0";
}

export function applyGeometryInteractions(
  sel: Selection<SVGPathElement, GeometryDatum, any, any>,
  specs: InteractionSpec[] | undefined,
  ctx: InteractionContext & { featureCollection?: any },
  baseStrokeWidth: number
) {
  if (!specs || specs.length === 0) {
    specs = [{ interaction: "hover-highlight", action: "highlight" }];
  }

  // clear previous namespaced handlers
  sel
    .on(".hover-highlight", null)
    .on(".hover-tooltip", null)
    .on(".click-remove", null)
    .on(".click-modify", null);

  for (const spec of specs) {
    const { interaction, action } = spec;

    // ───────────────────── HOVER-HIGHLIGHT (+ optional show) ─────────────────────
    if (interaction === "hover-highlight") {
      const wantsShow = action === "highlight+show";

      sel
        .on("mouseover.hover-highlight", function (event, d) {
          d3.select(this).style("stroke-width", baseStrokeWidth + 1.5);
          d3.select(this).style("cursor", "pointer");

          if (!wantsShow) return;

          const content = spec.tooltipAccessor?.(d) ?? d?.properties?.id ?? "";

          if (!content) return;

          if (ctx.showTooltip) {
            ctx.showTooltip(content, event as MouseEvent, d);
          } else {
            defaultShowTooltip(content, event as MouseEvent);
          }
        })
        .on("mousemove.hover-highlight", function (event, d) {
          if (!wantsShow) return;

          const content = spec.tooltipAccessor?.(d) ?? d?.properties?.id ?? "";

          if (!content) return;

          if (ctx.showTooltip) {
            ctx.showTooltip(content, event as MouseEvent, d);
          } else {
            defaultShowTooltip(content, event as MouseEvent);
          }
        })
        .on("mouseout.hover-highlight", function () {
          d3.select(this).style("stroke-width", baseStrokeWidth);
          if (ctx.hideTooltip) ctx.hideTooltip();
          else defaultHideTooltip();
        });
    }

    // ───────────────────────────── HOVER-TOOLTIP ONLY ────────────────────────────
    if (interaction === "hover-tooltip" && !action) {
      sel
        .on("mouseover.hover-tooltip", function (event, d) {
          const content = spec.tooltipAccessor?.(d) ?? d?.properties?.id ?? "";
          if (!content) return;

          if (ctx.showTooltip) {
            ctx.showTooltip(content, event as MouseEvent, d);
          } else {
            defaultShowTooltip(content, event as MouseEvent);
          }
        })
        .on("mousemove.hover-tooltip", function (event, d) {
          const content = spec.tooltipAccessor?.(d) ?? d?.properties?.id ?? "";
          if (!content) return;

          if (ctx.showTooltip) {
            ctx.showTooltip(content, event as MouseEvent, d);
          } else {
            defaultShowTooltip(content, event as MouseEvent);
          }
        })
        .on("mouseout.hover-tooltip", function () {
          if (ctx.hideTooltip) ctx.hideTooltip();
          else defaultHideTooltip();
        });
    }

    // ─────────────────────────────── CLICK + REMOVE ──────────────────────────────
    if (interaction === "click" && action === "remove") {
      sel.on("click.click-remove", function (_event, d) {
        if (ctx.shouldHandleClick && !ctx.shouldHandleClick()) {
          return;
        }
        // remove SVG
        d3.select(this).remove();

        // update in-memory GeoJSON if provided
        const fc = ctx.featureCollection;
        if (fc && Array.isArray(fc.features)) {
          fc.features = fc.features.filter((f: any) => f !== d);

          ctx.onCollectionChange?.({
            featureCollection: fc,
          });
        }

        // user callback
      });
    }

    // ───────────── CLICK + USER_INPUT -> MODIFY_FEATURE (modify_feature) ─────────
    if (interaction === "click+user_input" && action === "modify_feature") {
      sel.on("click.click-modify", function (_event, d) {
        if (ctx.shouldHandleClick && !ctx.shouldHandleClick()) {
          return;
        }
        const cfg = spec.modifyPrompt?.(d) ?? {
          prop: "value",
          message: "Enter new value:",
        };

        const input = window.prompt(cfg.message, "");
        if (input == null) return;

        const updates: Record<string, unknown> = { [cfg.prop]: input };

        if (d && d.properties) {
          Object.assign(d.properties, updates);
        }

        const fc = ctx.featureCollection;
        if (fc && Array.isArray(fc.features)) {
          ctx.onCollectionChange?.({
            featureCollection: fc,
          });
        }
      });
    }
  }

  return sel;
}
