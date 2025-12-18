// import { ParsedView } from "./types";
import type { InteractionDef } from "./types";

// export function parseView(raw: any): ParsedView[] {
//   if (!raw?.view || !Array.isArray(raw.view)) return [];

//   return raw.view.map((v: any) => {
//     const base: ParsedView = {
//       physicalLayerRef: v.physical_layer?.ref,
//       thematicLayerRef: v.thematic_layer?.ref,
//       operation: v.operation ?? undefined,
//       file_type: v["file_type"],
//       type: v.type,
//       zoom_level: v.zoom_level ?? 16,
//     };

//     // if (v.type === "raster") {
//     //   (base as ParsedView).opacity = v.opacity ?? 1; // should be inside style
//     // }

//     // if (v.type === "raster") {
//     //   // Have to later set colormap for pngs as well but from backend
//     //   (base as ParsedView).colormap = v.colormap ?? "Reds"; // should be inside style
//     // }

//     if (v.type === "raster") {
//       const style = v.style;

//     if (v.type == "vector") {
//       // base.layers = v.layers.map((lyr: any) => {
//         const geomType = v["geom-type"];

//         const style = v.style;
//         const isPolygon = geomType === "polygon" || geomType === "multipolygon";
//         const isPoint = geomType === "point";
//         const isLine = geomType === "linestring";

//         // const layer: any = {
//         //   tag: lyr.tag,
//         //   "geom-type": geomType,
//         //   opacity: style.opacity ?? 1,
//         //   zIndex: style["z-index"] ?? 1,
//         // };

//         if (isPolygon) {
//           let fillSpec: any;

//           if (typeof style.fill === "object") {
//             fillSpec = {
//               attribute: style.fill.feature,
//               colormap: style.fill.colormap ?? "viridis",
//             };
//           } else {
//             fillSpec = style.fill ?? "#6aa9ff";
//           }

//           style.fill = fillSpec;

//           if (style["stroke-color"] || style["stroke-width"]) {
//             style.stroke = {
//               color: style["stroke-color"] ?? "#000",
//               width: style["stroke-width"] ?? 1,
//             };
//           }
//         }

//         if (isLine) {
//           style.stroke = {
//             color: style["stroke-color"] ?? "#000",
//             width: style["stroke-width"] ?? 1,
//           };

//           if (style["border-color"] || style["border-width"]) {
//             style.border = {
//               color: style["border-color"] ?? "#ddd",
//               width: style["border-width"] ?? 1,
//             };
//           }
//         }

//         if (isPoint) {
//           let fillSpec: any;

//           if (typeof style.fill === "object") {
//             // attribute-based point color
//             fillSpec = {
//               attribute: style.fill.feature,
//               colormap: style.fill.colormap ?? "viridis",
//             };
//           } else {
//             // solid color
//             fillSpec = style.fill ?? "#6aa9ff";
//           }

//           style.fill = fillSpec;

//           if (style["stroke-color"] || style["stroke-width"]) {
//             style.stroke = {
//               color: style["stroke-color"] ?? "#000",
//               width: style["stroke-width"] ?? 1,
//             };
//           }

//           style.size = style["radius"] ?? 4; // radius in px
//         }

//         // return layer;
//       // });
//     }

//     return base;
//   });
// }

export function parseInteraction(raw: any): InteractionDef[] {
  if (!raw?.interaction || !Array.isArray(raw.interaction)) return [];

  return raw.interaction
    .map((it: any): InteractionDef | null => {
      if (!it) return null;

      const layer = it.layer ?? {};

      const def: InteractionDef = {
        id: String(it.id ?? ""),
        type: String(it.type ?? ""),
        action: String(it.action ?? ""),
        physicalLayerRef: String(it.physicalLayerRef ?? ""),
        layer: {
          tag: String(layer.tag ?? ""),
          ...(layer.feature != null ? { feature: String(layer.feature) } : {}),
        },
      };

      return def;
    })
    .filter(
      (d): d is InteractionDef =>
        !!d &&
        !!d.id &&
        !!d.type &&
        !!d.action &&
        !!d.physicalLayerRef &&
        !!d.layer?.tag
    );
}

// parser for widget def. should parse according to widget type and then we can pass it to the widgetview node and render accordingly
