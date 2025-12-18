// src/nodes/utils/renderLayers.ts
import * as d3 from "d3";
import L from "leaflet";
import { getPropertyRangeFromGeoJSON, pickInterpolator } from "./helper";
import { applyGeometryInteractions } from "./geomInteractions";
import type { InteractionSpec } from "./geomInteractions";
import type {
  ViewDef,
  // ParsedView,
  ParsedInteraction,
} from "./types";

import * as GeoTIFF from "geotiff";

type TagGroup = d3.Selection<SVGGElement, unknown, null, undefined>;
const rasterOverlays = new Set<L.Layer>();
/**
 * Map ParsedInteraction -> InteractionSpec for a given layer.
 */

function tileBoundsFromXYZ(x: number, y: number, z: number, map: L.Map) {
  const tileSize = 256;

  const nwPoint = L.point(x * tileSize, y * tileSize);
  const sePoint = L.point((x + 1) * tileSize, (y + 1) * tileSize);

  const nwLatLng = map.unproject(nwPoint, z);
  const seLatLng = map.unproject(sePoint, z);

  return L.latLngBounds(nwLatLng, seLatLng);
}

function colorToRgb(str: string): [number, number, number] {
  if (!str) return [0, 0, 0];

  const s = str.trim();

  // Handle hex: #rgb or #rrggbb
  if (s[0] === "#") {
    let hex = s.slice(1);
    if (hex.length === 3) {
      // e.g. "f0a" -> "ff00aa"
      hex = hex
        .split("")
        .map((c) => c + c)
        .join("");
    }
    if (hex.length === 6) {
      const r = parseInt(hex.slice(0, 2), 16);
      const g = parseInt(hex.slice(2, 4), 16);
      const b = parseInt(hex.slice(4, 6), 16);
      return [
        Number.isFinite(r) ? r : 0,
        Number.isFinite(g) ? g : 0,
        Number.isFinite(b) ? b : 0,
      ];
    }
  }

  // Handle "rgb(r,g,b)" / "rgba(r,g,b,a)"
  if (s.startsWith("rgb")) {
    const nums = s
      .replace(/[^\d,]/g, "")
      .split(",")
      .map((x) => Number(x));
    return [nums[0] ?? 0, nums[1] ?? 0, nums[2] ?? 0];
  }

  // Fallback: try to extract any numbers in order
  const nums = s
    .replace(/[^\d,]/g, "")
    .split(",")
    .map((x) => Number(x));
  return [nums[0] ?? 0, nums[1] ?? 0, nums[2] ?? 0];
}

function buildInteractionSpecsForLayer(opts: {
  interactions: ParsedInteraction[];
  plId: string;
  tag: string;
  attr?: string;
}): InteractionSpec[] {
  const { interactions, plId, tag, attr } = opts;
  const relevant = interactions.filter(
    (i) => i.physicalLayerRef === plId && i.layer.tag === tag
  );

  const specs: InteractionSpec[] = [];

  for (const i of relevant) {
    // HOVER
    if (i.type === "hover") {
      if (i.action === "highlight") {
        specs.push({
          interaction: "hover-highlight",
          action: "highlight",
        });
        continue;
      }

      if (i.action === "highlight+show") {
        const featureKey = i.feature || attr || "height";

        specs.push({
          interaction: "hover-highlight",
          action: "highlight+show",
          tooltipAccessor: (d: any) => {
            const raw = d?.properties?.[featureKey];
            const num = Number(raw);

            if (Number.isFinite(num)) {
              return `${featureKey}: ${num.toFixed(2)}`;
            }
            if (raw != null) {
              return `${featureKey}: ${raw}`;
            }
            return tag;
          },
        });
        continue;
      }
    }

    // CLICK
    if (i.type === "click") {
      if (i.action === "remove") {
        specs.push({
          interaction: "click",
          action: "remove",
        });
        continue;
      }

      if (i.action === "modify_feature") {
        // You can extend InteractionSpec later to carry feature info / UI config.
        specs.push({
          interaction: "click",
          action: "modify_feature",
        } as any);
        continue;
      }
    }
  }

  return specs;
}

// Helper: render a raster layer (physical or thematic) and update unionBounds.
async function renderRasterForView(opts: {
  map: L.Map;
  view: ViewDef;
  layerId: string;
  unionBounds: L.LatLngBounds | null;
}): Promise<L.LatLngBounds | null> {
  const { map, view, layerId, unionBounds } = opts;
  const z = (view as any).style.zoom_level ?? 16;

  const tiles: string[] = await fetch(
    `http://127.0.0.1:5000/api/list-rasters/${layerId}`
  ).then((r) => r.json());

  const cacheBust = Date.now();

  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;

  for (const name of tiles) {
    // ignore malformed tiles like "12345_.png"
    if (name.endsWith("_.png")) continue;

    const [xStr, yStr] = name.replace(".png", "").split("_");
    const x = Number(xStr);
    const y = Number(yStr);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;

    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);

    const url = `http://127.0.0.1:5000/generated/raster/${layerId}/${name}?v=${cacheBust}`;
    // console.log("raster tile url:", url);

    const bounds = tileBoundsFromXYZ(x, y, z, map);

    const overlay = L.imageOverlay(url, bounds, {
      opacity: (view as any).style.opacity ?? 1,
    });

    overlay.addTo(map);
    rasterOverlays.add(overlay);
  }

  if (minX === Infinity) {
    return unionBounds;
  }

  const tileSize = 256;
  const nwPoint = L.point(minX * tileSize, minY * tileSize);
  const sePoint = L.point((maxX + 1) * tileSize, (maxY + 1) * tileSize);

  const nwLatLng = map.unproject(nwPoint, z);
  const seLatLng = map.unproject(sePoint, z);

  const rasterBounds = L.latLngBounds(nwLatLng, seLatLng);

  return unionBounds ? unionBounds.extend(rasterBounds) : rasterBounds;
}

async function renderGeoTiffForView(opts: {
  map: L.Map;
  view: ViewDef;
  layerId: string;
  unionBounds: L.LatLngBounds | null;
}): Promise<L.LatLngBounds | null> {
  const { map, view, layerId, unionBounds } = opts;

  const cacheBust = Date.now();

  // console.log(`[Viewport] Rendering GeoTIFF for layer ${layerId}`);
  const url = `http://127.0.0.1:5000/generated/raster/${layerId}.tif?v=${cacheBust}`;

  const colormapName = (view as any).style.colormap || undefined;

  // console.log("[GeoTIFF] colormapName =", colormapName);

  const interpolator =
    colormapName != null ? pickInterpolator(colormapName) : null;

  let tiff: GeoTIFF.GeoTIFF;
  try {
    tiff = await GeoTIFF.fromUrl(url);
  } catch (err) {
    console.error(`[Viewport] Failed to load GeoTIFF ${url}`, err);
    return unionBounds;
  }

  let image: GeoTIFF.IFD;
  try {
    image = await tiff.getImage();
  } catch (err) {
    console.error("[Viewport] Failed to get GeoTIFF image", err);
    return unionBounds;
  }

  const width = image.getWidth();
  const height = image.getHeight();
  const samplesPerPixel = image.getSamplesPerPixel();

  const [minX, minY, maxX, maxY] = image.getBoundingBox();
  const bounds = L.latLngBounds([minY, minX], [maxY, maxX]);

  let rasters: any[];
  try {
    rasters = (await image.readRasters()) as any[];
  } catch (err) {
    console.error("[Viewport] Failed to read GeoTIFF rasters", err);
    return unionBounds;
  }

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    console.error("[Viewport] Could not get 2D context for GeoTIFF canvas");
    return unionBounds;
  }

  const imageData = ctx.createImageData(width, height);
  const data = imageData.data;

  const band0 = rasters[0] as any;

  if (interpolator) {
    // ---------- scalar + colormap path (Viridis, Plasma, etc.) ----------
    let min = Infinity;
    let max = -Infinity;

    for (let i = 0; i < band0.length; i++) {
      const v = band0[i];
      if (!Number.isFinite(v)) continue;
      if (v < min) min = v;
      if (v > max) max = v;
    }

    // console.log("[GeoTIFF] band0 min/max =", min, max);

    const hasRange = max > min && Number.isFinite(min) && Number.isFinite(max);

    for (let i = 0; i < width * height; i++) {
      const v = band0[i];

      if (!Number.isFinite(v) || !hasRange) {
        data[4 * i + 3] = 0; // transparent nodata / degenerate
        continue;
      }

      const t = (v - min) / (max - min); // 0..1
      const color = interpolator(t); // "#21918c" or "rgb(...)"
      const [r, g, b] = colorToRgb(color);

      data[4 * i + 0] = r;
      data[4 * i + 1] = g;
      data[4 * i + 2] = b;
      data[4 * i + 3] = 255;
    }
  } else if (samplesPerPixel >= 3) {
    // ---------- RGB path (no colormap requested) ----------
    const rBand = rasters[0];
    const gBand = rasters[1];
    const bBand = rasters[2];

    for (let i = 0; i < width * height; i++) {
      const r = rBand[i];
      const g = gBand[i];
      const b = bBand[i];

      data[4 * i + 0] = Number.isFinite(r) ? Math.max(0, Math.min(255, r)) : 0;
      data[4 * i + 1] = Number.isFinite(g) ? Math.max(0, Math.min(255, g)) : 0;
      data[4 * i + 2] = Number.isFinite(b) ? Math.max(0, Math.min(255, b)) : 0;
      data[4 * i + 3] = 255;
    }
  } else {
    // ---------- single-band grayscale fallback ----------
    let min = Infinity;
    let max = -Infinity;

    for (let i = 0; i < band0.length; i++) {
      const v = band0[i];
      if (!Number.isFinite(v)) continue;
      if (v < min) min = v;
      if (v > max) max = v;
    }

    const hasRange = max > min && Number.isFinite(min) && Number.isFinite(max);
    const scale = hasRange ? 255 / (max - min) : 1;

    for (let i = 0; i < width * height; i++) {
      const v = band0[i];
      let value = 0;
      if (Number.isFinite(v) && hasRange) {
        value = (v - min) * scale;
      }
      value = Math.max(0, Math.min(255, value));

      data[4 * i + 0] = value;
      data[4 * i + 1] = value;
      data[4 * i + 2] = value;
      data[4 * i + 3] = Number.isFinite(v) ? 255 : 0;
    }
  }

  ctx.putImageData(imageData, 0, 0);

  const dataUrl = canvas.toDataURL("image/png");
  const opacity = (view as any).style.opacity ?? 1;

  const overlay = L.imageOverlay(dataUrl, bounds, { opacity });
  overlay.addTo(map);
  rasterOverlays.add(overlay);

  return unionBounds ? unionBounds.extend(bounds) : bounds;
}

export async function renderLayers(opts: {
  id: string;
  map: L.Map;
  parsedViews: ViewDef[];
  parsedInteractions: ParsedInteraction[];
  // physicalLayers: PhysicalLayerDef[];
  clearAllSvgLayers: () => void;
  makeLeafletPath: (map: L.Map) => d3.GeoPath<any, d3.GeoPermissibleObjects>;
  getOrCreateTagGroup: (tag: string) => TagGroup;
  onDirty?: (args: {
    plId: string;
    tag: string;
    featureCollection: any;
  }) => void;
  shouldHandleClick: () => boolean;
}) {
  const {
    id,
    map,
    parsedViews,
    parsedInteractions,
    // physicalLayers,
    clearAllSvgLayers,
    makeLeafletPath,
    getOrCreateTagGroup,
    onDirty,
    shouldHandleClick,
  } = opts;

  // Remove old raster overlays from previous renders
  for (const overlay of rasterOverlays) {
    map.removeLayer(overlay);
  }
  rasterOverlays.clear();

  clearAllSvgLayers();

  // const physicalViews = parsedViews.filter((v) => v.physical_layer?.ref);
  // const thematicViews = parsedViews.filter((v) => v.thematic_layer?.ref);

  const views = parsedViews;
  // if (!physicalViews.length && !thematicViews.length) {
  //   return;
  // }

  if (!views.length) {
    return;
  }

  let unionBounds: L.LatLngBounds | null = null;
  const path = makeLeafletPath(map);

  // --- Thematic (non-physical) views ---
  for (const view of views) {
    const ref = view.ref;
    const ref_base = view.ref_base;
    const ref_comp = view.ref_comp;
    if (!ref && !ref_base && !ref_comp) {
      continue;
    }

    if (ref) {
      if (view.type === "raster" && view.file_type === "png") {
        unionBounds = await renderRasterForView({
          map,
          view,
          layerId: ref,
          unionBounds,
        });
      } else if (view.file_type === "tif" || view.file_type === "tiff") {
        // console.log("Rendering thematic GeoTIFF view:", thId);
        unionBounds = await renderGeoTiffForView({
          map,
          view,
          layerId: ref,
          unionBounds,
        });
      } else if (view.type === "vector" && view.file_type === "geojson") {
        // It should be without tags: if osm id_buildings, id_roads, etc. if not simply

        // Also the zIndex like this does not make sense!
        // const layers = [...(view.layers ?? [])].sort((a: any, b: any) => {
        //   const za = (a as any).zIndex ?? 0;
        //   const zb = (b as any).zIndex ?? 0;
        //   return za - zb;
        // });

        // for (const lyr of layers) {
        // const tag = lyr.tag;
        // const url = `http://127.0.0.1:5000/generated/vector/${plId}_${tag}.geojson`;

        const url = `http://127.0.0.1:5000/generated/vector/${ref}.geojson`;
        let fc: any;
        try {
          const res = await fetch(url);
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          fc = await res.json();
        } catch (err) {
          console.error(`[Viewport ${id}] Failed to fetch ${url}`, err);
          continue;
        }

        // --- styling ---
        const geomType = view["geom_type"];

        const isPolygonLayer =
          geomType === "polygon" || geomType === "multipolygon";
        const isLineLayer = geomType === "linestring";
        const isPointLayer = geomType === "point";

        let attr: string | undefined;
        let colormapName: string | undefined;
        let solidFill: string | undefined;

        if ((isPolygonLayer || isPointLayer) && view.style.fill) {
          if (typeof view.style.fill === "string") {
            // solid color fill
            solidFill = view.style.fill;
          } else if ("feature" in (view.style.fill as any)) {
            attr = (view.style.fill as any).feature;
            colormapName = (view.style.fill as any).colormap;
          }
        }

        console.log(isPolygonLayer);

        const interp = attr ? pickInterpolator(colormapName) : null;
        const ext = attr ? getPropertyRangeFromGeoJSON(fc, attr) : null;
        const colorScale =
          attr && interp && ext
            ? d3.scaleSequential(interp).domain(ext ?? [0, 1])
            : null;

        const strokeColor = view.style.stroke?.color ?? "#000";
        const strokeWidth = view.style.stroke?.width ?? 1;
        const layerOpacity = view.style.opacity ?? 1;

        // point radius from parser (default 4px)

        if (isPointLayer && typeof (path as any).pointRadius === "function") {
          // d3-geo point rendering uses this
          const pointRadius = (view.style as any).size ?? 4;
          (path as any).pointRadius(pointRadius);
        }

        // const nTag = `${plId}::${tag}`;
        const gTag = getOrCreateTagGroup(ref);

        const configuredRadius = (view.style as any).size ?? 4; // radius in px

        // Store metadata for redrawAll
        (gTag as any)._geomType = geomType;
        (gTag as any)._isPointLayer = isPointLayer;
        (gTag as any)._pointRadius = configuredRadius;

        const keyFn = (d: any, i: number) =>
          d.id ?? d.properties?.id ?? d.properties?.osm_id ?? i;

        const sel = gTag
          .selectAll<SVGPathElement, any>("path.geom")
          .data(fc.features, keyFn);

        sel.exit().remove();

        if (isLineLayer) {
          if (view.style["border-color"] || view.style["border-width"]) {
            const borderColor = view.style["border-color"] ?? "#fff"; // or whatever default
            const borderWidth = view.style["border-width"] ?? 0;

            const borderSel = gTag
              .selectAll<SVGPathElement, any>("path.geom-border")
              .data(fc.features, keyFn);

            borderSel.exit().remove();

            const borderEnter = borderSel
              .enter()
              .append("path")
              .attr("class", "geom-border");

            borderEnter
              .merge(borderSel as any)
              .attr("d", path as any)
              .style("fill", "none")
              .style("stroke", borderColor)
              .style("stroke-width", borderWidth) // slightly thicker than inner line
              .style("stroke-opacity", layerOpacity)
              .style("vector-effect", "non-scaling-stroke")
              .style("pointer-events", "none"); // so clicks go to the inner path
          }
        }

        const enter = sel.enter().append("path").attr("class", "geom");
        // console.log(strokeColor, strokeWidth);
        const geomSel = enter
          .merge(sel as any)
          .attr("d", path as any)
          .style("fill", (d: any) => {
            const gType = d?.geometry?.type;
            const isLineFeature =
              gType === "LineString" ||
              gType === "MultiLineString" ||
              isLineLayer;

            // Lines: no fill
            if (isLineFeature) return "none";

            // Solid fill from parser (polygons or points)
            if (solidFill) return solidFill;

            // Polygon: attribute-based colormap from parser
            if (attr && colorScale) {
              const v = Number(d?.properties?.[attr]);
              if (Number.isFinite(v)) return colorScale(v);
            }

            // Fallback polygon fill
            return "none";
          })
          .style("fill-opacity", (d: any) => {
            const gType = d?.geometry?.type;
            const isLineFeature =
              gType === "LineString" ||
              gType === "MultiLineString" ||
              isLineLayer;
            return isLineFeature ? 0 : layerOpacity;
          })
          .style("stroke", strokeColor)
          .style("stroke-width", strokeWidth)
          .style("stroke-opacity", layerOpacity)
          .style("vector-effect", "non-scaling-stroke")
          .style("pointer-events", "all");

        // --- interactions from parsedInteractions ---
        // const interactions = buildInteractionSpecsForLayer({
        //   interactions: parsedInteractions,
        //   plId,
        //   tag,
        //   attr,
        // });
        // if (interactions.length) {
        //   applyGeometryInteractions(
        //     geomSel,
        //     interactions,
        //     {
        //       tag,
        //       featureCollection: fc,
        //       shouldHandleClick,
        //       onCollectionChange: ({ tag: changedTag, featureCollection }) => {
        //         if (!onDirty) return;
        //         onDirty({
        //           plId,
        //           tag: changedTag,
        //           featureCollection,
        //         });
        //       },
        //     },
        //     strokeWidth
        //   );
        // }

        // --- bounds ---
        const tmp = L.geoJSON(fc);
        const b = tmp.getBounds();
        if (b.isValid()) {
          unionBounds = unionBounds ? unionBounds.extend(b) : b;
        }
        tmp.remove();
      }
    } else if (ref_base && ref_comp) {
      if (
        view.type === "raster" &&
        view.file_type === "png"
        // &&
        // view.style.operation === "difference"
      ) {
        const diffThId = ref_comp + "_minus_" + ref_base;

        await fetch("http://127.0.0.1:5000/api/diff-png", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            dir1: ref_base,
            dir2: ref_comp,
            colormap: view.style.colormap || "Reds",
          }),
        });

        unionBounds = await renderRasterForView({
          map,
          view,
          layerId: diffThId,
          unionBounds,
        });
      } else if (
        view.type === "raster" &&
        (view.file_type === "tif" || view.file_type === "tiff")
        // &&
        // view.style.operation === "difference"
      ) {
        const diffThId = ref_comp + "_minus_" + ref_base;
        await fetch("http://127.0.0.1:5000/api/diff-tif", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            tif1: ref_base,
            tif2: ref_comp,
            colormap: view.style.colormap || "Reds",
          }),
        });

        unionBounds = await renderGeoTiffForView({
          map,
          view,
          layerId: diffThId,
          unionBounds,
        });
      }
    }
  }

  // }

  if (unionBounds && unionBounds.isValid()) {
    map.fitBounds(unionBounds, { padding: [12, 12] });
  }
}
