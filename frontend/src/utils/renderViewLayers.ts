// src/nodes/utils/renderLayers.ts
import * as d3 from "d3";
import L from "leaflet";
import {
  getPropertyRangeFromGeoJSON,
  pickInterpolator,
  tileBoundsFromXYZ,
  colorToRgb,
  getGeometryTypeFromGeoJSON,
} from "./helper";
import { applyGeometryInteractions } from "./geomInteractions";
import type { InteractionSpec } from "./geomInteractions";
import type { ViewDef, InteractionDef } from "./types";

import * as GeoTIFF from "geotiff";

type TagGroup = d3.Selection<SVGGElement, unknown, null, undefined>;
const rasterOverlays = new Set<L.Layer>();

function buildInteractionSpecsForLayer(opts: {
  interactions: InteractionDef[];
  ref: string;
}): InteractionSpec[] {
  const { interactions, ref } = opts;
  const relevant = interactions.filter((i) => i.ref === ref);

  const specs: InteractionSpec[] = [];

  for (const i of relevant) {
    // HOVER
    if (i.itype === "hover") {
      if (i.action === "highlight") {
        specs.push({
          interaction: "hover-highlight",
          action: "highlight",
        });
        continue;
      }

      if (i.action === "tooltip_highlight") {
        const featureKey = i.attribute;

        specs.push({
          interaction: "hover-highlight",
          action: "highlight+show",
          tooltipAccessor: (d: any) => {
            const raw = d?.properties?.[featureKey!];
            const num = Number(raw);

            if (Number.isFinite(num)) {
              return `${featureKey}: ${num.toFixed(2)}`;
            }
            if (raw != null) {
              return `${featureKey}: ${raw}`;
            }
            return ref;
          },
        });
        continue;
      }
    }

    // CLICK
    if (i.itype === "click") {
      if (i.action === "remove") {
        specs.push({
          interaction: "click",
          action: "remove",
        });
        continue;
      }

      if (i.action === "modify") {
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

async function renderPngForView(opts: {
  map: L.Map;
  view: ViewDef;
  ref: string;
  unionBounds: L.LatLngBounds | null;
}): Promise<L.LatLngBounds | null> {
  const { map, view, ref, unionBounds } = opts;

  const cmap = (view as any).style?.colormap ?? "reds";

  const tiles: string[] = await fetch(
    `http://127.0.0.1:5000/api/list-rasters/${ref}`
  ).then((r) => r.json());

  const cacheBust = Date.now();

  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;

  let z_ = 16;
  for (const name of tiles) {
    if (name.endsWith("_.png")) continue;
    const parts = name.replace(/\.png$/i, "").split("_");

    const yStr = parts[parts.length - 1];
    const xStr = parts[parts.length - 2];
    const zStr = parts[parts.length - 3];

    const x = Number(xStr);
    const y = Number(yStr);
    const z = Number(zStr);
    // if (!Number.isFinite(x) || !Number.isFinite(y)) continue;

    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);

    // const url = `http://127.0.0.1:5000/generated/raster/${ref}/${name}?v=${cacheBust}`;
    const url =
      `http://127.0.0.1:5000/generated/raster/${ref}/${name}` +
      `?v=${cacheBust}&cmap=${encodeURIComponent(cmap)}`;

    console.log(url);
    const bounds = tileBoundsFromXYZ(x, y, z, map);

    const overlay = L.imageOverlay(url, bounds, {
      opacity: (view as any).style.opacity ?? 1,
    });

    overlay.addTo(map);
    rasterOverlays.add(overlay);
    z_ = z;
  }

  if (minX === Infinity) {
    return unionBounds;
  }

  const tileSize = 256;
  const nwPoint = L.point(minX * tileSize, minY * tileSize);
  const sePoint = L.point((maxX + 1) * tileSize, (maxY + 1) * tileSize);

  const nwLatLng = map.unproject(nwPoint, z_);
  const seLatLng = map.unproject(sePoint, z_);

  const rasterBounds = L.latLngBounds(nwLatLng, seLatLng);

  return unionBounds ? unionBounds.extend(rasterBounds) : rasterBounds;
}

async function renderGeoTiffForView(opts: {
  map: L.Map;
  view: ViewDef;
  ref: string;
  unionBounds: L.LatLngBounds | null;
}): Promise<L.LatLngBounds | null> {
  const { map, view, ref, unionBounds } = opts;

  const cacheBust = Date.now();
  const url = `http://127.0.0.1:5000/generated/raster/${ref}.tif?v=${cacheBust}`;

  const colormapName = (view as any).style.colormap || undefined;

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
  views: ViewDef[];
  interactions: InteractionDef[];
  clearAllSvgLayers: () => void;
  makeLeafletPath: (map: L.Map) => d3.GeoPath<any, d3.GeoPermissibleObjects>;
  getOrCreateTagGroup: (tag: string) => TagGroup;
  onDirty?: (args: { ref: string; featureCollection: any }) => void;
  shouldHandleClick: () => boolean;
}) {
  const {
    id,
    map,
    views,
    interactions,
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

  if (!views.length) {
    return;
  }

  let unionBounds: L.LatLngBounds | null = null;
  const path = makeLeafletPath(map);

  for (const view of views) {
    const ref = view.ref;
    const ref_base = view.ref_base;
    const ref_comp = view.ref_comp;
    if (!ref && !ref_base && !ref_comp) {
      continue;
    }

    if (ref) {
      const res = await fetch("http://127.0.0.1:5000/api/infer-filetype", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ref }),
      });
      const info = await res.json();

      if (info.file_type === ".png") {
        unionBounds = await renderPngForView({
          map,
          view,
          ref,
          unionBounds,
        });
      } else if (info.file_type === ".tif" || info.file_type === ".tiff") {
        unionBounds = await renderGeoTiffForView({
          map,
          view,
          ref,
          unionBounds,
        });
      } else if (info.file_type === ".geojson") {
        // --- render GeoJSON for view. Maybe later function ---
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

        // --- geometry type ---
        const geomType = getGeometryTypeFromGeoJSON(fc)?.toLowerCase();
        // console.log("[Viewport] geometry type =", gType);

        // const geomType = view["geom_type"];
        const isPolygonLayer =
          geomType === "polygon" || geomType === "multipolygon";
        const isLineLayer = geomType === "linestring";
        const isPointLayer = geomType === "point";

        let attr: string | undefined;
        let colormapName: string | undefined;
        let solidFill: string | undefined;

        // --- solid fill or attribute-based fill ---
        if ((isPolygonLayer || isPointLayer) && view.style.fill) {
          if (typeof view.style.fill === "string") {
            // Solid fill
            solidFill = view.style.fill;
          } else if ("feature" in (view.style.fill as any)) {
            // Attribute based fill. Currently only does numeric continuous
            attr = (view.style.fill as any).feature;
            colormapName = (view.style.fill as any).colormap;
          }
        }

        // --- range of values for attribute-based fills ---
        const interp = attr ? pickInterpolator(colormapName) : null;

        let ext = null;

        const fill = view.style?.fill;

        if (attr && fill && typeof fill === "object" && "range" in fill) {
          // Explicit range
          const [min, max] = fill.range;
          ext = [min, max];
        } else {
          // Compute from data
          ext = attr ? getPropertyRangeFromGeoJSON(fc, attr) : null;
        }

        // --- colorscale for attribute-based fills ---
        const colorScale =
          attr && interp && ext
            ? d3.scaleSequential(interp).domain(ext ?? [0, 1])
            : null;

        // --- other style properties and default values ---
        const strokeColor = view.style.stroke?.color ?? "#000";
        const strokeWidth = view.style.stroke?.width ?? 1;
        const layerOpacity = view.style.opacity ?? 1;

        if (isPointLayer && typeof (path as any).pointRadius === "function") {
          const pointRadius = (view.style as any).size ?? 4;
          (path as any).pointRadius(pointRadius);
        }

        const gTag = getOrCreateTagGroup(ref);

        const configuredRadius = (view.style as any).size ?? 4; // radius in px

        //  --- Store metadata for redrawAll ---
        (gTag as any)._geomType = geomType;
        (gTag as any)._isPointLayer = isPointLayer;
        (gTag as any)._pointRadius = configuredRadius;

        // --- unique ids ---
        const keyFn = (d: any, i: number) =>
          d.id ?? d.properties?.id ?? d.properties?.osm_id ?? i;

        const sel = gTag
          .selectAll<SVGPathElement, any>("path.geom")
          .data(fc.features, keyFn);

        sel.exit().remove();

        if (isLineLayer) {
          if (view.style["border-color"] || view.style["border-width"]) {
            const borderColor = view.style["border-color"] ?? "#fff";
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

        // --- interactions ---
        const interactions_ = buildInteractionSpecsForLayer({
          interactions,
          ref,
        });
        if (interactions_.length) {
          applyGeometryInteractions(
            geomSel,
            interactions_,
            {
              featureCollection: fc,
              shouldHandleClick,
              onCollectionChange: ({ featureCollection }) => {
                if (!onDirty) return;
                onDirty({
                  ref,
                  featureCollection,
                });
              },
            },
            strokeWidth
          );
        }

        // --- bounds ---
        const tmp = L.geoJSON(fc);
        const b = tmp.getBounds();
        if (b.isValid()) {
          unionBounds = unionBounds ? unionBounds.extend(b) : b;
        }
        tmp.remove();
      }
    } else if (ref_base && ref_comp) {
      // go to server side. check inside served-raster and served-vector and infer filetype

      const res_base = await fetch("http://127.0.0.1:5000/api/infer-filetype", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ref_base }),
      });
      const info_base = await res_base.json();

      const res_comp = await fetch("http://127.0.0.1:5000/api/infer-filetype", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ref_comp }),
      });
      const info_comp = await res_comp.json();

      if (info_base.file_type === ".png" && info_comp.file_type === ".png") {
        const diff = ref_comp + "_minus_" + ref_base;

        await fetch("http://127.0.0.1:5000/api/diff-png", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            dir1: ref_base,
            dir2: ref_comp,
            colormap: view.style.colormap || "Reds",
          }),
        });

        unionBounds = await renderPngForView({
          map,
          view,
          ref: diff,
          unionBounds,
        });
      } else if (
        (info_base.file_type === ".tif" || info_base.file_type === ".tiff") &&
        (info_comp.file_type === ".tif" || info_comp.file_type === ".tiff")
      ) {
        const diff = ref_base + "_minus_" + ref_comp;
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
          ref: diff,
          unionBounds,
        });
      }
    }
  }

  if (unionBounds && unionBounds.isValid()) {
    map.fitBounds(unionBounds, { padding: [12, 12] });
  }
}
