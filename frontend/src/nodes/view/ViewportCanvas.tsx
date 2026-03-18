import { memo, useCallback, useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import * as d3 from "d3";
import type { ViewDef, InteractionDef } from "../../utils/types";
import { renderLayers } from "../../utils/renderViewLayers";

type ViewportCanvasProps = {
  id: string;
  center?: [number, number];
  view?: ViewDef[];
  interactions?: InteractionDef[];
  showBasemap?: boolean;
  className?: string;
  onDirty?: (payload: { ref: string; featureCollection: any }) => void;
};

const ViewportCanvas = memo(function ViewportCanvas({
  id,
  center = [41.881, -87.63],
  view = [],
  interactions = [],
  showBasemap = false,
  className = "vpnode__map nodrag nowheel",
  onDirty,
}: ViewportCanvasProps) {
  const mapRef = useRef<HTMLDivElement | null>(null);
  const leafletRef = useRef<L.Map | null>(null);
  const baseLayerRef = useRef<L.TileLayer | null>(null);

  const wasDraggedRef = useRef(false);

  const svgLayerRef = useRef<L.SVG | null>(null);
  const overlaySvgRef = useRef<d3.Selection<
    SVGSVGElement,
    unknown,
    null,
    undefined
  > | null>(null);
  const gRootRef = useRef<d3.Selection<
    SVGGElement,
    unknown,
    null,
    undefined
  > | null>(null);
  const gByTagRef = useRef<
    Map<string, d3.Selection<SVGGElement, unknown, null, undefined>>
  >(new Map());

  const shouldHandleClick = useCallback(() => {
    if (wasDraggedRef.current) {
      wasDraggedRef.current = false;
      return false;
    }
    return true;
  }, []);

  const makeLeafletPath = useCallback((map: L.Map) => {
    const projectPoint = function (this: any, x: number, y: number) {
      const point = map.latLngToLayerPoint([y, x]);
      this.stream.point(point.x, point.y);
    };

    const transform = d3.geoTransform({ point: projectPoint as any });
    return d3.geoPath(transform as any);
  }, []);

  const getOrCreateTagGroup = useCallback((tag: string) => {
    const m = gByTagRef.current;
    if (m.has(tag)) return m.get(tag)!;

    const gRoot = gRootRef.current!;
    const g = gRoot.append("g").attr("class", `tag-${tag}`);
    m.set(tag, g);
    return g;
  }, []);

  const clearAllSvgLayers = useCallback(() => {
    gByTagRef.current.forEach((g) => g.remove());
    gByTagRef.current.clear();
  }, []);

  const redrawAll = useCallback(() => {
    const map = leafletRef.current;
    if (!map) return;

    gByTagRef.current.forEach((g) => {
      const isPointLayer = (g as any)._isPointLayer;
      const configuredRadius = (g as any)._pointRadius;

      const path = makeLeafletPath(map);

      if (isPointLayer && typeof (path as any).pointRadius === "function") {
        const pointRadius = configuredRadius ?? 4;
        (path as any).pointRadius(pointRadius);
      }

      g.selectAll<SVGPathElement, any>("path.geom, path.geom-border").attr(
        "d",
        path as any,
      );
    });
  }, [makeLeafletPath]);

  const loadFromView = useCallback(async () => {
    const map = leafletRef.current;
    if (!map) return;

    await renderLayers({
      id,
      map,
      views: view ?? [],
      interactions: interactions ?? [],
      clearAllSvgLayers,
      makeLeafletPath,
      getOrCreateTagGroup,
      onDirty: ({ ref, featureCollection }) => {
        onDirty?.({ ref, featureCollection });
      },
      shouldHandleClick,
    });

    map.invalidateSize();
  }, [
    id,
    view,
    interactions,
    clearAllSvgLayers,
    makeLeafletPath,
    getOrCreateTagGroup,
    onDirty,
    shouldHandleClick,
  ]);

  useEffect(() => {
    if (!mapRef.current || leafletRef.current) return;

    const map = L.map(mapRef.current, {
      attributionControl: false,
      preferCanvas: true,
    });

    map.setView(center, 14);

    const baseLayer = L.tileLayer(
      "https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
      {
        maxZoom: 19,
        attribution: "&copy; OpenStreetMap contributors",
        opacity: 0,
      },
    ).addTo(map);

    baseLayerRef.current = baseLayer;

    const svgLayer = L.svg().addTo(map);
    const overlaySvg = d3.select(svgLayer._container as SVGSVGElement);
    const gRoot = overlaySvg.append("g").attr("class", "d3-layer");

    leafletRef.current = map;
    svgLayerRef.current = svgLayer;
    overlaySvgRef.current = overlaySvg;
    gRootRef.current = gRoot;

    const onMoveStart = () => {
      wasDraggedRef.current = true;
    };

    const onMove = () => redrawAll();

    map.on("movestart", onMoveStart);
    map.on("zoomstart", onMoveStart);
    map.on("zoom viewreset moveend", onMove);

    setTimeout(() => map.invalidateSize(), 0);

    return () => {
      try {
        map.off("movestart", onMoveStart);
        map.off("zoomstart", onMoveStart);
        map.off("zoom viewreset moveend", onMove);

        clearAllSvgLayers();
        gRootRef.current?.remove();
        gRootRef.current = null;
        overlaySvgRef.current = null;

        if (svgLayerRef.current) {
          map.removeLayer(svgLayerRef.current);
          svgLayerRef.current = null;
        }

        map.remove();
      } catch {
        //
      } finally {
        leafletRef.current = null;
      }
    };
  }, [clearAllSvgLayers, redrawAll, center]);

  useEffect(() => {
    if (baseLayerRef.current) {
      baseLayerRef.current.setOpacity(showBasemap ? 0.5 : 0);
    }
  }, [showBasemap]);

  useEffect(() => {
    if (baseLayerRef.current) {
      baseLayerRef.current.setOpacity(showBasemap ? 0.5 : 0);
    }
  }, [showBasemap]);

  useEffect(() => {
    if (!leafletRef.current) return;

    const observer = new ResizeObserver(() =>
      leafletRef.current?.invalidateSize(),
    );

    if (mapRef.current) observer.observe(mapRef.current);

    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!leafletRef.current) return;

    (async () => {
      try {
        await loadFromView();
      } catch (e) {
        console.error(e);
      }
    })();
  }, [loadFromView]);

  return (
    <div
      ref={mapRef}
      className={className}
      aria-label={`Leaflet map for ${id}`}
      onPointerDown={(e) => e.stopPropagation()}
      onMouseDown={(e) => e.stopPropagation()}
      onTouchStart={(e) => e.stopPropagation()}
      // style={{ width: "100%", height: "100%" }}
    />
  );
});

export default ViewportCanvas;
