import { useEffect, useRef } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { validateBbox } from "./api.js";

const ZONE_FALLBACK = [68.0, 19.0, 89.0, 32.5];

function bboxPolygon([w, s, e, n]) {
  return {
    type: "Feature",
    geometry: {
      type: "Polygon",
      coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
    },
    properties: {},
  };
}

// World polygon with the supported zone cut out — dims everything the
// classifier can't serve.
function zoneMask([w, s, e, n]) {
  return {
    type: "Feature",
    geometry: {
      type: "Polygon",
      coordinates: [
        [[-180, -85], [180, -85], [180, 85], [-180, 85], [-180, -85]],
        [[w, s], [w, n], [e, n], [e, s], [w, s]],
      ],
    },
    properties: {},
  };
}

const STYLE = {
  version: 8,
  glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
  sources: {
    esri: {
      type: "raster",
      tiles: [
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      ],
      tileSize: 256,
      maxzoom: 18,
      attribution: "Imagery © Esri, Maxar, Earthstar Geographics",
    },
    osm: {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      maxzoom: 19,
      attribution: "© OpenStreetMap contributors",
    },
  },
  layers: [
    { id: "basemap-esri", type: "raster", source: "esri" },
    {
      id: "basemap-osm",
      type: "raster",
      source: "osm",
      layout: { visibility: "none" },
    },
  ],
};

export default function MapView({
  limits,
  bbox,
  onBbox,
  drawMode,
  onDrawModeEnd,
  focusRequest, // {bounds:[w,s,e,n], nonce} -> fly there
  basemap, // "esri" | "osm"
}) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const loadedRef = useRef(false);
  const drawRef = useRef({ active: false, start: null, dragging: false });
  const onBboxRef = useRef(onBbox);
  const onDrawEndRef = useRef(onDrawModeEnd);
  onBboxRef.current = onBbox;
  onDrawEndRef.current = onDrawModeEnd;

  // ---- map init (once) ----
  useEffect(() => {
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: STYLE,
      bounds: [
        [ZONE_FALLBACK[0] - 1.5, ZONE_FALLBACK[1] - 1.5],
        [ZONE_FALLBACK[2] + 1.5, ZONE_FALLBACK[3] + 1.5],
      ],
      attributionControl: { compact: true },
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }));
    map.addControl(
      new maplibregl.ScaleControl({ maxWidth: 120, unit: "metric" }),
      "bottom-right"
    );
    map.boxZoom.disable();

    map.on("load", () => {
      map.addSource("zone-mask", { type: "geojson", data: zoneMask(ZONE_FALLBACK) });
      map.addSource("zone-line", { type: "geojson", data: bboxPolygon(ZONE_FALLBACK) });
      map.addSource("pilots", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
      map.addSource("aoi", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });

      map.addLayer({
        id: "zone-mask-fill",
        type: "fill",
        source: "zone-mask",
        paint: { "fill-color": "#0d0d0d", "fill-opacity": 0.45 },
      });
      map.addLayer({
        id: "zone-outline",
        type: "line",
        source: "zone-line",
        paint: {
          "line-color": "#c3c2b7",
          "line-width": 1.5,
          "line-dasharray": [3, 2],
        },
      });
      map.addLayer({
        id: "pilots-fill",
        type: "fill",
        source: "pilots",
        paint: { "fill-color": "#0ca30c", "fill-opacity": 0.07 },
      });
      map.addLayer({
        id: "pilots-line",
        type: "line",
        source: "pilots",
        paint: { "line-color": "#0ca30c", "line-width": 1.5 },
      });
      map.addLayer({
        id: "pilots-label",
        type: "symbol",
        source: "pilots",
        layout: {
          "text-field": ["get", "label"],
          "text-font": ["Open Sans Semibold"],
          "text-size": 12,
          "text-anchor": "top-left",
          "text-offset": [0.4, 0.4],
        },
        paint: {
          "text-color": "#7ee787",
          "text-halo-color": "#0d0d0d",
          "text-halo-width": 1.4,
        },
      });
      map.addLayer({
        id: "aoi-fill",
        type: "fill",
        source: "aoi",
        paint: { "fill-color": "#3987e5", "fill-opacity": 0.18 },
      });
      map.addLayer({
        id: "aoi-line",
        type: "line",
        source: "aoi",
        paint: { "line-color": "#3987e5", "line-width": 2 },
      });
      loadedRef.current = true;
      map.fire("ps6:ready");
    });

    // ---- draw-a-box interaction ----
    const dr = drawRef.current;
    const setBoxFrom = (a, b) => {
      const box = [
        Math.min(a.lng, b.lng),
        Math.min(a.lat, b.lat),
        Math.max(a.lng, b.lng),
        Math.max(a.lat, b.lat),
      ];
      onBboxRef.current(box);
    };
    map.on("mousedown", (e) => {
      if (!dr.active) return;
      e.preventDefault();
      dr.start = e.lngLat;
      dr.dragging = true;
    });
    map.on("mousemove", (e) => {
      if (dr.dragging && dr.start) setBoxFrom(dr.start, e.lngLat);
    });
    map.on("mouseup", (e) => {
      if (!dr.dragging) return;
      setBoxFrom(dr.start, e.lngLat);
      dr.dragging = false;
      dr.start = null;
      onDrawEndRef.current();
    });
    map.on("touchstart", (e) => {
      if (!dr.active || e.points.length !== 1) return;
      e.preventDefault();
      dr.start = e.lngLat;
      dr.dragging = true;
    });
    map.on("touchmove", (e) => {
      if (dr.dragging && dr.start) {
        e.preventDefault();
        setBoxFrom(dr.start, e.lngLat);
      }
    });
    map.on("touchend", () => {
      if (!dr.dragging) return;
      dr.dragging = false;
      dr.start = null;
      onDrawEndRef.current();
    });

    mapRef.current = map;
    if (import.meta.env.DEV) window.__ps6map = map;
    return () => {
      map.remove();
      mapRef.current = null;
      loadedRef.current = false;
    };
  }, []);

  // run fn now if style is loaded, else after load
  const whenReady = (fn) => {
    const map = mapRef.current;
    if (!map) return;
    if (loadedRef.current) fn(map);
    else map.once("ps6:ready", () => fn(map));
  };

  // ---- limits -> zone + pilots ----
  useEffect(() => {
    if (!limits) return;
    whenReady((map) => {
      map.getSource("zone-mask").setData(zoneMask(limits.supported_bbox));
      map.getSource("zone-line").setData(bboxPolygon(limits.supported_bbox));
      const feats = Object.entries(limits.pilots).map(([name, bb]) => {
        const f = bboxPolygon(bb);
        f.properties.label = `${name} · VALIDATED`;
        return f;
      });
      map.getSource("pilots").setData({
        type: "FeatureCollection",
        features: feats,
      });
    });
  }, [limits]);

  // ---- bbox -> AOI layer ----
  useEffect(() => {
    whenReady((map) => {
      const src = map.getSource("aoi");
      if (!bbox) {
        src.setData({ type: "FeatureCollection", features: [] });
        return;
      }
      src.setData(bboxPolygon(bbox));
      const v = validateBbox(bbox, limits);
      const color = v.ok ? (v.validated ? "#0ca30c" : "#3987e5") : "#e66767";
      map.setPaintProperty("aoi-fill", "fill-color", color);
      map.setPaintProperty("aoi-line", "line-color", color);
    });
  }, [bbox, limits]);

  // ---- draw mode toggling ----
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    drawRef.current.active = drawMode;
    map.getCanvas().style.cursor = drawMode ? "crosshair" : "";
    if (drawMode) {
      map.dragPan.disable();
      map.touchZoomRotate.disable();
    } else {
      map.dragPan.enable();
      map.touchZoomRotate.enable();
      drawRef.current.dragging = false;
      drawRef.current.start = null;
    }
  }, [drawMode]);

  // ---- external focus requests (pilot buttons, demo AOI) ----
  useEffect(() => {
    if (!focusRequest) return;
    const map = mapRef.current;
    if (!map) return;
    const [w, s, e, n] = focusRequest.bounds;
    map.fitBounds([[w, s], [e, n]], { padding: 60, duration: 900 });
  }, [focusRequest]);

  // ---- basemap toggle ----
  useEffect(() => {
    whenReady((map) => {
      map.setLayoutProperty(
        "basemap-esri",
        "visibility",
        basemap === "esri" ? "visible" : "none"
      );
      map.setLayoutProperty(
        "basemap-osm",
        "visibility",
        basemap === "osm" ? "visible" : "none"
      );
    });
  }, [basemap]);

  return <div ref={containerRef} className="map-container" />;
}
