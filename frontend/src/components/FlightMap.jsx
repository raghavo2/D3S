import { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Polyline, CircleMarker, Popup, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import './FlightMap.css';

/**
 * Auto-fit the map bounds to the flight path.
 */
function FitBounds({ bounds }) {
  const map = useMap();
  useEffect(() => {
    if (bounds) {
      map.fitBounds([
        [bounds.lat_min, bounds.lon_min],
        [bounds.lat_max, bounds.lon_max],
      ], { padding: [30, 30] });
    }
  }, [bounds, map]);
  return null;
}

/**
 * Map an altitude value to a color (blue=low → red=high).
 */
function altitudeColor(alt, altMin, altMax) {
  const range = altMax - altMin || 1;
  const t = (alt - altMin) / range; // 0..1
  const r = Math.round(255 * t);
  const b = Math.round(255 * (1 - t));
  return `rgb(${r}, 80, ${b})`;
}

/**
 * Interactive flight path map using Leaflet + OpenStreetMap.
 * Renders GPS coordinates as a colored polyline with altitude gradient.
 */
export default function FlightMap({ gpsData, compact = false }) {
  if (!gpsData || !gpsData.points || gpsData.points.length === 0) {
    return (
      <div className={`flight-map-empty ${compact ? 'compact' : ''}`}>
        <span>📡</span>
        <span>No GPS data available</span>
      </div>
    );
  }

  const { points, bounds } = gpsData;
  const altMin = bounds ? Math.min(...points.map(p => p.altitude)) : 0;
  const altMax = bounds ? Math.max(...points.map(p => p.altitude)) : 100;
  const center = bounds
    ? [(bounds.lat_min + bounds.lat_max) / 2, (bounds.lon_min + bounds.lon_max) / 2]
    : [0, 0];

  const pathCoords = points.map(p => [p.latitude, p.longitude]);

  return (
    <div className={`flight-map-container ${compact ? 'compact' : ''}`}>
      <div className="flight-map-header">
        <span className="flight-map-title">🗺️ Flight Path</span>
        <span className="flight-map-stats">
          {points.length} pts · {altMin.toFixed(0)}m—{altMax.toFixed(0)}m
        </span>
      </div>
      <MapContainer
        center={center}
        zoom={17}
        className="flight-map"
        scrollWheelZoom={!compact}
        zoomControl={!compact}
        dragging={!compact ? true : true}
        attributionControl={false}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* Flight path polyline */}
        <Polyline
          positions={pathCoords}
          color="#f0f0f0"
          weight={3}
          opacity={0.8}
        />

        {/* Frame position markers */}
        {points.map((point, i) => (
          <CircleMarker
            key={i}
            center={[point.latitude, point.longitude]}
            radius={compact ? 3 : 5}
            fillColor={altitudeColor(point.altitude, altMin, altMax)}
            fillOpacity={0.9}
            stroke={false}
          >
            {!compact && (
              <Popup>
                <div className="flight-map-popup">
                  <strong>{point.filename || `Point ${i + 1}`}</strong>
                  <br />
                  Lat: {point.latitude.toFixed(6)}°
                  <br />
                  Lon: {point.longitude.toFixed(6)}°
                  <br />
                  Alt: {point.altitude.toFixed(1)}m
                </div>
              </Popup>
            )}
          </CircleMarker>
        ))}

        {/* Start marker */}
        <CircleMarker
          center={[points[0].latitude, points[0].longitude]}
          radius={compact ? 5 : 8}
          fillColor="#10b981"
          fillOpacity={1}
          color="#fff"
          weight={2}
        >
          <Popup>Start</Popup>
        </CircleMarker>

        {/* End marker */}
        <CircleMarker
          center={[points[points.length - 1].latitude, points[points.length - 1].longitude]}
          radius={compact ? 5 : 8}
          fillColor="#ef4444"
          fillOpacity={1}
          color="#fff"
          weight={2}
        >
          <Popup>End</Popup>
        </CircleMarker>

        <FitBounds bounds={bounds} />
      </MapContainer>

      {/* Altitude Legend */}
      {!compact && (
        <div className="flight-map-legend">
          <span style={{ color: 'rgb(0, 80, 255)' }}>▮ Low</span>
          <div className="flight-map-gradient" />
          <span style={{ color: 'rgb(255, 80, 0)' }}>High ▮</span>
        </div>
      )}
    </div>
  );
}
