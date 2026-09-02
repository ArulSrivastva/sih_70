// -----------------------------------------------------------------------
// API client
//
// The backend integration API (integration_api, mounted on the phase-6
// FastAPI app) is live: USE_MOCK is false and every function POSTs the
// validated 5-observation history contract to /api/*.
//
//   POST /api/analyze   -> full dashboard payload (single call)
//   POST /api/detect    -> detection block only
//   POST /api/classify  -> classification block only
//   POST /api/forecast  -> forecast block only
//
// Every UI component consumes the return value of these functions — the
// response shapes mirror src/api/mockData.js field for field.
// -----------------------------------------------------------------------

import { mockAnalyzeResponse } from './mockData';

// A 24h, 6-hourly demo history consistent with the ANIKA (BOB07) narrative.
// This is demo input data — the integration API echoes it back into the
// histories/track and drives the audited EXP005 forecast from it.
export const DEMO_HISTORY = [
  { timestamp: "2026-08-25T00:00:00Z", latitude: 13.80, longitude: 80.10, wind_speed_kmh: 105, pressure_hpa: 986, sst: 28.2, wind_u: 10.0, wind_v: 7.0 },
  { timestamp: "2026-08-25T06:00:00Z", latitude: 14.35, longitude: 80.55, wind_speed_kmh: 114, pressure_hpa: 979, sst: 28.3, wind_u: 10.5, wind_v: 6.8 },
  { timestamp: "2026-08-25T12:00:00Z", latitude: 14.98, longitude: 81.10, wind_speed_kmh: 122, pressure_hpa: 971, sst: 28.5, wind_u: 11.2, wind_v: 6.5 },
  { timestamp: "2026-08-25T18:00:00Z", latitude: 15.62, longitude: 81.58, wind_speed_kmh: 133, pressure_hpa: 962, sst: 28.5, wind_u: 11.5, wind_v: 6.6 },
  { timestamp: "2026-08-26T00:00:00Z", latitude: 16.52, longitude: 82.31, wind_speed_kmh: 145, pressure_hpa: 950, sst: 28.6, wind_u: 11.8, wind_v: 6.7 },
];

export const DEMO_META = {
  systemId: "BOB07",
  systemName: "Cyclonic Storm ANIKA",
  basin: "Bay of Bengal",
  lastPass: "2026-08-26T05:30:00Z",
};

export const USE_MOCK = false;
const API_BASE = '/api';

function wait(ms){ return new Promise(res => setTimeout(res, ms)); }

async function postJSON(path, body){
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if(!res.ok){
    let detail = `HTTP ${res.status}`;
    try{
      const err = await res.json();
      if(err?.error?.message) detail = `${err.error.code} · ${err.error.message}`;
    }catch(_){ /* non-JSON body */ }
    throw new Error(`Request to ${path} failed: ${detail}`);
  }
  return res.json();
}

/**
 * P2 + P3 on a real user-uploaded satellite image via POST /api/image
 * (Option A). The backend validates the file (type/size/decodability), runs
 * the P2 detector and P3 image classifier on it and returns a dedicated
 * image-only response. The filename is used only as a display label — it is
 * never treated as a filesystem path on the server.
 */
export async function uploadSatelliteImage(file){
  if(USE_MOCK){
    await wait(200);
    return {
      status: 'success',
      detection: mockAnalyzeResponse.detection,
      classification: mockAnalyzeResponse.classification,
      satellite: mockAnalyzeResponse.satellite,
      provenance: {
        ...mockAnalyzeResponse.provenance,
        image_source: 'USER-UPLOADED IMAGE',
        notes: ['image_source: USER-UPLOADED IMAGE (mock)'],
      },
      sourceLabel: 'USER-UPLOADED IMAGE',
    };
  }
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch(`${API_BASE}/image`, { method: 'POST', body: fd });
  if(!res.ok){
    let detail = `HTTP ${res.status}`;
    try{
      const err = await res.json();
      if(err?.error?.message) detail = `${err.error.code} · ${err.error.message}`;
    }catch(_){ /* non-JSON body */ }
    throw new Error(`Image upload to /api/image failed: ${detail}`);
  }
  const body = await res.json();
  // Surface the source label so the UI can distinguish a user image from the
  // bundled reference frame.
  return { ...body, sourceLabel: body?.provenance?.image_source || 'USER-UPLOADED IMAGE' };
}

/** Detection stage only — cyclone presence, confidence, location. */
export async function fetchDetect(history, meta){
  if(USE_MOCK){ await wait(150); return mockAnalyzeResponse.detection; }
  return postJSON('/detect', { history: history || DEMO_HISTORY, meta: meta || DEMO_META });
}

/** Classification stage only — category, wind, pressure, confidence. */
export async function fetchClassify(history, meta){
  if(USE_MOCK){ await wait(150); return mockAnalyzeResponse.classification; }
  return postJSON('/classify', { history: history || DEMO_HISTORY, meta: meta || DEMO_META });
}

/** Forecast stage only — projected positions at +6h / +12h / +24h. */
export async function fetchForecast(history, meta){
  if(USE_MOCK){ await wait(150); return mockAnalyzeResponse.forecast; }
  const body = await postJSON('/forecast', { history: history || DEMO_HISTORY, meta: meta || DEMO_META });
  return body.forecast;
}

export async function fetchHealth(){
  const res = await fetch(`${API_BASE}/health`);
  if(!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export async function fetchSampleImages(){
  const res = await fetch(`${API_BASE}/sample_images`);
  if(!res.ok) throw new Error(`Could not fetch sample images: ${res.status}`);
  return res.json();
}

export async function analyzeSampleImage(key){
  const res = await fetch(`${API_BASE}/sample_images/${key}`);
  if(!res.ok) throw new Error(`Could not load sample image ${key}: ${res.status}`);
  const blob = await res.blob();
  const file = new File([blob], `${key}_MOSDAC_GENUINE_CROP.png`, { type: 'image/png' });
  return await uploadSatelliteImage(file);
}

/**
 * Combined result for the whole dashboard — a single POST /api/analyze.
 * Accepts optional validated history/meta (e.g. from the HistoryEditor).
 * Falls back to combining the three separate endpoints if analyze fails.
 */
export async function fetchAnalyze(history, meta){
  if(USE_MOCK){
    await wait(300);
    return mockAnalyzeResponse;
  }
  return await postJSON('/analyze', { history: history || DEMO_HISTORY, meta: meta || DEMO_META });
}