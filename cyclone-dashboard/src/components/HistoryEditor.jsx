import { useMemo, useState } from 'react';
import CardShell from './CardShell';
import { DEMO_HISTORY } from '../api/client';

const FIELDS = [
  { key: 'timestamp', label: 'Timestamp (ISO-UTC)', type: 'text' },
  { key: 'latitude', label: 'Lat', type: 'number' },
  { key: 'longitude', label: 'Lon', type: 'number' },
  { key: 'wind_speed_kmh', label: 'Wind km/h', type: 'number' },
  { key: 'pressure_hpa', label: 'Pressure hPa', type: 'number' },
  { key: 'sst', label: 'SST °C', type: 'number' },
  { key: 'wind_u', label: 'Wind-u', type: 'number' },
  { key: 'wind_v', label: 'Wind-v', type: 'number' },
];

function clone(rows){ return rows.map(r => ({ ...r })); }

function blankRow(){
  return {
    timestamp: '',
    latitude: Number.NaN,
    longitude: Number.NaN,
    wind_speed_kmh: Number.NaN,
    pressure_hpa: Number.NaN,
    sst: Number.NaN,
    wind_u: Number.NaN,
    wind_v: Number.NaN,
  };
}

function validate(rows){
  const errors = [];
  if(!Array.isArray(rows) || rows.length !== 5){
    return ['History must contain exactly 5 observations (found ' + (rows?.length || 0) + ').'];
  }
  rows.forEach((r, i) => {
    const ts = Date.parse(r.timestamp);
    if(Number.isNaN(ts)) errors.push(`Observation ${i + 1}: timestamp is not a valid ISO-UTC time.`);
    if(typeof r.latitude !== 'number' || !isFinite(r.latitude) || r.latitude < -90 || r.latitude > 90)
      errors.push(`Observation ${i + 1}: latitude must be between -90 and 90.`);
    if(typeof r.longitude !== 'number' || !isFinite(r.longitude) || r.longitude < -180 || r.longitude > 180)
      errors.push(`Observation ${i + 1}: longitude must be between -180 and 180.`);
    if(typeof r.wind_speed_kmh !== 'number' || !isFinite(r.wind_speed_kmh) || r.wind_speed_kmh < 0)
      errors.push(`Observation ${i + 1}: wind speed must be ≥ 0.`);
    if(typeof r.pressure_hpa !== 'number' || !isFinite(r.pressure_hpa) || r.pressure_hpa < 850 || r.pressure_hpa > 1050)
      errors.push(`Observation ${i + 1}: pressure must be between 850 and 1050 hPa.`);
    if(typeof r.sst !== 'number' || !isFinite(r.sst))
      errors.push(`Observation ${i + 1}: SST must be a number.`);
  });
  // chronological + 6h spacing
  for(let i = 1; i < rows.length; i++){
    const a = Date.parse(rows[i - 1].timestamp);
    const b = Date.parse(rows[i].timestamp);
    if(!Number.isNaN(a) && !Number.isNaN(b)){
      if(b <= a) errors.push(`Observations must be strictly chronological (obs ${i} must be later than obs ${i + 1 - 1}).`);
      else {
        const h = (b - a) / 3600000;
        if(Math.abs(h - 6) > 0.00001) errors.push(`Observations must be spaced 6h apart (obs ${i - 1 + 1}→obs ${i + 1} is ${h}h).`);
      }
    }
  }
  return errors;
}

/**
 * P0-2: minimal 5-observation history editor.
 *
 * Lets a user supply a real, valid history instead of being locked to the
 * demo input. Edits are validated on the client; on submit the raw values are
 * sent through the existing, unchanged POST /api/analyze contract (the server
 * re-validates with the audited phase-6 rules).
 */
const PRESETS = [
  {
    name: "Default (Andhra/Odisha)",
    data: DEMO_HISTORY,
  },
  {
    name: "BIPARJOY (Arabian Sea / Gujarat)",
    data: [
      { timestamp: "2023-06-11T00:00:00Z", latitude: 17.50, longitude: 67.40, wind_speed_kmh: 165, pressure_hpa: 945, sst: 29.5, wind_u: 8.5, wind_v: 12.0 },
      { timestamp: "2023-06-11T06:00:00Z", latitude: 18.20, longitude: 67.60, wind_speed_kmh: 160, pressure_hpa: 950, sst: 29.4, wind_u: 8.0, wind_v: 11.5 },
      { timestamp: "2023-06-11T12:00:00Z", latitude: 18.90, longitude: 67.80, wind_speed_kmh: 155, pressure_hpa: 955, sst: 29.2, wind_u: 7.5, wind_v: 11.0 },
      { timestamp: "2023-06-11T18:00:00Z", latitude: 19.60, longitude: 68.00, wind_speed_kmh: 150, pressure_hpa: 960, sst: 29.0, wind_u: 7.0, wind_v: 10.5 },
      { timestamp: "2023-06-12T00:00:00Z", latitude: 20.30, longitude: 68.30, wind_speed_kmh: 145, pressure_hpa: 965, sst: 28.8, wind_u: 6.8, wind_v: 10.0 },
    ],
  },
  {
    name: "AMPHAN (Bengal / Sundarbans)",
    data: [
      { timestamp: "2020-05-18T00:00:00Z", latitude: 13.20, longitude: 86.30, wind_speed_kmh: 190, pressure_hpa: 935, sst: 30.2, wind_u: 9.0, wind_v: 15.0 },
      { timestamp: "2020-05-18T06:00:00Z", latitude: 14.10, longitude: 86.40, wind_speed_kmh: 220, pressure_hpa: 920, sst: 30.1, wind_u: 8.8, wind_v: 15.5 },
      { timestamp: "2020-05-18T12:00:00Z", latitude: 15.00, longitude: 86.50, wind_speed_kmh: 240, pressure_hpa: 910, sst: 30.0, wind_u: 8.5, wind_v: 16.0 },
      { timestamp: "2020-05-18T18:00:00Z", latitude: 16.00, longitude: 86.70, wind_speed_kmh: 230, pressure_hpa: 918, sst: 29.8, wind_u: 8.0, wind_v: 15.8 },
      { timestamp: "2020-05-19T00:00:00Z", latitude: 17.20, longitude: 87.00, wind_speed_kmh: 215, pressure_hpa: 925, sst: 29.7, wind_u: 7.8, wind_v: 15.2 },
    ],
  },
  {
    name: "FANI (Odisha Coast / Puri)",
    data: [
      { timestamp: "2019-05-01T00:00:00Z", latitude: 12.80, longitude: 84.80, wind_speed_kmh: 175, pressure_hpa: 942, sst: 29.8, wind_u: 6.5, wind_v: 13.0 },
      { timestamp: "2019-05-01T06:00:00Z", latitude: 13.70, longitude: 84.70, wind_speed_kmh: 190, pressure_hpa: 934, sst: 29.8, wind_u: 6.2, wind_v: 13.5 },
      { timestamp: "2019-05-01T12:00:00Z", latitude: 14.60, longitude: 84.60, wind_speed_kmh: 205, pressure_hpa: 926, sst: 29.7, wind_u: 6.0, wind_v: 14.0 },
      { timestamp: "2019-05-01T18:00:00Z", latitude: 15.50, longitude: 84.60, wind_speed_kmh: 215, pressure_hpa: 920, sst: 29.6, wind_u: 5.8, wind_v: 14.2 },
      { timestamp: "2019-05-02T00:00:00Z", latitude: 16.50, longitude: 84.80, wind_speed_kmh: 200, pressure_hpa: 930, sst: 29.5, wind_u: 5.5, wind_v: 13.8 },
    ],
  },
  {
    name: "TAUKTAE (West Coast / Gujarat)",
    data: [
      { timestamp: "2021-05-16T00:00:00Z", latitude: 14.50, longitude: 72.80, wind_speed_kmh: 150, pressure_hpa: 960, sst: 29.6, wind_u: 5.0, wind_v: 14.0 },
      { timestamp: "2021-05-16T06:00:00Z", latitude: 15.60, longitude: 72.50, wind_speed_kmh: 165, pressure_hpa: 952, sst: 29.5, wind_u: 4.8, wind_v: 14.5 },
      { timestamp: "2021-05-16T12:00:00Z", latitude: 16.70, longitude: 72.10, wind_speed_kmh: 180, pressure_hpa: 944, sst: 29.4, wind_u: 4.5, wind_v: 15.0 },
      { timestamp: "2021-05-16T18:00:00Z", latitude: 17.80, longitude: 71.70, wind_speed_kmh: 195, pressure_hpa: 936, sst: 29.3, wind_u: 4.2, wind_v: 15.2 },
      { timestamp: "2021-05-17T00:00:00Z", latitude: 18.80, longitude: 71.30, wind_speed_kmh: 185, pressure_hpa: 942, sst: 29.2, wind_u: 4.0, wind_v: 14.8 },
    ],
  }
];

export default function HistoryEditor({ onAnalyze }){
  const [rows, setRows] = useState(() => clone(DEMO_HISTORY));
  const [activePreset, setActivePreset] = useState("Default (Andhra/Odisha)");
  const [showManualEditor, setShowManualEditor] = useState(false);
  const [dirty, setDirty] = useState(false);
  const errors = useMemo(() => validate(rows), [rows]);

  function loadPreset(preset){
    setActivePreset(preset.name);
    setRows(clone(preset.data));
    setDirty(false);
    onAnalyze(preset.data);
  }

  function updateCell(rowIdx, key, raw){
    setDirty(true);
    setActivePreset(null);
    setRows(prev => {
      const next = clone(prev);
      let value;
      if(key === 'timestamp') value = raw;
      else if(raw.trim() === '') value = Number.NaN;
      else value = Number(raw);
      next[rowIdx] = { ...next[rowIdx], [key]: value };
      return next;
    });
  }

  function reset(){
    setRows(clone(DEMO_HISTORY));
    setActivePreset("Default (Andhra/Odisha)");
    setDirty(false);
    onAnalyze(DEMO_HISTORY);
  }

  function addRow(){
    setDirty(true);
    setActivePreset(null);
    setRows(prev => [...prev, blankRow()]);
  }

  function deleteRow(rowIdx){
    setDirty(true);
    setActivePreset(null);
    setRows(prev => prev.filter((_, i) => i !== rowIdx));
  }

  function clear(){
    setDirty(true);
    setActivePreset(null);
    setRows([]);
  }

  function submit(){
    if(errors.length) return;
    const hist = rows.map(r => ({
      timestamp: r.timestamp,
      latitude: r.latitude,
      longitude: r.longitude,
      wind_speed_kmh: r.wind_speed_kmh,
      pressure_hpa: r.pressure_hpa,
      sst: r.sst,
      wind_u: r.wind_u,
      wind_v: r.wind_v,
    }));
    onAnalyze(hist);
    setDirty(false);
  }

  return (
    <CardShell
      eyebrow="Input History &amp; Spatial Track"
      title="Cyclone Track &amp; Environmental Data"
      right={
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[9.5px] font-bold text-ink-faint border border-border/60 bg-card-alt/40 rounded px-2 py-1 uppercase tracking-wider font-mono">
            {dirty ? "CUSTOM COORDINATES" : (activePreset ? `PRESET: ${activePreset.split(' ')[0]}` : "DEMO DATA")}
          </span>
          <button
            onClick={reset}
            className="text-[11.5px] font-semibold text-ink border border-border bg-card rounded-lg px-3 py-1.5 hover:border-accent hover:bg-card-alt transition-all cursor-pointer"
          >
            Reset
          </button>
          <button
            onClick={submit}
            disabled={errors.length > 0}
            className="text-[11.5px] font-semibold text-white bg-accent-strong rounded-lg px-4 py-1.5 hover:opacity-90 active:scale-95 transition-all shadow-sm disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer flex items-center gap-1.5"
          >
            <span>⚡</span>
            Re-run AI Forecast
          </button>
        </div>
      }
    >
      <div className="space-y-4">
        {/* Preset Selector */}
        <div>
          <span className="text-[9.5px] uppercase tracking-wider text-ink-faint font-bold block mb-1.5">
            Load Historical Cyclone Track Preset (Moves Map &amp; Runs Real Forecast):
          </span>
          <div className="flex flex-wrap gap-1.5">
            {PRESETS.map(p => (
              <button
                key={p.name}
                onClick={() => loadPreset(p)}
                className={`text-[11px] font-medium px-2.5 py-1 rounded-lg border transition-all cursor-pointer ${
                  activePreset === p.name
                    ? 'bg-accent-strong text-white border-accent-strong shadow-sm font-bold'
                    : 'bg-card border-border text-ink hover:border-accent hover:bg-card-alt'
                }`}
              >
                {p.name}
              </button>
            ))}
          </div>
        </div>

        {/* Collapsible Manual History Editor */}
        <div className="border-t border-border/50 pt-3">
          <div className="flex items-center justify-between">
            <button
              onClick={() => setShowManualEditor(prev => !prev)}
              className="text-[11.5px] font-bold text-accent-strong hover:underline flex items-center gap-1.5 cursor-pointer"
            >
              <span>{showManualEditor ? '▼' : '▶'}</span>
              <span>{showManualEditor ? 'Hide Advanced Coordinate Editor' : '⚙️ Advanced: Inspect & Edit Raw 5-Observation Timesteps'}</span>
            </button>

            {showManualEditor && (
              <div className="flex items-center gap-2">
                <button
                  onClick={addRow}
                  className="text-[10.5px] font-semibold text-ink border border-border bg-card rounded px-2.5 py-1 hover:border-accent"
                >
                  + Add observation
                </button>
                <button
                  onClick={clear}
                  className="text-[10.5px] font-semibold text-ink border border-border bg-card rounded px-2.5 py-1 hover:border-accent"
                >
                  Clear
                </button>
              </div>
            )}
          </div>

          {showManualEditor && (
            <div className="mt-3 space-y-3">
              <div className="overflow-x-auto">
                <table className="w-full text-[11px] border-collapse">
                  <thead>
                    <tr>
                      {FIELDS.map(f => (
                        <th key={f.key} className="text-left text-ink-faint font-semibold px-1.5 py-1 border-b border-border">{f.label}</th>
                      ))}
                      <th className="text-left text-ink-faint font-semibold px-1.5 py-1 border-b border-border">Del</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row, i) => (
                      <tr key={i} className="align-top">
                        {FIELDS.map(f => (
                          <td key={f.key} className="px-1 py-1 border-b border-border/40">
                            <input
                              type={f.type}
                              step={f.type === 'number' ? 'any' : undefined}
                              value={f.key === 'timestamp' ? row[f.key] : (Number.isFinite(row[f.key]) ? String(row[f.key]) : '')}
                              onChange={e => updateCell(i, f.key, e.target.value)}
                              className="w-full text-[11px] bg-card-alt/60 border border-border/50 rounded px-1.5 py-0.5 text-ink focus:border-accent outline-none font-mono"
                            />
                          </td>
                        ))}
                        <td className="px-1 py-1 border-b border-border/40">
                          <button
                            onClick={() => deleteRow(i)}
                            title="Delete this observation"
                            className="text-[10px] font-bold text-risk-high bg-risk-high/10 border border-risk-high/25 rounded px-1.5 py-0.5 hover:bg-risk-high/20 cursor-pointer"
                          >
                            ✕
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {errors.length > 0 && (
                <div className="text-[11px] text-risk-high bg-risk-high/10 border border-risk-high/25 rounded-lg px-3 py-2 font-mono">
                  {errors.map((e, i) => <div key={i}>• {e}</div>)}
                </div>
              )}

              <p className="text-[10px] text-ink-faint leading-relaxed font-mono">
                Observation history requires 5 points spaced 6h apart. Sent directly to POST /api/analyze for causal feature matrix generation and EXP005 GRU inference.
              </p>
            </div>
          )}
        </div>
      </div>
    </CardShell>
  );
}
