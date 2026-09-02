import { useRef, useState } from 'react';
import CardShell from './CardShell';
import { formatDateTime } from '../lib/format';
import { uploadSatelliteImage, analyzeSampleImage } from '../api/client';

const SAMPLE_LIST = [
  { key: 'FANI', label: 'FANI (2019 · Odisha)', file: 'p7_mosdac_FANI_2019_0000_Very_Severe_Cyclonic_Storm.png' },
  { key: 'AMPHAN', label: 'AMPHAN (2020 · Bengal)', file: 'p7_mosdac_AMPHAN_2020_0001_Very_Severe_Cyclonic_Storm.png' },
  { key: 'BIPARJOY', label: 'BIPARJOY (2023 · Gujarat)', file: 'p7_mosdac_BIPARJOY_2023_0015_Very_Severe_Cyclonic_Storm.png' },
  { key: 'TAUKTAE', label: 'TAUKTAE (2021 · West Coast)', file: 'p7_mosdac_TAUKTAE_2021_0011_Very_Severe_Cyclonic_Storm.png' },
];

export default function SatelliteViewer({ satellite, meta, loading, onImageResult }){
  const [preview, setPreview] = useState(null);     // object-URL of the local preview
  const [tensorPreview, setTensorPreview] = useState(null); // base64 of 224x224
  const [viewMode, setViewMode] = useState('raw');  // 'raw' | 'tensor'
  const [scale, setScale] = useState(1);
  const [state, setState] = useState('idle');       // idle|selected|uploading|done|failed
  const [fileMeta, setFileMeta] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [sourceLabel, setSourceLabel] = useState('REFERENCE FRAME');
  const [activeSample, setActiveSample] = useState(null);
  const inputRef = useRef(null);

  async function handleFile(e){
    const file = e.target.files?.[0];
    if(!file) return;
    setActiveSample(null);
    if(preview) URL.revokeObjectURL(preview);
    setPreview(URL.createObjectURL(file));
    setTensorPreview(null);
    setFileMeta({ name: file.name, size: (file.size / 1024 / 1024).toFixed(2) + " MB" });
    setState('uploading');
    setErrorMsg(null);
    setSourceLabel('USER IMAGE · ANALYZING');
    try {
      const result = await uploadSatelliteImage(file);
      setSourceLabel(result.sourceLabel || 'USER-UPLOADED IMAGE');
      if (result.preprocessedPreview) setTensorPreview(result.preprocessedPreview);
      setState('done');
      if(onImageResult) onImageResult(result);
    } catch (err) {
      setErrorMsg(err.message);
      setState('failed');
      setSourceLabel('USER IMAGE · FAILED');
    }
  }

  async function handleSampleSelect(sample){
    setActiveSample(sample.key);
    setState('uploading');
    setErrorMsg(null);
    setSourceLabel(`MOSDAC · ${sample.key} (ANALYZING)`);
    try {
      const result = await analyzeSampleImage(sample.key);
      setPreview(`/api/sample_images/${sample.key}`);
      if (result.preprocessedPreview) setTensorPreview(result.preprocessedPreview);
      setFileMeta({ name: sample.file, size: "0.08 MB (Genuine MOSDAC)" });
      setSourceLabel(`MOSDAC GENUINE CROP · ${sample.key}`);
      setState('done');
      if(onImageResult) onImageResult(result);
    } catch (err) {
      setErrorMsg(err.message);
      setState('failed');
      setSourceLabel(`MOSDAC · ${sample.key} FAILED`);
    }
  }

  const bbox = satellite?.boundingBox;
  const refSource = satellite?.source || '';

  return (
    <CardShell
      eyebrow="Primary Input Feeds"
      title="Satellite Observation Viewer &amp; Preprocessing Inspector"
      right={
        <div className="flex items-center gap-2">
          <button
            onClick={() => inputRef.current?.click()}
            className="text-[11.5px] font-semibold text-ink border border-border bg-card rounded-lg px-3 py-1.5 hover:border-accent hover:bg-card-alt transition-colors cursor-pointer"
          >
            Upload Custom Image
          </button>
          <input ref={inputRef} type="file" accept="image/*" onChange={handleFile} className="hidden" />
        </div>
      }
    >
      <div className="space-y-3">
        {/* Genuine Sample Cyclone Image Selector */}
        <div className="bg-card-alt/40 border border-border/50 rounded-xl p-2.5 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-bold text-ink-faint uppercase tracking-wider">
              Genuine MOSDAC Samples:
            </span>
            <div className="flex flex-wrap gap-1">
              {SAMPLE_LIST.map(s => (
                <button
                  key={s.key}
                  onClick={() => handleSampleSelect(s)}
                  className={`text-[11px] px-2.5 py-1 rounded-lg border font-mono transition-all cursor-pointer ${
                    activeSample === s.key
                      ? 'bg-accent-strong text-white border-accent-strong font-bold shadow-sm'
                      : 'bg-card border-border text-ink hover:border-accent hover:bg-card-alt'
                  }`}
                >
                  {s.label}
                </button>
              ))}
            </div>
          </div>

          {/* Preprocessing Inspector Toggle */}
          <div className="flex items-center bg-card border border-border rounded-lg p-0.5 text-[10.5px]">
            <button
              onClick={() => setViewMode('raw')}
              className={`px-2.5 py-1 rounded-md transition-colors cursor-pointer font-medium ${
                viewMode === 'raw' ? 'bg-accent-strong text-white font-bold' : 'text-ink-soft hover:text-ink'
              }`}
            >
              Raw IR View
            </button>
            <button
              onClick={() => setViewMode('tensor')}
              className={`px-2.5 py-1 rounded-md transition-colors cursor-pointer font-medium ${
                viewMode === 'tensor' ? 'bg-accent-strong text-white font-bold' : 'text-ink-soft hover:text-ink'
              }`}
            >
              224×224 E9-2 Input
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">

          {/* Image Container Panel */}
          <div className="md:col-span-3 relative rounded-xl overflow-hidden border border-border-soft bg-[#0d1420] h-[320px] select-none">
            <div
              className="absolute inset-0 flex items-center justify-center"
              style={{ transform: `scale(${scale})`, transformOrigin: 'center center', transition: 'transform 0.2s ease' }}
            >
              {viewMode === 'tensor' && tensorPreview ? (
                <div className="flex flex-col items-center justify-center h-full w-full bg-black/90 p-2">
                  <img src={tensorPreview} alt="224x224 Normalized Tensor Input" className="w-[224px] h-[224px] object-contain border border-white/20 shadow-lg" />
                  <span className="text-[9.5px] font-mono text-ink-faint mt-1 uppercase tracking-wider">
                    MobileNetV3 224×224 Normalization Transform
                  </span>
                </div>
              ) : preview ? (
                <img src={preview} alt="Satellite image" className="max-w-none w-full h-full object-cover" />
              ) : (
                <SyntheticSatelliteImage />
              )}

              {bbox && !loading && (
                <div
                  className="absolute border-2 rounded-sm shadow-[0_0_10px_rgba(242,169,59,0.5)]"
                  style={{
                    borderColor: '#F2A93B',
                    left: `${bbox.x * 100}%`, top: `${bbox.y * 100}%`,
                    width: `${bbox.w * 100}%`, height: `${bbox.h * 100}%`,
                  }}
                >
                  <span className="absolute -top-6 left-0 text-[10px] font-bold text-[#F2A93B] bg-[#0d1420]/95 px-2 py-0.5 rounded border border-[#F2A93B]/40 whitespace-nowrap uppercase tracking-wider font-mono">
                    Cyclone Center: {bbox.confidence}%
                  </span>
                </div>
              )}

              {!loading && (
                <div className="absolute top-2 left-2 flex items-center gap-1.5">
                  {state === 'uploading' && (
                    <span className="text-[10px] font-bold text-amber-300 bg-[#0d1420]/90 px-2 py-0.5 rounded border border-amber-300/40 uppercase tracking-wider font-mono animate-pulse">
                      Analyzing through E9-2…
                    </span>
                  )}
                  {state === 'failed' && (
                    <span className="text-[10px] font-bold text-red-400 bg-[#0d1420]/90 px-2 py-0.5 rounded border border-red-400/40 uppercase tracking-wider font-mono">
                      Analysis failed — see metadata
                    </span>
                  )}
                </div>
              )}
            </div>

            {/* Zoom controls */}
            <div className="absolute bottom-3 right-3 flex items-center gap-1.5 bg-[#0d1420]/80 backdrop-blur rounded-lg px-2 py-1.5 border border-white/10 z-10">
              <button onClick={() => setScale(s => Math.max(1, +(s - 0.25).toFixed(2)))} className="w-7 h-7 rounded-md text-white/90 hover:bg-white/10 text-sm leading-none font-bold">−</button>
              <span className="text-[11px] text-white/80 w-9 text-center font-mono tabular-nums">{Math.round(scale * 100)}%</span>
              <button onClick={() => setScale(s => Math.min(3, +(s + 0.25).toFixed(2)))} className="w-7 h-7 rounded-md text-white/90 hover:bg-white/10 text-sm leading-none font-bold">+</button>
              <button onClick={() => setScale(1)} className="ml-1 text-[10.5px] text-white/60 hover:text-white px-1.5 font-semibold">Reset</button>
            </div>
          </div>

        {/* Info panel */}
        <div className="md:col-span-1 flex flex-col justify-between p-3.5 bg-card-alt/60 border border-border/50 rounded-xl space-y-4 text-[11.5px] overflow-hidden min-w-0">
          <div className="space-y-3 min-w-0">
            <span className="text-[9.5px] uppercase tracking-wider text-ink-faint font-bold block border-b border-border/50 pb-1">Image Metadata</span>

            <div className="min-w-0">
              <span className="text-ink-faint text-[9.5px] uppercase block tracking-wider">Source Platform</span>
              <span className="text-ink font-semibold break-words">
                {preview ? "User-uploaded satellite image" : "INSAT-3D IR (reference frame)"}
              </span>
            </div>

            <div className="min-w-0">
              <span className="text-ink-faint text-[9.5px] uppercase block tracking-wider">Frame</span>
              <span className="text-ink font-mono text-[10px] break-all leading-tight block">
                {preview ? (fileMeta?.name || "user image") : (refSource || "—")}
              </span>
            </div>

            <div className="min-w-0">
              <span className="text-ink-faint text-[9.5px] uppercase block tracking-wider">Source</span>
              <span className="text-ink font-mono text-[10px] font-medium leading-tight break-words block">
                {sourceLabel}
              </span>
            </div>

            <div className="min-w-0">
              <span className="text-ink-faint text-[9.5px] uppercase block tracking-wider">Last Pass</span>
              <span className="text-ink font-mono font-medium leading-tight block break-words">
                {preview
                  ? (fileMeta?.name ? new Date().toISOString().replace('T', ' ').substring(0, 19) + ' UTC' : "Recent")
                  : (meta?.lastPass ? formatDateTime(meta.lastPass) : "—")}
              </span>
            </div>

            <div className="min-w-0">
              <span className="text-ink-faint text-[9.5px] uppercase block tracking-wider">Cyclone Center Box</span>
              <span className="text-ink font-semibold break-words">{bbox ? `ML localizer ${bbox.confidence}%` : "Not available (no audited localizer)"}</span>
            </div>

            {state === 'failed' && errorMsg && (
              <div className="text-[10px] text-red-300 font-mono leading-relaxed border border-red-400/30 bg-red-400/5 rounded-lg p-2 break-all">
                {errorMsg}
              </div>
            )}
          </div>

          <div className="text-[10px] text-ink-faint font-mono leading-relaxed pt-2 border-t border-border/50 break-words min-w-0 overflow-hidden">
            {preview ? (
              <div className="space-y-1">
                <div>Uploaded image IS analyzed by backend (P2 detection + P3 classification on uploaded bytes).</div>
                <div className="text-ink font-sans text-[10.5px] break-all">
                  <span className="font-semibold text-ink-faint font-mono text-[9px] uppercase">File: </span>
                  {fileMeta?.name || "uploaded image"} <span className="text-ink-faint font-mono text-[9px]">({fileMeta?.size || "—"})</span>
                </div>
              </div>
            ) : (
              "Illustrative reference frame from the artifact bundle. Detection/classification run locally on this one fixed frame."
            )}
            <div className="mt-2 text-[9px] break-words">
              STATUS: {state === 'done' ? 'USER IMAGE · BACKEND ANALYZED' : (preview ? 'USER IMAGE · LOCAL PREVIEW' : 'REFERENCE FRAME')} · NOT LIVE
            </div>
          </div>
        </div>

      </div>
      </div>
    </CardShell>
  );
}

// Procedural placeholder rendering
function SyntheticSatelliteImage(){
  return (
    <svg viewBox="0 0 400 320" className="w-full h-full">
      <defs>
        <radialGradient id="satBg" cx="50%" cy="45%" r="75%">
          <stop offset="0%" stopColor="#1a2534" />
          <stop offset="100%" stopColor="#0a1119" />
        </radialGradient>
        <filter id="satBlur"><feGaussianBlur stdDeviation="6" /></filter>
      </defs>
      <rect width="400" height="320" fill="url(#satBg)" />
      {[70, 55, 42, 30, 20].map((r, i) => (
        <circle key={r} cx={190 + i * 4} cy={150 - i * 2} r={r} fill="none"
          stroke="#cfd8e3" strokeOpacity={0.12 + i * 0.03} strokeWidth={10} filter="url(#satBlur)" />
      ))}
      <circle cx="196" cy="146" r="14" fill="#0a1119" stroke="#cfd8e3" strokeOpacity="0.3" strokeWidth="2" />
    </svg>
  );
}
