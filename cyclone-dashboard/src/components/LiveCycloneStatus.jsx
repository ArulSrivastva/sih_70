import { colorForWind } from '../api/mockData';

export default function LiveCycloneStatus({ data, loading }) {
  if (loading) {
    return (
      <div className="card p-5 animate-pulse bg-card-alt flex items-center justify-center h-24">
        <span className="text-[12.5px] font-mono text-ink-soft">Loading system status...</span>
      </div>
    );
  }

  const detection = data?.detection;
  const classification = data?.classification;

  if (!detection || !detection.detected) {
    return (
      <div className="card p-5 bg-card border border-border flex items-center gap-3">
        <span className="w-3 h-3 rounded-full bg-risk-low pulse-dot" />
        <div>
          <h2 className="font-serif text-[15px] font-bold text-ink">No Active Tropical Systems Detected</h2>
          <p className="text-[11.5px] text-ink-faint">Atmospheric and satellite input feeds reflect stable environmental conditions.</p>
        </div>
      </div>
    );
  }

  const currentBand = colorForWind(classification?.windSpeedKmh ?? 0);

  return (
    <div className="card p-5 bg-card border border-risk-high/30 shadow-md relative overflow-hidden">
      {/* Background visual warning slash */}
      <div className="absolute top-0 right-0 w-2.5 h-full bg-risk-high" />

      <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-5">
        
        {/* Left Side: System identification and core alert state */}
        <div className="space-y-1.5">
          <div className="flex flex-wrap items-center gap-2">
            <span className="flex items-center gap-1.5 bg-risk-high/10 text-risk-high px-2.5 py-0.5 rounded-full text-[11px] font-bold uppercase tracking-wider">
              <span className="w-2 h-2 rounded-full bg-risk-high animate-ping" />
              Cyclone Alert Active
            </span>
            <span 
              className="text-[11px] font-bold px-2.5 py-0.5 rounded-full border"
              style={{ color: currentBand.color, borderColor: `${currentBand.color}40`, backgroundColor: `${currentBand.color}10` }}
            >
              {classification?.category || "Unclassified System"}
            </span>
          </div>

          <h2 className="font-serif text-[22px] font-bold text-ink flex items-baseline gap-2">
            {data?.meta?.systemName || "Unnamed System"}
            <span className="text-sm font-sans text-ink-soft font-normal">({data?.meta?.systemId || "N/A"})</span>
          </h2>
          
          <p className="text-[11.5px] text-ink-soft flex items-center gap-1.5">
            Region: <span className="font-semibold text-ink">{data?.meta?.basin || "Bay of Bengal"}</span>
            <span className="text-ink-faint">·</span>
            Confidence: <span className="font-mono text-accent-strong font-semibold">{detection.confidence}% (P2 {data?.meta?.source?.includes('USER') ? 'user image' : 'reference frame'})</span>
          </p>
        </div>

        {/* Right Side: Quick-glance operational grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 lg:gap-8 bg-card-alt/50 border border-border/50 rounded-xl p-3.5 flex-1 lg:max-w-3xl w-full">
          
          <div>
            <span className="text-[9.5px] uppercase tracking-wider text-ink-faint font-semibold block">Coordinates</span>
            <span className="text-[14px] font-bold text-ink font-mono mt-0.5 block tabular-nums">
              {detection.location?.lat.toFixed(2)}°N {detection.location?.lon.toFixed(2)}°E
            </span>
          </div>

          <div>
            <span className="text-[9.5px] uppercase tracking-wider text-ink-faint font-semibold block">Max Sustained Wind</span>
            <span className="text-[14px] font-bold text-ink font-mono mt-0.5 block tabular-nums">
              {classification?.windSpeedKmh || "—"} km/h
            </span>
          </div>

          <div>
            <span className="text-[9.5px] uppercase tracking-wider text-ink-faint font-semibold block">Central Pressure</span>
            <span className="text-[14px] font-bold text-ink font-mono mt-0.5 block tabular-nums">
              {classification?.pressureHpa || "—"} hPa
            </span>
          </div>

          <div>
            <span className="text-[9.5px] uppercase tracking-wider text-ink-faint font-semibold block">Translation Speed</span>
            <span className="text-[14px] font-bold text-ink font-mono mt-0.5 block tabular-nums">
              {detection.movementSpeedKmh ? `${detection.movementSpeedKmh} km/h` : "Awaiting data"}
            </span>
            {detection.movementDirection && (
              <span className="text-[10px] text-ink-soft block font-sans truncate">{detection.movementDirection}</span>
            )}
          </div>

        </div>

      </div>
    </div>
  );
}
