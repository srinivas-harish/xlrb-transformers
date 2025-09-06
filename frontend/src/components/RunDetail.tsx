import type { RunDetail as RunDetailT } from '../lib/types'
import MetricCard from './MetricCard'
import ChartLine from './ChartLine'

export default function RunDetail({ data }: { data: RunDetailT }) {
  const cfg = data.result?.cfg
  const epochs = data.result?.epochs || []

  return (
    <div className="space-y-6">
      <div className="grid md:grid-cols-4 gap-3">
        <MetricCard label="Status" value={data.status} />
        <MetricCard label="Best val_acc" value={data.result?.best?.val_acc?.toFixed?.(4) ?? '—'}
          sub={data.result?.best ? `epoch ${data.result.best.epoch}` : ''} />
        <MetricCard label="Device" value={data.device ?? 'auto'} />
        <MetricCard label="Ablation" value={data.ablation ?? '(custom)'} />
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        <ChartLine
          data={(data.epochs_rows || []).map(e => ({ epoch: e.epoch, val_acc: e.val_acc, train_acc: e.train_acc }))}
          xKey="epoch" yKey="val_acc" y2Key="train_acc" yLabel="accuracy"
        />
        <ChartLine
          data={epochs.map(e => ({ epoch: e.epoch, f1_macro: e.val_f1_macro }))}
          xKey="epoch" yKey="f1_macro" yLabel="F1 (macro)"
        />
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
          <div className="text-sm font-semibold mb-2">Overrides</div>
          <pre className="text-xs overflow-auto bg-zinc-950 p-3 rounded border border-zinc-800">
            {JSON.stringify(data.overrides ?? {}, null, 2)}
          </pre>
        </div>
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
          <div className="text-sm font-semibold mb-2">BridgeCfg (effective)</div>
          <pre className="text-xs overflow-auto bg-zinc-950 p-3 rounded border border-zinc-800">
            {JSON.stringify(cfg ?? {}, null, 2)}
          </pre>
        </div>
      </div>

      <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
        <div className="text-sm font-semibold mb-2">Artifacts</div>
        {data.artifacts && data.artifacts.length > 0 ? (
          <ul className="text-sm list-disc pl-6">
            {data.artifacts.map(a => (
              <li key={a.id}>
                <span className="font-mono">{a.path}</span>
                {typeof a.size === 'number' && <span className="text-zinc-400 ml-2">({(a.size/1024).toFixed(1)} KB)</span>}
              </li>
            ))}
          </ul>
        ) : (
          <div className="text-sm text-zinc-400">No artifacts recorded.</div>
        )}
      </div>

      {data.error && (
        <div className="bg-red-950 border border-red-800 rounded-lg p-4">
          <div className="text-sm font-semibold mb-2">Error</div>
          <pre className="text-xs overflow-auto">{data.error}</pre>
        </div>
      )}
    </div>
  )
}
