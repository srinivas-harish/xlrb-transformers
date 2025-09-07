import type { RunDetail } from '../lib/types'

export default function EpochsTable({ data }: { data: RunDetail }) {
  const rows = data.epochs_rows || []
  const bestEpoch = rows.reduce<{epoch:number,val:number}|null>((best, r) => {
    const v = typeof r.val_acc === 'number' ? r.val_acc : -Infinity
    if (!best || v > best.val) return { epoch: r.epoch, val: v }
    return best
  }, null)

  return (
    <div className="rounded-lg bg-zinc-900 border border-zinc-800 overflow-hidden">
      <table className="w-full text-sm">
        <thead className="bg-zinc-800 text-zinc-300">
          <tr>
            <th className="text-left px-3 py-2">Epoch</th>
            <th className="text-left px-3 py-2">Val acc</th>
            <th className="text-left px-3 py-2">Train acc</th>
            <th className="text-left px-3 py-2">F1 macro</th>
            <th className="text-left px-3 py-2">LR</th>
            <th className="text-left px-3 py-2">Time (s)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((e) => (
            <tr key={e.epoch} className="border-t border-zinc-800 hover:bg-zinc-800/50">
              <td className="px-3 py-2 font-mono text-xs">{e.epoch}</td>
              <td className={"px-3 py-2 " + (bestEpoch?.epoch === e.epoch ? 'font-semibold text-emerald-400' : '')}>
                {e.val_acc?.toFixed?.(4) ?? '—'}
              </td>
              <td className="px-3 py-2">{e.train_acc?.toFixed?.(4) ?? '—'}</td>
              <td className="px-3 py-2">{e.val_f1_macro?.toFixed?.(4) ?? '—'}</td>
              <td className="px-3 py-2">{e.lr?.toFixed?.(6) ?? '—'}</td>
              <td className="px-3 py-2">{e.time_sec?.toFixed?.(1) ?? '—'}</td>
            </tr>
          ))}
          {rows.length === 0 && (
            <tr><td colSpan={6} className="px-3 py-6 text-center text-zinc-400">No epochs logged yet.</td></tr>
          )}
        </tbody>
      </table>
    </div>
  )
}

