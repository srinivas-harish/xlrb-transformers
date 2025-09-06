import { useEffect, useState } from 'react'
import { listRuns } from '../lib/api'
import type { RunRow } from '../lib/types'
import { relTime } from '../lib/fmt'
import { clsx } from 'clsx'

export default function RunList({ onOpen }: { onOpen: (id: string) => void }) {
  const [rows, setRows] = useState<RunRow[]>([])
  useEffect(() => {
    let tm: any
    const poll = async () => {
      try { setRows(await listRuns(undefined, 100)) } catch {}
      tm = setTimeout(poll, 1500)
    }
    poll()
    return () => tm && clearTimeout(tm)
  }, [])

  return (
    <div className="rounded-lg bg-zinc-900 border border-zinc-800 overflow-hidden">
      <table className="w-full text-sm">
        <thead className="bg-zinc-800 text-zinc-300">
          <tr>
            <th className="text-left px-3 py-2">ID</th>
            <th className="text-left px-3 py-2">Status</th>
            <th className="text-left px-3 py-2">Ablation</th>
            <th className="text-left px-3 py-2">Epochs</th>
            <th className="text-left px-3 py-2">Batch</th>
            <th className="text-left px-3 py-2">Best acc</th>
            <th className="text-left px-3 py-2">Created</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {rows.map(r => (
            <tr key={r.id} className="border-t border-zinc-800 hover:bg-zinc-800/50">
              <td className="px-3 py-2 font-mono text-xs">{r.id.slice(0,8)}…</td>
              <td className="px-3 py-2">
                <span className={clsx(
                  'px-2 py-1 rounded text-xs font-bold',
                  r.status === 'COMPLETE' ? 'bg-emerald-700' :
                  r.status === 'FAILED' ? 'bg-red-700' :
                  r.status === 'STARTED' ? 'bg-amber-700' : 'bg-zinc-700'
                )}>{r.status}</span>
              </td>
              <td className="px-3 py-2">{r.ablation ?? '—'}</td>
              <td className="px-3 py-2">{r.epochs}</td>
              <td className="px-3 py-2">{r.batch_size}</td>
              <td className="px-3 py-2">{r.best_val_acc?.toFixed?.(4) ?? '—'}</td>
              <td className="px-3 py-2">{relTime(r.created_at)}</td>
              <td className="px-3 py-2 text-right">
                <button className="px-2 py-1 rounded bg-zinc-800 hover:bg-zinc-700" onClick={() => onOpen(r.id)}>
                  Open
                </button>
              </td>
            </tr>
          ))}
          {rows.length === 0 && (
            <tr><td colSpan={8} className="px-3 py-6 text-center text-zinc-400">No runs yet.</td></tr>
          )}
        </tbody>
      </table>
    </div>
  )
}
