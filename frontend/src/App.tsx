import { useEffect, useState } from 'react'
import RunForm from './components/RunForm'
import RunList from './components/RunList'
import RunDetail from './components/RunDetail'
import RunProgress from './components/RunProgress'
import { getRun } from './lib/api'
import type { RunDetail as RunDetailT } from './lib/types'
import { clsx } from 'clsx'

type Tab = 'form' | 'runs' | 'detail'

export default function App() {
  const [tab, setTab] = useState<Tab>('form')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [selected, setSelected] = useState<RunDetailT | null>(null)

  useEffect(() => {
    const id = localStorage.getItem('last_run_id')
    if (id) {
      setSelectedId(id)
      setTab('detail')
    }
  }, [])

  useEffect(() => {
    let tm: any
    if (tab === 'detail' && selectedId) {
      const poll = async () => {
        try {
          const d = await getRun(selectedId)
          setSelected(d)
        } catch {}
        tm = setTimeout(poll, 1000)
      }
      poll()
    }
    return () => tm && clearTimeout(tm)
  }, [tab, selectedId])

  return (
    <div className="max-w-7xl mx-auto p-6">
      <header className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-extrabold tracking-tight">
          Ablation Dashboard
          <span className="ml-2 text-zinc-400">• RoBERTa RTE</span>
        </h1>
        <nav className="flex gap-2">
          {(['form','runs','detail'] as Tab[]).map(t => (
            <button
              key={t}
              className={clsx(
                'px-3 py-1 rounded-md text-sm font-semibold',
                tab===t ? 'bg-zinc-800' : 'bg-zinc-900 hover:bg-zinc-800'
              )}
              onClick={() => setTab(t)}
            >
              {t === 'form' ? 'New Run' : t === 'runs' ? 'Runs' : 'Details'}
            </button>
          ))}
        </nav>
      </header>

      {tab === 'form' && (
        <RunForm
          onSubmitted={(runId) => {
            localStorage.setItem('last_run_id', runId)
            setSelectedId(runId)
            setTab('detail')
          }}
        />
      )}

      {tab === 'runs' && (
        <RunList
          onOpen={(id) => {
            localStorage.setItem('last_run_id', id)
            setSelectedId(id)
            setTab('detail')
          }}
        />
      )}

      {tab === 'detail' && selectedId && (
        <>
          <RunProgress runId={selectedId} run={selected || undefined} />
          <div className="mt-6">
            {selected ? (
              <RunDetail data={selected} />
            ) : (
              <div className="text-sm text-zinc-400">Loading run {selectedId}…</div>
            )}
          </div>
        </>
      )}
    </div>
  )
}
