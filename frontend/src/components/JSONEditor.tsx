import { useEffect, useState } from 'react'
import { ZodError } from 'zod'

export default function JSONEditor({
  value,
  onChange,
  placeholder = '{\n  "hdim_proj_dim": 32,\n  "route_pool": "cls",\n  "lr_encoder": 1.5e-5\n}',
}: {
  value: any
  onChange: (v: any) => void
  placeholder?: string
}) {
  const [text, setText] = useState<string>(JSON.stringify(value ?? {}, null, 2))
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    setText(JSON.stringify(value ?? {}, null, 2))
  }, [value])

  const parse = (s: string) => {
    try {
      const v = s.trim() ? JSON.parse(s) : {}
      setErr(null)
      onChange(v)
    } catch (e: any) {
      setErr(e.message)
    }
  }

  return (
    <div>
      <textarea
        className="w-full h-40 font-mono text-sm bg-zinc-950 border border-zinc-800 rounded-md p-3"
        spellCheck={false}
        value={text}
        onChange={(e) => {
          setText(e.target.value)
          parse(e.target.value)
        }}
        placeholder={placeholder}
      />
      {err && <p className="text-xs text-red-400 mt-1">JSON error: {err}</p>}
    </div>
  )
}
