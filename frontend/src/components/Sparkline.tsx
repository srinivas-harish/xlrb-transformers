import { LineChart, Line, ResponsiveContainer } from 'recharts'

export default function Sparkline({ data, dataKey = 'y', stroke = '#10b981', height = 36 }: {
  data: Array<Record<string, number | null | undefined>>
  dataKey?: string
  stroke?: string
  height?: number
}) {
  return (
    <div className="h-9 w-full">
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 4, bottom: 0, left: 0, right: 0 }}>
          <Line type="monotone" dataKey={dataKey} stroke={stroke} dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )}

