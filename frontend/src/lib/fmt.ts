import { formatDistanceToNowStrict, parseISO } from 'date-fns'

export const relTime = (iso: string) => {
  try { return formatDistanceToNowStrict(parseISO(iso), { addSuffix: true }) }
  catch { return iso }
}

export const pct = (x: number, digits = 0) => `${(x * 100).toFixed(digits)}%`
