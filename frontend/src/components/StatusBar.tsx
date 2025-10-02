import { AlertTriangle, CheckCircle2, WifiOff } from 'lucide-react'
import clsx from 'clsx'

interface StatusBarProps {
  status: 'idle' | 'connecting' | 'success' | 'error'
  message?: string
}

export function StatusBar({ status, message }: StatusBarProps) {
  const iconMap = {
    idle: <CheckCircle2 size={16} />,
    connecting: <WifiOff size={16} />,
    success: <CheckCircle2 size={16} />,
    error: <AlertTriangle size={16} />,
  } as const

  const labelMap = {
    idle: 'Waiting for your prompt',
    connecting: 'Connecting to SatGraffin backend…',
    success: 'Response received',
    error: message ?? 'Something went wrong',
  } as const

  return (
    <div className={clsx('status-bar', `status-bar--${status}`)}>
      {iconMap[status]}
      <span>{labelMap[status]}</span>
    </div>
  )
}
