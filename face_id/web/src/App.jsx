import { useState } from 'react'
import VerifyPanel from './VerifyPanel.jsx'
import EnrollPanel from './EnrollPanel.jsx'

export default function App() {
  const [mode, setMode] = useState('verify')

  return (
    <div style={{ fontFamily: 'sans-serif', padding: 20 }}>
      <h1>Face ID — Test Console</h1>

      <div style={{ marginBottom: 20 }}>
        <button
          onClick={() => setMode('verify')}
          disabled={mode === 'verify'}
          style={{ marginRight: 8 }}
        >
          Verify
        </button>
        <button onClick={() => setMode('enroll')} disabled={mode === 'enroll'}>
          Enroll
        </button>
      </div>

      {mode === 'verify' ? <VerifyPanel /> : <EnrollPanel />}
    </div>
  )
}
