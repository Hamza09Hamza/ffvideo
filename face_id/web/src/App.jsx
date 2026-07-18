import { useState } from 'react'
import VerifyPanel from './VerifyPanel.jsx'
import EnrollPanel from './EnrollPanel.jsx'
import './styles.css'

export default function App() {
  const [mode, setMode] = useState('verify')
  const [serverUrl, setServerUrl] = useState('https://ibsai.tailff4c89.ts.net:8000')
  const [apiKey, setApiKey] = useState('')

  return (
    <div className="app">
      <h1>Face ID — Test Console</h1>

      <div className="config-row">
        <input placeholder="Server URL" value={serverUrl} onChange={(e) => setServerUrl(e.target.value)} />
        <input
          placeholder="API key"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          type="password"
        />
      </div>

      <div className="tabs">
        <button className={mode === 'verify' ? 'active' : ''} onClick={() => setMode('verify')}>
          Verify
        </button>
        <button className={mode === 'enroll' ? 'active' : ''} onClick={() => setMode('enroll')}>
          Enroll
        </button>
      </div>

      {mode === 'verify' ? (
        <VerifyPanel serverUrl={serverUrl} apiKey={apiKey} />
      ) : (
        <EnrollPanel serverUrl={serverUrl} apiKey={apiKey} />
      )}
    </div>
  )
}
