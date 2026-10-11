import { useEffect, useState } from 'react'

async function api(path, options) {
  const response = await fetch(`/api${path}`, options)
  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = body.detail ? JSON.stringify(body.detail) : response.statusText
    throw new Error(detail)
  }
  return body
}

export default function App() {
  const [health, setHealth] = useState(null)
  const [sources, setSources] = useState([])
  const [files, setFiles] = useState([])
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState(null)
  const [status, setStatus] = useState('')
  const [busy, setBusy] = useState(false)

  function refresh() {
    api('/health').then(setHealth).catch(() => setHealth(null))
    api('/sources').then((body) => setSources(body.sources)).catch(() => setSources([]))
  }

  useEffect(() => {
    refresh()
  }, [])

  async function ingest(event) {
    event.preventDefault()
    if (!files.length) return
    const form = new FormData()
    for (const file of files) form.append('files', file)
    setBusy(true)
    setStatus('Indexing…')
    try {
      const body = await api('/ingest', { method: 'POST', body: form })
      const skipped = body.skipped.length ? `; skipped ${body.skipped.join(', ')}` : ''
      setStatus(`Indexed ${body.chunks} chunks from ${body.files.length} file(s)${skipped}`)
      refresh()
    } catch (error) {
      setStatus(`Error: ${error.message}`)
    } finally {
      setBusy(false)
    }
  }

  async function ask(event) {
    event.preventDefault()
    if (question.trim().length < 3) return
    setBusy(true)
    setResult(null)
    setStatus('Thinking…')
    try {
      const body = await api('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })
      setResult(body)
      setStatus('')
    } catch (error) {
      setStatus(`Error: ${error.message}`)
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="wrap">
      <header>
        <h1>docs-rag</h1>
        <p className="muted">
          {health ? `${health.chunks} chunks indexed · ${health.model}` : 'API offline'}
        </p>
      </header>

      <section className="card">
        <h2>1. Add documents</h2>
        <form onSubmit={ingest} className="row">
          <input
            type="file"
            multiple
            accept=".pdf,.md,.txt,.png,.jpg,.jpeg,.webp,.gif"
            onChange={(event) => setFiles([...event.target.files])}
          />
          <button disabled={busy || !files.length}>Index</button>
        </form>
        {sources.length > 0 && (
          <ul className="sources">
            {sources.map((s) => (
              <li key={s.source}>
                {s.source} <span className="muted">({s.chunks} chunks)</span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="card">
        <h2>2. Ask a question</h2>
        <form onSubmit={ask}>
          <textarea
            rows={3}
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="What does the document say about…"
          />
          <button disabled={busy || question.trim().length < 3}>Ask</button>
        </form>
      </section>

      {status && <p className="status">{status}</p>}

      {result && (
        <section className="card">
          <h2>Answer</h2>
          <p className="answer">{result.answer}</p>
          <h3>Sources</h3>
          <ol>
            {result.sources.map((s) => (
              <li key={s.n}>
                <strong>{s.source}</strong>
                {s.page ? ` · page ${s.page}` : ''}
                <span className="muted"> · score {s.score}</span>
                <div className="snippet">{s.snippet}</div>
              </li>
            ))}
          </ol>
          <p className="muted">
            {result.latency_ms} ms · {result.usage.input_tokens ?? '?'} in /{' '}
            {result.usage.output_tokens ?? '?'} out tokens · {result.model}
          </p>
        </section>
      )}
    </main>
  )
}
