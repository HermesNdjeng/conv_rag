import { useEffect, useRef, useState } from 'react'
import { streamMessage, endSession } from './api'
import './App.css'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

// TODO(auth): userId is a dev placeholder — a per-browser random UUID the backend trusts blindly
// (IDOR risk). With auth it must come from the authenticated identity (token), not the front.
// threadId can stay client-managed, but is better minted by a backend POST /sessions endpoint.
/** Read an id from localStorage, creating and persisting one on first visit. */
function loadOrCreateId(key: string): string {
  let value = localStorage.getItem(key)
  if (!value) {
    value = crypto.randomUUID()
    localStorage.setItem(key, value)
  }
  return value
}

function App() {
  const [userId] = useState(() => loadOrCreateId('userId'))
  const [threadId, setThreadId] = useState(() => loadOrCreateId('threadId'))
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const messagesRef = useRef<HTMLDivElement>(null)

  // Keep the latest message in view as bubbles are added and tokens stream in.
  useEffect(() => {
    const el = messagesRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages])

  // Replace the last message (the in-progress assistant bubble) with an updated copy.
  function updateLast(update: (msg: Message) => Message) {
    setMessages((prev) => prev.map((msg, i) => (i === prev.length - 1 ? update(msg) : msg)))
  }

  async function handleSend() {
    const text = input.trim()
    if (!text || busy) return
    setInput('')
    setBusy(true)
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: text },
      { role: 'assistant', content: '' },
    ])

    try {
      for await (const token of streamMessage(userId, threadId, text)) {
        updateLast((msg) => ({ ...msg, content: msg.content + token }))
      }
    } catch (err) {
      updateLast(() => ({ role: 'assistant', content: `⚠️ Erreur : ${String(err)}` }))
    } finally {
      setBusy(false)
    }
  }

  async function handleNewConversation() {
    if (busy) return
    try {
      await endSession(userId, threadId)
    } catch {
      // Consolidation is best-effort; starting fresh must not be blocked by its failure.
    }
    const next = crypto.randomUUID()
    localStorage.setItem('threadId', next)
    setThreadId(next)
    setMessages([])
  }

  return (
    <div className="chat">
      <header className="chat__header">
        <h1>RAG — Histoire du Cameroun</h1>
        <button onClick={handleNewConversation} disabled={busy}>
          Nouvelle conversation
        </button>
      </header>

      <div className="chat__messages" ref={messagesRef}>
        {messages.map((msg, i) => (
          <div key={i} className={`msg msg--${msg.role}`}>
            {msg.content || '…'}
          </div>
        ))}
      </div>

      <form
        className="chat__composer"
        onSubmit={(e) => {
          e.preventDefault()
          void handleSend()
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Posez votre question sur l'histoire du Cameroun…"
          disabled={busy}
        />
        <button type="submit" disabled={busy || !input.trim()}>
          Envoyer
        </button>
      </form>
    </div>
  )
}

export default App
