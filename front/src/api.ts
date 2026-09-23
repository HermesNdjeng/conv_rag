// Client for the FastAPI backend. Base URL is injected at build time via VITE_API_URL.
const API_URL = import.meta.env.VITE_API_URL

/**
 * Send one turn and stream the agent's answer token by token (SSE).
 *
 * Yields each text token as it arrives. Throws if the request fails or the server emits an
 * error event mid-stream. Pass an AbortSignal to cancel the stream (e.g. on unmount).
 */
export async function* streamMessage(
  userId: string,
  threadId: string,
  message: string,
  signal?: AbortSignal,
): AsyncGenerator<string> {
  const url = `${API_URL}/users/${encodeURIComponent(userId)}/sessions/${encodeURIComponent(threadId)}/messages`
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({ message }),
    signal,
  })
  if (!response.ok || !response.body) {
    throw new Error(`Request failed (${response.status})`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    // SSE frames are separated by a blank line; keep the trailing partial frame in the buffer.
    const frames = buffer.split('\n\n')
    buffer = frames.pop() ?? ''
    for (const frame of frames) {
      const line = frame.trim()
      if (!line.startsWith('data:')) continue
      const payload = JSON.parse(line.slice(5).trim())
      if (payload.error) throw new Error(payload.error)
      if (payload.token) yield payload.token as string
    }
  }
}

/** End a session: consolidate the whole conversation into episodic memory. */
export async function endSession(userId: string, threadId: string): Promise<void> {
  const url = `${API_URL}/users/${encodeURIComponent(userId)}/sessions/${encodeURIComponent(threadId)}`
  const response = await fetch(url, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ status: 'ended' }),
  })
  if (!response.ok) {
    throw new Error(`End session failed (${response.status})`)
  }
}
