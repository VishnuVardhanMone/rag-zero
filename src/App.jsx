import { useState } from 'react'
import * as pdfjsLib from 'pdfjs-dist'
import axios from 'axios'
import './App.css'

pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url
).toString()

// --- TF-IDF + Cosine Similarity RAG Engine ---

function tokenize(text) {
  return text.toLowerCase().replace(/[^a-z0-9\s]/g, '').split(/\s+/).filter(Boolean)
}

function chunkText(text, size = 300) {
  const words = text.split(/\s+/)
  const chunks = []
  for (let i = 0; i < words.length; i += size) {
    chunks.push(words.slice(i, i + size).join(' '))
  }
  return chunks
}

function computeTFIDF(chunks) {
  const tf = chunks.map(chunk => {
    const tokens = tokenize(chunk)
    const freq = {}
    tokens.forEach(t => freq[t] = (freq[t] || 0) + 1)
    Object.keys(freq).forEach(k => freq[k] /= tokens.length)
    return freq
  })

  const df = {}
  tf.forEach(freq => Object.keys(freq).forEach(t => df[t] = (df[t] || 0) + 1))

  const N = chunks.length
  return tf.map(freq => {
    const tfidf = {}
    Object.keys(freq).forEach(t => {
      tfidf[t] = freq[t] * Math.log(N / (df[t] || 1))
    })
    return tfidf
  })
}

function cosineSimilarity(vecA, vecB) {
  const keys = new Set([...Object.keys(vecA), ...Object.keys(vecB)])
  let dot = 0, magA = 0, magB = 0
  keys.forEach(k => {
    const a = vecA[k] || 0
    const b = vecB[k] || 0
    dot += a * b
    magA += a * a
    magB += b * b
  })
  return dot / (Math.sqrt(magA) * Math.sqrt(magB) || 1)
}

function retrieveTopChunks(query, chunks, tfidfVecs, topK = 3) {
  const qTokens = tokenize(query)
  const qFreq = {}
  qTokens.forEach(t => qFreq[t] = (qFreq[t] || 0) + 1 / qTokens.length)

  const scores = tfidfVecs.map((vec, i) => ({
    index: i,
    score: cosineSimilarity(qFreq, vec)
  }))

  scores.sort((a, b) => b.score - a.score)
  return scores.slice(0, topK).map(s => chunks[s.index])
}

// --- Main App ---

export default function App() {
  const [inputText, setInputText] = useState('')
  const [chunks, setChunks] = useState([])
  const [tfidfVecs, setTfidfVecs] = useState([])
  const [indexed, setIndexed] = useState(false)
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [docName, setDocName] = useState('')

  function indexDocument(text, name) {
    const c = chunkText(text)
    const vecs = computeTFIDF(c)
    setChunks(c)
    setTfidfVecs(vecs)
    setIndexed(true)
    setDocName(name)
    setMessages([])
  }

  function handleTextIndex() {
    if (!inputText.trim()) return
    indexDocument(inputText, 'Pasted Text')
  }

  async function handlePDF(e) {
    const file = e.target.files[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = async (ev) => {
      const pdf = await pdfjsLib.getDocument({ data: ev.target.result }).promise
      let fullText = ''
      for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i)
        const content = await page.getTextContent()
        fullText += content.items.map(item => item.str).join(' ') + ' '
      }
      indexDocument(fullText, file.name)
    }
    reader.readAsArrayBuffer(file)
  }

  async function handleAsk() {
    if (!query.trim() || !indexed) return
    const userMsg = { role: 'user', content: query }
    const newMessages = [...messages, userMsg]
    setMessages(newMessages)
    setQuery('')
    setLoading(true)

    const topChunks = retrieveTopChunks(query, chunks, tfidfVecs)
    const context = topChunks.join('\n\n---\n\n')

    const systemPrompt = `You are a helpful assistant. Answer the user's question using ONLY the context below. If the answer isn't in the context, say "I couldn't find that in the document."

Context:
${context}`

    try {
      const res = await axios.post(
        'http://localhost:3001/api/chat',
        {
          model: 'claude-haiku-4-5-20251001',
          max_tokens: 1024,
          system: systemPrompt,
          messages: newMessages.map(m => ({ role: m.role, content: m.content }))
        }
      )
      const reply = res.data.content[0].text
      setMessages([...newMessages, { role: 'assistant', content: reply }])
    } catch (err) {
      setMessages([...newMessages, { role: 'assistant', content: 'Error: ' + (err.response?.data?.error?.message || err.message) }])
    }
    setLoading(false)
  }

  return (
    <div className="app">
      <header className="header">
        <h1>RAG<span>.zero</span></h1>
        <p>Chat with any document — runs entirely in your browser</p>
      </header>

      {!indexed ? (
        <div className="upload-section">
          <div className="upload-card">
            <h2>Upload a PDF</h2>
            <label className="pdf-btn">
              Choose PDF
              <input type="file" accept=".pdf" onChange={handlePDF} hidden />
            </label>
          </div>
          <div className="divider">or</div>
          <div className="upload-card">
            <h2>Paste Text</h2>
            <textarea
              placeholder="Paste any text here..."
              value={inputText}
              onChange={e => setInputText(e.target.value)}
              rows={8}
            />
            <button onClick={handleTextIndex} disabled={!inputText.trim()}>
              Index Text
            </button>
          </div>
        </div>
      ) : (
        <div className="chat-section">
          <div className="doc-badge">
            <span>📄 {docName}</span>
            <button onClick={() => { setIndexed(false); setMessages([]); setInputText('') }}>
              ✕ New Document
            </button>
          </div>
          <div className="messages">
            {messages.length === 0 && (
              <div className="empty-state">Document indexed! Ask anything about it 👇</div>
            )}
            {messages.map((m, i) => (
              <div key={i} className={`message ${m.role}`}>
                <div className="bubble">{m.content}</div>
              </div>
            ))}
            {loading && (
              <div className="message assistant">
                <div className="bubble typing">Thinking...</div>
              </div>
            )}
          </div>
          <div className="input-row">
            <input
              type="text"
              placeholder="Ask a question about your document..."
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleAsk()}
            />
            <button onClick={handleAsk} disabled={loading || !query.trim()}>
              Ask
            </button>
          </div>
        </div>
      )}
    </div>
  )
}