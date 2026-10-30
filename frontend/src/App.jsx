import { useCallback, useEffect, useRef, useState } from 'react'
import {
  getCurrentUser,
  loginUser,
  registerUser,
  uploadDocument,
  sendChatMessage,
  getDocuments,
  deleteDocument,
  getConversations,
  getConversationMessages,
  getDocumentChunk,
} from './api'
import './App.css'

const TOKEN_KEY = 'enterprise_ai_access_token'

function App() {
  const [token, setToken] = useState(() =>
    sessionStorage.getItem(TOKEN_KEY),
  )
  const [user, setUser] = useState(null)
  const [authMode, setAuthMode] = useState('login')
  const [activeView, setActiveView] = useState('chat')

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [checkingSession, setCheckingSession] = useState(Boolean(token))
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  const [selectedFile, setSelectedFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState('')
  const [uploadSuccess, setUploadSuccess] = useState('')
  const [uploadedDocuments, setUploadedDocuments] = useState([])
  const [documentsLoading, setDocumentsLoading] = useState(false)
  const [documentsError, setDocumentsError] = useState('')
  const fileInputRef = useRef(null)

  const [chatQuestion, setChatQuestion] = useState('')
  const [chatMessages, setChatMessages] = useState([])
  const [chatLoading, setChatLoading] = useState(false)
  const [chatError, setChatError] = useState('')
  const [conversations, setConversations] = useState([])
  const [selectedConversationId, setSelectedConversationId] = useState(null)
  const [conversationsLoading, setConversationsLoading] = useState(false)
  const [conversationsError, setConversationsError] = useState('')
  const [historyLoading, setHistoryLoading] = useState(false)

  const [expandedSources, setExpandedSources] = useState({})
  const [sourceLoading, setSourceLoading] = useState({})
  const [sourceErrors, setSourceErrors] = useState({})
  const [sourceContent, setSourceContent] = useState({})

  useEffect(() => {
    let cancelled = false

    async function restoreSession() {
      if (!token) {
        setCheckingSession(false)
        return
      }

      try {
        const currentUser = await getCurrentUser(token)
        if (!cancelled) setUser(currentUser)
      } catch {
        sessionStorage.removeItem(TOKEN_KEY)
        if (!cancelled) {
          setToken(null)
          setUser(null)
        }
      } finally {
        if (!cancelled) setCheckingSession(false)
      }
    }

    void restoreSession()

    return () => {
      cancelled = true
    }
  }, [token])

  const loadDocuments = useCallback(async () => {
    if (!token) return

    setDocumentsLoading(true)
    setDocumentsError('')

    try {
      const result = await getDocuments(token)
      setUploadedDocuments(result.documents ?? [])
    } catch (err) {
      setDocumentsError(
        err.message || 'Unable to load your documents. Please try again.',
      )
    } finally {
      setDocumentsLoading(false)
    }
  }, [token])

  const loadConversations = useCallback(async () => {
    if (!token) return

    setConversationsLoading(true)
    setConversationsError('')

    try {
      const result = await getConversations(token)
      setConversations(result)
    } catch (err) {
      setConversationsError(
        err.message || 'Unable to load conversation history.',
      )
    } finally {
      setConversationsLoading(false)
    }
  }, [token])

  useEffect(() => {
    if (!token || !user || activeView !== 'documents') return

    const timeoutId = setTimeout(() => {
      void loadDocuments()
    }, 0)

    return () => clearTimeout(timeoutId)
  }, [token, user, activeView, loadDocuments])

  useEffect(() => {
    if (!token || !user) return

    const timeoutId = setTimeout(() => {
      void loadConversations()
    }, 0)

    return () => clearTimeout(timeoutId)
  }, [token, user, loadConversations])

  async function handleAuthSubmit(event) {
    event.preventDefault()
    setError('')
    setNotice('')

    const normalizedEmail = email.trim().toLowerCase()

    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(normalizedEmail)) {
      setError('Enter a valid email address.')
      return
    }

    if (
      authMode === 'register' &&
      (password.length < 8 || password.length > 128)
    ) {
      setError('Password must contain 8-128 characters.')
      return
    }

    if (password.length === 0 || password.length > 128) {
      setError('Enter a password of up to 128 characters.')
      return
    }

    setBusy(true)

    try {
      if (authMode === 'register') {
        await registerUser(normalizedEmail, password)
      }

      const result = await loginUser(normalizedEmail, password)

      if (!result.access_token) {
        throw new Error('The server did not return an access token.')
      }

      sessionStorage.setItem(TOKEN_KEY, result.access_token)
      setToken(result.access_token)
      setPassword('')
      setNotice('')
      setActiveView('chat')
    } catch (err) {
      setError(
        authMode === 'register' && err.status === 409
          ? 'This email is already registered. Switch to Login.'
          : err.message || 'Authentication failed. Please try again.',
      )
    } finally {
      setBusy(false)
    }
  }

  async function handleDocumentUpload(event) {
    event.preventDefault()
    if (!selectedFile || uploading) return

    setUploading(true)
    setUploadError('')
    setUploadSuccess('')

    try {
      const filename = selectedFile.name
      await uploadDocument(token, selectedFile)
      await loadDocuments()
      setUploadSuccess(`${filename} uploaded successfully.`)
      setSelectedFile(null)

      if (fileInputRef.current) fileInputRef.current.value = ''
    } catch (err) {
      setUploadError(
        err.message || 'Document upload failed. Please try again.',
      )
    } finally {
      setUploading(false)
    }
  }

  async function handleDocumentDelete(document) {
    const confirmed = window.confirm(
      `Are you sure you want to delete "${document.filename}"? This action cannot be undone.`,
    )

    if (!confirmed) return

    setDocumentsError('')
    setUploadError('')
    setUploadSuccess('')

    try {
      await deleteDocument(token, document.id)
      setUploadSuccess(`${document.filename} deleted successfully.`)
      await loadDocuments()
    } catch (err) {
      setDocumentsError(
        err.message || 'Unable to delete the document. Please try again.',
      )
    }
  }

  function handleNewChat() {
    setSelectedConversationId(null)
    setChatMessages([])
    setChatQuestion('')
    setChatError('')
    setActiveView('chat')
  }

  async function handleSelectConversation(conversationId) {
    setSelectedConversationId(conversationId)
    setChatMessages([])
    setChatError('')
    setHistoryLoading(true)
    setActiveView('chat')

    try {
      const result = await getConversationMessages(token, conversationId)
      setChatMessages(result.messages ?? [])
    } catch (err) {
      setChatError(err.message || 'Unable to load this conversation.')
    } finally {
      setHistoryLoading(false)
    }
  }
  async function handleSourceClick(messageId, source) {
  const sourceKey = `${messageId}-${source.document_id}-${source.chunk_id}`

  // Close source if it is already expanded.
  if (expandedSources[sourceKey]) {
    setExpandedSources((previous) => ({
      ...previous,
      [sourceKey]: false,
    }))
    return
  }

  // Expand source.
  setExpandedSources((previous) => ({
    ...previous,
    [sourceKey]: true,
  }))

  // Use cached content if already fetched.
  if (sourceContent[sourceKey]) return

  setSourceLoading((previous) => ({
    ...previous,
    [sourceKey]: true,
  }))

  setSourceErrors((previous) => ({
    ...previous,
    [sourceKey]: '',
  }))

  try {
    const result = await getDocumentChunk(
      token,
      source.document_id,
      source.chunk_id,
    )

    setSourceContent((previous) => ({
      ...previous,
      [sourceKey]: result.content,
    }))
  } catch (err) {
    setSourceErrors((previous) => ({
      ...previous,
      [sourceKey]: err.message || 'Unable to load source text.',
    }))
  } finally {
    setSourceLoading((previous) => ({
      ...previous,
      [sourceKey]: false,
    }))
  }
}

  async function handleChatSubmit(event) {
    event.preventDefault()

    const question = chatQuestion.trim()
    if (!question || chatLoading) return

    setChatLoading(true)
    setChatError('')

    // Show the user's question immediately while waiting for the AI.
    setChatMessages((previous) => [
      ...previous,
      {
        id: `pending-${Date.now()}`,
        role: 'user',
        content: question,
      },
    ])
    setChatQuestion('')

    try {
      const result = await sendChatMessage(
        token,
        question,
        5,
        selectedConversationId,
      )

      setSelectedConversationId(result.conversation_id)

      // Reload persisted messages so the UI reflects PostgreSQL state.
      const history = await getConversationMessages(
        token,
        result.conversation_id,
      )
      setChatMessages(history.messages ?? [])
      await loadConversations()
    } catch (err) {
      setChatError(err.message || 'Unable to get an answer. Please try again.')
      // Remove the temporary question if the request failed.
      setChatMessages((previous) =>
        previous.filter(
          (message) =>
            !(typeof message.id === 'string' &&
              message.id.startsWith('pending-')),
        ),
      )
      setChatQuestion(question)
    } finally {
      setChatLoading(false)
    }
  }

  function handleLogout() {
    sessionStorage.removeItem(TOKEN_KEY)
    setToken(null)
    setUser(null)
    setEmail('')
    setPassword('')
    setError('')
    setNotice('')
    setAuthMode('login')
    setActiveView('chat')
    setUploadedDocuments([])
    setChatMessages([])
    setChatQuestion('')
    setConversations([])
    setSelectedConversationId(null)
    setCheckingSession(false)
  }

  if (checkingSession) {
    return (
      <div className="auth-loading">
        <div className="brand-mark">E</div>
        <p>Checking your session...</p>
      </div>
    )
  }

  if (!token || !user) {
    return (
      <main className="auth-page">
        <section className="auth-brand-panel">
          <div className="brand">
            <div className="brand-mark">E</div>
            <div>
              <h2>Enterprise AI</h2>
              <p>Knowledge &amp; Data Agent</p>
            </div>
          </div>

          <div className="auth-pitch">
            <div className="welcome-icon">✳</div>
            <p className="eyebrow">YOUR PRIVATE AI WORKSPACE</p>
            <h1>Knowledge that works for you.</h1>
            <p>
              Search your documents, ask questions, and get answers grounded
              in your own knowledge base.
            </p>
            <div className="auth-benefit">✓ Private, user-isolated documents</div>
            <div className="auth-benefit">✓ Local LLM answer generation</div>
            <div className="auth-benefit">✓ Answers with source references</div>
          </div>

          <p className="auth-footnote">
            Enterprise AI Knowledge &amp; Data Agent
          </p>
        </section>

        <section className="auth-form-panel">
          <form className="auth-form" onSubmit={handleAuthSubmit}>
            <p className="eyebrow">
              {authMode === 'login' ? 'WELCOME BACK' : 'GET STARTED'}
            </p>
            <h2>
              {authMode === 'login'
                ? 'Sign in to your workspace'
                : 'Create your account'}
            </h2>
            <p className="auth-subtitle">
              {authMode === 'login'
                ? 'Enter your details to continue.'
                : 'Register to create your private workspace.'}
            </p>

            <label htmlFor="email">Email address</label>
            <input
              id="email"
              type="email"
              autoComplete="email"
              placeholder="you@example.com"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              required
              maxLength={254}
              disabled={busy}
            />

            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              autoComplete={
                authMode === 'login' ? 'current-password' : 'new-password'
              }
              placeholder={
                authMode === 'register'
                  ? 'At least 8 characters'
                  : 'Enter your password'
              }
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              required
              minLength={authMode === 'register' ? 8 : 1}
              maxLength={128}
              disabled={busy}
            />

            {error && (
              <div className="form-message error-message" role="alert">
                {error}
              </div>
            )}
            {notice && (
              <div className="form-message success-message" role="status">
                {notice}
              </div>
            )}

            <button
              className="primary-button auth-submit"
              disabled={busy}
              type="submit"
            >
              {busy
                ? 'Please wait...'
                : authMode === 'login'
                  ? 'Sign in'
                  : 'Create account'}
            </button>

            <p className="auth-switch">
              {authMode === 'login'
                ? "Don't have an account?"
                : 'Already have an account?'}{' '}
              <button
                type="button"
                className="text-button"
                disabled={busy}
                onClick={() => {
                  setAuthMode(authMode === 'login' ? 'register' : 'login')
                  setError('')
                  setNotice('')
                }}
              >
                {authMode === 'login' ? 'Create account' : 'Sign in'}
              </button>
            </p>

            <p className="auth-security-note">
              Your documents are isolated by authenticated user.
            </p>
          </form>
        </section>
      </main>
    )
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">E</div>
          <div>
            <h2>Enterprise AI</h2>
            <p>Knowledge &amp; Data Agent</p>
          </div>
        </div>

        <div className="workspace-label">WORKSPACE</div>

        <nav className="navigation">
          <button
            className={activeView === 'chat' ? 'nav-item active' : 'nav-item'}
            onClick={() => setActiveView('chat')}
            type="button"
          >
            <span>◈</span> Knowledge Chat
          </button>
          <button
            className={
              activeView === 'documents' ? 'nav-item active' : 'nav-item'
            }
            onClick={() => setActiveView('documents')}
            type="button"
          >
            <span>▤</span> Documents
          </button>
        </nav>

        {activeView === 'chat' && (
          <div className="conversation-history">
            <div className="conversation-history-heading">
              <span>CHAT HISTORY</span>
              <button type="button" onClick={handleNewChat}>
                + New
              </button>
            </div>

            {conversationsLoading && <p>Loading conversations...</p>}
            {conversationsError && (
              <p className="history-error">{conversationsError}</p>
            )}

            {!conversationsLoading && conversations.length === 0 && (
              <p className="history-empty">Your conversations will appear here.</p>
            )}

            <div className="conversation-list">
              {conversations.map((conversation) => (
                <button
                  key={conversation.id}
                  type="button"
                  className={
                    selectedConversationId === conversation.id
                      ? 'conversation-item selected'
                      : 'conversation-item'
                  }
                  onClick={() => handleSelectConversation(conversation.id)}
                  title={conversation.title || 'Untitled conversation'}
                >
                  <span className="conversation-item-icon">◷</span>
                  <span className="conversation-item-title">
                    {conversation.title || 'Untitled conversation'}
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="sidebar-footer">
          <div className="status-dot" />
          <div className="user-info">
            <strong>{user.email}</strong>
            <p>Signed in · Local AI</p>
          </div>
        </div>

        <button
          className="logout-button"
          onClick={handleLogout}
          type="button"
        >
          Sign out
        </button>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div>
            <p className="eyebrow">ENTERPRISE KNOWLEDGE PLATFORM</p>
            <h1>
              {activeView === 'chat' ? 'Knowledge Chat' : 'Your Documents'}
            </h1>
          </div>
          <div className="local-badge">
            <span className="status-dot" />
            Running locally
          </div>
        </header>

        
        
{activeView === 'chat' ? (
  <section className="chat-workspace">
    <div className="chat-intro">
      <div className="welcome-icon">✳</div>
      <p className="eyebrow">YOUR PRIVATE AI WORKSPACE</p>

      <h2>
        {selectedConversationId
          ? conversations.find(
              (item) => item.id === selectedConversationId,
            )?.title || 'Conversation'
          : 'Ask your documents anything.'}
      </h2>

      <p className="welcome-description">
        Ask questions about your uploaded documents and receive
        answers grounded in retrieved context.
      </p>

      <button
        type="button"
        className="secondary-button"
        onClick={() => setActiveView('documents')}
      >
        Open document workspace →
      </button>
    </div>

    {historyLoading && <p>Loading conversation messages...</p>}

    {!historyLoading && chatMessages.length > 0 && (
      <div className="chat-message-list">
        {chatMessages.map((message) => (
          <article
            className={`chat-message chat-message-${message.role}`}
            key={message.id}
          >
            <p className="chat-message-role">
              {message.role === 'user' ? 'You' : 'AI Assistant'}
            </p>

            <div className="chat-message-content">
              {message.content}
            </div>

            {message.role === 'assistant' &&
              Array.isArray(message.sources) &&
              message.sources.length > 0 && (
                <div className="chat-message-sources">
                  <p className="chat-message-sources-title">
                    Sources ({message.sources.length})
                  </p>

                  {message.sources.map((source, index) => {
                    const sourceKey = `${message.id}-${source.document_id}-${source.chunk_id}`
                    const isExpanded = Boolean(
                      expandedSources[sourceKey],
                    )

                    return (
                      <div
                        className="chat-message-source"
                        key={sourceKey}
                      >
                        <button
                          type="button"
                          className="chat-source-toggle"
                          onClick={() =>
                            handleSourceClick(message.id, source)
                          }
                          aria-expanded={isExpanded}
                        >
                          <span className="chat-source-heading">
                            <strong>
                              {source.filename || 'Unknown document'}
                            </strong>
                            <span>
                              Chunk {source.chunk_index ?? index}
                            </span>
                          </span>

                          <span className="chat-source-chevron">
                            {isExpanded ? '−' : '+'}
                          </span>
                        </button>

                        <div className="chat-source-metadata">
                          <span>
                            Document ID: {source.document_id}
                          </span>
                          <span>
                            Chunk ID: {source.chunk_id}
                          </span>

                          {typeof source.score === 'number' && (
                            <span>
                              Relevance: {source.score.toFixed(3)}
                            </span>
                          )}
                        </div>

                        {isExpanded && (
                          <div className="chat-source-preview">
                            {sourceLoading[sourceKey] && (
                              <p>Loading source text...</p>
                            )}

                            {sourceErrors[sourceKey] && (
                              <p className="chat-source-error">
                                {sourceErrors[sourceKey]}
                              </p>
                            )}

                            {sourceContent[sourceKey] && (
                              <p>{sourceContent[sourceKey]}</p>
                            )}
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              )}
          </article>
        ))}
      </div>
    )}

    <form className="chat-form" onSubmit={handleChatSubmit}>
      <label htmlFor="chat-question">Your question</label>

      <textarea
        id="chat-question"
        value={chatQuestion}
        onChange={(event) => setChatQuestion(event.target.value)}
        placeholder="Ask a question about your documents..."
        rows={3}
        maxLength={2000}
        required
        disabled={chatLoading || historyLoading}
      />

      {chatError && (
        <div className="form-message error-message" role="alert">
          {chatError}
        </div>
      )}

      <div className="chat-form-footer">
        <span>{chatQuestion.length}/2000 characters</span>

        <button
          type="submit"
          className="primary-button"
          disabled={
            !chatQuestion.trim() || chatLoading || historyLoading
          }
        >
          {chatLoading ? 'Thinking...' : 'Ask AI →'}
        </button>
      </div>
    </form>
  </section>
) : (
  <section className="documents-placeholder">
    <div className="welcome-icon">▤</div>
    <h2>Build your knowledge base</h2>

    <p>
      Upload documents to index them for semantic search and grounded
      AI answers.
    </p>

    <form onSubmit={handleDocumentUpload} className="upload-form">
      <label htmlFor="document-file">Choose a document</label>

      <input
        ref={fileInputRef}
        id="document-file"
        type="file"
        accept=".pdf,.docx,.txt,.md"
        onChange={(event) => {
          setSelectedFile(event.target.files?.[0] ?? null)
          setUploadError('')
          setUploadSuccess('')
        }}
        disabled={uploading}
        required
      />

      <p className="upload-hint">
        Supported formats: PDF, DOCX, TXT, Markdown. Maximum size: 10 MB.
      </p>

      {uploadError && (
        <div className="form-message error-message" role="alert">
          {uploadError}
        </div>
      )}

      {uploadSuccess && (
        <div className="form-message success-message" role="status">
          {uploadSuccess}
        </div>
      )}

      <button
        type="submit"
        className="primary-button"
        disabled={!selectedFile || uploading}
      >
        {uploading ? 'Uploading and indexing...' : 'Upload document'}
      </button>
    </form>

    <div className="uploaded-documents">
      <h3>Your documents</h3>

      {documentsLoading && <p>Loading your documents...</p>}

      {documentsError && (
        <div className="form-message error-message" role="alert">
          {documentsError}
        </div>
      )}

      {!documentsLoading &&
        !documentsError &&
        uploadedDocuments.length === 0 && (
          <p>No documents uploaded yet.</p>
        )}

      {uploadedDocuments.length > 0 && (
        <ul>
          {uploadedDocuments.map((document, index) => (
            <li
              key={
                document.id ??
                `${document.filename ?? 'document'}-${index}`
              }
            >
              <span>
                {document.filename ?? 'Untitled document'}
              </span>

              <span className="status-dot" />
              <span>{document.status ?? 'Uploaded'}</span>

              <button
                type="button"
                className="delete-button"
                onClick={() => handleDocumentDelete(document)}
              >
                Delete
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>

    <button
      type="button"
      className="secondary-button"
      onClick={() => setActiveView('chat')}
    >
      Back to knowledge chat
    </button>
  </section>
)}

      </main>
    </div>
  )
}

export default App
