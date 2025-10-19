import { useState, useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import './App.css'

function App() {
  // State management
  const [messages, setMessages] = useState<Array<{role: string, content: string}>>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [useAgent, setUseAgent] = useState(false)
  
  // Ref for auto-scrolling
  const messagesEndRef = useRef<HTMLDivElement>(null)
  
  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  // Handle sending message to backend
  const handleSend = async () => {
    if (!input.trim()) return
    
    // Add user message to chat immediately
    const userMessage = { role: 'user', content: input }
    const newMessages = [...messages, userMessage]
    setMessages(newMessages)
    setInput('')
    setLoading(true)

    try {
      // Call FastAPI backend
      const response = await fetch('http://localhost:8000/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: input,
          use_agent: useAgent
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      
      // Add assistant response to chat (backend returns 'answer' not 'response')
      const assistantMessage = { role: 'assistant', content: data.answer }
      setMessages([...newMessages, assistantMessage])
    } catch (error) {
      console.error('Error calling backend:', error)
      const errorMessage = { 
        role: 'assistant', 
        content: 'Sorry, something went wrong. Make sure the backend is running on http://localhost:8000' 
      }
      setMessages([...newMessages, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  // Handle Enter key press
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !loading && input.trim()) {
      handleSend()
    }
  }

  return (
    <div className="app-container">
      {/* Header Section */}
      <div className="header">
        <h1>RAG Chat Interface</h1>
        
        {/* Toggle for Agent vs Simple RAG */}
        <div className="mode-selector">
          <label className="agent-toggle">
            <input 
              type="checkbox" 
              checked={useAgent}
              onChange={(e) => setUseAgent(e.target.checked)}
            />
            <span className="toggle-label">Enable Agent with Web Tools</span>
          </label>
          <p className="mode-description">
            {useAgent ? (
              <>
                <strong>Agent Mode:</strong> Searches internal documents (RAG) + web search (Tavily) + academic papers (Arxiv)
              </>
            ) : (
              <>
                <strong>RAG Only:</strong> Searches only internal documents
              </>
            )}
          </p>
        </div>
      </div>

      {/* Messages Display Area */}
      <div className="messages-container">
        {messages.length === 0 ? (
          <p className="messages-empty">
            No messages yet. Ask a question!
          </p>
        ) : (
          <div className="messages-wrapper">
            {messages.map((msg, index) => (
              <div 
                key={index}
                className={`message ${msg.role === 'user' ? 'message-user' : 'message-assistant'}`}
              >
                <strong>{msg.role === 'user' ? 'You' : 'Assistant'}:</strong>
                <div className="message-content">
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                </div>
              </div>
            ))}
            {loading && (
              <div className="loading-indicator">
                <p>Thinking...</p>
              </div>
            )}
            {/* Invisible element for auto-scroll */}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input Section */}
      <div className="input-section">
        <input 
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question..."
          className="input-field"
          disabled={loading}
        />
        <button 
          onClick={handleSend}
          disabled={loading || !input.trim()}
          className="send-button"
        >
          {loading ? 'Sending...' : 'Send'}
        </button>
      </div>
    </div>
  )
}

export default App
