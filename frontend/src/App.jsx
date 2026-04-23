import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import { LayoutDashboard, MessageSquare, Library, Server, Settings, Cpu, Database, Zap } from 'lucide-react';
import axios from 'axios';

// --- API Service ---
const API_BASE = 'http://localhost:8080/v1';

// --- Components ---

const Sidebar = () => (
  <div className="sidebar">
    <div className="headline" style={{ fontSize: '1.5rem', marginBottom: '2rem', color: 'var(--primary)' }}>RKLLama</div>
    <nav>
      <NavLink to="/" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
        <LayoutDashboard size={20} /> Dashboard
      </NavLink>
      <NavLink to="/chat" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
        <MessageSquare size={20} /> AI Chat
      </NavLink>
      <NavLink to="/models" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
        <Library size={20} /> Models
      </NavLink>
      <div className="nav-item"><Server size={20} /> Local Server</div>
      <div className="nav-item"><Settings size={20} /> Settings</div>
    </nav>
  </div>
);

const Dashboard = () => {
  const [stats, setStats] = useState({ npu: 15, ram: 82, temp: 54 });

  return (
    <div className="main-content">
      <div className="top-bar">
        <div className="stat-pill"><Cpu size={14} /> NPU: {stats.npu}%</div>
        <div className="stat-pill"><Database size={14} /> RAM: {stats.ram}%</div>
        <div className="stat-pill"><Zap size={14} /> TEMP: {stats.temp}°C</div>
      </div>

      <div className="glass-card" style={{ marginTop: '2rem', borderLeft: '4px solid var(--primary)' }}>
        <h2 className="headline">Active Model</h2>
        <div style={{ marginTop: '1rem', display: 'flex', justifyContent: 'space-between' }}>
          <div>
            <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>Llama-3-8B-RKLLM</div>
            <div style={{ color: 'var(--on-surface-variant)' }}>Quantization: Q4_K_M</div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ color: 'var(--primary)', fontWeight: 'bold' }}>12.5 tokens/s</div>
            <div style={{ fontSize: '0.8rem' }}>Memory: 4.2 GB</div>
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginTop: '2rem' }}>
        <div className="glass-card">
          <h3 className="headline">Local Library</h3>
          <div className="list-item">Phi-3 Mini (3.8B) <span>2.4 GB</span></div>
          <div className="list-item">Mistral 7B v0.3 <span>4.1 GB</span></div>
          <div className="list-item">Gemma 2B <span>1.6 GB</span></div>
        </div>
        <div className="glass-card">
          <h3 className="headline">Server Status</h3>
          <div style={{ marginTop: '1rem' }}>
            <div className="pulse-dot"></div> OpenAI API Running
            <div style={{ fontSize: '0.8rem', color: 'var(--on-surface-variant)', marginTop: '0.5rem' }}>http://localhost:8080</div>
          </div>
        </div>
      </div>
    </div>
  );
};

const Chat = () => {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! I am RKLLama, running on your Rockchip NPU. How can I help you today?' }
  ]);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const newMessages = [...messages, { role: 'user', content: input }];
    setMessages(newMessages);
    setInput('');
    setStreaming(true);

    try {
      const response = await fetch(`${API_BASE}/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'llama-3-8b',
          messages: newMessages,
          stream: true
        })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantMessage = '';
      setMessages([...newMessages, { role: 'assistant', content: '' }]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6);
            if (dataStr === '[DONE]') break;
            try {
              const data = JSON.parse(dataStr);
              const text = data.choices[0].delta.content || '';
              assistantMessage += text;
              setMessages([...newMessages, { role: 'assistant', content: assistantMessage }]);
            } catch (e) {}
          }
        }
      }
    } catch (error) {
      console.error('Chat error:', error);
    } finally {
      setStreaming(false);
    }
  };

  return (
    <div className="main-content" style={{ display: 'flex', flexDirection: 'column', height: '100vh', paddingBottom: '0' }}>
      <div className="chat-header">
        <div className="headline">AI Chat - Llama-3-8B</div>
      </div>
      
      <div className="message-area">
        {messages.map((m, i) => (
          <div key={i} className={`message ${m.role}`}>
            <div className="message-bubble">{m.content}</div>
          </div>
        ))}
      </div>

      <div className="input-area">
        <div className="input-container">
          <input 
            value={input} 
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="Type a message..." 
            disabled={streaming}
          />
          <button onClick={sendMessage} className="primary-btn" disabled={streaming}>Send</button>
        </div>
      </div>
    </div>
  );
};

const App = () => (
  <Router>
    <div style={{ display: 'flex' }}>
      <Sidebar />
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="/models" element={<div className="main-content"><h2 className="headline">Model Manager</h2><p>Coming Soon...</p></div>} />
      </Routes>
    </div>
  </Router>
);

export default App;
