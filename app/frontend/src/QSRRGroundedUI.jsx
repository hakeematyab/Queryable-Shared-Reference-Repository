import React, { useState, useEffect, useRef } from 'react';
import { Send, User, Loader2, Code } from 'lucide-react';

// Markdown renderer from Sherlock
const MarkdownRenderer = ({ content }) => {
  const renderMarkdown = (text) => {
    if (!text) return null;
    const lines = text.split('\n');
    const elements = [];
    let currentList = [];
    let currentListType = null;
    let codeBlock = [];
    let inCodeBlock = false;
    let codeLanguage = '';
    
    const flushList = () => {
      if (currentList.length > 0) {
        const ListComponent = currentListType === 'ol' ? 'ol' : 'ul';
        elements.push(
          <ListComponent key={elements.length} className={currentListType === 'ol' ? 'list-decimal ml-6 my-2' : 'list-disc ml-6 my-2'}>
            {currentList.map((item, idx) => <li key={idx} className="mb-1">{renderInline(item)}</li>)}
          </ListComponent>
        );
        currentList = [];
        currentListType = null;
      }
    };
    
    const renderInline = (text) => {
      text = text.replace(/`([^`]+)`/g, '<code class="bg-slate-700 px-1 py-0.5 rounded text-sm font-mono text-emerald-300">$1</code>');
      text = text.replace(/\*\*([^*]+)\*\*/g, '<strong class="font-semibold text-slate-100">$1</strong>');
      text = text.replace(/\*([^*]+)\*/g, '<em class="italic">$1</em>');
      text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" class="text-blue-400 hover:underline" target="_blank">$1</a>');
      return <span dangerouslySetInnerHTML={{ __html: text }} />;
    };
    
    lines.forEach((line) => {
      if (line.startsWith('```')) {
        if (!inCodeBlock) {
          inCodeBlock = true;
          codeLanguage = line.slice(3).trim();
          codeBlock = [];
        } else {
          inCodeBlock = false;
          elements.push(
            <div key={elements.length} className="my-3">
              <div className="bg-slate-900 rounded-lg overflow-hidden border border-slate-700">
                {codeLanguage && (
                  <div className="bg-slate-800 px-3 py-1 text-xs text-slate-400 flex items-center justify-between">
                    <span>{codeLanguage}</span>
                    <Code size={12} />
                  </div>
                )}
                <pre className="p-3 overflow-x-auto">
                  <code className="text-sm text-slate-100 font-mono">{codeBlock.join('\n')}</code>
                </pre>
              </div>
            </div>
          );
          codeLanguage = '';
        }
        return;
      }
      if (inCodeBlock) { codeBlock.push(line); return; }
      if (line.startsWith('### ')) { flushList(); elements.push(<h3 key={elements.length} className="text-base font-bold mt-2 mb-1 text-slate-100">{renderInline(line.slice(4))}</h3>); }
      else if (line.startsWith('## ')) { flushList(); elements.push(<h2 key={elements.length} className="text-lg font-bold mt-3 mb-2 text-slate-100">{renderInline(line.slice(3))}</h2>); }
      else if (line.startsWith('# ')) { flushList(); elements.push(<h1 key={elements.length} className="text-xl font-bold mt-4 mb-2 text-slate-100">{renderInline(line.slice(2))}</h1>); }
      else if (line.match(/^\d+\.\s/)) { if (currentListType !== 'ol') { flushList(); currentListType = 'ol'; } currentList.push(line.replace(/^\d+\.\s/, '')); }
      else if (line.match(/^[-*+]\s/)) { if (currentListType !== 'ul') { flushList(); currentListType = 'ul'; } currentList.push(line.replace(/^[-*+]\s/, '')); }
      else if (line.startsWith('> ')) { flushList(); elements.push(<blockquote key={elements.length} className="border-l-4 border-slate-600 pl-4 my-2 italic text-slate-400">{renderInline(line.slice(2))}</blockquote>); }
      else if (line.trim()) { flushList(); elements.push(<p key={elements.length} className="mb-2">{renderInline(line)}</p>); }
    });
    flushList();
    return elements;
  };
  return <div className="markdown-content">{renderMarkdown(content)}</div>;
};

// Loading indicator with quirky messages
const LoadingIndicator = () => {
  const [idx, setIdx] = useState(0);
  const msgs = ["🔍 Searching the archives...", "📚 Reading through papers...", "🔎 Following citations...", "💭 Synthesizing insights...", "🧩 Connecting the dots...", "🎯 Grounding the response..."];
  useEffect(() => { const i = setInterval(() => setIdx(p => (p + 1) % msgs.length), 2000); return () => clearInterval(i); }, []);
  return <div className="flex items-center gap-2 text-slate-400 text-sm italic"><Loader2 size={14} className="animate-spin" /><span>{msgs[idx]}</span></div>;
};

const QSRRGroundedUI = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const chatRef = useRef(null);

  useEffect(() => { if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight; }, [messages]);

  const getGroundingBadge = (level) => {
    // -1 = grounded, 0 = grey, 1 = hallucinated
    if (level === -1) return { gradient: 'from-emerald-500/20 to-green-500/20', border: 'border-emerald-500/40', text: 'text-emerald-300', iconBg: 'bg-emerald-500/20', label: 'Backed by sources - verify key claims', icon: '✓', glow: 'shadow-emerald-500/20' };
    if (level === 0) return { gradient: 'from-amber-500/20 to-yellow-500/20', border: 'border-amber-500/40', text: 'text-amber-300', iconBg: 'bg-amber-500/20', label: 'Weak source support - check carefully', icon: '⚠', glow: 'shadow-amber-500/20' };
    return { gradient: 'from-rose-500/20 to-red-500/20', border: 'border-rose-500/40', text: 'text-rose-300', iconBg: 'bg-rose-500/20', label: 'No source backing - likely fabricated', icon: '✕', glow: 'shadow-rose-500/20' };
  };

  const sendMessage = async () => {
    if (!input.trim() || isStreaming) return;
    const userMsg = { id: Date.now(), type: 'user', text: input.trim() };
    setMessages(p => [...p, userMsg]);
    setInput('');
    setIsStreaming(true);

    const botId = Date.now() + 1;
    setMessages(p => [...p, { id: botId, type: 'bot', text: '', isStreaming: true }]);

    try {
      const res = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_query: userMsg.text, thread_id: `thread-${botId}` })
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let accText = '', citations = [], hallucinationScore = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        for (const line of chunk.split('\n')) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.error) { setMessages(p => p.map(m => m.id === botId ? { ...m, text: '⚠️ Error occurred. Please try again.', isStreaming: false } : m)); setIsStreaming(false); return; }
              if (data.token) { accText += data.token; setMessages(p => p.map(m => m.id === botId ? { ...m, text: accText } : m)); }
              if (data.done) {
                citations = data.citations || [];
                hallucinationScore = data.hallucination_score ?? null; // -1, 0, or 1
                setMessages(p => p.map(m => m.id === botId ? { ...m, text: accText || 'Found relevant information.', citations, groundingLevel: hallucinationScore, isStreaming: false } : m));
                setIsStreaming(false);
              }
            } catch (e) { console.error('Parse error:', e); }
          }
        }
      }
    } catch (err) {
      console.error('Stream error:', err);
      // Fallback response for testing UI without backend
      const fallback = "**This is a fallback response for UI testing.**\n\nThe backend server isn't running at `http://localhost:8000`. Once connected, responses will stream here with:\n\n- Full markdown support\n- Citation references\n- Hallucination detection badges";
      setMessages(p => p.map(m => m.id === botId ? { ...m, text: fallback, citations: ['Sample Paper - Section 2.1', 'Reference Document - Page 15'], groundingLevel: 0, isStreaming: false } : m));
      setIsStreaming(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Header */}
      <div className="backdrop-blur-xl bg-slate-900/50 border-b border-slate-700/50 px-6 py-4 shadow-lg">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full blur-md opacity-50"></div>
            <img src="/logo.png" alt="QSRR" className="relative w-10 h-10 rounded-full object-cover" />
          </div>
          <div>
            <h1 className="text-xl font-semibold text-slate-100">QSRR</h1>
            <p className="text-xs text-slate-400">Queryable Shared Reference Repository</p>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div ref={chatRef} className="flex-1 overflow-y-auto px-6 py-8 space-y-6">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center space-y-4">
              <div className="relative inline-block">
                <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full blur-2xl opacity-20 animate-pulse"></div>
                <img src="/logo.png" alt="QSRR" className="relative w-24 h-24 rounded-full object-cover" />
              </div>
              <p className="text-slate-300 text-xl font-medium">Welcome to QSRR</p>
              <p className="text-slate-500 text-base">Ask questions about your research papers</p>
            </div>
          </div>
        ) : messages.map((msg) => (
          <div key={msg.id} className="max-w-3xl mx-auto">
            <div className={`flex gap-3 ${msg.type === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
              <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center shadow-lg overflow-hidden ${msg.type === 'user' ? 'bg-gradient-to-br from-blue-500 to-purple-500' : ''}`}>
                {msg.type === 'user' ? <User size={20} className="text-white" /> : <img src="/logo.png" alt="QSRR" className="w-full h-full object-cover" />}
              </div>
              <div className={`flex flex-col flex-1 ${msg.type === 'user' ? 'items-end' : 'items-start'}`}>
                <div className={`rounded-2xl px-5 py-4 backdrop-blur-xl shadow-xl max-w-full ${msg.type === 'user' ? 'bg-gradient-to-br from-blue-500 to-purple-500 text-white' : 'bg-slate-800/80 border border-slate-700/50 text-slate-100'}`}>
                  {msg.type === 'user' ? <p className="text-base leading-relaxed">{msg.text}</p> : msg.isStreaming ? <LoadingIndicator /> : <div className="text-base leading-relaxed"><MarkdownRenderer content={msg.text} /></div>}
                  {msg.citations?.length > 0 && !msg.isStreaming && (
                    <div className="mt-4 pt-4 border-t border-slate-700/50 space-y-1">
                      <p className="text-sm font-semibold text-slate-400 mb-2">Sources</p>
                      {msg.citations.map((c, i) => <p key={i} className="text-sm text-slate-400 flex items-start gap-2"><span className="text-blue-400 font-mono">[{i+1}]</span><span>{c}</span></p>)}
                    </div>
                  )}
                </div>
                {msg.groundingLevel !== undefined && msg.groundingLevel !== null && !msg.isStreaming && (() => {
                  const b = getGroundingBadge(msg.groundingLevel);
                  return (
                    <div className={`mt-3 inline-flex items-center gap-2.5 px-4 py-2 rounded-xl border backdrop-blur-xl bg-gradient-to-r ${b.gradient} ${b.border} shadow-lg ${b.glow}`}>
                      <span className={`${b.iconBg} ${b.text} w-6 h-6 rounded-full flex items-center justify-center text-sm font-bold`}>{b.icon}</span>
                      <span className={`text-sm font-medium ${b.text}`}>{b.label}</span>
                    </div>
                  );
                })()}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Input */}
      <div className="backdrop-blur-xl bg-slate-900/50 border-t border-slate-700/50 px-6 py-5 shadow-2xl">
        <div className="flex gap-3 max-w-3xl mx-auto">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !isStreaming && sendMessage()}
            placeholder={isStreaming ? "Generating response..." : "Ask about your research papers..."}
            disabled={isStreaming}
            className="flex-1 px-5 py-4 bg-slate-800/80 backdrop-blur-xl border border-slate-700/50 rounded-2xl focus:outline-none focus:ring-2 focus:ring-blue-500/50 text-slate-200 text-base shadow-xl disabled:opacity-50"
          />
          <button onClick={sendMessage} disabled={!input.trim() || isStreaming} className="px-6 py-4 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-2xl hover:from-blue-600 hover:to-purple-600 disabled:from-slate-700 disabled:to-slate-600 disabled:cursor-not-allowed transition-all flex items-center shadow-xl hover:shadow-blue-500/25">
            {isStreaming ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
          </button>
        </div>
      </div>
    </div>
  );
};

export default QSRRGroundedUI;