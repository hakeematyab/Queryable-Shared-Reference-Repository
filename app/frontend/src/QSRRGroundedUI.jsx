import React, { useState, useEffect, useRef, useCallback, memo } from 'react';
import { Send, User, Loader2, LogIn, Plus, MessageSquare, MoreHorizontal, FileText, X, Copy, Check, ChevronDown, ChevronUp } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

// Simple hash function to create thread_id from username
const hashUsername = async (username) => {
  const encoder = new TextEncoder();
  const data = encoder.encode(username.toLowerCase().trim());
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  return hashHex.slice(0, 16);
};

// Generate chat ID based on timestamp (numeric, matches backend format)
const generateChatId = () => {
  return Date.now().toString();
};

// Username welcome modal
const UsernameModal = ({ onSubmit }) => {
  const [username, setUsername] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!username.trim()) return;
    setIsLoading(true);
    const userHash = await hashUsername(username);
    onSubmit(username.trim(), userHash);
  };

  return (
    <div className="fixed inset-0 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 flex items-center justify-center z-50">
      <div className="relative">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-500 rounded-3xl blur-xl opacity-20 animate-pulse"></div>
        <div className="relative bg-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-3xl p-8 shadow-2xl max-w-md w-full mx-4">
          <div className="text-center mb-8">
            <div className="relative inline-block mb-4">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full blur-2xl opacity-30 animate-pulse"></div>
              <img src="/logo.png" alt="QSRR" className="relative w-20 h-20 rounded-full object-cover" />
            </div>
            <h1 className="text-2xl font-bold text-slate-100 mb-2">Welcome to QSRR</h1>
            <p className="text-slate-400">Enter your username to continue</p>
          </div>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="relative">
              <User size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" />
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter your username..."
                autoFocus
                className="w-full pl-12 pr-4 py-4 bg-slate-900/80 border border-slate-700/50 rounded-2xl focus:outline-none focus:ring-2 focus:ring-blue-500/50 text-slate-200 text-base"
              />
            </div>
            <button
              type="submit"
              disabled={!username.trim() || isLoading}
              className="w-full py-4 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-2xl hover:from-blue-600 hover:to-purple-600 disabled:from-slate-700 disabled:to-slate-600 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2 font-medium shadow-xl hover:shadow-blue-500/25"
            >
              {isLoading ? <Loader2 size={18} className="animate-spin" /> : <LogIn size={18} />}
              <span>{isLoading ? 'Loading...' : 'Continue'}</span>
            </button>
          </form>
          <p className="text-center text-slate-500 text-sm mt-6">Queryable Shared Reference Repository</p>
        </div>
      </div>
    </div>
  );
};

// Markdown renderer using react-markdown (memoized for perf)
const MarkdownRenderer = memo(({ content }) => {
  if (!content) return null;
  
  return (
    <div className="markdown-content break-words">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          // Headings
          h1: ({ children }) => <h1 className="text-lg font-bold mt-3 mb-2 text-slate-100">{children}</h1>,
          h2: ({ children }) => <h2 className="text-base font-bold mt-2 mb-1.5 text-slate-100">{children}</h2>,
          h3: ({ children }) => <h3 className="text-sm font-bold mt-2 mb-1 text-slate-100">{children}</h3>,
          h4: ({ children }) => <h4 className="text-xs font-bold mt-1 mb-0.5 text-slate-100">{children}</h4>,
          // Paragraphs
          p: ({ children }) => <p className="mb-2">{children}</p>,
          // Lists
          ul: ({ children }) => <ul className="list-disc ml-6 my-2">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal ml-6 my-2">{children}</ol>,
          li: ({ children }) => <li className="mb-1">{children}</li>,
          // Inline styles
          strong: ({ children }) => <strong className="font-semibold text-slate-100">{children}</strong>,
          em: ({ children }) => <em className="italic">{children}</em>,
          code: ({ inline, children }) => 
            inline ? (
              <code className="bg-slate-700 px-1 py-0.5 rounded text-sm font-mono text-emerald-300">{children}</code>
            ) : (
              <code className="text-sm text-slate-100 font-mono">{children}</code>
            ),
          // Code blocks
          pre: ({ children }) => (
            <div className="my-3">
              <div className="bg-slate-900 rounded-lg overflow-hidden border border-slate-700">
                <pre className="p-3 overflow-x-auto">{children}</pre>
              </div>
            </div>
          ),
          // Links
          a: ({ href, children }) => (
            <a href={href} className="text-blue-400 hover:underline" target="_blank" rel="noopener noreferrer">{children}</a>
          ),
          // Blockquotes
          blockquote: ({ children }) => (
            <blockquote className="border-l-4 border-slate-600 pl-4 my-2 italic text-slate-400">{children}</blockquote>
          ),
          // Horizontal rules
          hr: () => <hr className="my-4 border-slate-600" />,
          // Tables
          table: ({ children }) => (
            <div className="overflow-x-auto my-3">
              <table className="min-w-full border-collapse border border-slate-600 text-sm">{children}</table>
            </div>
          ),
          thead: ({ children }) => <thead className="bg-slate-800">{children}</thead>,
          tbody: ({ children }) => <tbody>{children}</tbody>,
          tr: ({ children }) => <tr className="border-b border-slate-700">{children}</tr>,
          th: ({ children }) => <th className="px-3 py-2 text-left font-semibold text-slate-200 border border-slate-600">{children}</th>,
          td: ({ children }) => <td className="px-3 py-2 text-slate-300 border border-slate-600">{children}</td>,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
});

// Loading indicator with quirky messages (memoized)
const LoadingIndicator = memo(() => {
  const [idx, setIdx] = useState(0);
  const msgs = ["🔍 Searching the archives...", "📚 Reading through papers...", "🔎 Following citations...", "💭 Synthesizing insights...", "🧩 Connecting the dots...", "🎯 Grounding the response..."];
  useEffect(() => { const i = setInterval(() => setIdx(p => (p + 1) % msgs.length), 2000); return () => clearInterval(i); }, []);
  return <div className="flex items-center gap-2 text-slate-400 text-sm italic"><Loader2 size={14} className="animate-spin" /><span>{msgs[idx]}</span></div>;
});

// Copy button with feedback (memoized)
const CopyButton = memo(({ text, className = "" }) => {
  const [copied, setCopied] = useState(false);
  
  const handleCopy = async (e) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Copy failed:', err);
    }
  };
  
  return (
    <button
      onClick={handleCopy}
      className={`p-1.5 rounded-lg transition-all ${copied ? 'text-emerald-400 bg-emerald-500/20' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'} ${className}`}
      title={copied ? "Copied!" : "Copy to clipboard"}
    >
      {copied ? <Check size={14} /> : <Copy size={14} />}
    </button>
  );
});

// Expandable citation item (memoized)
const CitationItem = memo(({ citation, index }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const previewLength = 80;
  const needsExpansion = citation.length > previewLength;
  const displayText = isExpanded || !needsExpansion ? citation : citation.slice(0, previewLength) + '...';
  
  return (
    <div className="flex items-start gap-2 py-0.5">
      <span className="text-blue-400 font-mono text-xs flex-shrink-0">[{index + 1}]</span>
      <div className="flex-1 min-w-0">
        <div 
          className={`text-xs text-slate-400 ${needsExpansion ? 'cursor-pointer hover:text-slate-300' : ''}`}
          onClick={() => needsExpansion && setIsExpanded(!isExpanded)}
        >
          <span>{displayText}</span>
          {needsExpansion && (
            <button className="ml-1 text-blue-400 hover:text-blue-300 inline-flex items-center">
              {isExpanded ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
            </button>
          )}
        </div>
      </div>
    </div>
  );
});

// Collapsible citations section (memoized)
const CitationsSection = memo(({ citations }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  if (!citations || citations.length === 0) return null;
  
  return (
    <div className="mt-3 pt-3 border-t border-slate-700/50">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center justify-between w-full text-left group"
      >
        <span className="text-xs font-semibold text-slate-400 group-hover:text-slate-300">Sources ({citations.length})</span>
        <div className="flex items-center gap-2">
          <CopyButton text={citations.map((c, i) => `[${i+1}] ${c}`).join('\n')} />
          {isExpanded ? <ChevronUp size={14} className="text-slate-400" /> : <ChevronDown size={14} className="text-slate-400" />}
        </div>
      </button>
      {isExpanded && (
        <div className="mt-2">
          {citations.map((c, i) => <CitationItem key={i} citation={c} index={i} />)}
        </div>
      )}
    </div>
  );
});

// Add Papers Panel Component
const AddPapersPanel = ({ onClose, onFilesSelected, pendingFiles }) => {
  const fileInputRef = useRef(null);

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;
    onFilesSelected(files);
    onClose();
  };

  return (
    <div className="absolute bottom-full left-0 mb-2 bg-slate-800/95 backdrop-blur-xl border border-slate-700/50 rounded-xl shadow-2xl p-4 min-w-[200px]">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium text-slate-200">Actions</span>
        <button onClick={onClose} className="text-slate-400 hover:text-slate-200 transition-colors">
          <X size={14} />
        </button>
      </div>
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".pdf,.txt,.md,.doc,.docx"
        onChange={handleFileSelect}
        className="hidden"
      />
      <button
        onClick={() => fileInputRef.current?.click()}
        className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-slate-700/50 transition-colors text-slate-300 hover:text-slate-100"
      >
        <FileText size={16} />
        <span className="text-sm">Add New Papers</span>
      </button>
      {pendingFiles.length > 0 && (
        <div className="mt-2 text-xs text-slate-400">
          {pendingFiles.length} file(s) ready to index
        </div>
      )}
    </div>
  );
};

// Sidebar Component
const Sidebar = ({ chats, activeChat, onSelectChat, onNewChat, onLoadMore, hasMore, isLoadingMore, username }) => {
  return (
    <div className="w-64 h-full bg-slate-900/80 backdrop-blur-xl border-r border-slate-700/50 flex flex-col">
      {/* New Chat Button */}
      <div className="p-3 border-b border-slate-700/50">
        <button
          onClick={onNewChat}
          className="w-full flex items-center gap-2 px-4 py-3 bg-gradient-to-r from-blue-500/20 to-purple-500/20 hover:from-blue-500/30 hover:to-purple-500/30 border border-slate-700/50 rounded-xl text-slate-200 transition-all"
        >
          <Plus size={18} />
          <span className="font-medium">New Chat</span>
        </button>
      </div>
      
      {/* Chat List */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {chats.map((chat) => (
          <button
            key={chat.id}
            onClick={() => onSelectChat(chat)}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all text-left ${
              activeChat?.id === chat.id
                ? 'bg-slate-700/60 text-slate-100'
                : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200'
            }`}
          >
            <MessageSquare size={16} className="flex-shrink-0" />
            <span className="text-sm truncate">{chat.title || 'New Chat'}</span>
          </button>
        ))}
      </div>
      
      {/* Load More Button */}
      <div className="p-3 border-t border-slate-700/50">
        {hasMore ? (
          <button
            onClick={onLoadMore}
            disabled={isLoadingMore}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 rounded-lg transition-all disabled:opacity-50"
          >
            {isLoadingMore ? <Loader2 size={16} className="animate-spin" /> : <MoreHorizontal size={16} />}
            <span className="text-sm">{isLoadingMore ? 'Loading...' : 'Load More'}</span>
          </button>
        ) : (
          <div className="text-center text-slate-500 text-xs py-2">
            {chats.length > 0 ? 'No more chats' : 'No previous chats'}
          </div>
        )}
      </div>
    </div>
  );
};

const QSRRGroundedUI = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [username, setUsername] = useState(null);
  const [userHash, setUserHash] = useState(null);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [chats, setChats] = useState([]);
  const [activeChat, setActiveChat] = useState(null);
  const [chatId, setChatId] = useState(null);
  const [hasMoreChats, setHasMoreChats] = useState(false);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [chatOffset, setChatOffset] = useState(0);
  const [showAddPanel, setShowAddPanel] = useState(false);
  const [pendingFiles, setPendingFiles] = useState([]);
  const [isIndexing, setIsIndexing] = useState(false);
  const chatRef = useRef(null);

  useEffect(() => { if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight; }, [messages]);

  // Memoize grounding badge lookup to avoid recalc on every render
  const getGroundingBadge = useCallback((level) => {
    if (level === -1) return { gradient: 'from-emerald-500/20 to-green-500/20', border: 'border-emerald-500/40', text: 'text-emerald-300', iconBg: 'bg-emerald-500/20', label: 'Backed by sources - verify key claims', icon: '✓', glow: 'shadow-emerald-500/20' };
    if (level === 0) return { gradient: 'from-amber-500/20 to-yellow-500/20', border: 'border-amber-500/40', text: 'text-amber-300', iconBg: 'bg-amber-500/20', label: 'Weak source support - check carefully', icon: '⚠', glow: 'shadow-amber-500/20' };
    return { gradient: 'from-rose-500/20 to-red-500/20', border: 'border-rose-500/40', text: 'text-rose-300', iconBg: 'bg-rose-500/20', label: 'No source backing - likely fabricated', icon: '✕', glow: 'shadow-rose-500/20' };
  }, []);

  const fetchChats = async (hash, offset = 0) => {
    try {
      const res = await fetch(`http://localhost:8000/chats/${hash}?limit=10&offset=${offset}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      // Map backend response to frontend chat structure
      const chats = (data.threads || []).map(t => ({
        id: t.thread_id,
        thread_id: t.thread_id,
        title: t.preview || 'New conversation',
        timestamp: t.created_at
      }));
      return { chats, hasMore: data.has_more || false };
    } catch (err) {
      console.log('Could not fetch chats:', err.message);
      return { chats: [], hasMore: false };
    }
  };

  const fetchChatHistory = async (threadId) => {
    try {
      const res = await fetch(`http://localhost:8000/history/${threadId}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (data.messages && Array.isArray(data.messages)) {
        const loadedMessages = data.messages.map((msg, idx) => ({
          id: Date.now() + idx,
          type: msg.role === 'user' ? 'user' : 'bot',
          text: msg.content,
          citations: msg.citations || [],
          groundingLevel: msg.hallucination_score ?? null,
          isStreaming: false
        }));
        setMessages(loadedMessages);
      }
    } catch (err) {
      console.log('Could not fetch history, starting fresh:', err.message);
      setMessages([]);
    }
  };

  const handleLogin = async (name, hash) => {
    setUsername(name);
    setUserHash(hash);
    const { chats: loadedChats, hasMore } = await fetchChats(hash, 0);
    setChats(loadedChats);
    setHasMoreChats(hasMore);
    setChatOffset(loadedChats.length);
    setActiveChat(null);
    setChatId(null);
    setMessages([]);
    setIsLoggedIn(true);
  };

  const handleLoadMore = async () => {
    setIsLoadingMore(true);
    const { chats: moreChats, hasMore } = await fetchChats(userHash, chatOffset);
    setChats(prev => [...prev, ...moreChats]);
    setHasMoreChats(hasMore);
    setChatOffset(prev => prev + moreChats.length);
    setIsLoadingMore(false);
  };

  const handleSelectChat = async (chat) => {
    setActiveChat(chat);
    setChatId(chat.thread_id);
    await fetchChatHistory(chat.thread_id);
  };

  const handleNewChat = () => {
    setActiveChat(null);
    setChatId(null);
    setMessages([]);
    setPendingFiles([]);
  };

  const handleFilesSelected = (files) => {
    // Store actual File objects so we can upload their content
    setPendingFiles(prev => [...prev, ...files]);
  };

  const removePendingFile = (index) => {
    setPendingFiles(prev => prev.filter((_, i) => i !== index));
  };



  const getThreadId = () => {
    if (chatId) return chatId;
    return null;
  };

  if (!isLoggedIn) {
    return <UsernameModal onSubmit={handleLogin} />;
  }

  const sendMessage = async () => {
    // If we have pending files, only do indexing (no chat)
    if (pendingFiles.length > 0) {
      if (isIndexing) return;
      
      // Create new chat if needed
      if (!chatId) {
        const timestamp = generateChatId();
        const newThreadId = `${userHash}_${timestamp}`;
        setChatId(newThreadId);
        const docNames = pendingFiles.map(f => f.name).join(', ');
        const chatTitle = `Index: ${docNames}`.slice(0, 30) + (docNames.length > 25 ? '...' : '');
        const newChat = { id: newThreadId, thread_id: newThreadId, title: chatTitle, timestamp: Date.now() };
        setChats(prev => [newChat, ...prev]);
        setActiveChat(newChat);
      }
      
      // Capture files to upload before clearing state
      const filesToUpload = [...pendingFiles];
      const docNames = filesToUpload.map(f => f.name).join(', ');
      const userMsg = { id: Date.now(), type: 'user', text: `Index ${docNames}` };
      setMessages(p => [...p, userMsg]);
      setInput('');
      setPendingFiles([]); // Clear immediately so chips disappear
      
      // Step 1: Upload files to server
      setIsIndexing(true);
      const botId = Date.now() + 1;
      setMessages(p => [...p, { id: botId, type: 'bot', text: '📤 Uploading files...', isStreaming: true }]);
      
      let serverPaths = [];
      try {
        const formData = new FormData();
        filesToUpload.forEach(file => formData.append('files', file));
        
        const uploadRes = await fetch('http://localhost:8000/upload', {
          method: 'POST',
          body: formData
        });
        
        if (!uploadRes.ok) throw new Error(`Upload failed: HTTP ${uploadRes.status}`);
        
        const uploadData = await uploadRes.json();
        if (uploadData.errors && uploadData.errors.length > 0) {
          console.warn('Some files failed to upload:', uploadData.errors);
        }
        serverPaths = uploadData.saved_paths;
        
        if (serverPaths.length === 0) {
          setMessages(p => p.map(m => m.id === botId ? { 
            ...m, 
            text: '⚠️ No files were uploaded successfully.', 
            isStreaming: false 
          } : m));
          setIsIndexing(false);
          return;
        }
        
        setMessages(p => p.map(m => m.id === botId ? { ...m, text: `📤 Uploaded ${serverPaths.length} file(s). Starting indexing...` } : m));
      } catch (err) {
        console.error('Upload error:', err);
        setMessages(p => p.map(m => m.id === botId ? { 
          ...m, 
          text: '⚠️ Failed to upload files. The backend may be offline.', 
          isStreaming: false 
        } : m));
        setIsIndexing(false);
        return;
      }
      
      // Step 2: Index the uploaded files
      try {
        const res = await fetch('http://localhost:8000/index', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ paper_paths: serverPaths })
        });
        
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let progressText = '';
        
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          const chunk = decoder.decode(value, { stream: true });
          for (const line of chunk.split('\n')) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                
                if (data.status === 'error') {
                  // Lock error - another indexing in progress
                  setMessages(p => p.map(m => m.id === botId ? { 
                    ...m, 
                    text: `⚠️ ${data.error}`, 
                    isStreaming: false 
                  } : m));
                  setIsIndexing(false);
                  return;
                }
                
                if (data.status === 'progress') {
                  const icon = data.success ? '✓' : '✗';
                  progressText = `📚 Indexing papers...\n\nProcessing ${data.current}/${data.total}: ${data.file} ${icon}`;
                  if (!data.success && data.error) {
                    progressText += `\n  Error: ${data.error}`;
                  }
                  setMessages(p => p.map(m => m.id === botId ? { ...m, text: progressText } : m));
                }
                
                if (data.status === 'done') {
                  let finalText = `📚 Indexing complete!\n\n✓ Successfully indexed: ${data.indexed} paper(s)`;
                  if (data.failed > 0) {
                    finalText += `\n✗ Failed: ${data.failed} paper(s)`;
                    data.failed_files?.forEach(f => {
                      finalText += `\n  - ${f.file}: ${f.error}`;
                    });
                  }
                  setMessages(p => p.map(m => m.id === botId ? { 
                    ...m, 
                    text: finalText, 
                    isStreaming: false 
                  } : m));
                  setIsIndexing(false);
                }
              } catch (e) {
                console.error('Parse error:', e);
              }
            }
          }
        }
      } catch (err) {
        console.error('Index error:', err);
        setMessages(p => p.map(m => m.id === botId ? { 
          ...m, 
          text: '⚠️ Failed to index papers. The backend may be offline.', 
          isStreaming: false 
        } : m));
        setIsIndexing(false);
      }
      return;
    }
    
    // Normal chat flow (no pending files)
    if (!input.trim() || isStreaming) return;
    
    let currentChatId = chatId;
    let currentThreadId = getThreadId();
    
    if (!currentChatId) {
      const timestamp = generateChatId();
      currentThreadId = `${userHash}_${timestamp}`;
      currentChatId = currentThreadId;
      setChatId(currentThreadId);
      const newChat = { id: currentThreadId, thread_id: currentThreadId, title: input.trim().slice(0, 30) + (input.trim().length > 30 ? '...' : ''), timestamp: Date.now() };
      setChats(prev => [newChat, ...prev]);
      setActiveChat(newChat);
    }
    
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
        body: JSON.stringify({ user_query: userMsg.text, thread_id: currentThreadId })
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
                hallucinationScore = data.hallucination_score ?? null;
                setMessages(p => p.map(m => m.id === botId ? { ...m, text: accText || 'Found relevant information.', citations, groundingLevel: hallucinationScore, isStreaming: false } : m));
                setIsStreaming(false);
              }
            } catch (e) { console.error('Parse error:', e); }
          }
        }
      }
    } catch (err) {
      console.error('Stream error:', err);
      const fallback = "**This is a fallback response for UI testing.**\n\nThe backend server isn't running at `http://localhost:8000`. Once connected, responses will stream here with:\n\n- Full markdown support\n- Citation references\n- Hallucination detection badges";
      setMessages(p => p.map(m => m.id === botId ? { ...m, text: fallback, citations: ['Sample Paper - Section 2.1', 'Reference Document - Page 15'], groundingLevel: 0, isStreaming: false } : m));
      setIsStreaming(false);
    }
  };

  return (
    <div className="flex h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Sidebar */}
      <Sidebar
        chats={chats}
        activeChat={activeChat}
        onSelectChat={handleSelectChat}
        onNewChat={handleNewChat}
        onLoadMore={handleLoadMore}
        hasMore={hasMoreChats}
        isLoadingMore={isLoadingMore}
        username={username}
      />
      
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="backdrop-blur-xl bg-slate-900/50 border-b border-slate-700/50 px-6 py-4 shadow-lg">
          <div className="flex items-center justify-between">
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
            <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-800/60 rounded-full border border-slate-700/50">
              <User size={14} className="text-slate-400" />
              <span className="text-sm text-slate-300">{username}</span>
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
                <p className="text-slate-300 text-xl font-medium">Welcome back, {username}!</p>
                <p className="text-slate-500 text-base">Ask questions about your research papers</p>
              </div>
            </div>
          ) : messages.map((msg) => (
            <div key={msg.id} className="max-w-3xl mx-auto group/msg">
              <div className={`flex gap-3 ${msg.type === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center shadow-lg overflow-hidden ${msg.type === 'user' ? 'bg-gradient-to-br from-blue-500 to-purple-500' : ''}`}>
                  {msg.type === 'user' ? <User size={20} className="text-white" /> : <img src="/logo.png" alt="QSRR" className="w-full h-full object-cover" />}
                </div>
                <div className={`flex flex-col flex-1 ${msg.type === 'user' ? 'items-end' : 'items-start'}`}>
                  <div className={`relative rounded-2xl px-4 py-3 backdrop-blur-xl shadow-xl max-w-full ${msg.type === 'user' ? 'bg-gradient-to-br from-blue-500 to-purple-500 text-white' : 'bg-slate-800/80 border border-slate-700/50 text-slate-100'}`}>
                    {msg.type === 'user' ? (
                      <p className="text-sm leading-relaxed pr-8">{msg.text}</p>
                    ) : msg.isStreaming ? (
                      <LoadingIndicator />
                    ) : (
                      <div className="text-sm leading-relaxed pr-8"><MarkdownRenderer content={msg.text} /></div>
                    )}
                    {!msg.isStreaming && (
                      <CopyButton 
                        text={msg.text} 
                        className={`absolute top-3 right-3 ${msg.type === 'user' ? 'text-white/70 hover:text-white hover:bg-white/20' : ''}`}
                      />
                    )}
                    {msg.citations?.length > 0 && !msg.isStreaming && (
                      <CitationsSection citations={msg.citations} />
                    )}
                  </div>
                  {msg.type === 'bot' && !msg.isStreaming && (() => {
                    if (msg.groundingLevel !== undefined && msg.groundingLevel !== null) {
                      const b = getGroundingBadge(msg.groundingLevel);
                      return (
                        <div className={`mt-3 inline-flex items-center gap-2.5 px-4 py-2 rounded-xl border backdrop-blur-xl bg-gradient-to-r ${b.gradient} ${b.border} shadow-lg ${b.glow}`}>
                          <span className={`${b.iconBg} ${b.text} w-6 h-6 rounded-full flex items-center justify-center text-sm font-bold`}>{b.icon}</span>
                          <span className={`text-sm font-medium ${b.text}`}>{b.label}</span>
                        </div>
                      );
                    } else {
                      return (
                        <div className="mt-3 inline-flex items-center gap-2.5 px-4 py-2 rounded-xl border backdrop-blur-xl bg-gradient-to-r from-slate-500/20 to-slate-600/20 border-slate-500/40 shadow-lg">
                          <span className="bg-slate-500/20 text-slate-400 w-6 h-6 rounded-full flex items-center justify-center text-sm font-bold">?</span>
                          <span className="text-sm font-medium text-slate-400">Hallucination rating unavailable</span>
                        </div>
                      );
                    }
                  })()}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Input */}
        <div className="backdrop-blur-xl bg-slate-900/50 border-t border-slate-700/50 px-6 py-5 shadow-2xl">
          {/* Pending Files Display */}
          {pendingFiles.length > 0 && (
            <div className="max-w-3xl mx-auto mb-3">
              <div className="flex flex-wrap gap-2">
                {pendingFiles.map((file, idx) => (
                  <div key={idx} className="flex items-center gap-2 px-3 py-1.5 bg-slate-800/80 border border-slate-700/50 rounded-lg">
                    <FileText size={14} className="text-blue-400" />
                    <span className="text-sm text-slate-300 max-w-[150px] truncate">{file.name}</span>
                    <button
                      onClick={() => removePendingFile(idx)}
                      className="text-slate-500 hover:text-rose-400 transition-colors"
                    >
                      <X size={14} />
                    </button>
                  </div>
                ))}
              </div>
              <p className="text-xs text-slate-500 mt-2">Papers will be indexed when you send your message</p>
            </div>
          )}
          <div className="flex gap-3 max-w-3xl mx-auto">
            <div className="relative">
              <button
                onClick={() => setShowAddPanel(!showAddPanel)}
                className={`h-full px-4 py-4 bg-slate-800/80 border border-slate-700/50 rounded-2xl hover:bg-slate-700/80 transition-all ${pendingFiles.length > 0 ? 'text-blue-400' : 'text-slate-400 hover:text-slate-200'}`}
              >
                <Plus size={18} />
              </button>
              {showAddPanel && <AddPapersPanel onClose={() => setShowAddPanel(false)} onFilesSelected={handleFilesSelected} pendingFiles={pendingFiles} />}
            </div>
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && !isStreaming && !isIndexing && sendMessage()}
              placeholder={isIndexing ? "Indexing papers..." : pendingFiles.length > 0 ? "Press Enter to index papers..." : isStreaming ? "Generating response..." : "Ask about your research papers..."}
              disabled={isStreaming || isIndexing || pendingFiles.length > 0}
              className="flex-1 px-5 py-4 bg-slate-800/80 backdrop-blur-xl border border-slate-700/50 rounded-2xl focus:outline-none focus:ring-2 focus:ring-blue-500/50 text-slate-200 text-base shadow-xl disabled:opacity-50"
            />
            <button onClick={sendMessage} disabled={(pendingFiles.length === 0 && !input.trim()) || isStreaming || isIndexing} className="px-6 py-4 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-2xl hover:from-blue-600 hover:to-purple-600 disabled:from-slate-700 disabled:to-slate-600 disabled:cursor-not-allowed transition-all flex items-center shadow-xl hover:shadow-blue-500/25">
              {isIndexing ? <Loader2 size={18} className="animate-spin" /> : isStreaming ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default QSRRGroundedUI;