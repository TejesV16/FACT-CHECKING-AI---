import { ChangeEvent, FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import { Paperclip, Clipboard, History, Menu, MessageCircle, MoreHorizontal, SquarePen, Plus, Search, ShieldCheck, Sparkles, Trash2, X } from "lucide-react";

type Evidence = { title: string; url: string; stance: "entailment" | "contradiction" | "neutral"; excerpt: string; quality: string; retrieval_role: string; source_date?: string | null; number_check?: { claimed: string; found: string } | null };
type CheckResult = { claim: string; verdict: "True" | "False" | "Misleading" | "Unverified"; strength: string; summary: string; sources: { title: string; url: string; quality?: string }[]; evidence?: Evidence[]; independent_sources?: number };
type Conversation = { id: string; title: string; prompt: string; createdAt: number; result?: CheckResult[]; elapsed?: number };
const starter: Conversation[] = [
  ["viral", "Is the viral claim about…", "Is the viral claim about a new health study true?"],
  ["jobs", "AI and job replacement…", "Will AI replace all jobs in the next decade?"],
  ["climate", "Climate change facts…", "What evidence supports climate change?"],
  ["health", "Health claim verification…", "Can a health claim be verified from reliable sources?"],
].map(([id, title, prompt], index) => ({ id, title, prompt, createdAt: Date.now() - index }));
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

const readConversations = (): Conversation[] => {
  try { const saved = localStorage.getItem("factai-conversations"); return saved ? JSON.parse(saved) : starter; } catch { return starter; }
};

function App() {
  const [conversations, setConversations] = useState<Conversation[]>(readConversations);
  const [prompt, setPrompt] = useState("Live sources. Local reasoning. Verifiable verdicts.");
  const [search, setSearch] = useState("");
  const [activeId, setActiveId] = useState<string | null>(null);
  const [attachments, setAttachments] = useState<File[]>([]);
  const [panel, setPanel] = useState<"" | "features" | "examples" | "pricing" | "docs" | "onboarding">("");
  const [menuId, setMenuId] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [drawerOpen, setDrawerOpen] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => { localStorage.setItem("factai-conversations", JSON.stringify(conversations)); }, [conversations]);

  const newChat = () => { setActiveId(null); setPrompt(""); setAttachments([]); setPanel(""); setDrawerOpen(false); requestAnimationFrame(() => inputRef.current?.focus()); };
  const activeChat = conversations.find((chat) => chat.id === activeId);
  const selectChat = (chat: Conversation) => { setActiveId(chat.id); setPrompt(chat.prompt); setError(""); setMenuId(null); setDrawerOpen(false); requestAnimationFrame(() => inputRef.current?.focus()); };
  const submit = async () => {
    const cleaned = prompt.trim(); if ((!cleaned && attachments.length === 0) || isSubmitting) return;
    setIsSubmitting(true); setError("");
    try {
      const media = attachments.find((file) => file.type.startsWith("audio/") || file.type.startsWith("video/"));
      const request = media
        ? { url: "/api/check-media", body: { filename: media.name, content: await fileToBase64(media) } }
        : { url: "/api/check", body: { text: cleaned } };
      const response = await fetch(`${API_BASE_URL}${request.url}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request.body) });
      const data = await response.json() as { error?: string; claims?: CheckResult[]; elapsed?: number; transcript?: string };
      if (!response.ok || !data.claims) throw new Error(data.error || "The checker did not return a result.");
      const checkedText = data.transcript || cleaned;
      const next: Conversation = { id: crypto.randomUUID(), title: checkedText.length > 31 ? `${checkedText.slice(0, 31)}…` : checkedText, prompt: checkedText, createdAt: Date.now(), result: data.claims, elapsed: data.elapsed };
      setConversations((items) => [next, ...items]); setActiveId(next.id); setAttachments([]);
    } catch (requestError) { setError(requestError instanceof Error ? requestError.message : "Unable to reach the local fact-checking service."); }
    finally { setIsSubmitting(false); }
  };
  const keyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); submit(); } };
  const chooseFiles = (event: ChangeEvent<HTMLInputElement>) => setAttachments(Array.from(event.target.files ?? []).slice(0, 3));
  const rename = (id: string) => { const chat = conversations.find((item) => item.id === id); const title = window.prompt("Conversation name", chat?.title); if (title?.trim()) setConversations((items) => items.map((item) => item.id === id ? { ...item, title: title.trim() } : item)); setMenuId(null); };
  const remove = (id: string) => { if (window.confirm("Delete this conversation?")) setConversations((items) => items.filter((item) => item.id !== id)); setMenuId(null); };
  const copy = async (chat: Conversation) => { await navigator.clipboard.writeText(chat.prompt); setMenuId(null); };
  const filtered = conversations.filter((chat) => `${chat.title} ${chat.prompt}`.toLowerCase().includes(search.toLowerCase()));

  return <div className="factai-shell">
    <button className="mobile-menu" aria-label="Open conversations" onClick={() => setDrawerOpen(true)}><Menu size={20}/></button>
    <Sidebar open={drawerOpen} close={() => setDrawerOpen(false)} conversations={filtered} activeId={activeId} search={search} setSearch={setSearch} newChat={newChat} selectChat={selectChat} menuId={menuId} setMenuId={setMenuId} rename={rename} remove={remove} copy={copy}/>
    <main className="workspace">
      <TopNav openPanel={setPanel} newChat={newChat}/>
      <section className="hero" aria-labelledby="hero-title">
        <h1 id="hero-title">Give a fact. I’ll <span>check it.</span></h1>
        <form className="composer" onSubmit={(event: FormEvent) => { event.preventDefault(); submit(); }}>
          <textarea ref={inputRef} value={prompt} onChange={(event) => setPrompt(event.target.value)} onKeyDown={keyDown} aria-label="Enter a fact to check" placeholder="Live sources. Local reasoning. Verifiable verdicts." rows={2}/>
          <div className="toolbar">
            <label className="icon-button" title="Attach a document, audio, or video file"><Paperclip size={19}/><input type="file" multiple accept=".pdf,.doc,.docx,.png,.jpg,.jpeg,.mp3,.wav,.m4a,.ogg,.flac,.aac,.mp4,.mov,.mkv,.webm,.avi,.mpeg" onChange={chooseFiles}/></label>
            <div className="toolbar-spacer"/>
            <button type="button" className="edit-button" title="Start a fresh prompt" aria-label="Start a fresh prompt" onClick={newChat}><SquarePen size={19}/></button>
            <button className="submit-button" aria-label="Submit prompt" disabled={(!prompt.trim() && attachments.length === 0) || isSubmitting}>{isSubmitting ? <span className="spinner"/> : <Plus size={21}/>}</button>
          </div>
          {attachments.length > 0 && <div className="attachments">{attachments.map((file, index) => <span key={file.name + index}>{file.name}<button type="button" onClick={() => setAttachments((items) => items.filter((_, i) => i !== index))} aria-label={`Remove ${file.name}`}><X size={12}/></button></span>)}</div>}
        </form>
        {isSubmitting && <p className="mock-response"><Sparkles size={15}/> Checking sources and weighing evidence…</p>}
        {error && <p className="check-error">{error} Start the local API with <code>.venv\Scripts\python api_server.py</code>.</p>}
        {activeChat?.result && <Results results={activeChat.result} elapsed={activeChat.elapsed}/>} 
      </section>
    </main>
    {panel && <Panel name={panel} close={() => setPanel("")} useExample={(text) => { setPrompt(text); setPanel(""); inputRef.current?.focus(); }}/>} 
  </div>;
}

function Sidebar(props: { open: boolean; close: () => void; conversations: Conversation[]; activeId: string | null; search: string; setSearch: (value: string) => void; newChat: () => void; selectChat: (chat: Conversation) => void; menuId: string | null; setMenuId: (id: string | null) => void; rename: (id: string) => void; remove: (id: string) => void; copy: (chat: Conversation) => void }) {
  return <aside className={`sidebar ${props.open ? "open" : ""}`} aria-label="Conversations"><div className="sidebar-top"><div className="brand"><ShieldCheck/><div><strong>FACT<span>AI</span></strong><small>Truth. Verified.</small></div></div><button className="new-chat" onClick={props.newChat}><Plus size={18}/>New Chat</button><label className="side-search"><Search size={20}/><input value={props.search} onChange={(event) => props.setSearch(event.target.value)} placeholder="Search chats" aria-label="Search chats"/></label><button className="history-button"><History size={19}/>History</button><hr/><p className="today">Today</p></div><div className="conversations">{props.conversations.length ? props.conversations.map((chat) => <div className={`conversation ${props.activeId === chat.id ? "selected" : ""}`} key={chat.id}><button className="conversation-main" onClick={() => props.selectChat(chat)}><MessageCircle size={18}/><span>{chat.title}</span></button><button className="more" aria-label={`Actions for ${chat.title}`} onClick={() => props.setMenuId(props.menuId === chat.id ? null : chat.id)}><MoreHorizontal size={18}/></button>{props.menuId === chat.id && <div className="chat-menu"><button onClick={() => props.rename(chat.id)}>Rename</button><button onClick={() => props.copy(chat)}><Clipboard size={14}/>Copy link</button><button className="danger" onClick={() => props.remove(chat.id)}><Trash2 size={14}/>Delete</button></div>}</div>) : <p className="empty-history">No conversations match this search.</p>}</div><button className="close-drawer" onClick={props.close}>Close</button></aside>;
}

function TopNav({ openPanel, newChat }: { openPanel: (panel: "features" | "examples" | "pricing" | "docs" | "onboarding") => void; newChat: () => void }) { return <nav className="top-nav" aria-label="Main navigation"><button className="nav-active" onClick={newChat}><Plus size={17}/>Chat check</button><button onClick={() => openPanel("features")}>Features</button><button onClick={() => openPanel("examples")}>Examples</button><div/><button className="get-started" onClick={() => openPanel("onboarding")}>Get Started</button></nav>; }

function fileToBase64(file: File): Promise<string> { return new Promise((resolve, reject) => { const reader = new FileReader(); reader.onerror = () => reject(new Error("Could not read the selected file.")); reader.onload = () => resolve(String(reader.result).split(",")[1] || ""); reader.readAsDataURL(file); }); }

function Results({ results, elapsed }: { results: CheckResult[]; elapsed?: number }) { return <section className="results" aria-live="polite"><p className="results-meta">{results.length} claim{results.length === 1 ? "" : "s"} checked{elapsed ? ` · ${elapsed}s total` : ""}</p>{results.map((result) => <article className="result-card" key={result.claim}><div className="result-top"><h2>{result.claim}</h2><span className={`verdict ${result.verdict.toLowerCase()}`}>{result.verdict}</span></div><p>{result.summary}</p><small>{result.strength}{result.independent_sources ? ` · ${result.independent_sources} independent domain${result.independent_sources === 1 ? "" : "s"}` : ""}</small>{result.sources.length > 0 && <div className="sources">{result.sources.map((source) => <a key={source.url} href={source.url} target="_blank" rel="noreferrer">{source.title}{source.quality ? <em>{source.quality}</em> : null}</a>)}</div>}{result.evidence && <details className="evidence-details"><summary>Evidence and checks</summary>{result.evidence.map((item) => <div className="evidence-row" key={item.url}><a href={item.url} target="_blank" rel="noreferrer">{item.title}</a><span className={`evidence-stance ${item.stance}`}>{item.stance}</span><p>{item.excerpt || "No extractable passage; the result was assessed from the search snippet."}</p><small>{item.quality} · {item.retrieval_role}{item.source_date ? ` · visible date: ${item.source_date}` : ""}{item.number_check ? ` · number check: claim says ${item.number_check.claimed}; source says ${item.number_check.found}` : ""}</small></div>)}</details>}</article>)}</section>; }

function Panel({ name, close, useExample }: { name: string; close: () => void; useExample: (value: string) => void }) { const content: Record<string, React.ReactNode> = { features: <>FactAI extracts factual claims, retrieves public web sources, and uses its local NLI model to weigh the evidence.</>, examples: <div className="examples"><button onClick={() => useExample("The Earth revolves around the Sun.")}>Earth orbits the Sun</button><button onClick={() => useExample("Antibiotics are effective against viruses.")}>Antibiotics and viruses</button></div>, pricing: <>This local workspace uses your installed FactAI backend.</>, docs: <>Enter a factual claim and press Enter. FactAI will retrieve web evidence and return a verdict with sources.</>, onboarding: <>Welcome to FACTAI. Start the local API, then enter a factual statement to check it against live web evidence.</> }; return <div className="modal-backdrop" role="presentation" onMouseDown={close}><section className="info-panel" role="dialog" aria-modal="true" aria-label={`${name} information`} onMouseDown={(event) => event.stopPropagation()}><button className="panel-close" onClick={close} aria-label="Close"><X/></button><h2>{name === "onboarding" ? "Get started" : name[0].toUpperCase() + name.slice(1)}</h2><div>{content[name]}</div></section></div>; }

export default App;
