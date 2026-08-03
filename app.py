"""
GraphRAG Engine - Interactive Streamlit Dashboard
Explore the knowledge graph, search entities, and visualize communities.

Usage:
    streamlit run app.py
"""

import json
import os
import streamlit as st
import networkx as nx
from pyvis.network import Network
import tempfile
from dotenv import load_dotenv

load_dotenv()

# ─── Page Config ───
st.set_page_config(
    page_title="GraphRAG Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Design System CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"],.stApp{font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif!important}
#MainMenu,footer{visibility:hidden}
.block-container{padding-top:1.5rem!important;padding-bottom:2rem!important;max-width:1400px!important}
.stApp{background:#080c14}
section[data-testid="stSidebar"]{background:#070b12!important;border-right:1px solid #141f30!important}
.app-header{margin-bottom:1.5rem;padding-bottom:1rem;border-bottom:1px solid #141f30}
.app-title{font-size:1.55rem;font-weight:700;color:#f1f5f9;letter-spacing:-.4px;margin:0;line-height:1.2}
.app-title span{color:#3b82f6}
.app-subtitle{font-size:.78rem;color:#334155;font-weight:400;margin:5px 0 0 1px}
.metrics-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:1.5rem}
.metric-card{background:#0c1220;border:1px solid #141f30;border-radius:10px;padding:18px 20px;position:relative;overflow:hidden;transition:border-color 150ms ease,transform 150ms ease;cursor:default}
.metric-card::after{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#1d4ed8,#3b82f6);opacity:0;transition:opacity 150ms ease}
.metric-card:hover{border-color:#1e3a5f;transform:translateY(-1px)}
.metric-card:hover::after{opacity:1}
.metric-value{font-size:2rem;font-weight:700;color:#f1f5f9;line-height:1;font-variant-numeric:tabular-nums;letter-spacing:-1px}
.metric-label{font-size:.68rem;color:#334155;text-transform:uppercase;letter-spacing:1.4px;font-weight:600;margin-top:7px}
.stTabs [data-baseweb="tab-list"]{gap:0!important;border-bottom:1px solid #141f30!important;background:transparent!important;padding:0!important}
.stTabs [data-baseweb="tab"]{background:transparent!important;border:none!important;border-bottom:2px solid transparent!important;border-radius:0!important;padding:10px 22px!important;color:#334155!important;font-size:.82rem!important;font-weight:500!important;transition:color 150ms ease,border-color 150ms ease!important;margin-bottom:-1px!important}
.stTabs [data-baseweb="tab"]:hover{color:#64748b!important;background:transparent!important}
.stTabs [aria-selected="true"]{color:#f1f5f9!important;border-bottom-color:#3b82f6!important;background:transparent!important}
.stTabs [data-baseweb="tab-panel"]{padding-top:1.5rem!important}
hr{border-color:#141f30!important;margin:1.5rem 0!important}
.stTextInput>div>div>input{background:#0c1220!important;border:1px solid #1e2d45!important;border-radius:8px!important;color:#e2e8f0!important;font-size:.88rem!important;transition:border-color 150ms ease!important;padding:10px 14px!important}
.stTextInput>div>div>input:focus{border-color:#3b82f6!important;box-shadow:0 0 0 3px rgba(59,130,246,.12)!important}
.stTextInput>div>div>input::placeholder{color:#1e3a5f!important}
[data-baseweb="select"]>div{background:#0c1220!important;border:1px solid #1e2d45!important;border-radius:8px!important;color:#e2e8f0!important;font-size:.85rem!important;transition:border-color 150ms ease!important}
.stRadio label{color:#94a3b8!important;font-size:.83rem!important}
.stButton>button{font-weight:600!important;font-size:.83rem!important;border-radius:8px!important;transition:background 150ms ease,box-shadow 150ms ease,transform 100ms ease!important;cursor:pointer!important}
.stButton>button:active{transform:scale(.98)!important}
.stButton>button[kind="primary"]{background:#2563eb!important;border:none!important;color:#fff!important}
.stButton>button[kind="primary"]:hover{background:#1d4ed8!important;box-shadow:0 0 0 3px rgba(37,99,235,.22)!important}
.stSuccess{background:#052e16!important;border-color:#10b981!important;color:#6ee7b7!important;border-radius:8px!important}
.stError,.stWarning{border-radius:8px!important}
.stInfo{background:#0c1a30!important;border-color:#1e3a5f!important;border-radius:8px!important;color:#94a3b8!important}
.answer-card{background:#0c1220;border:1px solid #1e2d45;border-left:3px solid #3b82f6;border-radius:10px;padding:20px 24px;margin-top:12px;line-height:1.75;color:#e2e8f0;font-size:.88rem}
.answer-label{font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:2px;color:#3b82f6;margin-bottom:12px}
.citations-label{font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:2px;color:#334155;margin:20px 0 10px}
.citation-row{display:flex;gap:12px;align-items:flex-start;padding:8px 0;border-bottom:1px solid #111827;font-size:.8rem}
.citation-row:last-of-type{border-bottom:none}
.citation-entity{font-weight:600;color:#93c5fd;min-width:130px;flex-shrink:0}
.citation-detail{color:#475569;line-height:1.5}
.modes-row{display:flex;gap:12px;margin-top:1rem}
.mode-card{flex:1;background:#0c1220;border:1px solid #141f30;border-radius:8px;padding:14px 16px}
.mode-title{font-size:.78rem;font-weight:600;color:#64748b;margin-bottom:5px}
.mode-body{font-size:.76rem;color:#334155;line-height:1.6}
.sb-header{background:#0c1220;border-radius:10px;padding:18px 20px;border:1px solid #1e2d45;margin-bottom:8px;box-shadow:0 4px 6px -1px rgba(0,0,0,0.1)}
.sb-header h2{font-size:1.05rem;font-weight:700;color:#f8fafc;margin:0;letter-spacing:-.3px}
.sb-header p{font-size:.75rem;color:#64748b;margin:5px 0 0 0;line-height:1.4}
.sb-section-label{font-size:.62rem;font-weight:700;text-transform:uppercase;letter-spacing:2.2px;color:#1e3a5f;margin:18px 0 8px 1px}
.sb-card{background:#0c1220;border-radius:8px;padding:13px 15px;border:1px solid #141f30;margin-bottom:6px}
.sb-stat-row{display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid #0a101c;font-size:.78rem}
.sb-stat-row:last-of-type{border-bottom:none}
.sb-stat-label{color:#475569}
.sb-stat-value{font-weight:600;color:#f1f5f9;font-variant-numeric:tabular-nums}
.sb-badge{margin-top:10px;padding:6px 11px;background:#031a0e;border-radius:6px;border-left:2px solid #10b981;font-size:.74rem;color:#6ee7b7;font-weight:500}
.sb-about{padding:10px 14px;background:#05080f;border-radius:8px;border:1px solid #0c1220;font-size:.72rem;color:#334155;line-height:1.7}
[data-testid="stFileUploader"]{width:100%}
[data-testid="stFileUploaderDropzone"]{background:#0a0f18!important;border:2px dashed #1e2d45!important;border-radius:12px!important;padding:24px 16px!important;transition:border-color 150ms ease,background 150ms ease!important;min-height:140px;display:flex;align-items:center;justify-content:center}
[data-testid="stFileUploaderDropzone"]:hover{border-color:#3b82f6!important;background:#0d1522!important}
[data-testid="stFileUploaderDropzoneInstructions"]>div>span{font-size:.8rem!important;color:#94a3b8!important;font-weight:500!important}
[data-testid="stFileUploaderDropzoneInstructions"]>div>small{font-size:.7rem!important;color:#475569!important}
[data-testid="stFileUploaderDropzoneInstructions"] svg{fill:#3b82f6!important;width:28px!important;height:28px!important;margin-bottom:8px!important}
[data-testid="stFileUploader"] section>button{background:#0c1220!important;border:1px solid #1e2d45!important;color:#64748b!important;border-radius:6px!important;font-size:.73rem!important;padding:3px 12px!important;transition:all 150ms ease!important}
[data-testid="stFileUploader"] section>button:hover{background:#1e3a5f!important;border-color:#2563eb!important;color:#93c5fd!important}
[data-testid="stFileUploaderFile"]{background:#0c1220!important;border-radius:6px!important;border:1px solid #141f30!important;padding:5px 10px!important;margin-top:5px!important}
[data-testid="stFileUploaderFileName"]{color:#94a3b8!important;font-size:.74rem!important}
[data-testid="stFileUploaderFileSize"]{color:#334155!important;font-size:.68rem!important}
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:#080c14}
::-webkit-scrollbar-thumb{background:#141f30;border-radius:4px}
::-webkit-scrollbar-thumb:hover{background:#1e3a5f}
.stDataFrame{border:1px solid #141f30!important;border-radius:8px!important;overflow:hidden!important}
</style>
""", unsafe_allow_html=True)

# ─── Color Palette ───
TYPE_COLORS = {
    "person": "#f472b6", "organization": "#60a5fa", "technology": "#34d399",
    "algorithm": "#fbbf24", "concept": "#a78bfa", "tool": "#f97316",
    "dataset": "#06b6d4", "event": "#ef4444", "unknown": "#6b7280",
}

COMM_COLORS = [
    "#60a5fa", "#f472b6", "#34d399", "#fbbf24", "#a78bfa", "#f97316",
    "#06b6d4", "#ef4444", "#84cc16", "#ec4899", "#14b8a6", "#f59e0b",
    "#8b5cf6", "#10b981", "#e11d48",
]


@st.cache_data
def load_graph_data():
    """Load all artifacts from output directory."""
    stats = json.load(open("output/graph_stats.json", encoding="utf-8"))
    summaries = json.load(open("output/community_summaries.json", encoding="utf-8"))
    summaries = {int(k): v for k, v in summaries.items()}
    G = nx.read_gml("output/knowledge_graph.gml")
    return G, stats, summaries


def render_pyvis_graph(G, filter_community=None, height="600px"):
    """Render an interactive PyVis network graph."""
    net = Network(height=height, width="100%", bgcolor="#0f172a", font_color="#e2e8f0")
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=150)

    for node, data in G.nodes(data=True):
        comm = data.get("community", 0)
        if filter_community is not None and comm != filter_community:
            continue
        ntype = data.get("type", "unknown")
        color = TYPE_COLORS.get(ntype, "#6b7280")
        degree = G.degree(node)
        size = min(10 + degree * 3, 40)
        title = f"<b>{node}</b><br>Type: {ntype}<br>Community: {comm}<br>{data.get('description', '')}"
        net.add_node(node, label=node, color=color, size=size, title=title)

    for src, tgt, data in G.edges(data=True):
        if filter_community is not None:
            src_comm = G.nodes[src].get("community", -1)
            tgt_comm = G.nodes[tgt].get("community", -1)
            if src_comm != filter_community and tgt_comm != filter_community:
                continue
        if src in [n["id"] for n in net.nodes] and tgt in [n["id"] for n in net.nodes]:
            descs = data.get("descriptions", [""])
            weight = data.get("weight", 1)
            net.add_edge(src, tgt, title=descs[0] if descs else "", width=min(weight, 5))

    # Save to temp file and display
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as f:
        net.save_graph(f.name)
        return f.name


# ─── Load Data ───
try:
    G, stats, summaries = load_graph_data()
except FileNotFoundError:
    st.error("No output data found. Run `python pipeline.py` first.")
    st.stop()

# ─── Header ───
st.markdown("""
<div class="app-header">
    <div class="app-title">GraphRAG <span>Engine</span></div>
    <div class="app-subtitle">Graph-based Retrieval Augmented Generation — built from scratch</div>
</div>
""", unsafe_allow_html=True)

# ─── Metrics Row ───
st.markdown(f"""
<div class="metrics-grid">
    <div class="metric-card"><div class="metric-value">{stats["num_entities"]}</div><div class="metric-label">Entities</div></div>
    <div class="metric-card"><div class="metric-value">{stats["num_relationships"]}</div><div class="metric-label">Relationships</div></div>
    <div class="metric-card"><div class="metric-value">{stats["num_communities"]}</div><div class="metric-label">Communities</div></div>
    <div class="metric-card"><div class="metric-value">{stats["num_documents"]}</div><div class="metric-label">Documents</div></div>
</div>
""", unsafe_allow_html=True)


# ─── Tabs ───
tab1, tab2, tab3, tab4 = st.tabs(["Knowledge Graph", "Search", "Communities", "Statistics"])

# ─── Tab 1: Knowledge Graph ───
with tab1:
    col_filter, col_legend = st.columns([1, 3])
    with col_filter:
        comm_options = ["All"] + [f"Community {i}" for i in sorted(stats["communities"].keys(), key=int)]
        selected = st.selectbox("Filter by Community", comm_options)
        filter_comm = None if selected == "All" else int(selected.split()[-1])

    with col_legend:
        legend_items = " ".join([
            f'<span style="color:{color};font-size:12px;font-weight:500;">&#9679; {t.capitalize()}</span>'
            for t, color in TYPE_COLORS.items()
            if stats["entity_types"].get(t, 0) > 0
        ])
        st.markdown(f'<div style="padding-top:8px;display:flex;flex-wrap:wrap;gap:10px;">{legend_items}</div>', unsafe_allow_html=True)

    html_path = render_pyvis_graph(G, filter_community=filter_comm)
    with open(html_path, "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=620, scrolling=False)
    os.unlink(html_path)

# ─── Tab 2: Search ───
with tab2:
    st.markdown('<div style="font-size:1rem;font-weight:600;color:#e2e8f0;margin-bottom:16px;">Query the Knowledge Graph</div>', unsafe_allow_html=True)
    col_search, col_type = st.columns([3, 1])
    with col_search:
        query = st.text_input("question", placeholder="e.g. Who created the Transformer architecture?", label_visibility="collapsed")
    with col_type:
        search_type = st.radio("Mode", ["Local", "Global"], horizontal=True, label_visibility="collapsed")

    if query:
        api_key = os.getenv("MISTRAL_API_KEY", "")
        if not api_key:
            st.warning("Set MISTRAL_API_KEY in .env to enable search")
        else:
            with st.spinner(f"Running {search_type} Search..."):
                from src.search import local_search, global_search
                chunks = []
                for f in sorted(os.listdir("data/documents")):
                    if f.endswith(".txt"):
                        chunks.append(open(os.path.join("data/documents", f), encoding="utf-8").read())

                if search_type == "Local":
                    result = local_search(query, G, summaries, chunks, api_key)
                else:
                    result = global_search(query, summaries, api_key)

            try:
                if isinstance(result, str):
                    import json
                    res_json = json.loads(result)
                else:
                    res_json = result

                answer_text = res_json.get("answer", str(result))
                citations = res_json.get("cited_entities", [])

                citation_html = ""
                if citations:
                    rows = "".join(
                        f'<div class="citation-row"><span class="citation-entity">{c.get("entity","")}</span><span class="citation-detail">{c.get("details","")}</span></div>'
                        for c in citations
                    )
                    citation_html = f'<div class="citations-label">Cited Entities</div>{rows}'

                st.markdown(f"""
                <div class="answer-card">
                    <div class="answer-label">Answer</div>
                    <div>{answer_text}</div>
                    {citation_html}
                </div>
                """, unsafe_allow_html=True)

            except Exception:
                st.markdown(f'<div class="answer-card"><div class="answer-label">Answer</div>{result}</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-top:2rem;margin-bottom:0.5rem;font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:2px;color:#1e3a5f;">Search Modes</div>
    <div class="modes-row">
        <div class="mode-card"><div class="mode-title">Local Search</div><div class="mode-body">Traverses entity neighborhoods in the graph. Best for specific questions about people, tools, or algorithms.</div></div>
        <div class="mode-card"><div class="mode-title">Global Search</div><div class="mode-body">Synthesizes across community summaries. Best for thematic questions about trends and patterns.</div></div>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown('<div style="font-size:1rem;font-weight:600;color:#e2e8f0;margin-bottom:16px;">Discovered Communities</div>', unsafe_allow_html=True)
    for cid in sorted(summaries.keys()):
        members = stats["communities"].get(str(cid), [])
        with st.expander(f"Community {cid}  ·  {len(members)} members", expanded=(cid < 3)):
            st.markdown(f'<div style="font-size:.82rem;color:#64748b;line-height:1.7;margin-bottom:12px;">{summaries[cid]}</div>', unsafe_allow_html=True)
            member_tags = ""
            for m in members:
                ntype = G.nodes[m].get("type", "unknown") if m in G.nodes else "unknown"
                color = TYPE_COLORS.get(ntype, "#6b7280")
                member_tags += f'<span style="background:{color}18;color:{color};padding:3px 10px;border-radius:20px;margin:3px 3px 0 0;display:inline-block;font-size:.74rem;font-weight:500;border:1px solid {color}30;">{m}</span>'
            st.markdown(f'<div style="margin-top:4px;">{member_tags}</div>', unsafe_allow_html=True)

# ─── Tab 4: Statistics ───
with tab4:
    import pandas as pd
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div style="font-size:.82rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">Entity Type Distribution</div>', unsafe_allow_html=True)
        type_df = pd.DataFrame(
            [(t, c) for t, c in sorted(stats["entity_types"].items(), key=lambda x: -x[1])],
            columns=["Type", "Count"]
        )
        st.bar_chart(type_df.set_index("Type"))

    with col_b:
        st.markdown('<div style="font-size:.82rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">Most Connected Entities</div>', unsafe_allow_html=True)
        degrees = sorted(G.degree(), key=lambda x: -x[1])[:15]
        deg_df = pd.DataFrame(degrees, columns=["Entity", "Connections"])
        st.dataframe(deg_df, use_container_width=True, hide_index=True)

    st.markdown('<div style="height:1px;background:#141f30;margin:1.5rem 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:.82rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">Community Sizes</div>', unsafe_allow_html=True)
    comm_sizes = pd.DataFrame(
        [(f"Community {k}", len(v)) for k, v in sorted(stats["communities"].items(), key=lambda x: int(x[0]))],
        columns=["Community", "Members"]
    )
    st.bar_chart(comm_sizes.set_index("Community"))

# ─── Sidebar ───
with st.sidebar:

    # Header card
    st.markdown("""
    <div class="sb-header">
        <h2>GraphRAG Control Panel</h2>
        <p>Build and explore knowledge graphs from your documents</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload Section ──
    st.markdown('<div class="sb-section-label">Build New Graph</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["txt", "pdf", "docx", "md"],
        accept_multiple_files=True,
        help="Supports .txt, .pdf, .docx, and .md files.",
        label_visibility="collapsed"
    )

    if uploaded_files:
        for uf in uploaded_files:
            ext = uf.name.lower().rsplit(".", 1)[-1]
            icon = {"pdf": "[PDF]", "docx": "[DOCX]", "md": "[MD]"}.get(ext, "[TXT]")
            st.caption(f"{icon} {uf.name}  —  {round(uf.size/1024, 1)} KB")

        if st.button("Process & Build Graph", type="primary", use_container_width=True):
            import shutil
            import io

            docs_dir = "data/documents"
            if os.path.exists(docs_dir):
                shutil.rmtree(docs_dir)
            os.makedirs(docs_dir, exist_ok=True)

            progress = st.progress(0, text="Saving documents...")
            for i, uf in enumerate(uploaded_files):
                ext = uf.name.lower().rsplit(".", 1)[-1]
                base_name = os.path.splitext(uf.name)[0] + ".txt"
                out_path = os.path.join(docs_dir, base_name)

                try:
                    if ext == "pdf":
                        import pypdf
                        reader = pypdf.PdfReader(io.BytesIO(uf.read()))
                        text = "\n".join(page.extract_text() or "" for page in reader.pages)
                    elif ext == "docx":
                        import docx as docxlib
                        doc = docxlib.Document(io.BytesIO(uf.read()))
                        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
                    else:  # .txt and .md — read as plain text
                        text = uf.read().decode("utf-8", errors="ignore")

                    with open(out_path, "w", encoding="utf-8") as fout:
                        fout.write(text)
                except Exception as e:
                    st.error(f"Could not read {uf.name}: {e}")
                    continue

                progress.progress((i + 1) / len(uploaded_files), text=f"Saved {uf.name}")

            with st.status("Running GraphRAG Pipeline...", expanded=True) as status_box:
                try:
                    import os
                    api_key = os.getenv("MISTRAL_API_KEY", "")
                    if not api_key:
                        status_box.update(label="Missing MISTRAL_API_KEY in environment.", state="error")
                    else:
                        from pipeline import run_pipeline
                        import sys
                        import contextlib
                        
                        class StreamlitStatusWriter:
                            def __init__(self, container):
                                self.container = container
                                self.buffer = ""
                            def write(self, text):
                                if text.strip() and not text.startswith("=") and not text.isspace():
                                    self.container.write(text.strip())
                            def flush(self): pass
                            
                        with contextlib.redirect_stdout(StreamlitStatusWriter(status_box)):
                            run_pipeline(input_dir="data/documents", output_dir="output", api_key=api_key)
                            
                        load_graph_data.clear()
                        status_box.update(label="Graph built successfully! Reloading...", state="complete", expanded=False)
                        st.rerun()
                except Exception as e:
                    status_box.update(label=f"Pipeline error: {e}", state="error")

    # ── Optimization Card ──
    st.markdown('<div class="sb-section-label">Cost Optimization</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-card">
        <div class="sb-stat-row">
            <span class="sb-stat-label">Standard approach</span>
            <span class="sb-stat-value">15 LLM calls</span>
        </div>
        <div class="sb-stat-row">
            <span class="sb-stat-label">Our approach</span>
            <span class="sb-stat-value">1 batched call</span>
        </div>
        <div class="sb-badge">~93% cost reduction via batched prompt</div>
    </div>
    """, unsafe_allow_html=True)

    # ── About ──
    st.markdown('<div class="sb-section-label">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-about">
        From-scratch implementation of Microsoft's GraphRAG architecture.<br><br>
        Built with &nbsp;<strong style="color:#e2e8f0">Python · NetworkX · Gemini API · Streamlit</strong>
    </div>
    """, unsafe_allow_html=True)
