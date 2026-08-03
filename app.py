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
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .metric-card {
        background: #1e293b; border-radius: 12px; padding: 20px;
        border: 1px solid #334155; text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #60a5fa; }
    .metric-label { font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b; border-radius: 8px; padding: 8px 16px;
        color: #e2e8f0; border: 1px solid #334155;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important; color: white !important;
    }
    /* ── Sidebar Styles ── */
    .sb-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #1e293b 100%);
        border-radius: 10px;
        padding: 14px 16px;
        border: 1px solid #2d4a6e;
        margin-bottom: 4px;
    }
    .sb-header h2 {
        font-size: 1rem; font-weight: 700; color: #e2e8f0;
        margin: 0; letter-spacing: 0.5px;
    }
    .sb-header p {
        font-size: 0.72rem; color: #64748b;
        margin: 2px 0 0 0; letter-spacing: 0.3px;
    }
    .sb-section-label {
        font-size: 0.68rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 1.8px;
        color: #475569; margin: 16px 0 8px 2px;
    }
    .sb-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 14px 16px;
        border: 1px solid #2d3748;
        margin-bottom: 8px;
    }
    .sb-stat-row {
        display: flex; justify-content: space-between;
        align-items: center; padding: 7px 0;
        border-bottom: 1px solid #263344;
        font-size: 0.82rem;
    }
    .sb-stat-row:last-of-type { border-bottom: none; }
    .sb-stat-label { color: #94a3b8; }
    .sb-stat-value { font-weight: 600; color: #f8fafc; }
    .sb-badge {
        margin-top: 10px;
        padding: 7px 12px;
        background: #0f2318;
        border-radius: 7px;
        border-left: 3px solid #22c55e;
        font-size: 0.78rem;
        color: #86efac;
        font-weight: 500;
    }
    .sb-about {
        padding: 10px 14px;
        background: #0f172a;
        border-radius: 8px;
        border: 1px solid #1e293b;
        font-size: 0.76rem;
        color: #64748b;
        line-height: 1.5;
    }
    /* ── File Uploader Overrides ── */
    [data-testid="stFileUploader"] {
        width: 100%;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: #0f172a !important;
        border: 2px dashed #334155 !important;
        border-radius: 10px !important;
        padding: 20px 12px !important;
        transition: border-color 0.2s ease, background 0.2s ease;
        text-align: center;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #60a5fa !important;
        background: #0d1f3c !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] > div > span {
        font-size: 0.78rem !important;
        color: #94a3b8 !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] > div > small {
        font-size: 0.7rem !important;
        color: #475569 !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] svg {
        fill: #3b82f6 !important;
        width: 28px !important;
        height: 28px !important;
    }
    [data-testid="stFileUploader"] section > button {
        background: #1e3a5f !important;
        border: 1px solid #2d4a6e !important;
        color: #93c5fd !important;
        border-radius: 8px !important;
        font-size: 0.78rem !important;
        padding: 4px 14px !important;
    }
    [data-testid="stFileUploader"] section > button:hover {
        background: #1e40af !important;
        border-color: #60a5fa !important;
        color: #fff !important;
    }
    /* Uploaded file item */
    [data-testid="stFileUploaderFile"] {
        background: #1e293b !important;
        border-radius: 8px !important;
        border: 1px solid #334155 !important;
        padding: 6px 10px !important;
        margin-top: 6px !important;
    }
    [data-testid="stFileUploaderFileName"] {
        color: #e2e8f0 !important;
        font-size: 0.78rem !important;
    }
    [data-testid="stFileUploaderFileSize"] {
        color: #64748b !important;
        font-size: 0.72rem !important;
    }
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
st.markdown('<p class="main-header">GraphRAG Engine</p>', unsafe_allow_html=True)
st.caption("From-scratch Graph-based Retrieval Augmented Generation")

# ─── Metrics Row ───
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{stats["num_entities"]}</div><div class="metric-label">Entities</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{stats["num_relationships"]}</div><div class="metric-label">Relationships</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{stats["num_communities"]}</div><div class="metric-label">Communities</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{stats["num_documents"]}</div><div class="metric-label">Documents</div></div>', unsafe_allow_html=True)

st.divider()

# ─── Tabs ───
tab1, tab2, tab3, tab4 = st.tabs(["🕸️ Knowledge Graph", "🔍 Search", "🏘️ Communities", "📊 Statistics"])

# ─── Tab 1: Knowledge Graph ───
with tab1:
    col_filter, col_legend = st.columns([1, 3])
    with col_filter:
        comm_options = ["All"] + [f"Community {i}" for i in sorted(stats["communities"].keys(), key=int)]
        selected = st.selectbox("Filter by Community", comm_options)
        filter_comm = None if selected == "All" else int(selected.split()[-1])

    with col_legend:
        legend_items = " ".join([
            f'<span style="color:{color};font-size:13px;">● {t.capitalize()}</span>'
            for t, color in TYPE_COLORS.items()
            if stats["entity_types"].get(t, 0) > 0
        ])
        st.markdown(f"**Legend:** {legend_items}", unsafe_allow_html=True)

    html_path = render_pyvis_graph(G, filter_community=filter_comm)
    with open(html_path, "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=620, scrolling=False)
    os.unlink(html_path)

# ─── Tab 2: Search ───
with tab2:
    st.markdown("### Query the Knowledge Graph")
    col_search, col_type = st.columns([3, 1])
    with col_search:
        query = st.text_input("Enter your question", placeholder="e.g. Who created the Transformer architecture?")
    with col_type:
        search_type = st.radio("Search Mode", ["Local", "Global"], horizontal=True)

    if query:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            st.warning("Set GEMINI_API_KEY in .env to enable search")
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

            st.markdown("### Answer")
            st.markdown(result)

    st.divider()
    st.markdown("**How search works:**")
    col_l, col_g = st.columns(2)
    with col_l:
        st.info("**Local Search** traverses entity neighborhoods in the graph. Best for specific questions about people, tools, or algorithms.")
    with col_g:
        st.info("**Global Search** synthesizes across community summaries. Best for thematic questions about trends and patterns.")

# ─── Tab 3: Communities ───
with tab3:
    st.markdown("### Discovered Communities")
    for cid in sorted(summaries.keys()):
        members = stats["communities"].get(str(cid), [])
        with st.expander(f"Community {cid} — {len(members)} members", expanded=(cid < 3)):
            st.markdown(f"**Summary:** {summaries[cid]}")
            st.markdown("**Members:**")
            member_tags = ""
            for m in members:
                ntype = G.nodes[m].get("type", "unknown") if m in G.nodes else "unknown"
                color = TYPE_COLORS.get(ntype, "#6b7280")
                member_tags += f'<span style="background:{color}22;color:{color};padding:2px 8px;border-radius:12px;margin:2px;display:inline-block;font-size:13px;">{m}</span> '
            st.markdown(member_tags, unsafe_allow_html=True)

# ─── Tab 4: Statistics ───
with tab4:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Entity Type Distribution")
        import pandas as pd
        type_df = pd.DataFrame(
            [(t, c) for t, c in sorted(stats["entity_types"].items(), key=lambda x: -x[1])],
            columns=["Type", "Count"]
        )
        st.bar_chart(type_df.set_index("Type"))

    with col_b:
        st.markdown("### Most Connected Entities")
        degrees = sorted(G.degree(), key=lambda x: -x[1])[:15]
        deg_df = pd.DataFrame(degrees, columns=["Entity", "Connections"])
        st.dataframe(deg_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Community Sizes")
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
        <h2>⚙️ GraphRAG Control Panel</h2>
        <p>Build and explore knowledge graphs from your documents</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload Section ──
    st.markdown('<div class="sb-section-label">📂 Build New Graph</div>', unsafe_allow_html=True)
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
            icon = {"pdf": "📑", "docx": "📝", "md": "📌"}.get(ext, "📄")
            st.caption(f"{icon} {uf.name}  —  {round(uf.size/1024, 1)} KB")

        if st.button("🚀 Process & Build Graph", type="primary", use_container_width=True):
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

            with st.spinner("🔄 Running pipeline — ~30–60 seconds..."):
                try:
                    import os
                    api_key = os.getenv("GEMINI_API_KEY", "")
                    if not api_key:
                        st.error("Missing GEMINI_API_KEY in environment.")
                    else:
                        from pipeline import run_pipeline
                        run_pipeline(input_dir="data/documents", output_dir="output", api_key=api_key)
                        load_graph_data.clear()
                        st.success("✅ Done! Explore the tabs above.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Pipeline error: {e}")

    # ── Optimization Card ──
    st.markdown('<div class="sb-section-label">⚡ Cost Optimization</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-card">
        <div class="sb-stat-row">
            <span class="sb-stat-label">🔴 Standard approach</span>
            <span class="sb-stat-value">15 LLM calls</span>
        </div>
        <div class="sb-stat-row">
            <span class="sb-stat-label">🟢 Our approach</span>
            <span class="sb-stat-value">1 batched call</span>
        </div>
        <div class="sb-badge">↓ ~93% cost reduction via batched prompt</div>
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
