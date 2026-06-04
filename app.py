import streamlit as st
import pickle
import pandas as pd

# 1. Page Configuration
st.set_page_config(
    page_title="ByteBrain AI - Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Premium Professional CSS Styling Injection
st.markdown("""
    <style>
    /* Import modern typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Clean sidebar adjustments */
    .css-110z7d5, .stSidebar {
        background-color: #f8f9fa;
        border-right: 1px solid #e9ecef;
    }
    
    /* Custom Stylized Cards for metrics and verdicts */
    .metric-card {
        background: #ffffff;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #eaeaea;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #111827;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 14px;
        font-weight: 500;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    
    /* Status indicators */
    .verdict-box {
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 24px;
        font-weight: 500;
    }
    .verdict-real {
        background-color: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
    }
    .verdict-fake {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        color: #991b1b;
    }
    
    /* Smooth transition for side history items */
    .history-item {
        padding: 10px 12px;
        border-radius: 6px;
        margin-bottom: 8px;
        background-color: #ffffff;
        border: 1px solid #f3f4f6;
        font-size: 13px;
        color: #374151;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Session State Core Setup
if "history" not in st.session_state:
    st.session_state.history = []

if "current_news_input" not in st.session_state:
    st.session_state.current_news_input = ""

# 4. Safe Resource Cache Loading
@st.cache_resource
def load_assets():
    try:
        with open("model.pkl", "rb") as file:
            model = pickle.load(file)
        with open("vectorizer.pkl", "rb") as file:
            vectorizer = pickle.load(file)
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_assets()

# 5. Dashboard Sidebar Interface Design
with st.sidebar:
    st.markdown("<h2 style='color:#1e3a8a; font-weight:800; margin-bottom: 2px;'>🧠 ByteBrain AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6b7280; font-size:12px; margin-bottom:20px;'>Advanced Intelligence Layer</p>", unsafe_allow_html=True)
    
    page = st.radio(
        "Application Navigation",
        ["Dashboard Analytics", "Fake News Detector", "Historical Logs", "About Project"],
        index=1
    )
    
    st.markdown("<br><hr style='margin:10px 0; border-color:#eaeaea;'><br>", unsafe_allow_html=True)
    
    # ChatGPT-Style Sidebar Feed
    st.markdown("<p style='font-weight:600; font-size:12px; color:#374151; text-transform:uppercase; letter-spacing:0.05em;'>Recent Workspace Streams</p>", unsafe_allow_html=True)
    if not st.session_state.history:
        st.caption("Workspace history is currently empty.")
    else:
        for item in reversed(st.session_state.history[-4:]):
            color_dot = "🟢" if item["result"] == "Real News" else "🔴"
            trunc = item["news"][:22] + "..." if len(item["news"]) > 22 else item["news"]
            st.markdown(f"<div class='history-item'>{color_dot} {trunc}</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Reset All Sessions", use_container_width=True, type="secondary"):
            st.session_state.history = []
            st.session_state.current_news_input = ""
            st.rerun()

# 6. Core Application Pages Routing

# ----------------- PAGE 1: DETECTOR -----------------
if page == "Fake News Detector":
    st.title("📰 Fake News Detection System")
    st.write("Check whether a news article appears Real or Fake.")
    
    # Modern Input UI Workspace block
    with st.container():
        news_text = st.text_area(
            "Paste News Article Here",
            value=st.session_state.current_news_input,
            placeholder="Paste raw text string or entire article data here to evaluate standard variations...",
            height=200,
            label_visibility="visible"
        )
        
        act_col1, act_col2 = st.columns([1, 6])
        with act_col1:
            analyze_btn = st.button("Analyze News", type="primary", use_container_width=True)
        with act_col2:
            if st.button("Clear Text Window", type="secondary"):
                st.session_state.current_news_input = ""
                st.rerun()

    # Calculation logic processing
    if analyze_btn:
        if not news_text.strip():
            st.warning("Please enter some news text.")
        elif model is None or vectorizer is None:
            st.error("Engine failure: Asset dependency arrays 'model.pkl' or 'vectorizer.pkl' were not located in root.")
        else:
            with st.spinner("Processing vectorized elements against baseline clusters..."):
                # Run ML Prediction
                news_vector = vectorizer.transform([news_text])
                prediction = model.predict(news_vector)
                probability = model.predict_proba(news_vector)
                confidence = round(max(probability[0]) * 100, 2)
                
                result_str = "Real News" if prediction[0] == 1 else "Fake News"
                word_count = len(news_text.split())
                char_count = len(news_text)
                
                # Dynamic context assignment
                category = "General"
                l_text = news_text.lower()
                if any(x in l_text for x in ["govt", "government", "election", "policy", "senate", "law"]): category = "Politics"
                elif any(x in l_text for x in ["game", "player", "sports", "league", "championship", "cricket"]): category = "Sports"
                elif any(x in l_text for x in ["medical", "health", "clinical", "fda", "virus", "treatment", "hospital"]): category = "Health"
                elif any(x in l_text for x in ["crypto", "inflation", "market", "stocks", "revenue"]): category = "Finance"

                # Store metadata to global session state
                st.session_state.history.append({
                    "news": news_text, "result": result_str, "confidence": confidence,
                    "words": word_count, "chars": char_count, "category": category
                })

            st.markdown("<br><h3 style='font-weight:700; letter-spacing:-0.02em;'>Result</h3>", unsafe_allow_html=True)
            
            # Layout Response Panels professionally
            col_panel_left, col_panel_right = st.columns([4, 3])
            
            with col_panel_left:
                if result_str == "Real News":
                    st.markdown(f"""
                        <div class='verdict-box verdict-real'>
                            <h3 style='margin:0 0 4px 0; font-weight:700; color:#065f46;'>✅ Real News</h3>
                            The content structurally aligns with verified informational distribution characteristics.
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class='verdict-box verdict-fake'>
                            <h3 style='margin:0 0 4px 0; font-weight:700; color:#991b1b;'>❌ Fake News</h3>
                            Linguistic vectors show heavy deviation from conventional informational frameworks.
                        </div>
                    """, unsafe_allow_html=True)
                
                # Confidence Meter Container
                with st.container(border=True):
                    st.markdown(f"<p style='margin:0; font-size:14px; font-weight:500; color:#4b5563;'>Confidence Score: {confidence}%</p>", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.progress(int(confidence))
            
            with col_panel_right:
                # Metric display using premium custom card grid layout
                st.markdown("<h3 style='font-weight:700; margin-top:0;'>News Statistics</h3>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div class='metric-card' style='padding: 15px;'>
                        <div class='metric-label' style='font-size: 12px;'>Detected Category</div>
                        <div class='metric-value' style='font-size: 20px; color:#2563eb;'>🔹 {category}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                c_1, c_2 = st.columns(2)
                c_1.markdown(f"<div class='metric-card' style='padding: 15px;'><div class='metric-label' style='font-size: 12px;'>Words</div><div class='metric-value' style='font-size: 24px;'>{word_count}</div></div>", unsafe_allow_html=True)
                c_2.markdown(f"<div class='metric-card' style='padding: 15px;'><div class='metric-label' style='font-size: 12px;'>Characters</div><div class='metric-value' style='font-size: 24px;'>{char_count}</div></div>", unsafe_allow_html=True)


# ----------------- PAGE 2: ANALYTICAL DASHBOARD -----------------
elif page == "Dashboard Analytics":
    st.title("📊 Dashboard")
    st.write("Overview of current verification trends processed during this session.")
    
    total_scans = len(st.session_state.history)
    
    if total_scans == 0:
        st.info("System logs are clear. Once text blocks are processed in the Fake News Detector page, charts will render.")
    else:
        fake_c = sum(1 for x in st.session_state.history if x["result"] == "Fake News")
        real_c = sum(1 for x in st.session_state.history if x["result"] == "Real News")
        avg_c = round(sum(x["confidence"] for x in st.session_state.history) / total_scans, 1)
        
        # High Level KPI Grid Cards
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        kpi_col1.markdown(f"<div class='metric-card'><div class='metric-label'>Total Searches</div><div class='metric-value'>{total_scans}</div></div>", unsafe_allow_html=True)
        kpi_col2.markdown(f"<div class='metric-card'><div class='metric-label'>Fake News Detected</div><div class='metric-value' style='color:#dc2626;'>{fake_c}</div></div>", unsafe_allow_html=True)
        kpi_col3.markdown(f"<div class='metric-card'><div class='metric-label'>Real News Detected</div><div class='metric-value' style='color:#16a34a;'>{real_c}</div></div>", unsafe_allow_html=True)
        kpi_col4.markdown(f"<div class='metric-card'><div class='metric-label'>Mean Confidence</div><div class='metric-value' style='color:#2563eb;'>{avg_c}%</div></div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Streamlit Native Clean Graphic Display Chart area
        graph_col1, graph_col2 = st.columns([5, 2])
        with graph_col1:
            st.markdown("<h4 style='font-weight:600; margin-bottom:15px;'>Volume Classification Split Proportions</h4>", unsafe_allow_html=True)
            df_chart = pd.DataFrame({
                'Classification Label': ['Genuine Articles', 'Flagged Misinformation'],
                'Aggregate Matches': [real_c, fake_c]
            })
            st.bar_chart(data=df_chart, x='Classification Label', y='Aggregate Matches', color="#2563eb", use_container_width=True)
        with graph_col2:
            st.markdown("<h4 style='font-weight:600; margin-bottom:15px;'>Diagnostic Parameters</h4>", unsafe_allow_html=True)
            st.caption("Active models use computational frequency ratios (TF-IDF Matrices) mapping logic across multidimensional vectors to compute decision margins.")


# ----------------- PAGE 3: HISTORICAL RECORDS LOG -----------------
elif page == "Historical Logs":
    st.title("📜 Search History")
    st.write("Browse, audit, and reload previously processed articles.")
    
    if len(st.session_state.history) == 0:
        st.info("The history registry is clean. Run assertions inside the primary detector console.")
    else:
        for idx, item in enumerate(reversed(st.session_state.history), start=1):
            lbl_color = "🟢 Genuine" if item['result'] == "Real News" else "🔴 Flagged"
            
            with st.expander(f"Audit #{len(st.session_state.history) - idx + 1} — Result Vector: {lbl_color} ({item['confidence']}% certainty)"):
                st.markdown(f"**Extracted Classification Sub-Domain:** `{item['category']}` | **Words:** {item['words']} | **Characters:** {item['chars']}")
                st.markdown("<p style='font-size:12px; font-weight:600; color:#6b7280; margin-bottom:4px;'>RAW TEXT STREAM STRING:</p>", unsafe_allow_html=True)
                st.info(f"\"{item['news']}\"")
                
                # High-Fidelity UI Interaction Action
                if st.button("Reload parameters back to main scanner workspace", key=f"re_load_k_{idx}"):
                    st.session_state.current_news_input = item['news']
                    st.toast("Data payload queued successfully!", icon="📥")


# ----------------- PAGE 4: ABOUT PROJECT -----------------
elif page == "About Project":
    st.title("ℹ️ About Project")
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.write("""
        Fake News Detection System

        Technologies Used:
        - Python
        - Streamlit
        - Machine Learning
        - TF-IDF
        - Logistic Regression

        Developed as an AI Course Project.
        """)
    with col_right:
        with st.container(border=True):
            st.markdown("<h4 style='margin-top:0; font-weight:700;'>Infrastructure Matrix Blueprint</h4>", unsafe_allow_html=True)
            st.markdown("""
            * **Application Engine UI:** Streamlit Framework Core
            * **Compiler Platform:** Python Standard Runtime
            * **Vectorization Algorithm:** Term Frequency - Inverse Document Frequency
            * **Underlying Classifier Model:** Trained Probabilistic Estimator Model File
            * **Branding Stack Engine:** ByteBrain AI Ecosystem
            """)