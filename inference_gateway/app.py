import hashlib
import time
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import streamlit as st
from engine.pipeline import InferencePipeline, FastMLBackend

st.set_page_config(
    page_title="Low-Latency ML Inference Gateway",
    page_icon="⚡",
    layout="wide",
)

if "cache_capacity" not in st.session_state:
    st.session_state.cache_capacity = 5

if "pipeline" not in st.session_state or st.session_state.pipeline.cache.capacity != st.session_state.cache_capacity:
    if "pipeline" in st.session_state:
        st.session_state.pipeline.stop()
    st.session_state.pipeline = InferencePipeline(
        model=FastMLBackend(),
        cache_capacity=st.session_state.cache_capacity,
        max_batch_size=8,
        batch_timeout_ms=10.0,
    )

if "history" not in st.session_state:
    st.session_state.history = []

pipeline: InferencePipeline = st.session_state.pipeline

st.title("⚡ Low-Latency ML Inference Gateway")
st.markdown("Sub-millisecond inference gateway pairing a custom thread-safe LRU Cache with dynamic request batching.")

with st.sidebar:
    st.header("Gateway Configuration")
    new_capacity = st.slider("LRU Cache Capacity", min_value=2, max_value=20, value=st.session_state.cache_capacity)
    if new_capacity != st.session_state.cache_capacity:
        st.session_state.cache_capacity = new_capacity
        st.session_state.pipeline.stop()
        st.session_state.pipeline = InferencePipeline(
            model=FastMLBackend(),
            cache_capacity=new_capacity,
            max_batch_size=8,
            batch_timeout_ms=10.0,
        )
        st.session_state.history = []
        st.rerun()

    st.markdown("---")
    st.subheader("Manual Cache Controls")
    if st.button("Trigger Eviction (Overflow Cache)", use_container_width=True):
        overflow_samples = [
            f"Automated test prompt #{i}: system latency benchmark check"
            for i in range(st.session_state.cache_capacity + 2)
        ]
        for prompt in overflow_samples:
            res = st.session_state.pipeline.predict(prompt)
            st.session_state.history.append(res)
        st.success(f"Dispatched {len(overflow_samples)} queries. Oldest entries evicted.")
        st.rerun()

    if st.button("Clear Cache & History", use_container_width=True):
        st.session_state.pipeline.cache.clear()
        st.session_state.pipeline.overall_metrics.clear()
        st.session_state.pipeline.hit_metrics.clear()
        st.session_state.pipeline.miss_metrics.clear()
        st.session_state.history = []
        st.rerun()

cache_stats = pipeline.cache.stats()
pcts = pipeline.overall_metrics.compute_percentiles([50.0, 95.0, 99.0])

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Cache Entries", f"{cache_stats['size']} / {cache_stats['capacity']}")
with col2:
    st.metric("Cache Hit Ratio", f"{cache_stats['hit_ratio'] * 100:.1f}%")
with col3:
    st.metric("Total Evictions", str(cache_stats["evictions"]))
with col4:
    st.metric("p50 Latency", f"{pcts.get('p50', 0.0):.3f} ms")
with col5:
    st.metric("p95 Latency", f"{pcts.get('p95', 0.0):.3f} ms")

st.markdown("---")

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Inference Request")
    sample_queries = [
        "Select a sample query...",
        "This product is amazing and incredibly fast!",
        "Terrible experience, slow performance and buggy behavior.",
        "Clean implementation with very smooth execution.",
        "Horrible lag and frustrating broken interface.",
    ]
    selected_sample = st.selectbox("Sample Inputs", sample_queries)

    default_text = selected_sample if selected_sample != sample_queries[0] else ""
    user_input = st.text_area("Input Text", value=default_text, placeholder="Type sentence for sentiment classification...")

    run_btn = st.button("Submit Inference", type="primary", use_container_width=True)

    if run_btn and user_input.strip():
        result = pipeline.predict(user_input.strip())
        st.session_state.history.append(result)

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            status_text = "HIT (Fast Path)" if result.cached else "MISS (Slow Path)"
            st.metric("Cache Status", status_text)
        with res_col2:
            st.metric("Latency", f"{result.latency_ms:.4f} ms")
        with res_col3:
            st.metric("Prediction", f"{result.label} ({result.score:.2%})")

with right_col:
    st.subheader("LRU Doubly Linked List State")
    ordered_items = pipeline.cache.get_items_ordered()

    if not ordered_items:
        st.info("Cache is currently empty. Run an inference query to populate.")
    else:
        st.caption("MRU (Head) ➔ LRU (Tail, Next for Eviction)")
        for idx, (h_key, payload) in enumerate(ordered_items):
            is_mru = idx == 0
            is_lru = idx == len(ordered_items) - 1
            tag = " [MRU Head]" if is_mru else (" [LRU Tail - Will Evict]" if is_lru else "")
            with st.container():
                st.code(
                    f"Node #{idx + 1}{tag}\n"
                    f"├── Key:   {h_key[:16]}...\n"
                    f"├── Label: {payload.get('label')}\n"
                    f"└── Score: {payload.get('score'):.4f}",
                    language="text",
                )

st.markdown("---")

st.subheader("Recent Request History")
if st.session_state.history:
    history_data = []
    for r in reversed(st.session_state.history[-10:]):
        history_data.append({
            "Query": r.text[:60] + "..." if len(r.text) > 60 else r.text,
            "Cached": "⚡ HIT" if r.cached else "🔄 MISS",
            "Sentiment": r.label,
            "Confidence": f"{r.score:.2%}",
            "Latency (ms)": f"{r.latency_ms:.4f}",
        })
    st.dataframe(history_data, use_container_width=True)
else:
    st.text("No requests processed yet.")
