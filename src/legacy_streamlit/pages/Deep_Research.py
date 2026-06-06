import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import asyncio
import json
import os
from pathlib import Path as PathLib

from st_copy import copy_button
import streamlit as st

from streamlit_helper import model_selector
from config import OLLAMA_BASE_URL
from lib.research_config import build_research_config

DEPTH_DEFAULT = 2
BREADTH_DEFAULT = 4

CONFIG_PATH = PathLib(__file__).resolve().parent.parent.parent.parent / ".gpt-researcher-config.json"


def _get_event_loop() -> asyncio.AbstractEventLoop:
    if "dr_event_loop" not in st.session_state:
        loop = asyncio.new_event_loop()
        st.session_state.dr_event_loop = loop
    return st.session_state.dr_event_loop


def _to_litellm_model(raw: str) -> str:
    return f"ollama_chat/{raw.replace('ollama/', '')}"


def _write_config(fast: str, smart: str, strategic: str, depth: int, breadth: int, reasoning: str) -> str:
    config = build_research_config(
        fast_model=fast,
        smart_model=smart,
        strategic_model=strategic,
        depth=depth,
        breadth=breadth,
        reasoning_effort=reasoning,
    )
    with CONFIG_PATH.open("w") as f:
        json.dump(config, f)
    return str(CONFIG_PATH)


def _apply_runtime_env(reasoning: str) -> None:
    os.environ["OLLAMA_API_BASE"] = OLLAMA_BASE_URL
    os.environ["OLLAMA_BASE_URL"] = OLLAMA_BASE_URL
    if reasoning != "none":
        os.environ["REASONING_EFFORT"] = reasoning


async def _run_research(query: str, report_type: str, config_path: str) -> tuple[str, list[str], float]:
    from gpt_researcher import GPTResearcher

    researcher = GPTResearcher(query=query, report_type=report_type, config_path=config_path)
    await researcher.conduct_research()
    report = await researcher.write_report()
    sources = researcher.get_source_urls()
    costs = researcher.get_costs()
    return report, sources, costs


def deep_research_sidebar() -> None:
    with st.sidebar:
        with st.expander("Models", expanded=False):
            st.caption("Summarization, sub-queries, scraping")
            fast_raw = model_selector(key="dr_fast")
            st.markdown("---")
            st.caption("Report generation, main reasoning")
            smart_raw = model_selector(key="dr_smart")
            st.markdown("---")
            st.caption("Planning, research strategy")
            strategic_raw = model_selector(key="dr_strategic")

        fast = _to_litellm_model(fast_raw)
        smart = _to_litellm_model(smart_raw)
        strategic = _to_litellm_model(strategic_raw)

        st.markdown("---")
        with st.expander("Research Parameters", expanded=True):
            reasoning = st.selectbox(
                "Reasoning effort",
                options=["none", "low", "medium", "high"],
                index=2,
                help="LLM reasoning depth. 'none' omits the parameter.",
            )
            depth = st.slider(
                "Depth",
                min_value=1,
                max_value=4,
                value=DEPTH_DEFAULT,
                help="How many levels deep to explore.",
            )
            breadth = st.slider(
                "Breadth",
                min_value=2,
                max_value=6,
                value=BREADTH_DEFAULT,
                help="Parallel research paths per level.",
            )
            report_type = st.radio(
                "Report type",
                options=["deep", "research_report"],
                format_func=lambda x: "Deep (recursive)" if x == "deep" else "Standard (single-pass)",
            )

    st.session_state.dr_fast = fast
    st.session_state.dr_smart = smart
    st.session_state.dr_strategic = strategic
    st.session_state.dr_reasoning = reasoning
    st.session_state.dr_depth = depth
    st.session_state.dr_breadth = breadth
    st.session_state.dr_report_type = report_type


def _init_history() -> None:
    if "dr_queries" not in st.session_state:
        st.session_state.dr_queries = []
        st.session_state.dr_reports = []
        st.session_state.dr_sources_list = []
        st.session_state.dr_costs_list = []


def deep_research_main() -> None:
    _init_history()

    col, _ = st.columns([0.9, 0.1])
    with col:
        st.write("")
        messages_container = st.container()

    with messages_container:
        for _i, (query, report, sources, cost) in enumerate(
            zip(
                st.session_state.dr_queries,
                st.session_state.dr_reports,
                st.session_state.dr_sources_list,
                st.session_state.dr_costs_list,
                strict=False,
            )
        ):
            with st.chat_message("user"):
                st.markdown(query)
                copy_button(query)
            with st.chat_message("assistant"):
                st.markdown(report)
                copy_button(report)
                with st.expander("Sources & cost"):
                    st.caption(f"Estimated cost: ${cost:.4f}")
                    for url in sources:
                        st.markdown(f"- {url}")

    with st._bottom:
        query = st.chat_input("Research query...", key="dr_query")

    if query:
        config_path = _write_config(
            fast=st.session_state.dr_fast,
            smart=st.session_state.dr_smart,
            strategic=st.session_state.dr_strategic,
            depth=st.session_state.dr_depth,
            breadth=st.session_state.dr_breadth,
            reasoning=st.session_state.dr_reasoning,
        )
        _apply_runtime_env(reasoning=st.session_state.dr_reasoning)

        with st.chat_message("user"):
            st.markdown(query)
            copy_button(query)

        with st.chat_message("assistant"):
            with st.spinner("Researching..."):
                try:
                    loop = _get_event_loop()
                    report, sources, costs = loop.run_until_complete(
                        _run_research(query, st.session_state.dr_report_type, config_path)
                    )
                except Exception as e:
                    st.error(f"Research failed: {e}")
                    return

            st.markdown(report)
            copy_button(report)
            with st.expander("Sources & cost"):
                st.caption(f"Estimated cost: ${costs:.4f}")
                for url in sources:
                    st.markdown(f"- {url}")

        st.session_state.dr_queries.append(query)
        st.session_state.dr_reports.append(report)
        st.session_state.dr_sources_list.append(sources)
        st.session_state.dr_costs_list.append(costs)


if __name__ == "__main__":
    st.set_page_config(page_title="Deep Research", page_icon=":mag:", layout="wide")
    deep_research_sidebar()
    deep_research_main()
