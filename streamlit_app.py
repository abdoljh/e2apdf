"""
E2A PDF Translator — Streamlit App

Translate English PDF files to Arabic with full RTL support.
Supports mock (testing), Google Translate, DeepL, OpenAI, and Anthropic backends.
"""

import io
import os
import sys
import tempfile
import logging
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="E2A PDF Translator",
    page_icon="📄",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Title & description
# ---------------------------------------------------------------------------
st.title("📄 E2A PDF Translator")
st.markdown(
    "Translate **English PDF** files into **Arabic** with full RTL layout support, "
    "correct letter shaping, and Bidi reordering."
)
st.divider()

# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    backend = st.selectbox(
        "Translation backend",
        options=["mock", "google", "deepl", "llm-openai", "llm-anthropic"],
        format_func=lambda x: {
            "mock": "Mock (testing — no API key needed)",
            "google": "Google Cloud Translation",
            "deepl": "DeepL",
            "llm-openai": "OpenAI GPT",
            "llm-anthropic": "Anthropic Claude",
        }[x],
        help="Choose the translation service.",
    )

    api_key = ""
    model_name = ""

    if backend != "mock":
        env_var_hints = {
            "google": "GOOGLE_TRANSLATE_API_KEY",
            "deepl": "DEEPL_API_KEY",
            "llm-openai": "OPENAI_API_KEY",
            "llm-anthropic": "ANTHROPIC_API_KEY",
        }
        env_hint = env_var_hints.get(backend, "API_KEY")
        # Pre-fill from environment if available
        default_key = os.environ.get(env_hint, "")
        api_key = st.text_input(
            "API Key",
            value=default_key,
            type="password",
            help=f"Your API key (or set the `{env_hint}` environment variable).",
        )

    if backend in ("llm-openai", "llm-anthropic"):
        model_name = st.text_input(
            "Model name (optional)",
            value="",
            help="Override the default model, e.g. `claude-sonnet-4-6` or `gpt-4o`.",
        )

    st.subheader("Layout options")

    mirror_layout = st.checkbox("Mirror layout for RTL", value=True)
    preserve_positions = st.checkbox("Preserve original positions", value=True)
    add_page_numbers = st.checkbox("Add Arabic page numbers", value=True)

    line_spacing = st.slider(
        "Line spacing", min_value=1.0, max_value=2.5, value=1.4, step=0.1
    )
    margin = st.slider(
        "Page margin (pts)", min_value=20, max_value=100, value=50, step=5
    )

    st.subheader("Cache")
    use_cache = st.checkbox(
        "Enable translation cache",
        value=True,
        help="Cache translations to avoid re-calling the API for identical text.",
    )

# ---------------------------------------------------------------------------
# Main area — file upload
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload an English PDF",
    type=["pdf"],
    help="Upload the PDF you want translated to Arabic.",
)

if uploaded_file is not None:
    st.success(f"File loaded: **{uploaded_file.name}** ({uploaded_file.size:,} bytes)")

    if st.button("Translate to Arabic", type="primary", use_container_width=True):
        if backend != "mock" and not api_key.strip():
            st.error(
                f"Please enter your API key for the **{backend}** backend "
                "(or set the corresponding environment variable)."
            )
            st.stop()

        # ------------------------------------------------------------------
        # Run the pipeline
        # ------------------------------------------------------------------
        progress = st.progress(0, text="Starting translation pipeline…")
        status_area = st.empty()
        log_expander = st.expander("Pipeline log", expanded=False)
        log_box = log_expander.empty()
        log_lines: list[str] = []

        # Capture log output so we can show it in the UI
        class StreamlitLogHandler(logging.Handler):
            def emit(self, record: logging.LogRecord):
                msg = self.format(record)
                log_lines.append(msg)
                log_box.code("\n".join(log_lines[-40:]), language=None)

        root_logger = logging.getLogger()
        ui_handler = StreamlitLogHandler()
        ui_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
        root_logger.addHandler(ui_handler)
        root_logger.setLevel(logging.INFO)

        try:
            # Write uploaded PDF to a temp file
            with tempfile.TemporaryDirectory() as tmpdir:
                input_path = Path(tmpdir) / uploaded_file.name
                output_path = Path(tmpdir) / (Path(uploaded_file.name).stem + "_ar.pdf")
                cache_path = (
                    str(Path(tmpdir) / "translations.json") if use_cache else None
                )

                input_path.write_bytes(uploaded_file.getvalue())

                progress.progress(10, text="Importing translation engine…")
                status_area.info("Importing modules…")

                # Import here so Streamlit startup isn't blocked by heavy deps
                try:
                    from src.pipeline import E2APipeline, PipelineConfig
                except ImportError as exc:
                    st.error(
                        f"Failed to import the translation engine: {exc}\n\n"
                        "Make sure all dependencies are installed:\n"
                        "```\npip install pypdfium2 pypdf reportlab arabic-reshaper python-bidi requests\n```"
                    )
                    st.stop()

                progress.progress(20, text="Configuring pipeline…")
                status_area.info("Configuring pipeline…")

                config = PipelineConfig(
                    translation_backend=backend,
                    api_key=api_key.strip() or None,
                    model=model_name.strip() or None,
                    cache_path=cache_path,
                    mirror_layout=mirror_layout,
                    preserve_positions=preserve_positions,
                    add_page_numbers=add_page_numbers,
                    line_spacing=line_spacing,
                    margin=float(margin),
                    verbose=False,
                )

                progress.progress(30, text="Extracting text from PDF…")
                status_area.info("Step 1/3 — Extracting content…")

                pipeline = E2APipeline(config)

                # We monkey-patch the pipeline steps to update the progress bar
                # by running translate() directly and watching log output.
                # Since the pipeline is synchronous, we just run it and update after.

                progress.progress(40, text="Translating content…")
                status_area.info("Step 2/3 — Translating…  (this may take a while)")

                report = pipeline.translate(str(input_path), str(output_path))

                progress.progress(90, text="Finalising output PDF…")
                status_area.info("Step 3/3 — Rendering Arabic PDF…")

                # ----------------------------------------------------------
                # Show report
                # ----------------------------------------------------------
                progress.progress(100, text="Done!")
                status_area.empty()

                if report.success:
                    st.success("Translation complete!")

                    # Stats
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Pages", f"{report.translated_pages}/{report.total_pages}")
                    col2.metric("Translated blocks", report.translated_blocks)
                    col3.metric("From cache", report.cached_blocks)
                    col4.metric("Skipped", report.skipped_blocks)

                    st.metric("Duration", f"{report.duration_seconds:.1f} s")

                    # Download button
                    translated_bytes = output_path.read_bytes()
                    st.download_button(
                        label="Download Arabic PDF",
                        data=translated_bytes,
                        file_name=output_path.name,
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True,
                    )
                else:
                    st.error("Translation failed.")

                # Warnings & errors
                if report.warnings:
                    with st.expander(f"Warnings ({len(report.warnings)})", expanded=False):
                        for w in report.warnings:
                            st.warning(w)

                if report.errors:
                    with st.expander(f"Errors ({len(report.errors)})", expanded=True):
                        for e in report.errors:
                            st.error(e)

                # Full text report
                with st.expander("Full pipeline report", expanded=False):
                    st.code(report.summary(), language=None)

        except Exception as exc:
            progress.empty()
            status_area.empty()
            st.error(f"Unexpected error: {exc}")
            st.exception(exc)
        finally:
            root_logger.removeHandler(ui_handler)

else:
    st.info("Upload a PDF file above to get started.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Backends: **mock** (free, testing) · **Google** · **DeepL** · **OpenAI** · **Anthropic**  |  "
    "Source: [abdoljh/e2apdf](https://github.com/abdoljh/e2apdf)"
)
