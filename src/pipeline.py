"""
E2A PDF Translation Pipeline.

Orchestrates the full workflow:
  Input PDF → Extract → Translate → Render → Output PDF

Provides both programmatic API and CLI-friendly interface.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .extractor import PDFExtractor, ExtractionError
from .translator import Translator, TranslatorConfig, TranslationError
from .renderer import PDFRenderer, RendererConfig, RenderError, FontConfig
from .models import DocumentContent, TranslatedDocument

logger = logging.getLogger(__name__)


# ===========================================================================
# Pipeline Configuration
# ===========================================================================

@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    # Translation settings
    translation_backend: str = "mock"
    api_key: Optional[str] = None
    model: Optional[str] = None
    source_lang: str = "en"
    target_lang: str = "ar"
    cache_path: Optional[str] = ".e2a_cache/translations.json"

    # Extraction settings
    merge_paragraphs: bool = True
    min_text_length: int = 2

    # Rendering settings
    font_path: Optional[str] = None
    mirror_layout: bool = True
    preserve_positions: bool = False  # Flowing A4 layout by default
    body_font_size: float = 13.0
    add_page_numbers: bool = True
    line_spacing: float = 1.6
    margin: float = 50.0

    # Pipeline behavior
    continue_on_error: bool = True
    verbose: bool = False


# ===========================================================================
# Pipeline Report
# ===========================================================================

@dataclass
class PipelineReport:
    """Report of a translation pipeline run."""
    input_path: str = ""
    output_path: str = ""
    success: bool = False
    total_pages: int = 0
    translated_pages: int = 0
    total_blocks: int = 0
    translated_blocks: int = 0
    cached_blocks: int = 0
    skipped_blocks: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "E2A PDF Translation Report",
            "=" * 60,
            f"Input:            {self.input_path}",
            f"Output:           {self.output_path}",
            f"Status:           {'SUCCESS' if self.success else 'FAILED'}",
            f"Duration:         {self.duration_seconds:.1f}s",
            f"Pages:            {self.translated_pages}/{self.total_pages}",
            f"Text blocks:      {self.translated_blocks} translated, "
            f"{self.cached_blocks} from cache, "
            f"{self.skipped_blocks} skipped",
        ]
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            for w in self.warnings[:10]:
                lines.append(f"  - {w}")
            if len(self.warnings) > 10:
                lines.append(f"  ... and {len(self.warnings) - 10} more")
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            for e in self.errors[:10]:
                lines.append(f"  - {e}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ===========================================================================
# Main Pipeline
# ===========================================================================

class E2APipeline:
    """
    Main translation pipeline.

    Usage:
        config = PipelineConfig(translation_backend="mock")
        pipeline = E2APipeline(config)
        report = pipeline.translate("input.pdf", "output_ar.pdf")
        print(report.summary())
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._setup_logging()

    def _setup_logging(self):
        level = logging.DEBUG if self.config.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    def translate(
        self,
        input_path: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> PipelineReport:
        """
        Run the full translation pipeline.

        Args:
            input_path: Path to the English PDF.
            output_path: Path for the Arabic PDF output.
                         Defaults to input_name_ar.pdf.

        Returns:
            PipelineReport with statistics and any warnings/errors.
        """
        start_time = time.time()
        input_path = Path(input_path)

        if output_path is None:
            output_path = input_path.with_name(
                f"{input_path.stem}_ar{input_path.suffix}"
            )
        else:
            output_path = Path(output_path)

        report = PipelineReport(
            input_path=str(input_path),
            output_path=str(output_path),
        )

        # ---- Step 1: Extract ----
        logger.info(f"Step 1/3: Extracting content from {input_path.name}")
        try:
            extractor = PDFExtractor(
                merge_paragraphs=self.config.merge_paragraphs,
                min_text_length=self.config.min_text_length,
            )
            doc = extractor.extract(input_path)
            report.total_pages = len(doc.pages)
            report.warnings.extend(doc.extraction_warnings)

            if doc.is_scanned:
                report.warnings.append(
                    "Document appears to be scanned. "
                    "Text extraction may be incomplete."
                )

            total_blocks = sum(len(p.text_blocks) for p in doc.pages)
            logger.info(
                f"  Extracted {len(doc.pages)} pages, "
                f"{total_blocks} text blocks, "
                f"{sum(len(p.image_blocks) for p in doc.pages)} images"
            )

        except ExtractionError as e:
            report.errors.append(f"Extraction failed: {e}")
            report.duration_seconds = time.time() - start_time
            return report

        # ---- Step 2: Translate ----
        logger.info(f"Step 2/3: Translating ({self.config.translation_backend})")
        try:
            translator_config = TranslatorConfig(
                backend=self.config.translation_backend,
                api_key=self.config.api_key,
                model=self.config.model,
                cache_path=self.config.cache_path,
                source_lang=self.config.source_lang,
                target_lang=self.config.target_lang,
            )
            translator = Translator(translator_config)
            translated_doc = translator.translate_document(doc)

            report.warnings.extend(translated_doc.warnings)
            stats = translated_doc.stats
            report.total_blocks = stats.get("total_blocks", 0)
            report.translated_blocks = stats.get("translated", 0)
            report.cached_blocks = stats.get("cached", 0)
            report.skipped_blocks = stats.get("skipped", 0)

            logger.info(
                f"  Translated: {report.translated_blocks}, "
                f"Cached: {report.cached_blocks}, "
                f"Skipped: {report.skipped_blocks}"
            )

        except TranslationError as e:
            report.errors.append(f"Translation failed: {e}")
            report.duration_seconds = time.time() - start_time
            if not self.config.continue_on_error:
                return report
            # Create a minimal translated doc with original text
            translated_doc = TranslatedDocument(source_path=str(input_path))

        # ---- Step 3: Render ----
        logger.info(f"Step 3/3: Rendering Arabic PDF to {output_path.name}")
        try:
            font_config = None
            if self.config.font_path:
                font_config = FontConfig(regular=self.config.font_path)
            else:
                font_config = FontConfig().auto_detect()

            renderer_config = RendererConfig(
                font_config=font_config,
                mirror_layout=self.config.mirror_layout,
                preserve_positions=self.config.preserve_positions,
                body_font_size=self.config.body_font_size,
                add_page_numbers=self.config.add_page_numbers,
                line_spacing=self.config.line_spacing,
                margin_top=self.config.margin,
                margin_bottom=self.config.margin,
                margin_left=self.config.margin,
                margin_right=self.config.margin,
            )
            renderer = PDFRenderer(renderer_config)
            renderer.render(translated_doc, output_path)

            report.translated_pages = len(translated_doc.pages)
            report.success = True
            logger.info(f"  Output: {output_path}")

        except RenderError as e:
            report.errors.append(f"Rendering failed: {e}")

        report.duration_seconds = time.time() - start_time
        return report


# ===========================================================================
# Convenience function
# ===========================================================================

def translate_pdf(
    input_path: str,
    output_path: Optional[str] = None,
    backend: str = "mock",
    api_key: Optional[str] = None,
    **kwargs,
) -> PipelineReport:
    """
    Convenience function for quick translation.

    Args:
        input_path: Path to English PDF.
        output_path: Path for Arabic PDF (optional).
        backend: Translation backend ("mock", "google", "deepl", "llm-openai").
        api_key: API key for the translation service.
        **kwargs: Additional PipelineConfig options.

    Returns:
        PipelineReport with results.

    Example:
        report = translate_pdf("paper.pdf", backend="google", api_key="...")
        print(report.summary())
    """
    config = PipelineConfig(
        translation_backend=backend,
        api_key=api_key,
        **kwargs,
    )
    pipeline = E2APipeline(config)
    return pipeline.translate(input_path, output_path)
