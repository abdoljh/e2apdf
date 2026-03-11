"""
PDF content extractor.

Extracts text blocks (with font info and positions) and images from PDF files.
Uses pypdfium2 as the primary backend, with fallback to pypdf.

Design decisions:
- Adjacent text spans with same font properties are merged into blocks.
- Text blocks that are spatially close vertically are merged into paragraphs
  to give the translator better sentence context.
- Images are extracted with their bounding boxes for repositioning.
"""

from __future__ import annotations

import io
import logging
import struct
from pathlib import Path
from typing import Optional

from .models import (
    BBox,
    BlockType,
    DocumentContent,
    FontInfo,
    ImageBlock,
    PageContent,
    TextBlock,
    TextSpan,
)

logger = logging.getLogger(__name__)

# Vertical gap threshold: if two text lines are closer than this
# multiple of the font size, they're considered the same paragraph.
_PARA_GAP_FACTOR = 1.5

# Minimum text length to consider a block translatable
_MIN_TEXT_LENGTH = 1


class ExtractionError(Exception):
    """Raised when PDF extraction fails."""
    pass


class PDFExtractor:
    """
    Extracts structured content from PDF files.

    Usage:
        extractor = PDFExtractor()
        doc = extractor.extract("input.pdf")
        for page in doc.pages:
            for block in page.text_blocks:
                print(block.full_text)
    """

    def __init__(
        self,
        merge_paragraphs: bool = True,
        min_text_length: int = _MIN_TEXT_LENGTH,
    ):
        self.merge_paragraphs = merge_paragraphs
        self.min_text_length = min_text_length

    def extract(self, pdf_path: str | Path) -> DocumentContent:
        """
        Extract all content from a PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            DocumentContent with all pages, text blocks, and images.

        Raises:
            ExtractionError: If the file can't be read or is encrypted.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise ExtractionError(f"File not found: {pdf_path}")

        doc = DocumentContent(source_path=str(pdf_path))

        try:
            doc = self._extract_with_pypdfium2(pdf_path, doc)
        except Exception as e:
            logger.warning(f"pypdfium2 extraction failed: {e}. Trying pypdf fallback.")
            try:
                doc = self._extract_with_pypdf(pdf_path, doc)
            except Exception as e2:
                raise ExtractionError(
                    f"All extraction backends failed. "
                    f"pypdfium2: {e}, pypdf: {e2}"
                ) from e2

        # Check for scanned PDF (all pages have no/minimal text)
        total_text = sum(
            len(block.full_text)
            for page in doc.pages
            for block in page.text_blocks
        )
        if total_text < 50 and len(doc.pages) > 0:
            doc.is_scanned = True
            doc.extraction_warnings.append(
                "PDF appears to be scanned (minimal text found). "
                "OCR preprocessing is recommended before translation."
            )

        return doc

    # ------------------------------------------------------------------
    # pypdfium2 backend
    # ------------------------------------------------------------------
    def _extract_with_pypdfium2(
        self, pdf_path: Path, doc: DocumentContent
    ) -> DocumentContent:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(pdf_path))

        # Metadata
        try:
            for key in ("Title", "Author", "Subject", "Creator"):
                val = pdf.get_metadata_value(key)
                if val:
                    doc.metadata[key.lower()] = val
        except Exception:
            pass

        for page_idx in range(len(pdf)):
            try:
                page_content = self._extract_page_pypdfium2(pdf, page_idx)
                doc.pages.append(page_content)
            except Exception as e:
                logger.error(f"Failed to extract page {page_idx + 1}: {e}")
                doc.extraction_warnings.append(
                    f"Page {page_idx + 1} extraction failed: {e}"
                )
                # Add empty page as placeholder
                page = pdf[page_idx]
                doc.pages.append(
                    PageContent(
                        page_number=page_idx + 1,
                        width=page.get_width(),
                        height=page.get_height(),
                    )
                )

        pdf.close()
        return doc

    def _extract_page_pypdfium2(
        self, pdf, page_idx: int
    ) -> PageContent:
        import pypdfium2 as pdfium

        page = pdf[page_idx]
        width = page.get_width()
        height = page.get_height()

        page_content = PageContent(
            page_number=page_idx + 1,
            width=width,
            height=height,
        )

        # --- Extract text ---
        textpage = page.get_textpage()
        full_text = textpage.get_text_bounded()

        # Extract character-level info to build spans
        n_chars = textpage.count_chars()
        if n_chars > 0:
            spans = self._build_spans_from_textpage(textpage, n_chars, height)
            blocks = self._spans_to_blocks(spans)
            if self.merge_paragraphs:
                blocks = self._merge_paragraph_blocks(blocks)
            page_content.text_blocks = [
                b for b in blocks
                if len(b.full_text.strip()) >= self.min_text_length
            ]

        textpage.close()

        # --- Extract images ---
        try:
            page_content.image_blocks = self._extract_images_pypdfium2(
                pdf, page, page_idx, width, height
            )
        except Exception as e:
            logger.warning(f"Image extraction failed on page {page_idx + 1}: {e}")

        page.close()
        return page_content

    def _build_spans_from_textpage(
        self, textpage, n_chars: int, page_height: float
    ) -> list[TextSpan]:
        """Build text spans from pypdfium2 textpage character data."""
        spans: list[TextSpan] = []
        current_text: list[str] = []
        current_font_size: Optional[float] = None
        current_bbox: Optional[list[float]] = None  # [x0, y0, x1, y1]

        for i in range(n_chars):
            try:
                char = textpage.get_text_range(i, 1)
                if not char or char == "\x00":
                    continue

                # Get character rectangle
                try:
                    rect = textpage.get_charbox(i)
                    # rect is (left, bottom, right, top) in PDF coords
                    x0, y0, x1, y1 = rect
                except Exception:
                    # If we can't get the rect, flush current and skip
                    if current_text:
                        spans.append(TextSpan(
                            text="".join(current_text),
                            font=FontInfo(size=current_font_size or 12.0),
                            bbox=BBox(*current_bbox) if current_bbox else BBox(0, 0, 0, 0),
                        ))
                        current_text = []
                        current_bbox = None
                    continue

                font_size = abs(y1 - y0) if abs(y1 - y0) > 0.5 else 12.0

                # Check if this char continues the current span
                # (same approximate font size, on same line)
                if (
                    current_font_size is not None
                    and abs(font_size - current_font_size) < 1.0
                    and current_bbox is not None
                    and abs(y0 - current_bbox[1]) < font_size * 0.5
                ):
                    current_text.append(char)
                    current_bbox[2] = max(current_bbox[2], x1)
                    current_bbox[3] = max(current_bbox[3], y1)
                    current_bbox[1] = min(current_bbox[1], y0)
                else:
                    # Flush previous span
                    if current_text:
                        spans.append(TextSpan(
                            text="".join(current_text),
                            font=FontInfo(size=current_font_size or 12.0),
                            bbox=BBox(*current_bbox) if current_bbox else BBox(0, 0, 0, 0),
                        ))
                    current_text = [char]
                    current_font_size = font_size
                    current_bbox = [x0, y0, x1, y1]

            except Exception:
                continue

        # Flush last span
        if current_text:
            spans.append(TextSpan(
                text="".join(current_text),
                font=FontInfo(size=current_font_size or 12.0),
                bbox=BBox(*current_bbox) if current_bbox else BBox(0, 0, 0, 0),
            ))

        return spans

    def _extract_images_pypdfium2(
        self, pdf, page, page_idx: int, page_width: float, page_height: float
    ) -> list[ImageBlock]:
        """Extract images from a page using pypdfium2."""
        images: list[ImageBlock] = []
        try:
            for obj_idx in range(page.count_objs()):
                try:
                    obj = page.get_obj(obj_idx)
                    if obj.type == 2:  # Image object
                        # Get position matrix
                        try:
                            matrix = obj.get_matrix()
                            # matrix = (a, b, c, d, e, f) - transformation matrix
                            x = matrix.e
                            y = matrix.f
                            w = matrix.a
                            h = matrix.d
                            bbox = BBox(x, y, x + w, y + h)
                        except Exception:
                            bbox = BBox(0, 0, page_width, page_height)

                        # Try to get image bitmap
                        try:
                            bitmap = obj.get_bitmap()
                            pil_image = bitmap.to_pil()
                            buf = io.BytesIO()
                            pil_image.save(buf, format="PNG")
                            img_bytes = buf.getvalue()

                            images.append(ImageBlock(
                                image_bytes=img_bytes,
                                bbox=bbox,
                                extension="png",
                            ))
                        except Exception as img_err:
                            logger.debug(f"Could not extract image bitmap: {img_err}")
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"Image object enumeration failed: {e}")

        return images

    # ------------------------------------------------------------------
    # pypdf fallback backend
    # ------------------------------------------------------------------
    def _extract_with_pypdf(
        self, pdf_path: Path, doc: DocumentContent
    ) -> DocumentContent:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))

        if reader.is_encrypted:
            raise ExtractionError("PDF is encrypted. Please decrypt it first.")

        # Metadata
        if reader.metadata:
            for key in ("title", "author", "subject", "creator"):
                val = getattr(reader.metadata, key, None)
                if val:
                    doc.metadata[key] = str(val)

        for page_idx, page in enumerate(reader.pages):
            try:
                mediabox = page.mediabox
                width = float(mediabox.width)
                height = float(mediabox.height)

                text = page.extract_text() or ""

                page_content = PageContent(
                    page_number=page_idx + 1,
                    width=width,
                    height=height,
                )

                # pypdf doesn't give us position info easily,
                # so create one block per paragraph
                if text.strip():
                    paragraphs = text.split("\n\n")
                    y_pos = height - 50  # Start near top

                    for para in paragraphs:
                        para = para.strip()
                        if not para:
                            continue
                        block = TextBlock(
                            spans=[TextSpan(
                                text=para,
                                font=FontInfo(size=12.0),
                                bbox=BBox(50, y_pos - 20, width - 50, y_pos),
                            )],
                            bbox=BBox(50, y_pos - 20, width - 50, y_pos),
                        )
                        page_content.text_blocks.append(block)
                        y_pos -= 30

                doc.pages.append(page_content)

            except Exception as e:
                logger.error(f"Failed on page {page_idx + 1}: {e}")
                doc.extraction_warnings.append(f"Page {page_idx + 1} failed: {e}")
                doc.pages.append(PageContent(
                    page_number=page_idx + 1, width=612, height=792
                ))

        return doc

    # ------------------------------------------------------------------
    # Block merging utilities
    # ------------------------------------------------------------------
    def _spans_to_blocks(self, spans: list[TextSpan]) -> list[TextBlock]:
        """Group spans into text blocks based on vertical proximity."""
        if not spans:
            return []

        # Sort by vertical position (top to bottom), then horizontal
        spans.sort(key=lambda s: (-s.bbox.y1, s.bbox.x0))

        blocks: list[TextBlock] = []
        current_spans: list[TextSpan] = [spans[0]]

        for span in spans[1:]:
            prev = current_spans[-1]
            # Same line if close vertically
            vert_gap = abs(span.bbox.y1 - prev.bbox.y1)
            if vert_gap < prev.font.size * 0.6:
                current_spans.append(span)
            else:
                blocks.append(self._make_block(current_spans))
                current_spans = [span]

        if current_spans:
            blocks.append(self._make_block(current_spans))

        return blocks

    def _make_block(self, spans: list[TextSpan]) -> TextBlock:
        """Create a TextBlock from a list of spans."""
        x0 = min(s.bbox.x0 for s in spans)
        y0 = min(s.bbox.y0 for s in spans)
        x1 = max(s.bbox.x1 for s in spans)
        y1 = max(s.bbox.y1 for s in spans)

        return TextBlock(
            spans=spans,
            bbox=BBox(x0, y0, x1, y1),
        )

    def _merge_paragraph_blocks(self, blocks: list[TextBlock]) -> list[TextBlock]:
        """
        Merge vertically adjacent blocks with similar font into paragraphs.
        This gives the translator more context for better translations.
        """
        if len(blocks) <= 1:
            return blocks

        merged: list[TextBlock] = [blocks[0]]

        for block in blocks[1:]:
            prev = merged[-1]
            prev_font = prev.primary_font
            cur_font = block.primary_font

            # Merge if: similar font size, close vertical gap, similar x range
            vert_gap = abs(block.bbox.y1 - prev.bbox.y0)
            size_match = abs(prev_font.size - cur_font.size) < 2.0
            x_overlap = (
                abs(block.bbox.x0 - prev.bbox.x0) < prev_font.size * 3
            )

            if (
                size_match
                and x_overlap
                and vert_gap < prev_font.size * _PARA_GAP_FACTOR
            ):
                # Merge into previous block
                prev.spans.extend(block.spans)
                prev.bbox = BBox(
                    min(prev.bbox.x0, block.bbox.x0),
                    min(prev.bbox.y0, block.bbox.y0),
                    max(prev.bbox.x1, block.bbox.x1),
                    max(prev.bbox.y1, block.bbox.y1),
                )
            else:
                merged.append(block)

        return merged
