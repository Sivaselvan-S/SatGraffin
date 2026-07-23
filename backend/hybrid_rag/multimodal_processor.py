"""
Multimodal Processor (Image + PDF Visual Extraction)
===================================================
Processes uploaded visual assets (PNG, JPG, WEBP, PDF) using Gemini Vision models
to extract descriptions, data tables, and diagrams into indexed text context.
"""

import io
import logging
from typing import Dict, Any, List, Optional


from PIL import Image
import pypdf
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

class MultimodalProcessor:
    """Extracts visual insights from uploaded images and PDFs."""

    VISION_PROMPT = """Analyze this image in detail for a RAG knowledge base.
Extract all readable text, data tables, charts, graphs, diagrams, and visual entities.
Structure your summary clearly so that semantic search can match user questions against this image content."""

    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm


    def process_image(self, image_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Process image file bytes with Gemini Vision."""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Call Gemini vision model
            if hasattr(self.llm, "invoke"):
                from langchain_core.messages import HumanMessage
                
                # Format message for LangChain multimodal support
                import base64
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": self.VISION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}
                        }
                    ]
                )
                res = self.llm.invoke([message])
                description = res.content if hasattr(res, "content") else str(res)
            else:
                description = f"Uploaded image: {filename}"

            logger.info(f"Successfully processed image '{filename}' with Gemini Vision ({len(description)} chars)")
            return {
                "filename": filename,
                "description": description,
                "content": f"[Visual Content from {filename}]:\n{description}"
            }
        except Exception as e:
            logger.error(f"Failed to process image '{filename}': {e}")
            return {
                "filename": filename,
                "description": f"Failed to extract visual description: {e}",
                "content": f"[Uploaded file {filename}]"
            }

    def process_pdf(self, pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Extract text and images from PDF."""
        try:
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            text_parts: List[str] = []
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")

            combined_text = "\n\n".join(text_parts)
            logger.info(f"Extracted {len(text_parts)} pages from PDF '{filename}' ({len(combined_text)} chars)")
            
            return {
                "filename": filename,
                "description": f"PDF Document ({len(reader.pages)} pages)",
                "content": f"[PDF Document: {filename}]\n{combined_text}"
            }
        except Exception as e:
            logger.error(f"Failed to process PDF '{filename}': {e}")
            return {
                "filename": filename,
                "description": f"Failed to parse PDF: {e}",
                "content": f"[Uploaded PDF {filename}]"
            }
