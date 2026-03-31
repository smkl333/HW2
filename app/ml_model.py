import logging
from PIL import Image
from pix2tex.cli import LatexOCR
import sympy
from sympy.parsing.latex import parse_latex
import easyocr
import numpy as np
import re

logger = logging.getLogger(__name__)

class MathOCRModel:
    def __init__(self, handwriting_reader=None):
        """
        Initialize the ML model. 
        """
        logger.info("Loading Vision Transformer model (pix2tex) into memory...")
        self.model = LatexOCR()
        self.handwriting_reader = handwriting_reader # Reference to EasyOCR reader
        logger.info("Math OCR Model loaded successfully.")

    def _clean_math_text(self, text: str) -> str:
        """Helper to clean OCR text for math parsing."""
        # Remove spaces and common handwriting OCR misinterpretations
        text = text.replace(" ", "")
        # Basic mapping for handwritten math symbols
        mapping = {'x': '*', 'l': '1', '|': '1', 'o': '0', 'X': '*'}
        for old, new in mapping.items():
            text = text.replace(old, new)
        return text

    def predict(self, image: Image.Image) -> dict:
        """
        Intelligent ensemble prediction: Try pix2tex first, fallback to EasyOCR for handwriting.
        """
        # 1. Primary Inference (pix2tex - LaTeX Optimized)
        recognized_math = ""
        used_fallback = False
        
        try:
            recognized_math = self.model(image)
        except Exception as e:
            logger.warning(f"Primary inference failed: {e}. Trying fallback...")
            recognized_math = ""

        # 2. Intelligent Fallback (If pix2tex result is suspect or empty)
        # Often messy handwriting like "1+1" returns empty or just "1" in pix2tex
        if not recognized_math or len(recognized_math.strip()) < 3:
            if self.handwriting_reader:
                try:
                    logger.info("Suspected handwritten or simple input. Running EasyOCR fallback...")
                    image_np = np.array(image.convert("RGB"))
                    ocr_results = self.handwriting_reader.readtext(image_np)
                    fallback_text = "".join([res[1] for res in ocr_results])
                    
                    if fallback_text:
                        recognized_math = self._clean_math_text(fallback_text)
                        used_fallback = True
                        logger.info(f"Fallback successful: {recognized_math}")
                except Exception as e:
                    logger.error(f"Fallback inference failed: {e}")

        if not recognized_math:
             raise RuntimeError("Could not recognize any mathematical expression in the image.")

        # 3. Evaluation (Parsing to Result)
        solved_result = None
        try:
            # If we used fallback, it's plain text math. If not, it's LaTeX.
            if used_fallback:
                # Basic python-style math evaluation via sympy
                expr = sympy.sympify(recognized_math)
            else:
                # LaTeX parsing
                expr = parse_latex(recognized_math)
            
            simplified_expr = sympy.simplify(expr)
            solved_result = str(simplified_expr)
            
        except Exception as e:
            logger.warning(f"Could not solve the equation: {e}")
            solved_result = "Recognized, but cannot solve. Check the formula."

        return {
            "formula": recognized_math,
            "solved": solved_result,
            "method": "Ensemble(EasyOCR)" if used_fallback else "Primary(pix2tex)"
        }

class HandwritingModel:
    def __init__(self):
        """
        Initialize EasyOCR for Korean and English text recognition.
        """
        logger.info("Loading EasyOCR models (Korean, English) into memory...")
        self.reader = easyocr.Reader(['ko', 'en'])
        logger.info("EasyOCR models loaded successfully.")

    def predict(self, image: Image.Image) -> dict:
        """
        Run OCR on the image to recognize handwritten or printed text.
        """
        try:
            # Convert PIL image to numpy array as required by EasyOCR
            image_np = np.array(image.convert("RGB"))
            results = self.reader.readtext(image_np)
            
            # results format: [([[x, y], [x, y], ...], text, confidence), ...]
            full_text = " ".join([res[1] for res in results])
            
            return {
                "text_raw": full_text,
                "segments": [{"text": res[1], "confidence": round(float(res[2]), 4)} for res in results]
            }
        except Exception as e:
            logger.error(f"Handwriting OCR failed: {e}")
            raise RuntimeError(f"Failed to extract text from image: {str(e)}")
