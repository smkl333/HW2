import logging
from PIL import Image, ImageEnhance
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

    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Boost contrast to make handwriting clearer for AI."""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(2.0) # Double the contrast

    def _clean_math_text(self, text: str) -> str:
        """Helper to clean OCR text specifically for math solver."""
        # 1. Basic mapping for common handwritten OCR mistakes
        mapping = {'X': '*', 'x': '*', '|': '1', 'l': '1', 'o': '0', 'O': '0', 'i': '1'}
        for old, new in mapping.items():
            text = text.replace(old, new)
        
        # 2. Keep only math-relevant characters (numbers, operators, dots, parens)
        cleaned = re.sub(r'[^0-9+\-*/().=^]', '', text)
        return cleaned

    def _try_solve(self, formula: str, is_latex: bool) -> str:
        """Internal helper to try and solve an equation or simplify expression."""
        try:
            logger.info(f"Processing Formula: '{formula}' (is_latex: {is_latex})")
            
            # 1. Detect and Handle Equality (=) for both LaTeX and Text
            if '=' in formula:
                lhs_str, rhs_str = formula.split('=', 1)
                if is_latex:
                    lhs = parse_latex(lhs_str.strip())
                    rhs = parse_latex(rhs_str.strip())
                else:
                    lhs = sympy.sympify(lhs_str.strip())
                    rhs = sympy.sympify(rhs_str.strip())
                expr = sympy.Eq(lhs, rhs)
            else:
                # 2. Handle simple expressions without '='
                if is_latex:
                    expr = parse_latex(formula)
                else:
                    expr = sympy.sympify(formula)

            # 3. Logic for Equation vs Expression
            # If it's an equation (Equality object from Sympy)
            if isinstance(expr, sympy.Equality):
                # Identify variables (Symbols)
                symbols = list(expr.free_symbols)
                if symbols:
                    # Solve for the primary variable (usually 'x')
                    # We sort symbols to prioritize 'x' if multiple exist
                    target_symbol = sorted(symbols, key=lambda s: str(s))[0]
                    solutions = sympy.solve(expr, target_symbol)
                    return f"{target_symbol} = {solutions}"
                else:
                    # No variables, just verify if true/false (e.g. 1 = 1)
                    return str(sympy.simplify(expr))
            
            # 4. Logic for simple expressions (Simplify)
            simplified = sympy.simplify(expr)
            return str(simplified)
        except Exception as e:
            logger.warning(f"Solving failed: {e}")
            return None

    def predict(self, image: Image.Image) -> dict:
        """
        Final Polish: Intelligent ensemble with contrast enhancement and aggressive fallback.
        """
        # 1. Primary Inference (pix2tex)
        recognized_math = ""
        solved_result = None
        used_fallback = False
        
        try:
            recognized_math = self.model(image)
            if recognized_math:
                solved_result = self._try_solve(recognized_math, is_latex=True)
        except Exception:
            recognized_math = ""

        # 2. High-Performance Fallback (If Primary failed or was unsolvable)
        # messy handwriting often fails pix2tex or returns nonsensical LaTeX
        if not solved_result:
            if self.handwriting_reader:
                try:
                    logger.info("Primary model failed or unsolvable. Boosting contrast and running EasyOCR fallback...")
                    # Pre-process: Enhance contrast for better handwriting recognition
                    enhanced_img = self._enhance_image(image)
                    image_np = np.array(enhanced_img.convert("RGB"))
                    
                    ocr_results = self.handwriting_reader.readtext(image_np)
                    fallback_text = "".join([res[1] for res in ocr_results])
                    
                    if fallback_text:
                        cleaned_math = self._clean_math_text(fallback_text)
                        fallback_solved = self._try_solve(cleaned_math, is_latex=False)
                        
                        if fallback_solved:
                            recognized_math = cleaned_math
                            solved_result = fallback_solved
                            used_fallback = True
                            logger.info(f"Fallback SUCCESS: {recognized_math} = {solved_result}")
                except Exception as e:
                    logger.error(f"Fallback failed: {e}")

        if not recognized_math:
             raise RuntimeError("Could not recognize any mathematical expression. Please try a clearer image.")

        return {
            "formula": recognized_math,
            "solved": solved_result if solved_result else "Recognized, but cannot solve. Check the formula.",
            "method": "Ensemble(EasyOCR+Enhanced)" if used_fallback else "Primary(pix2tex)"
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
