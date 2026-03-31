import logging
from PIL import Image
from pix2tex.cli import LatexOCR
import sympy
from sympy.parsing.latex import parse_latex
import easyocr
import numpy as np

logger = logging.getLogger(__name__)

class MathOCRModel:
    def __init__(self):
        """
        Initialize the ML model. 
        In an MLOps pipeline, model weights are loaded here (once during server startup).
        """
        logger.info("Loading Vision Transformer model (pix2tex) into memory...")
        self.model = LatexOCR() # Automatically downloads pretrained weights on first run
        logger.info("Model loaded successfully.")

    def predict(self, image: Image.Image) -> dict:
        """
        Run inference on the image and try to solve the parsed mathematical expression.
        """
        # 1. Inference (Image -> LaTeX string)
        try:
            recognized_latex = self.model(image)
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise RuntimeError(f"Failed to extract equation from image: {str(e)}")

        # 2. Evaluation / Post-processing (LaTeX -> Mathematical Result)
        solved_result = None
        try:
            # Parse LaTeX string into a SymPy object
            expr = parse_latex(recognized_latex)
            
            # Evaluate/Simplify the expression (e.g., '1+1' -> '2')
            simplified_expr = sympy.simplify(expr)
            solved_result = str(simplified_expr)
            
        except Exception as e:
            # Non-evaluable models like equations or complex calculus will fall here
            logger.warning(f"Could not solve the recognized equation: {e}")
            solved_result = "Equation parsed, but cannot evaluate automatically."

        return {
            "latex_raw": recognized_latex,
            "solved": solved_result
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
