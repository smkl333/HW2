# Math OCR MLOps Pipeline API

This project is a FastAPI server that acts as the prediction serving layer in an MLOps pipeline. It leverages the [pix2tex](https://github.com/lukas-blecher/LaTeX-OCR) (LaTeX-OCR) Vision Transformer to recognize mathematical expressions from uploaded images, and uses `sympy` to evaluate those expressions entirely locally.

## Project Structure

```text
.
├── app/
│   ├── __init__.py      # Marks directory as a python package
│   ├── main.py          # FastAPI application, App routing, and Model initialization
│   └── ml_model.py      # Core Inference logic (Image -> LaTeX -> Sympy Evaluation)
├── requirements.txt     # All required packages including specific ML libraries
└── README.md            # Pipeline overview and execution guide
```

## Setup & Running locally

1. **Set up Virtual Environment**
   To isolate dependencies, use a virtual environment. Open your terminal in this directory and run:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Prediction Server**
   Start the FastAPI app via `uvicorn`:
   ```bash
   uvicorn app.main:app --reload
   ```
   *(Note: Upon the very first startup, `pix2tex` will connect to the internet to download its pre-trained model checkpoint (weights). This will take a few moments depending on your network.)*

4. **Testing the Endpoint Using Swagger UI**
   - Open your browser to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
   - Find the `POST /solve` endpoint.
   - Click "Try it out", upload a test image file containing a simple printed or handwritten math calculation (like `2 + 2`), and hit Execute.
   - The API will respond with both the raw LaTeX representation and the solved answer.
