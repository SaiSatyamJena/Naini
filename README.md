# Agentic AI Geologist "Naini" V1 🏔️🧠

<p align="center">
  <img src="Naini_pic.png" alt="Naini AI Geologist Logo" width="400"/>
</p>

<p align="center">
  <i>Your intelligent assistant for analyzing geological documents locally and efficiently.</i>
</p>

---

Meet **NainiV1**, an agentic AI designed specifically for the challenges of geological data analysis. Traditional geological documents often contain complex tables, diagrams, and unstructured text, making automated data extraction difficult. Naini tackles this head-on using a **state-of-the-art, self-developed parsing pipeline**. This custom logic intelligently combines the strengths of tools like **Camelot** and **img2table** (among others) to meticulously extract information even from challenging visual formats.

The "brain" behind Naini is a highly **optimized Qwen2 7-billion parameter model**. This powerful LLM has been fine-tuned specifically for geological interpretation tasks, allowing it to understand context and provide insightful analysis based on the parsed data.

The result? NainiV1 is:
*   **💡 Intelligent:** Understands and interprets geological data.
*   **⚡ Lightning-Fast:** Optimized for rapid inference.
*   **🎯 Accurate:** Fine-tuned for high precision in geological tasks.
*   **💻 Fully Local:** Runs entirely on your own machine. No data leaves your system.
*   **💰 Cost-Free:** **Zero reliance on paid APIs.**
*   **📉 Lightweight:** Operates effectively with approximately **14GB of VRAM**.

NainiV1 empowers geoscientists and researchers by providing a powerful, private, and efficient tool for unlocking insights hidden within their documents.

---

## ✨ Naini in Action

See NainiV1 process a geological document in real-time!

<p align="center">
  <!-- IMPORTANT: Replace this placeholder link with the actual link to your GIF or video -->
  <img src="NainiAnimation.gif" alt="NainiV1 Demo Animation" width="2000"/>
</p>


---

## 🚀 Key Features

*   **🧠 Intelligent Geological Analysis:** Leverages a fine-tuned Qwen2 7B model for domain-specific understanding.
*   **📄 Advanced Data Parsing:** Employs a custom-built pipeline using Camelot, img2table, and other techniques for robust extraction from PDFs and images.
*   **⚡ Optimized Performance:** Designed for speed and efficiency during inference.
*   **💻 100% Local Operation:** Ensures data privacy and security by running entirely on your hardware.
*   **📉 Low Resource Requirement:** Operable on systems with ~14GB of VRAM, making advanced AI accessible.
*   **💰 Zero API Costs:** Completely free to run after initial setup.
*   **🧩 Agentic Workflow:** 

---

## 🛠️ Technology Stack

*   **Core Language:** Python 3.x
*   **LLM:** Fine-tuned Qwen2 (7 Billion Parameters)
*   **Parsing Libraries:** Camelot-py, img2table etc
*   **AI/ML Frameworks:** LangGraph,LangChain,PyTorch, Transformers (Hugging Face),YOLOv10, Vision Transformers (ViTs) for layout segmentation
*   **Core Data Handling:** Pandas, NumPy,ChromaDB, FAISS

---

## ✅ Requirements

- **Python:** 3.8 or higher recommended  
- **GPU:** NVIDIA GPU with CUDA support is highly recommended.  
- **VRAM:** Approximately 14 GB or more.  
- **RAM:** 16 GB+ recommended.  
- **OS:** Dockerfile pushed to GitHub, so it can run in any environment—just download the Docker app.  
- **Dependencies:** See `requirements.txt` for exact versions.  

**Required Libraries:**  

- **Core ML/Vision/LLM Libraries**  
  - torch  
  - torchvision  
  - transformers  
  - accelerate  
  - bitsandbytes  
  - timm  

- **LangChain & RAG Components**  
  - langchain  
  - langchain-community  
  - langchain-core  
  - langchain-huggingface  
  - sentence-transformers  
  - faiss-cpu  
  - rank_bm25  
  - tiktoken  

- **PDF & Image Handling**  
  - pdf2image  
  - Pillow  
  - pymupdf  
  - opencv-python  

- **OCR**  
  - pytesseract  

- **Table Extraction**  
  - img2table  
  - tabulate  
  - camelot-py[cv]  

- **UI**  
  - gradio  
  - fastapi  

- **Dev & Utility**  
  - jupyterlab  
  - ipykernel  
  - numpy  
  - pandas  
  - matplotlib  
  - joblib  
  - protobuf  
  - scipy  
  - typing-inspect  
  - typing_extensions  
  - pydantic



---


## 🤝 Contributing

Contributions are welcome! If you'd like to improve NainiV1, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/YourFeature`).
6.  Open a Pull Request.

Please report any bugs or suggest features by opening an issue on the GitHub repository.

---

## 📜 License

This project is licensed under the **Apache 2.0**.
COZ Sharing is Caring 
See the `LICENSE` file for details

---

## 📧 Contact

For questions or support, please "Open an issue on GitHub" or contact saisatyam2016@gmail.com

---
