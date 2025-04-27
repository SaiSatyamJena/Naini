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
  <img src="NainiAnimation.gif" alt="NainiV1 Demo Animation" width="700"/>
</p>

*(Replace the placeholder link above with a direct link to your hosted GIF. You can upload GIFs directly to GitHub issues/PRs and use that link, or host it elsewhere.)*

---

## 🚀 Key Features

*   **🧠 Intelligent Geological Analysis:** Leverages a fine-tuned Qwen2 7B model for domain-specific understanding.
*   **📄 Advanced Data Parsing:** Employs a custom-built pipeline using Camelot, img2table, and other techniques for robust extraction from PDFs and images.
*   **⚡ Optimized Performance:** Designed for speed and efficiency during inference.
*   **💻 100% Local Operation:** Ensures data privacy and security by running entirely on your hardware.
*   **📉 Low Resource Requirement:** Operable on systems with ~14GB of VRAM, making advanced AI accessible.
*   **💰 Zero API Costs:** Completely free to run after initial setup.
*   **🧩 Agentic Workflow:** Capable of multi-step reasoning based on extracted data (if applicable, customize this point based on Naini's capabilities).

---

## 🛠️ Technology Stack

*   **Core Language:** Python 3.x
*   **LLM:** Fine-tuned Qwen2 (7 Billion Parameters)
*   **Parsing Libraries:** Camelot-py, img2table, [List any other key parsing libs, e.g., PyPDF2, OpenCV]
*   **AI/ML Frameworks:** [List primary frameworks, e.g., PyTorch, Transformers (Hugging Face)]
*   **Core Data Handling:** Pandas, NumPy

---

## ✅ Requirements

*   **Python:** 3.8 or higher recommended
*   **GPU:** NVIDIA GPU with CUDA support is highly recommended.
*   **VRAM:** Approximately 14 GB or more.
*   **RAM:** 16 GB+ recommended.
*   **OS:** Linux, macOS, or Windows (Linux generally preferred for ML workloads).
*   **Dependencies:** See `requirements.txt`.

---

## ⚙️ Installation

Follow these steps to get NainiV1 running on your local machine:

1.  **Clone the Repository:**
    ```bash
    git clone [Your GitHub Repository Link]
    cd NainiV1 # Or your repository's directory name
    ```

2.  **Set up a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate the environment:
    # Linux/macOS:
    source venv/bin/activate
    # Windows:
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Depending on your system and CUDA version, you might need specific versions of PyTorch or other GPU-related libraries. Consult their official documentation if you encounter issues.*

4.  **Download the Fine-Tuned Qwen2 Model:**
    *   You need to obtain the fine-tuned Qwen2 7B model files used by NainiV1.
    *   **[IMPORTANT: Provide clear instructions here. Examples:]**
        *   *Option A (If hosting on Hugging Face):* "Download the model files from our Hugging Face repository: [Link to HF Repo]. Place them in the `models/` directory."
        *   *Option B (If providing download script):* "Run the download script: `python download_model.py`. The model will be saved to the `models/` directory."
        *   *Option C (Manual download link):* "Download the model archive [Link to model file/archive] and extract its contents into the `models/` directory within this project."
    *   Ensure the application knows where to find the model (you might need a configuration file or environment variable).

---

## ▶️ Usage

1.  **Prepare Your Input:** Ensure your geological document (e.g., PDF, image file) is ready.

2.  **Run the Application:**
    Execute the main script from your terminal. Adjust the command based on your specific script and arguments:

    ```bash
    python main.py --input_path /path/to/your/document.pdf [Other arguments like --output_path, etc.]
    ```

    *   Replace `/path/to/your/document.pdf` with the actual path to your input file.
    *   Include any other necessary command-line arguments (e.g., `--output_dir`, `--config_file`). List and explain the key arguments here:
        *   `--input_path`: **Required.** Path to the input geological document (PDF, JPG, PNG, etc.).
        *   `--output_dir`: *Optional.* Directory to save results (default: `output/`).
        *   `[--another_arg]`: *[Explain other important arguments]*.

3.  **Check the Output:**
    NainiV1 will process the document and generate its analysis. The output might include:
    *   Extracted tables (e.g., as CSV files).
    *   A summary report (e.g., a text file).
    *   Annotated images or diagrams.
    *   [Describe the specific output format(s) Naini produces].
    Check the specified output directory (or the default `output/` directory).

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

This project is licensed under the **[Specify Your License Name - e.g., MIT License, Apache 2.0]**. See the `LICENSE` file for details.

---

## 📧 Contact

For questions or support, please [Open an issue on GitHub] or contact [Your Name/Email/Preferred Contact Method - Optional].

---
