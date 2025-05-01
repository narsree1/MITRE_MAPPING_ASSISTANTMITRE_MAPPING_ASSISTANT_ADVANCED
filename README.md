
# MITRE ATT&CK Mapping Tool

A web-based Streamlit application that automatically maps your security use cases to the MITRE ATT&CK framework using natural language processing, library lookups, and similarity-based matching.

## Features

- **Automatic MITRE Mapping**: Uses NLP and semantic similarity to match use cases to MITRE ATT&CK Tactics and Techniques
- **Pre-Mapped Library Matching**: Matches use cases to a built-in library for high-confidence mappings
- **Suggestions Engine**: Recommends additional use cases based on uploaded log sources
- **Analytics Dashboard**: Visualizes MITRE coverage and use case distribution
- **MITRE Navigator Export**: Generates Navigator layer JSON for ATT&CK framework visualization

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/mitre-mapping-tool.git
cd mitre-mapping-tool
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**
```bash
streamlit run app.py
```

## Usage

1. Upload a CSV with the following columns:
   - `Use Case Name`
   - `Description`
   - `Log Source`

2. Click **Start Mapping** to begin mapping use cases to MITRE ATT&CK

3. Navigate through the app:
   - **Results**: View mapped tactics and techniques
   - **Suggestions**: Discover related use cases from the library
   - **Analytics**: Visualize coverage and metrics
   - **Export**: Download a Navigator layer file for visualization

## Technical Details

This app is built with:
- **Sentence Transformers**: `all-mpnet-base-v2` for NLP-based similarity
- **Streamlit**: For building the interactive web UI
- **PyTorch**: High-performance tensor calculations
- **Plotly**: Used for chart visualizations
- **MITRE STIX API**: Live ATT&CK framework data fetch with structured JSON fallback

## Requirements

- Python 3.8+
- Streamlit ≥ 1.24.0
- PyTorch ≥ 2.0
- Sentence Transformers ≥ 2.2.2
- See `requirements.txt` for a complete list

## Project Structure

```
mitre-mapping-tool/
│
├── app.py                   # Main application logic
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
│
├── analytics.py             # Renders analytics page
├── modules/                 # Optional modular structure (if refactored)
│   ├── mapper.py            # Core logic for MITRE mapping
│   ├── suggestions.py       # Logic to suggest related use cases
│   ├── utils.py             # Helper functions (e.g. Lottie, caching)
│
├── assets/                  # Static assets like animations or logos
│   └── mitre_cache.json     # Local MITRE ATT&CK JSON (optional fallback)
```

## License

[MIT License](LICENSE)

## Contributing

Feel free to fork, open issues, or submit PRs to improve the tool and expand its capabilities.

## Acknowledgements

- [MITRE ATT&CK Framework](https://attack.mitre.org/)
- [Sentence Transformers](https://www.sbert.net/)
- [HuggingFace Transformers](https://huggingface.co/sentence-transformers)
- [Streamlit](https://streamlit.io/)
