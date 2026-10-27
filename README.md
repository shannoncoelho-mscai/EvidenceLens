# EvidenceLens — Scientific Claim and Evidence Exploration System

EvidenceLens is a web-based application designed to help users explore scientific claims and examine relevant research evidence. It combines research-paper retrieval, evidence ranking, and Natural Language Inference (NLI) to provide an initial assessment of how retrieved scientific literature relates to a submitted claim.

The project demonstrates how information retrieval and natural language processing can be combined to support evidence-based exploration of scientific claims.

> **Important:** EvidenceLens is an exploratory research-support tool, not a definitive scientific fact-checker. Its outputs require human review and should not replace expert judgment or a systematic literature review.

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Interface Preview](#interface-preview)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation and Setup](#installation-and-setup)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Testing](#testing)
- [Research and Model Notes](#research-and-model-notes)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Responsible Use](#responsible-use)
- [Author](#author)
- [License](#license)

## Project Overview

Scientific claims can be difficult to evaluate because supporting and contradicting evidence may be distributed across many research publications. Finding relevant studies and understanding how their findings relate to a claim can require considerable time and effort.

EvidenceLens explores a software-based approach to this problem by retrieving research literature, ranking relevant evidence, and applying Natural Language Inference techniques to help interpret the relationship between a claim and retrieved research.

The application provides a web interface connected to a Flask backend. Research retrieval and evidence-analysis components process a submitted claim and return a structured response for the user to examine.

## Objectives

The main objectives of EvidenceLens are to:

- Retrieve potentially relevant scientific literature for a submitted claim.
- Rank retrieved evidence according to its relevance to the claim.
- Explore relationships between claims and evidence using Natural Language Inference.
- Present the results through a user-friendly web interface.
- Demonstrate the integration of research retrieval, machine learning, and web application development.
- Encourage transparent, evidence-informed investigation rather than unsupported conclusions.

## Interface Preview

Add a screenshot of your actual running application to the repository to display the interface here.

Save the screenshot using the following path:

`screenshots/interface.png`

Then add the image to GitHub using this Markdown:

![EvidenceLens user interface](screenshots/interface.png)

**To add the screenshot:**

1. Run EvidenceLens locally.
2. Open the application in your browser.
3. Capture a screenshot showing the interface.
4. Create a folder named `screenshots` in the project root.
5. Save the image as `interface.png` inside that folder.
6. Commit and push the screenshot along with this README.

If you have not added the screenshot yet, you can temporarily remove the image line above and add it later.

## Key Features

### 1. Scientific Claim Analysis

Accepts a scientific claim as input for an evidence-exploration workflow.

### 2. Research Literature Retrieval

Uses Europe PMC as a research literature source to identify potentially relevant publications.

### 3. Evidence Ranking

Uses text-based relevance scoring, including TF-IDF and cosine similarity, to rank retrieved evidence.

### 4. Natural Language Inference

Uses the `cross-encoder/nli-deberta-v3-small` model to examine the relationship between a claim and evidence text.

NLI results are model predictions and should not be interpreted as proof that a scientific claim is true or false.

### 5. Structured Evidence Reporting

Organizes the analysis results into a structured response to help users examine the retrieved evidence.

### 6. REST API Backend

Provides a Flask backend for processing requests and exposing application endpoints.

### 7. Automated Tests

Includes a pytest test suite for checking application behavior, API functionality, and research-search components.

## How It Works

The general workflow is:

1. **Claim submission:** The user enters a scientific claim through the web interface.
2. **Research retrieval:** The backend searches for potentially relevant scientific publications.
3. **Evidence ranking:** Retrieved material is ranked according to textual relevance.
4. **Inference analysis:** The NLI model estimates the relationship between claim and evidence text.
5. **Report generation:** The backend organizes the analysis into a structured result.
6. **Result presentation:** The frontend displays the response for the user to review.

```text
             User submits a claim
                      |
                      v
               Flask Backend
                      |
                      v
             Research Retrieval
                (Europe PMC)
                      |
                      v
               Evidence Ranking
             (TF-IDF / Cosine)
                      |
                      v
              NLI Model Analysis
                      |
                      v
              Evidence Report
                      |
                      v
              Results in Frontend
```

This diagram represents the intended high-level workflow. The exact output depends on the available research results and the behavior of the implemented components.

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, Flask |
| Research source | Europe PMC |
| Text relevance | TF-IDF, cosine similarity |
| Natural Language Inference | Hugging Face Transformers model |
| Testing | pytest |
| Version control | Git and GitHub |

## Project Structure

The repository is organized into frontend files, backend application modules, and automated tests.

```text
EvidenceLens/
├── backend/
│   ├── app.py
│   ├── config.py
│   ├── requirements.txt
│   ├── smoke_test_nli.py
│   ├── services/
│   │   ├── claim_analysis.py
│   │   ├── evidence_ranking.py
│   │   ├── evidence_report.py
│   │   └── research_search.py
│   └── tests/
│       ├── conftest.py
│       ├── test_analysis.py
│       ├── test_api.py
│       └── test_search.py
├── screenshots/
│   └── interface.png
├── index.html
├── script.js
├── style.css
├── .gitignore
└── README.md
```

The `screenshots/` folder is optional until you add an interface image. If your actual repository structure differs, update this tree to match the files in your project.

## Requirements

Before installing EvidenceLens, ensure you have:

- Python installed.
- Git installed.
- A web browser.
- A terminal such as PowerShell or the Cursor terminal.
- An internet connection for retrieving research publications and downloading any required model files.

The application uses Python packages listed in the backend's `requirements.txt` file.

## Installation and Setup

### 1. Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/shannoncoelho-mscai/EvidenceLens.git
cd EvidenceLens
```

### 2. Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the environment's Python executable directly instead.

### 3. Install Dependencies

Navigate to the backend folder:

```powershell
cd backend
```

Install the required packages:

```powershell
python -m pip install -r requirements.txt
```

If you need to return to the project root afterward, run:

```powershell
cd ..
```

## Running the Application

EvidenceLens has a frontend and a Flask backend. Start them in separate terminal windows.

### 1. Start the Backend

Open a terminal in the project root, activate the virtual environment, and navigate to the backend folder:

```powershell
.\.venv\Scripts\Activate.ps1
cd backend
python app.py
```

The backend is expected to run at:

`http://127.0.0.1:5000`

The exact host and port depend on the Flask configuration in your application.

### 2. Start the Frontend

Open a second terminal in the project root and activate the virtual environment if needed.

Start a local web server:

```powershell
python -m http.server 5500
```

Open the frontend in your browser:

`http://127.0.0.1:5500`

Make sure the frontend's API configuration points to the backend address and port.

If your backend or frontend uses a different configuration, follow the values defined in the project files.

## API Endpoints

The backend includes the following endpoints.

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Checks the backend's health status. |
| `/predict` | POST | Processes a claim-analysis request. |
| `/model-info` | GET | Returns information about the configured model. |

### Example: Health Check

```http
GET http://127.0.0.1:5000/health
```

### Example: Claim Analysis

```http
POST http://127.0.0.1:5000/predict
Content-Type: application/json
```

The exact request body and response schema should be checked in `backend/app.py` and the related service modules.

## Testing

EvidenceLens includes automated tests using pytest.

From the project root, activate your virtual environment and run:

```powershell
python -m pytest backend/tests -v
```

If your tests expect the backend folder to be the working directory, use:

```powershell
cd backend
python -m pytest tests -v
```

Run the test command that matches your current project configuration. A passing test suite indicates that the tested behaviors passed; it does not establish that every scientific assessment produced by the application is correct.

## Research and Model Notes

### Europe PMC

Europe PMC is used as a source for retrieving potentially relevant scientific literature. Retrieval results depend on the query, the source's available records, and service availability.

### TF-IDF and Cosine Similarity

TF-IDF represents text using term-weighting features, while cosine similarity compares the direction of text vectors. These techniques can help rank text by lexical relevance.

High textual similarity does not necessarily mean that a publication supports a claim.

### Natural Language Inference

The project uses `cross-encoder/nli-deberta-v3-small` for NLI analysis. Such models estimate relationships between text inputs according to learned patterns.

The model may misinterpret technical terminology, uncertainty, study limitations, negation, or context. Its predictions should therefore be treated as signals for further examination rather than definitive conclusions.

## Limitations

EvidenceLens has several important limitations:

- **Retrieval coverage:** Relevant publications may not be retrieved.
- **Textual relevance:** Similar wording does not guarantee scientific relevance.
- **Model uncertainty:** NLI predictions can be incorrect or misleading.
- **Research context:** A study's design, sample size, methodology, and limitations require careful interpretation.
- **Conflicting evidence:** Research findings may disagree or apply to different populations and conditions.
- **Service availability:** External research services may temporarily fail or return incomplete results.
- **No definitive verification:** The application does not establish scientific truth from model predictions alone.

These limitations mean that results should be interpreted cautiously and reviewed against the original publications.

## Future Improvements

Potential future improvements include:

- Improving research retrieval and query expansion.
- Adding clearer publication metadata and direct links to original studies.
- Displaying source passages that explain why evidence was ranked.
- Improving the handling of contradictory findings and uncertainty.
- Adding more comprehensive error handling and service-retry behavior.
- Expanding automated tests and integration testing.
- Improving accessibility and responsive interface design.
- Adding options for filtering publications by date, study type, or research topic.
- Evaluating model performance on a carefully selected scientific-claim dataset.

These are possible extensions, not claims about features already implemented.

## Responsible Use

EvidenceLens is intended to support research exploration and learning.

Users should:

- Read the original research publications.
- Consider study quality, limitations, and context.
- Check whether the retrieved evidence directly addresses the submitted claim.
- Avoid treating automated labels or scores as proof.
- Consult qualified professionals for consequential scientific, medical, or other high-stakes decisions.

The application should complement critical thinking and expert review, not replace them.

## Author

**Shannon Coelho**

M.Sc. Artificial Intelligence  
Goa University

- GitHub: [shannoncoelho-mscai](https://github.com/shannoncoelho-mscai)
- Project repository: [EvidenceLens](https://github.com/shannoncoelho-mscai/EvidenceLens)

## License

No license has been specified yet. Unless a license is added to the repository, others should not assume that they have permission to redistribute or modify the project.

If you intend to make the project open source, consider adding an appropriate license file, such as an MIT License, after deciding which license meets your needs.
