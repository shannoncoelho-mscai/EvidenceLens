
"use strict";

// ========================================
// EvidenceLens - Frontend JavaScript
// ========================================

const API_BASE_URL = "http://127.0.0.1:5000";
const ANALYZE_URL = `${API_BASE_URL}/api/analyze`;

// Wait until the page is ready before connecting event listeners.
document.addEventListener("DOMContentLoaded", () => {
    initializeEvidenceLens();
});

function initializeEvidenceLens() {
    const claimInput = document.getElementById("claim-input");
    const analyzeButton = document.getElementById("analyse-btn");
    const resultContainer = document.getElementById("checker-result");

    // Connect the Analyse button.
    if (analyzeButton) {
        analyzeButton.addEventListener("click", analyzeClaim);
    }

    // Allow Ctrl+Enter to submit a claim.
    if (claimInput) {
        claimInput.addEventListener("keydown", (event) => {
            if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
                event.preventDefault();
                analyzeClaim();
            }
        });
    }

    // Connect example-claim buttons.
    document.querySelectorAll(
        "[data-claim], .example-claim, .example-btn"
    ).forEach((button) => {
        button.addEventListener("click", () => {
            const example =
                button.dataset.claim ||
                button.dataset.example ||
                button.textContent.trim();

            if (claimInput) {
                claimInput.value = example;
                claimInput.focus();
            }
        });
    });

    // Initialize the evidence checklist.
    initializeChecklist();

    // Keep the results section hidden until needed.
    if (resultContainer) {
        resultContainer.hidden = true;
    }
}

// ========================================
// Claim analysis
// ========================================

async function analyzeClaim() {
    const claimInput = document.getElementById("claim-input");
    const analyzeButton = document.getElementById("analyse-btn");
    const resultContainer = document.getElementById("checker-result");

    if (!claimInput || !resultContainer) {
        console.error(
            "EvidenceLens: Could not find #claim-input or #checker-result."
        );
        return;
    }

    const claim = claimInput.value.trim();

    if (!claim) {
        showError("Please enter a scientific claim before analysing.");
        claimInput.focus();
        return;
    }

    setButtonLoading(analyzeButton, true);
    showLoading();

    try {
        const response = await fetch(ANALYZE_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            body: JSON.stringify({ claim: claim })
        });

        const responseText = await response.text();
        let data = {};

        try {
            data = responseText ? JSON.parse(responseText) : {};
        } catch {
            throw new Error(
                `The backend returned an invalid response (HTTP ${response.status}).`
            );
        }

        if (!response.ok) {
            throw new Error(
                data.error ||
                data.message ||
                `The backend returned HTTP ${response.status}.`
            );
        }

        if (data.error) {
            throw new Error(data.error);
        }

        renderAnalysisResults(data, claim);

    } catch (error) {
        console.error("EvidenceLens analysis error:", error);

        let message = error.message || "An unexpected error occurred.";

        if (
            error instanceof TypeError ||
            message.toLowerCase().includes("failed to fetch")
        ) {
            message =
                "Could not connect to the backend. Make sure Flask is running " +
                "at http://127.0.0.1:5000 and then try again.";
        }

        showError(message);
    } finally {
        setButtonLoading(analyzeButton, false);
    }
}

// ========================================
// Loading, error and success states
// ========================================

function showLoading() {
    const container = document.getElementById("checker-result");

    if (!container) return;

    container.hidden = false;
    container.innerHTML = `
        <section class="analysis-loading" role="status" aria-live="polite">
            <div class="loading-spinner" aria-hidden="true"></div>
            <h3>Analysing your claim...</h3>
            <p>Searching scientific literature and examining relevant evidence.
               This may take a little while.</p>
        </section>
    `;
}

function showError(message) {
    const container = document.getElementById("checker-result");

    if (!container) return;

    container.hidden = false;
    container.innerHTML = `
        <section class="analysis-error" role="alert">
            <h3>Analysis could not be completed</h3>
            <p>${escapeHtml(message)}</p>
            <p>Please check that the backend is running and try again.</p>
        </section>
    `;
}

function showSuccess() {
    const container = document.getElementById("checker-result");

    if (container) {
        container.hidden = false;
    }
}

// ========================================
// Display the analysis response
// ========================================

function renderAnalysisResults(data, submittedClaim) {
    const container = document.getElementById("checker-result");

    if (!container) return;

    const publications = getPublications(data);
    const assessment = getOverallAssessment(data, publications);
    const searchStatus = data.search_status || "unknown";

    let html = `
        <section class="analysis-results">
            <header class="results-header">
                <p class="results-eyebrow">ANALYSIS REPORT</p>
                <h2>Claim Analysis</h2>
                <p class="submitted-claim">
                    <strong>Your claim:</strong>
                    ${escapeHtml(submittedClaim)}
                </p>
                <div class="assessment assessment-${escapeHtml(
                    assessment.toLowerCase().replace(/[^a-z0-9-]/g, "-")
                )}">
                    <span class="assessment-label">Overall assessment</span>
                    <h3>${escapeHtml(formatLabel(assessment))}</h3>
                    <p>
                        This is an automated assessment of retrieved text,
                        not a definitive scientific verdict.
                    </p>
                </div>
            </header>
    `;

    // Search status and result count.
    html += `
        <div class="results-summary">
            <p>
                <strong>Publications found:</strong>
                ${publications.length}
            </p>
            <p>
                <strong>Literature search:</strong>
                ${escapeHtml(formatLabel(searchStatus))}
            </p>
        </div>
    `;

    // Show backend message if it provides one.
    const summary = data.summary || data.message;

    if (summary && typeof summary === "string") {
        html += `
            <div class="analysis-summary">
                <p>${escapeHtml(summary)}</p>
            </div>
        `;
    }

    // Explain that relevance and inference are different.
    html += `
        <section class="results-explanation">
            <h3>How to interpret these results</h3>
            <p>
                <strong>Relevance</strong> estimates how closely a publication's
                text matches your claim. It does not measure study quality.
            </p>
            <p>
                <strong>Evidence relationship</strong> is the language model's
                estimate of whether the available text supports, contradicts,
                or is inconclusive about the claim.
            </p>
        </section>
    `;

    if (publications.length === 0) {
        html += `
            <section class="no-results">
                <h3>No publications to display</h3>
                <p>
                    The search did not return any publications for display.
                    Try rephrasing the claim using more specific scientific
                    terms.
                </p>
            </section>
        `;
    } else {
        const groups = {
            supports: [],
            contradicts: [],
            inconclusive: []
        };

        publications.forEach((publication) => {
            const category = normalizeEvidenceCategory(publication);
            groups[category].push(publication);
        });

        html += renderEvidenceGroup(
            "Supporting evidence",
            groups.supports,
            "supports"
        );

        html += renderEvidenceGroup(
            "Contradicting evidence",
            groups.contradicts,
            "contradicts"
        );

        html += renderEvidenceGroup(
            "Inconclusive evidence",
            groups.inconclusive,
            "inconclusive"
        );
    }

    // Display any model information returned by the API.
    const nliInfo = data.nli_analysis;

    if (nliInfo && typeof nliInfo === "object") {
        const available = nliInfo.available;

        html += `
            <section class="model-status">
                <h3>Inference model status</h3>
                <p>
                    ${available === true
                        ? "The inference model was available for this analysis."
                        : available === false
                            ? "The inference model was unavailable for this analysis."
                            : "Model availability information was returned by the backend."
                    }
                </p>
            </section>
        `;
    }

    html += `
        <footer class="results-disclaimer">
            <h3>Important limitation</h3>
            <p>
                EvidenceLens is an educational research-assistance prototype.
                Its results depend on the publications retrieved and the
                model's interpretation of the available text. They do not
                establish scientific truth, causation, or medical advice.
                Open the original publications and evaluate their methods,
                limitations, and conclusions before drawing conclusions.
            </p>
        </footer>
        </section>
    `;

    container.innerHTML = html;
    showSuccess();
}

// ========================================
// Publication data helpers
// ========================================

function getPublications(data) {
    // Support the common response formats used by the API.
    const possibleArrays = [
        data.results,
        data.publications,
        data.evidence,
        data.articles,
        data.data?.results,
        data.data?.publications
    ];

    for (const value of possibleArrays) {
        if (Array.isArray(value)) {
            return value;
        }
    }

    return [];
}

function getOverallAssessment(data, publications) {
    const possibleValues = [
        data.overall_assessment,
        data.overallAssessment,
        data.assessment,
        data.verdict,
        data.overall?.assessment,
        data.report?.overall_assessment
    ];

    for (const value of possibleValues) {
        if (typeof value === "string" && value.trim()) {
            return normalizeAssessment(value);
        }
    }

    // If the backend does not supply a verdict, summarize the publication
    // labels without treating them as proof of truth or falsehood.
    const categories = publications.map(normalizeEvidenceCategory);

    const supportCount = categories.filter(
        (item) => item === "supports"
    ).length;

    const contradictCount = categories.filter(
        (item) => item === "contradicts"
    ).length;

    if (supportCount > 0 && contradictCount === 0) {
        return "supporting evidence found";
    }

    if (contradictCount > 0 && supportCount === 0) {
        return "contradicting evidence found";
    }

    return "inconclusive";
}

function normalizeAssessment(value) {
    const text = String(value).toLowerCase().trim();

    if (text.includes("inconclusive") || text.includes("uncertain")) {
        return "inconclusive";
    }

    if (text.includes("contradict")) {
        return "contradicting evidence found";
    }

    if (text.includes("support")) {
        return "supporting evidence found";
    }

    if (text === "true" || text === "false") {
        return "inconclusive";
    }

    return value;
}

function normalizeEvidenceCategory(publication) {
    const nli = publication.nli || publication.inference || {};
    const raw = String(
        publication.evidence_category ||
        publication.category ||
        publication.label ||
        publication.assessment ||
        nli.label ||
        nli.prediction ||
        publication.nli_label ||
        "inconclusive"
    ).toLowerCase();

    if (
        raw.includes("contradict") ||
        raw.includes("refute") ||
        raw.includes("against")
    ) {
        return "contradicts";
    }

    if (
        raw.includes("support") ||
        raw.includes("entail") ||
        raw.includes("favor")
    ) {
        return "supports";
    }

    return "inconclusive";
}

function renderEvidenceGroup(title, publications, category) {
    if (!publications.length) {
        return "";
    }

    let html = `
        <section class="evidence-group evidence-${category}">
            <h3>${escapeHtml(title)}
                <span class="evidence-count">${publications.length}</span>
            </h3>
            <div class="publication-list">
    `;

    publications.forEach((publication) => {
        html += renderPublication(publication, category);
    });

    html += `
            </div>
        </section>
    `;

    return html;
}

function renderPublication(publication, category) {
    const title =
        publication.title ||
        publication.article_title ||
        publication.name ||
        "Untitled publication";

    const authors = formatAuthors(
        publication.authors || publication.author
    );

    const year =
        publication.year ||
        publication.pubYear ||
        publication.publication_year ||
        publication.pub_year ||
        "";

    const journal =
        publication.journal ||
        publication.journalTitle ||
        publication.source ||
        "";

    const abstract =
        publication.abstract ||
        publication.description ||
        publication.text ||
        publication.snippet ||
        "";

    const relevance = getRelevanceScore(publication);
    const relationship = getRelationshipLabel(publication);
    const sourceUrl = getPublicationUrl(publication);

    let html = `
        <article class="publication-card">
            <div class="publication-meta">
                <span class="evidence-tag evidence-tag-${category}">
                    ${escapeHtml(formatLabel(category))}
                </span>
                ${year
                    ? `<span class="publication-year">${escapeHtml(String(year))}</span>`
                    : ""
                }
            </div>

            <h4>${escapeHtml(title)}</h4>
    `;

    if (authors) {
        html += `<p class="publication-authors">${escapeHtml(authors)}</p>`;
    }

    if (journal) {
        html += `<p class="publication-journal">${escapeHtml(journal)}</p>`;
    }

    if (relevance !== null) {
        html += `
            <p class="publication-relevance">
                <strong>Text relevance:</strong>
                ${escapeHtml(formatScore(relevance))}
            </p>
        `;
    }

    if (relationship) {
        html += `
            <p class="publication-relationship">
                <strong>Model assessment:</strong>
                ${escapeHtml(relationship)}
            </p>
        `;
    }

    if (abstract) {
        html += `
            <p class="publication-abstract">
                ${escapeHtml(limitText(String(abstract), 650))}
            </p>
        `;
    }

    if (sourceUrl) {
        html += `
            <a class="publication-link"
               href="${escapeHtml(sourceUrl)}"
               target="_blank"
               rel="noopener noreferrer">
                Read original publication ↗
            </a>
        `;
    }

    html += `</article>`;

    return html;
}

function getRelevanceScore(publication) {
    const possibleValues = [
        publication.relevance_score,
        publication.relevanceScore,
        publication.similarity,
        publication.score,
        publication.tfidf_score,
        publication.relevance
    ];

    for (const value of possibleValues) {
        if (typeof value === "number" && Number.isFinite(value)) {
            return value;
        }

        if (
            typeof value === "string" &&
            value.trim() !== "" &&
            Number.isFinite(Number(value))
        ) {
            return Number(value);
        }
    }

    return null;
}

function formatScore(score) {
    // Keep the original numeric scale; don't assume every score is a percent.
    return Number(score).toFixed(3);
}

function getRelationshipLabel(publication) {
    const nli = publication.nli || publication.inference || {};

    return (
        nli.label ||
        nli.prediction ||
        publication.nli_label ||
        publication.relationship ||
        publication.evidence_relationship ||
        ""
    );
}

function getPublicationUrl(publication) {
    const candidates = [
        publication.url,
        publication.link,
        publication.source_url,
        publication.article_url,
        publication.fullTextUrl,
        publication.full_text_url
    ];

    for (const candidate of candidates) {
        const safeUrl = validateHttpUrl(candidate);

        if (safeUrl) {
            return safeUrl;
        }
    }

    const doi = publication.doi;

    if (typeof doi === "string" && doi.trim()) {
        const cleanedDoi = doi.trim().replace(/^https?:\/\/doi\.org\//i, "");

        if (!cleanedDoi.includes(" ")) {
            return `https://doi.org/${encodeURIComponent(cleanedDoi).replace(/%2F/gi, "/")}`;
        }
    }

    const pmid = publication.pmid || publication.PMID;

    if (pmid && /^[0-9]+$/.test(String(pmid))) {
        return `https://pubmed.ncbi.nlm.nih.gov/${encodeURIComponent(String(pmid))}/`;
    }

    const pmcid = publication.pmcid || publication.pmcId;

    if (typeof pmcid === "string" && /^PMC[0-9]+$/i.test(pmcid)) {
        return `https://europepmc.org/articles/${encodeURIComponent(pmcid)}`;
    }

    return "";
}

function validateHttpUrl(value) {
    if (typeof value !== "string" || !value.trim()) {
        return "";
    }

    try {
        const parsed = new URL(value, window.location.href);

        if (parsed.protocol === "https:" || parsed.protocol === "http:") {
            return parsed.href;
        }
    } catch {
        // Ignore invalid URLs.
    }

    return "";
}

function formatAuthors(authors) {
    if (Array.isArray(authors)) {
        return authors
            .map((author) => {
                if (typeof author === "string") return author;
                return author.name || author.fullName || "";
            })
            .filter(Boolean)
            .slice(0, 5)
            .join(", ");
    }

    if (typeof authors === "string") {
        return authors;
    }

    return "";
}

// ========================================
// General helper functions
// ========================================

function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (character) => {
        const entities = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#39;"
        };

        return entities[character];
    });
}

function formatLabel(value) {
    return String(value ?? "Unknown")
        .replace(/[_-]+/g, " ")
        .replace(/\s+/g, " ")
        .trim()
        .replace(/\b\w/g, (character) => character.toUpperCase());
}

function limitText(value, maxLength) {
    if (value.length <= maxLength) {
        return value;
    }

    return `${value.slice(0, maxLength).trim()}...`;
}

function setButtonLoading(button, isLoading) {
    if (!button) return;

    if (isLoading) {
        button.dataset.originalText = button.textContent.trim();
        button.disabled = true;
        button.textContent = "Analysing...";
        button.setAttribute("aria-busy", "true");
    } else {
        button.disabled = false;
        button.textContent = button.dataset.originalText || "Analyse Claim";
        button.removeAttribute("aria-busy");
    }
}

// ========================================
// Evidence checklist
// ========================================

function initializeChecklist() {
    const checkboxes = Array.from(
        document.querySelectorAll(
            'input[type="checkbox"][data-evidence-check], ' +
            '.evidence-checklist input[type="checkbox"], ' +
            '#evidence-checklist input[type="checkbox"]'
        )
    );

    const progressElement =
        document.getElementById("checklist-progress") ||
        document.getElementById("progress-text");

    const progressBar =
        document.getElementById("checklist-progress-bar") ||
        document.querySelector(".checklist-progress-bar");

    const resetButton =
        document.getElementById("reset-checklist") ||
        document.getElementById("reset-checklist-btn");

    function updateProgress() {
        const total = checkboxes.length;
        const checked = checkboxes.filter((box) => box.checked).length;
        const percentage = total ? Math.round((checked / total) * 100) : 0;

        if (progressElement) {
            progressElement.textContent =
                `${checked} of ${total} completed (${percentage}%)`;
        }

        if (progressBar) {
            if ("value" in progressBar && progressBar.tagName === "PROGRESS") {
                progressBar.value = percentage;
            } else {
                progressBar.style.width = `${percentage}%`;
                progressBar.setAttribute("aria-valuenow", String(percentage));
            }
        }

        checkboxes.forEach((checkbox) => {
            const label = checkbox.closest("label");

            if (label) {
                label.classList.toggle("completed", checkbox.checked);
            }
        });
    }

    checkboxes.forEach((checkbox) => {
        checkbox.addEventListener("change", updateProgress);
    });

    if (resetButton) {
        resetButton.addEventListener("click", () => {
            checkboxes.forEach((checkbox) => {
                checkbox.checked = false;
            });

            updateProgress();
        });
    }

    updateProgress();
}
