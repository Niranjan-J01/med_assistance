const input = document.getElementById("queryInput");
const sendButton = document.getElementById("sendButton");
const messages = document.getElementById("messages");
const welcome = document.getElementById("welcome");


input.addEventListener("keydown", function (event) {

    if (event.key === "Enter" && !event.shiftKey) {

        event.preventDefault();

        sendMessage();
    }
});


function useExample(question) {

    input.value = question;

    input.focus();
}


function addUserMessage(text) {

    const message = document.createElement("div");

    message.className = "message user-message";

    message.textContent = text;

    messages.appendChild(message);
}


function addLoadingMessage() {

    const loading = document.createElement("div");

    loading.id = "loading";

    loading.className = "loading";

    loading.textContent = "Retrieving medical evidence and generating answer...";

    messages.appendChild(loading);
}


function removeLoadingMessage() {

    const loading = document.getElementById("loading");

    if (loading) {
        loading.remove();
    }
}


function parseCitation(citation) {

    const sourceMatch = citation.match(/SOURCE\s*:\s*(.+)/);

    const pageMatch = citation.match(/PAGE\s*:\s*(.+)/);


    return {

        source: sourceMatch
            ? sourceMatch[1].trim()
            : "Unknown Source",

        page: pageMatch
            ? pageMatch[1].trim()
            : "N/A"
    };
}


function addAssistantMessage(data) {

    const container = document.createElement("div");

    container.className = "assistant-container";


    if (
        data.safety_status === "emergency" ||
        data.safety_status === "unsafe_request"
    ) {

        const alert = document.createElement("div");

        alert.className = "safety-alert";

        alert.textContent = data.answer;

        container.appendChild(alert);

        messages.appendChild(container);

        return;
    }


    const answer = document.createElement("div");

    answer.className = "message assistant-message";

    answer.textContent = data.answer;

    container.appendChild(answer);


    const refusalDetected =
        data.confidence?.reason?.refusal_detected === true;


    if (data.confidence) {

        const metaRow = document.createElement("div");

        metaRow.className = "meta-row";


        const confidence = document.createElement("span");

        confidence.className =
            refusalDetected
                ? "confidence insufficient"
                : "confidence";


        if (refusalDetected) {

            confidence.textContent = "Insufficient Evidence";

        } else {

            const percent = Math.round(
                data.confidence.score * 100
            );

            confidence.textContent =
                `${data.confidence.level} Evidence Confidence — ${percent}%`;
        }


        metaRow.appendChild(confidence);

        container.appendChild(metaRow);


        const confidenceDetails =
            document.createElement("details");


        const confidenceSummary =
            document.createElement("summary");


        confidenceSummary.textContent =
            "Why this confidence?";


        confidenceDetails.appendChild(
            confidenceSummary
        );


        const reason =
            data.confidence.reason || {};


        const confidenceText =
            document.createElement("p");


        confidenceText.textContent =
            `Best reranker score: ${reason.best_cross_score ?? "N/A"}
Average similarity: ${reason.average_similarity ?? "N/A"}
Refusal detected: ${reason.refusal_detected ?? false}

Evidence Confidence is a heuristic based on retrieval and reranking signals. It is not the probability that the medical answer is correct.`;


        confidenceDetails.appendChild(
            confidenceText
        );


        container.appendChild(
            confidenceDetails
        );
    }


    if (
        !refusalDetected &&
        Array.isArray(data.citations) &&
        data.citations.length > 0
    ) {

        const uniqueCitations = [
            ...new Set(data.citations)
        ];


        const sourcesPanel =
            document.createElement("details");


        const sourcesSummary =
            document.createElement("summary");


        sourcesSummary.textContent =
            `Supporting Sources (${uniqueCitations.length})`;


        sourcesPanel.appendChild(
            sourcesSummary
        );


        uniqueCitations.forEach(citation => {

            const parsed =
                parseCitation(citation);


            const source =
                document.createElement("div");


            source.className = "source";


            source.textContent =
                `${parsed.source} — Page ${parsed.page}`;


            sourcesPanel.appendChild(source);
        });


        container.appendChild(
            sourcesPanel
        );
    }


    messages.appendChild(container);
}


function addErrorMessage() {

    const container =
        document.createElement("div");


    container.className =
        "assistant-container";


    const error =
        document.createElement("div");


    error.className =
        "safety-alert";


    error.textContent =
        "Unable to reach the medical assistant backend. Please check that the server is running.";


    container.appendChild(error);

    messages.appendChild(container);
}


async function sendMessage() {

    const query = input.value.trim();


    if (!query) {
        return;
    }


    welcome.style.display = "none";


    addUserMessage(query);


    input.value = "";

    sendButton.disabled = true;


    addLoadingMessage();


    window.scrollTo({
        top: document.body.scrollHeight,
        behavior: "smooth"
    });


    try {

        const response = await fetch("/chat", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                query: query
            })
        });


        if (!response.ok) {
            throw new Error(
                `Backend error: ${response.status}`
            );
        }


        const data = await response.json();


        removeLoadingMessage();


        addAssistantMessage(data);


    } catch (error) {

        console.error(error);

        removeLoadingMessage();

        addErrorMessage();

    } finally {

        sendButton.disabled = false;

        input.focus();


        window.scrollTo({
            top: document.body.scrollHeight,
            behavior: "smooth"
        });
    }
}