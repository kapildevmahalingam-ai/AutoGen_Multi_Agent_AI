# AutoGen Multi-Agent Customer Support Chatbot (OpenAI)

A Streamlit customer-support chatbot built with **Microsoft AutoGen AgentChat** and the **OpenAI API**.

## 1. Agent design

The application uses five agents:

| Agent | Responsibility |
|---|---|
| `CategoryAnalysisAgent` | Reads the customer's message, classifies it and selects the correct specialist |
| `AITechnologyAgent` | AI features, integrations, technical capabilities, setup and compatibility |
| `ProductReviewFeedbackAgent` | Product reviews, criticism, complaints and customer feedback |
| `ProductEnhancementAgent` | Feature requests, usability improvements and enhancement ideas |
| `PricingIssuesSupportAgent` | Pricing, plans, billing, troubleshooting, product issues and bugs |

The category agent routes each question to **one specialist**. The selected specialist writes the customer-facing answer.

## 2. Workflow

```text
Customer question
       |
       v
CategoryAnalysisAgent
       |
       +---- AI Technology ----------> AITechnologyAgent
       |
       +---- Product Review/Feedback -> ProductReviewFeedbackAgent
       |
       +---- Product Enhancement ----> ProductEnhancementAgent
       |
       +---- Pricing/Issues/Bugs -----> PricingIssuesSupportAgent
                                             |
                                             v
                                      Customer response
                                             |
                                             v
                                      SQLite memory
                                             |
                                             v
                                       Conversation End
```

Every specialist is instructed to end its response with `End`. The application also enforces this ending if the model omits it.

## 3. Requirements

- Python 3.10 or later
- An OpenAI API key
- Internet connection for OpenAI API calls

## 4. Installation

Open a terminal in this project directory.

### Windows

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 5. Configure OpenAI API key

Create a file named `.env` in the same folder as `app.py`:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4.1-mini
```

Create an API key here:

https://platform.openai.com/api-keys

API usage is billed separately from ChatGPT subscriptions. Check your API billing and usage here:

https://platform.openai.com/settings/organization/billing/overview

Do not upload your `.env` file to GitHub or share your API key.

## 6. Run the application

```bash
streamlit run app.py
```

Streamlit will show a local URL, normally:

```text
http://localhost:8501
```

## 7. Memory function

The application creates `customer_support_memory.db` automatically.

- Exact repeat questions are retrieved using a normalized question hash.
- Very similar questions are checked using Python's `difflib` similarity matching.
- A match returns the saved category and answer without calling OpenAI again.
- New questions and answers are stored in SQLite.
- Use **Delete saved memory** in the sidebar to clear the memory database.

This is lightweight local text matching, not semantic vector search. For a larger production knowledge base, consider embeddings and a vector database.

## 8. Console activity

The terminal running Streamlit prints workflow events, including:

- Category agent analyzing the message
- Selected category
- Specialist agent assigned
- Specialist response completed
- Memory hit or workflow error

The same activity events are also displayed in the Streamlit activity panel.

## 9. Example questions

1. "Can your product connect with other AI platforms?"
2. "I am unhappy with the latest product update."
3. "Please add dark mode to the dashboard."
4. "What plans and pricing do you offer?"
5. "The export button is not working. How can I fix it?"

## 10. Important notes

- This is a learning/prototype implementation. It does not connect to a real product catalogue, billing system, ticketing system or CRM.
- The agents must not invent prices, policies, product capabilities, bug fixes or roadmap commitments.
- Do not enter passwords, API keys, full payment-card details or other sensitive customer information.
- Review generated answers before using them in a live customer-support environment.
- OpenAI API usage is separate from ChatGPT plan billing.

## 11. Project files

```text
autogen_customer_support/
├── app.py
├── requirements.txt
├── .env.example
└── README.md
```
