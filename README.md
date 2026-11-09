# Medical Terminology Translator API

A lightweight **FastAPI + LangChain + OpenRouter/OpenAI-compatible LLM + MLflow** service that converts medical terminology into patient-friendly language.

The application is designed as a small, modular healthcare AI API with separate layers for:

- FastAPI routing
- Request/response models
- Translation service orchestration
- LLM configuration
- Prompt management
- Structured Pydantic output
- Logging with Loguru
- MLflow request/run tracking

> **Important:** This README is based on the source code contained in the supplied ZIP archive. The archive does not contain a `requirements.txt`, `.env.example`, Docker configuration, tests, or deployment manifests, so those are not documented as existing project files.

---

## 1. Project Overview

The application exposes a REST endpoint that accepts a medical term such as:

```text
Hypertension
```

and uses an LLM to produce a structured response containing:

- Original medical term
- Patient-friendly terminology
- Simple explanation
- Severity classification

Example conceptual output:

```json
{
  "medical_term": "Hypertension",
  "plain_language": "High blood pressure",
  "explanation": "Blood pressure that stays higher than normal.",
  "severity": "Moderate"
}
```

The application also initializes MLflow tracking for translation requests and uses Loguru for application and error logging.

---

## 2. High-Level Architecture

```text
                         Client
                           |
                           | POST /translator/translate
                           v
                +----------------------+
                | FastAPI Router       |
                | translator_router.py |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | TranslatorService    |
                | translator_service.py|
                +----------+-----------+
                           |
             +-------------+-------------+
             |                           |
             v                           v
   +-------------------+       +-------------------+
   | Translator Prompt |       | LLM Configuration |
   | translator_prompt |       | llm_config.py     |
   +---------+---------+       +---------+---------+
             |                           |
             +-------------+-------------+
                           |
                           v
                +----------------------+
                | ChatOpenAI / LLM     |
                | OpenRouter-compatible|
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Structured Output    |
                | MedicalTranslation   |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | MLflow Tracking      |
                +----------+-----------+
                           |
                           v
                       Response
```

---

## 3. Project Structure

```text
app/
├── agents/
│   └── # currently empty
│
├── config/
│   ├── llm_config.py
│   └── __pycache__/
│
├── implementations/
│   ├── translator_impl.py
│   └── __pycache__/
│
├── memory/
│   └── # currently empty
│
├── models/
│   ├── request_model.py
│   ├── response_model.py
│   └── __pycache__/
│
├── prompts/
│   ├── translator_prompt.py
│   └── __pycache__/
│
├── routers/
│   ├── translator_router.py
│   └── __pycache__/
│
├── services/
│   ├── translator_service.py
│   └── __pycache__/
│
├── utils/
│   ├── logger.py
│   └── __pycache__/
│
└── main.py
```

### Layer responsibilities

| Layer | File | Responsibility |
|---|---|---|
| Application | `main.py` | Creates FastAPI application and registers routes |
| Router | `routers/translator_router.py` | Defines HTTP API endpoint |
| Service | `services/translator_service.py` | Orchestrates prompt, LLM, structured output and MLflow |
| Configuration | `config/llm_config.py` | Creates configured `ChatOpenAI` client |
| Prompt | `prompts/translator_prompt.py` | Defines system and human prompts |
| Implementation | `implementations/translator_impl.py` | Defines `MedicalTranslation` structured schema |
| Request Model | `models/request_model.py` | Validates incoming API request |
| Response Model | `models/response_model.py` | Defines intended API response contract |
| Logging | `utils/logger.py` | Configures Loguru console/file/error logging |
| Memory | `memory/` | Directory exists but no memory implementation is currently present |
| Agents | `agents/` | Directory exists but no agent implementation is currently present |

---

# 4. API Endpoints

## 4.1 Root Endpoint

```http
GET /
```

Example response:

```json
{
  "application": "Medical Terminology Translator",
  "status": "Running"
}
```

---

## 4.2 Health Endpoint

```http
GET /health
```

Example response:

```json
{
  "status": "Healthy"
}
```

> The current health endpoint checks application availability only. It does not currently verify LLM connectivity, MLflow connectivity, or other external dependencies.

---

## 4.3 Translation Endpoint

```http
POST /translator/translate
```

### Request body

The request model is:

```python
class TranslatorRequest(BaseModel):
    medical_term: str
    language: str = "English"
    user_id: str = "anonymous"
    session_id: str = "session_001"
```

Example:

```json
{
  "medical_term": "Hypertension",
  "language": "English",
  "user_id": "USR001",
  "session_id": "SESSION001"
}
```

### Current implementation behavior

Although `language`, `user_id`, and `session_id` are accepted by the request model, the router currently passes only:

```python
medical_term=request.medical_term
```

to the service.

Therefore, the current implementation does **not yet use the requested language, user ID, or session ID** in the translation workflow.

### Conceptual response

The service returns the `MedicalTranslation` Pydantic object:

```json
{
  "medical_term": "Hypertension",
  "plain_language": "High blood pressure",
  "explanation": "Blood pressure that remains higher than normal.",
  "severity": "Moderate"
}
```

Note that the currently implemented router/service path returns `MedicalTranslation`; the separate `TranslatorResponse` model is defined but is not currently used by the endpoint.

---

# 5. LLM Configuration

LLM configuration is implemented in:

```text
app/config/llm_config.py
```

The application uses:

```python
ChatOpenAI(
    model=os.getenv("model"),
    api_key=os.getenv("openrouter_api_key"),
    base_url=os.getenv("base_url")
)
```

This allows an OpenAI-compatible provider such as OpenRouter to be configured through environment variables.

## Environment variables

The source code expects:

```env
model=<your-model-name>
openrouter_api_key=<your-api-key>
base_url=<openai-compatible-base-url>
```

The application calls:

```python
load_dotenv()
```

so a `.env` file can be used during local development.

### Security

Do not commit API keys to source control.

Recommended:

```text
.env
```

should be excluded from Git.

---

# 6. Translation Prompt

The prompt is defined in:

```text
app/prompts/translator_prompt.py
```

The system instruction positions the model as an expert medical terminology translator and asks it to:

- Translate medical terms into simple patient-friendly language
- Use plain English
- Avoid unnecessary medical jargon
- Keep explanations concise
- Return structured output

The human message supplies:

```text
Medical Term:
{medical_term}
```

The prompt is represented as a LangChain:

```python
ChatPromptTemplate
```

and is composed with the structured LLM using the pipe operator:

```python
translator_prompt | structured_llm
```

---

# 7. Structured LLM Output

The project defines the following Pydantic schema:

```python
class MedicalTranslation(BaseModel):

    medical_term: str
    plain_language: str
    explanation: str
    severity: str
```

The intended meaning is:

| Field | Description |
|---|---|
| `medical_term` | Original medical terminology |
| `plain_language` | Patient-friendly equivalent |
| `explanation` | Simple explanation |
| `severity` | Expected classification such as Low, Moderate, High |

The service uses:

```python
llm.with_structured_output(MedicalTranslation)
```

This is preferable to manually parsing arbitrary JSON because the LLM integration is instructed to produce the target Pydantic structure.

---

# 8. Translation Execution Flow

The main execution path is:

```text
POST /translator/translate
        |
        v
TranslatorRequest validation
        |
        v
TranslatorService.translate()
        |
        v
get_llm()
        |
        v
ChatOpenAI
        |
        v
with_structured_output(MedicalTranslation)
        |
        v
translator_prompt
        |
        v
chain.invoke()
        |
        v
MedicalTranslation
        |
        v
MLflow logging
        |
        v
API response
```

---

# 9. MLflow Integration

MLflow is initialized in:

```text
app/services/translator_service.py
```

Tracking URI:

```text
http://localhost:5000
```

Experiment:

```text
Medical_Terminology_Translator_Interface
```

Each translation creates an MLflow run.

The application records information including:

- Start time
- Input medical term
- Response
- Source
- Endpoint
- Execution time

Example tags:

```text
source = FastAPI
endpoint = translator/translate
```

## MLflow dependency

The current source assumes an MLflow server is available at:

```text
http://localhost:5000
```

For example, a local MLflow server can be started separately using a suitable MLflow installation/configuration.

---

# 10. Logging

Logging is implemented using **Loguru**.

The logger writes to:

### Console

INFO and higher messages are displayed on stdout.

### Application log

```text
logs/healthcare.log
```

Configuration:

- Rotation: 10 MB
- Retention: 30 days
- Compression: ZIP

### Error log

```text
logs/error.log
```

Configuration:

- Level: ERROR
- Rotation: 5 MB
- Retention: 60 days
- Compression: ZIP

The `logs/` directory is created automatically.

---

# 11. Error Handling

The router wraps the translation call in:

```python
try:
    ...
except Exception as ex:
    ...
```

and converts unexpected exceptions into:

```http
500 Internal Server Error
```

with:

```json
{
  "detail": "<error message>"
}
```

The service also logs and re-raises exceptions.

### Production consideration

Returning raw exception messages through an API can expose implementation details. In production, it is preferable to:

- Return a generic error message
- Generate a correlation/request ID
- Log the detailed exception internally
- Avoid exposing API/provider/database information

---

# 12. Request Validation

The request model applies:

```text
medical_term:
    minimum length = 2
    maximum length = 200
```

This prevents empty or excessively large medical-term inputs.

Default values are provided for:

```text
language = English
user_id = anonymous
session_id = session_001
```

FastAPI automatically exposes validation errors when an invalid request is supplied.

---

# 13. Running the Application

Because the ZIP archive does not contain a requirements file, install the dependencies appropriate to the source code.

Core packages include:

```bash
pip install fastapi uvicorn pydantic python-dotenv
pip install langchain-core langchain-openai
pip install mlflow loguru
```

Then start the application from the project directory.

Because the source imports modules such as:

```python
from utils.logger import logger
from routers.translator_router import router
```

the `app` directory needs to be on the Python module path.

One practical approach is:

```bash
cd app
uvicorn main:app --reload
```

The API should then be available at:

```text
http://localhost:8000
```

Swagger/OpenAPI documentation:

```text
http://localhost:8000/docs
```

ReDoc:

```text
http://localhost:8000/redoc
```

---

# 14. Example API Request

Using `curl`:

```bash
curl -X POST "http://localhost:8000/translator/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "medical_term": "Hypertension",
    "language": "English",
    "user_id": "USR001",
    "session_id": "SESSION001"
  }'
```

---

# 15. Example Translation

### Input

```json
{
  "medical_term": "Hypertension",
  "language": "English"
}
```

### Expected structure

```json
{
  "medical_term": "Hypertension",
  "plain_language": "High blood pressure",
  "explanation": "A condition where blood pressure stays higher than normal.",
  "severity": "Moderate"
}
```

The exact LLM-generated wording may differ.

---

# 16. Key Technologies

| Technology | Purpose |
|---|---|
| Python | Application language |
| FastAPI | REST API framework |
| Pydantic | Request/response and structured data validation |
| LangChain Core | Prompt and LLM orchestration |
| LangChain OpenAI | OpenAI-compatible chat model integration |
| OpenRouter-compatible API | LLM provider abstraction through configurable base URL |
| Loguru | Application logging |
| MLflow | Experiment/run tracking |
| python-dotenv | Environment variable loading |

---

# 17. Current Implementation Strengths

The current implementation provides a useful foundation for a healthcare terminology translation service.

### Separation of concerns

The code separates:

```text
Router
  -> Service
      -> Prompt
      -> LLM
      -> Structured Output
      -> MLflow
```

### Structured output

Using:

```python
with_structured_output(MedicalTranslation)
```

reduces dependence on fragile manual JSON parsing.

### Centralized LLM configuration

The LLM provider/model/base URL are externalized through environment variables.

### Observability foundation

The project already includes:

- Loguru application logging
- Error logging
- MLflow tracking
- Execution-time measurement

### API documentation

FastAPI automatically provides OpenAPI/Swagger documentation.

---

# 18. Important Issues Identified

The following items should be addressed before treating this as a production-ready healthcare application.

## 18.1 Language is not implemented

The request accepts:

```json
"language": "Telugu"
```

but the service currently only sends:

```python
medical_term
```

to the prompt.

The prompt itself asks for plain English.

### Recommended improvement

Pass `language` into the prompt:

```text
Medical Term:
{medical_term}

Target Language:
{language}
```

and explicitly instruct the model to produce the patient-friendly explanation in the requested language.

---

## 18.2 User/session information is not used

The request model accepts:

```text
user_id
session_id
```

but these values are not forwarded to the service or MLflow.

If session-level conversational memory is intended, this needs to be implemented explicitly.

---

## 18.3 Response model is unused

The project defines:

```text
models/response_model.py
```

with:

```python
TranslatorResponse
```

but the router does not specify:

```python
response_model=TranslatorResponse
```

and the service returns:

```python
MedicalTranslation
```

### Recommended improvement

Either:

1. Use `TranslatorResponse` as the API response contract, or
2. Remove the unused response model.

---

## 18.4 PydanticOutputParser is created but unused

`translator_impl.py` creates:

```python
translation_parser = PydanticOutputParser(
    pydantic_object=MedicalTranslation
)
```

but the service uses:

```python
llm.with_structured_output(MedicalTranslation)
```

Therefore, the explicit `PydanticOutputParser` is currently redundant.

Use one structured-output strategy consistently.

---

## 18.5 MLflow response logging should be reviewed

The source contains:

```python
mlflow.log_param("response", response)
```

`response` is a Pydantic object rather than a simple scalar parameter.

MLflow parameters are intended for parameter-like values. A safer implementation would serialize the structured result, or use an appropriate artifact/JSON logging mechanism.

For example:

```python
response.model_dump_json()
```

could be stored as an artifact or appropriately serialized value.

---

## 18.6 Execution-time log message is inconsistent

The code calculates:

```python
execution_time = end_ts - start_ts
```

which is in seconds, but logs:

```text
Translation completed in ... ms
```

The message should either say seconds or multiply the value by 1000 before reporting milliseconds.

---

## 18.7 MLflow configuration is hard-coded

The source contains:

```python
mlflow.set_tracking_uri("http://localhost:5000")
```

A production implementation should move this into configuration:

```env
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=Medical_Terminology_Translator_Interface
```

---

## 18.8 No authentication/authorization

The API currently has no:

- Authentication
- Authorization
- API key validation
- OAuth2/JWT
- RBAC

This is especially important if deployed in a healthcare environment.

---

## 18.9 No PHI/privacy controls

The service accepts user/session identifiers and medical terminology, but there is currently no explicit:

- PHI detection
- Data minimization
- Data masking
- Consent management
- Audit policy
- Retention policy
- Encryption policy

These should be considered before handling real patient data.

---

## 18.10 No automated tests

The ZIP archive does not contain a test suite.

Recommended tests include:

```text
tests/
├── test_health.py
├── test_translator_router.py
├── test_translator_service.py
├── test_request_model.py
└── test_translation_schema.py
```

---

# 19. Recommended Production Architecture

A more production-oriented design could evolve into:

```text
                       API Gateway
                            |
                            v
                     Authentication
                            |
                            v
                     FastAPI Router
                            |
                            v
                  Translator Service
                            |
              +-------------+-------------+
              |                           |
              v                           v
        Input Guardrails            Session Context
              |                           |
              +-------------+-------------+
                            |
                            v
                     Prompt Builder
                            |
                            v
                      LLM Gateway
                            |
                 +----------+----------+
                 |                     |
                 v                     v
             OpenRouter          Other LLM Provider
                 |
                 v
          Structured Output
                 |
                 v
        Output/Clinical Guardrails
                 |
                 v
             Response
                 |
        +--------+--------+
        |                 |
        v                 v
     Audit Log          MLflow
```

---

# 20. Recommended Healthcare Guardrails

For a healthcare-facing implementation, consider adding:

### Input guardrails

- Prompt injection detection
- Input length limits
- Malicious content detection
- Medical-term validation
- PHI handling rules

### Output guardrails

- Structured schema validation
- Unsupported medical claim detection
- Hallucination controls
- Medical terminology consistency
- Safety disclaimer where appropriate
- Language validation

### Clinical safety

The service should clearly distinguish:

```text
Terminology explanation
        !=
Medical diagnosis
        !=
Treatment recommendation
```

The translator should explain terminology without presenting itself as a diagnostic or treatment decision system.

---

# 21. Suggested Future Enhancements

## Phase 1 — API quality

- Use `TranslatorResponse`
- Add `response_model`
- Pass `language`
- Validate language
- Add request IDs
- Add structured error responses
- Add automated tests

## Phase 2 — LLM reliability

- Add structured output validation
- Add retry policies
- Add timeout handling
- Add model fallback
- Add token/cost tracking
- Add LLM latency metrics

## Phase 3 — Healthcare safety

- Add clinical guardrails
- Add PHI detection
- Add audit logging
- Add RBAC
- Add data retention policies
- Add consent handling

## Phase 4 — Agentic architecture

Potential agents could include:

```text
Translation Orchestrator
        |
        +-- Medical Term Agent
        |
        +-- Terminology Normalization Agent
        |
        +-- Patient Explanation Agent
        |
        +-- Language Translation Agent
        |
        +-- Safety/Clinical Guardrail Agent
        |
        +-- Quality Critic Agent
```

For this relatively simple task, however, a deterministic service + structured LLM workflow may be preferable to introducing multiple agents unless additional reasoning/tool use is required.

---

# 22. Observability Recommendations

The current application has basic logging and MLflow tracking.

A mature implementation could add:

```text
Request ID
    |
    +-- API latency
    +-- LLM latency
    +-- Prompt tokens
    +-- Completion tokens
    +-- Total tokens
    +-- Model
    +-- Provider
    +-- Estimated cost
    +-- Validation result
    +-- Retry count
    +-- Error category
```

Possible monitoring stack:

```text
FastAPI
   |
   +--> Loguru
   |
   +--> OpenTelemetry
   |
   +--> Prometheus
   |
   +--> Grafana
   |
   +--> MLflow
```

---

# 23. Security Recommendations

Before production deployment:

- Store secrets in a secret manager
- Never commit API keys
- Enable TLS/HTTPS
- Implement authentication
- Implement authorization
- Validate all inputs
- Avoid returning raw exception details
- Sanitize logs
- Avoid logging sensitive patient information
- Implement audit trails
- Apply rate limiting
- Configure secure CORS
- Restrict MLflow access
- Apply least-privilege permissions

---

# 24. Deployment Considerations

A production deployment could use:

```text
                    Load Balancer
                         |
                         v
                    API Gateway
                         |
                         v
                  FastAPI Containers
                   /      |      \
                  /       |       \
              App-1     App-2    App-3
                  \       |       /
                   \      |      /
                         |
                         v
                    LLM Provider
                         |
             +-----------+-----------+
             |                       |
             v                       v
         MLflow                  Observability
```

Containerization with Docker and orchestration with Kubernetes would be natural next steps if the service needs horizontal scaling.

---

# 25. Files Not Present in the Supplied Archive

The archive contains the Python application source but does not contain several files normally expected in a complete deployable project:

```text
requirements.txt
pyproject.toml
README.md
.env.example
Dockerfile
docker-compose.yml
tests/
.gitignore
k8s/
```

These can be added as the project matures.

---

# 26. Suggested `.env.example`

A useful starting point would be:

```env
# LLM
model=<your-model-name>
openrouter_api_key=<your-openrouter-api-key>
base_url=<your-openai-compatible-base-url>

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=Medical_Terminology_Translator_Interface
```

Do not put real credentials in this file.

---

# 27. Suggested Dependency Set

A starting dependency list based on the imports in the supplied source is:

```text
fastapi
uvicorn
pydantic
python-dotenv
langchain-core
langchain-openai
mlflow
loguru
```

Pin compatible versions for production rather than relying on unpinned latest releases.

---

# 28. API Contract Summary

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Application status |
| GET | `/health` | Health check |
| POST | `/translator/translate` | Translate medical terminology |

---

# 29. End-to-End Summary

The application implements a straightforward LLM-powered medical terminology translation workflow:

```text
Client
  |
  | Medical term
  v
FastAPI
  |
  v
Request validation
  |
  v
Translator Service
  |
  +--> Prompt
  |
  +--> Configured Chat LLM
  |
  +--> Pydantic structured output
  |
  +--> MLflow tracking
  |
  +--> Loguru logging
  |
  v
Patient-friendly translation
```

The codebase has a clean initial separation between API, service, configuration, prompts, models, implementation, and logging. The main work required for production readiness is around **configuration management, response-contract consistency, multilingual support, MLflow serialization, testing, security, healthcare privacy, clinical guardrails, and operational observability**.

---

## 30. Conclusion

This project is a good foundational implementation of a **Medical Terminology Translator API** using FastAPI and an OpenAI-compatible LLM through LangChain.

Its current architecture is intentionally simple:

```text
FastAPI
   ↓
Service
   ↓
Prompt + LLM
   ↓
Structured Pydantic Output
   ↓
MLflow + Loguru
```

The most important architectural observation is that the project currently behaves as an **LLM-powered service**, not yet as a multi-agent system. The empty `agents/` and `memory/` directories indicate potential extension points, but no agent or memory workflow is implemented in the supplied source.

For a production healthcare platform, the next architectural priorities should be **safety/guardrails, privacy/security, observability, robust error handling, configuration externalization, testing, and explicit clinical-safety boundaries**.
