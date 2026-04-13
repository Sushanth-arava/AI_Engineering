# LangChain Prompts — Research Paper Summarization Tool

A Streamlit-based web application that summarizes research papers using LangChain and OpenAI's chat models. Users select a paper, an explanation style, and a desired length; the app assembles a structured prompt and streams the result back through the UI.

---

## Project Files

| File | Role |
|---|---|
| `ui.py` | Streamlit frontend — builds the UI, loads the saved prompt, constructs and runs the LCEL chain |
| `prompt_generator.py` | One-time script — defines the `PromptTemplate` and serializes it to `template.json` |
| `template.json` | Serialized prompt template loaded at runtime by `ui.py` |

---

## LangChain Keywords Reference

### 1. `langchain_openai` (package)

**What it is:** The integration package that connects LangChain to OpenAI's API. It provides model wrappers that conform to LangChain's standard interfaces, making them composable with the rest of the framework.

**Used in this project (`ui.py`):**
```python
from langchain_openai import ChatOpenAI
```
This import is the entry point for the chat model used to process the final prompt.

---

### 2. `ChatOpenAI`

**What it is:** A LangChain chat model class that wraps OpenAI's Chat Completions API (e.g., `gpt-4o`, `gpt-3.5-turbo`). It accepts a sequence of messages and returns an `AIMessage` response object. When instantiated without arguments, it reads the `OPENAI_API_KEY` from the environment and uses the default model.

**Used in this project (`ui.py`):**
```python
model = ChatOpenAI()
```
This model instance is the second component in the LCEL chain. It receives the formatted prompt string and returns the model's response.

---

### 3. `langchain_core.prompts` (module)

**What it is:** The core sub-package in LangChain that contains all prompt-related primitives. It is part of `langchain-core`, the dependency-light foundation of the LangChain ecosystem. Using `langchain_core` directly (rather than `langchain`) keeps imports minimal and stable.

**Used in this project:**
```python
from langchain_core.prompts import PromptTemplate, load_prompt
```
Both `prompt_generator.py` and `ui.py` import from this module.

---

### 4. `PromptTemplate`

**What it is:** A class that represents a reusable, parameterized prompt string. It takes a template string containing named placeholders and a list of declared input variable names. At runtime, calling `.format()` or `.invoke()` on it substitutes the placeholders with actual values and produces a ready-to-send prompt.

**Used in this project (`prompt_generator.py`):**
```python
template = PromptTemplate(
    template="...{paper_input}...{style_input}...{length_input}...",
    input_variables=["paper_input", "style_input", "length_input"],
    validate_template=True,
)
```
The template encodes the full summarization instruction with three dynamic slots.

---

### 5. `input_variables`

**What it is:** A constructor parameter of `PromptTemplate` that explicitly declares which variable names the template string expects. LangChain uses this list for validation and to know what keys to look for when the template is invoked.

**Used in this project (`prompt_generator.py` and `template.json`):**
```python
input_variables=["paper_input", "style_input", "length_input"]
```
The three variables correspond to the three Streamlit `selectbox` widgets in `ui.py`. When `chain.invoke({"paper_input": ..., "style_input": ..., "length_input": ...})` is called, LangChain maps the dict keys to these declared variables.

---

### 6. `validate_template`

**What it is:** A boolean parameter on `PromptTemplate` that, when set to `True`, instructs LangChain to check at construction time that every placeholder in the template string has a matching entry in `input_variables`, and vice versa. This catches variable name mismatches early rather than at runtime.

**Used in this project (`prompt_generator.py` and `template.json`):**
```python
validate_template=True
```
Set to `true` in the serialized JSON as well, so validation also applies if the template is ever re-instantiated from the file.

---

### 7. `template_format`

**What it is:** A field on `PromptTemplate` (and in the serialized JSON) that specifies the string interpolation syntax used in the template. The supported values are `"f-string"` (Python-style `{variable}` braces) and `"jinja2"` (Jinja2 template syntax). The default is `"f-string"`.

**Used in this project (`template.json`):**
```json
"template_format": "f-string"
```
The template string uses `{paper_input}`, `{style_input}`, and `{length_input}` — standard Python f-string placeholder syntax — so `"f-string"` is the correct and expected value here.

---

### 8. `template.save()` — Prompt Serialization

**What it is:** An instance method on `PromptTemplate` (and other LangChain prompt classes) that serializes the template object to a JSON file. The saved file captures all configuration fields — `input_variables`, `template`, `template_format`, `validate_template`, `output_parser`, `partial_variables`, and the internal `_type` tag — so the template can be reconstructed exactly from disk without re-running the creation code.

**Used in this project (`prompt_generator.py`):**
```python
template.save("template.json")
```
Running `prompt_generator.py` once produces `template.json`. The UI then loads from that file at startup, keeping the prompt definition separate from the application logic.

---

### 9. `load_prompt`

**What it is:** A utility function from `langchain_core.prompts` that deserializes a previously saved prompt template from a JSON (or YAML) file back into a live LangChain prompt object. It reads the `_type` field in the file to determine which class to instantiate.

**Used in this project (`ui.py`):**
```python
template = load_prompt("template.json")
```
This line is called every time the Streamlit app loads. It reconstructs the `PromptTemplate` from `template.json`, making it ready to be wired into the LCEL chain.

---

### 10. `output_parser`

**What it is:** An optional field on `PromptTemplate` that can hold a reference to a LangChain output parser. Output parsers transform the raw text response from a model into a structured format (e.g., JSON, a Pydantic model, a list). When `null` (as in this project), no post-processing is applied to the model's response.

**In this project (`template.json`):**
```json
"output_parser": null
```
No output parser is used. The raw `AIMessage` object is returned from `chain.invoke()`, and `result.content` is accessed directly in `ui.py` to display the string response.

---

### 11. `partial_variables`

**What it is:** An optional dictionary parameter on `PromptTemplate` that allows certain variables to be pre-filled with fixed values at template-definition time. Partial variables are excluded from the `input_variables` list that callers must supply at invocation time. This is useful for injecting constants like a date, a persona, or a fixed instruction fragment.

**In this project (`template.json`):**
```json
"partial_variables": {}
```
No partial variables are used. All three variables (`paper_input`, `style_input`, `length_input`) are left open for the caller to supply at invocation time.

---

### 12. LCEL — LangChain Expression Language (the `|` pipe operator)

**What it is:** LangChain Expression Language (LCEL) is a declarative syntax for composing LangChain components into pipelines (called "chains") using the Python `|` (pipe) operator. Each component in a chain must implement the `Runnable` interface, which provides a consistent `.invoke()`, `.stream()`, and `.batch()` API. The pipe operator connects runnables so that the output of the left component becomes the input to the right component.

**Used in this project (`ui.py`):**
```python
chain = template | model
```
Data flow: `template` receives a dict of input variables, formats them into a prompt string (technically a `StringPromptValue`), and passes it to `model`. The `model` sends the prompt to OpenAI and returns an `AIMessage`.

---

### 13. `chain.invoke()`

**What it is:** The standard synchronous execution method on any LangChain `Runnable` or LCEL chain. It accepts a single input (a dict when the chain starts with a prompt template), passes it through all components in the chain in sequence, and returns the final output.

**Used in this project (`ui.py`):**
```python
result = chain.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input,
})
```
The dict keys must match the `input_variables` declared in the `PromptTemplate`. The return value is an `AIMessage` object; `result.content` holds the plain text of the model's response.

---

## End-to-End Data Flow

```
User selects values in Streamlit UI
        |
        v
chain.invoke({"paper_input": ..., "style_input": ..., "length_input": ...})
        |
        v
PromptTemplate  (loaded from template.json via load_prompt)
  - Substitutes {placeholders} using f-string formatting
  - Produces a StringPromptValue / formatted prompt string
        |
        v
ChatOpenAI  (reads OPENAI_API_KEY from .env via load_dotenv)
  - Sends the formatted prompt to the OpenAI Chat Completions API
  - Returns an AIMessage object
        |
        v
result.content  ->  st.write()  ->  Displayed in the browser
```

---

## Key Separation of Concerns

| Concern | Where handled |
|---|---|
| Prompt definition and validation | `prompt_generator.py` |
| Prompt persistence (serialization) | `template.json` (written by `template.save()`) |
| Prompt loading at runtime | `load_prompt("template.json")` in `ui.py` |
| Chain assembly | LCEL `template | model` in `ui.py` |
| Model interaction | `ChatOpenAI` in `ui.py` |
| User interface | Streamlit widgets in `ui.py` |

---

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:
   ```
   pip install langchain-core langchain-openai streamlit python-dotenv
   ```
3. Create a `.env` file at the project root and add your OpenAI key:
   ```
   OPENAI_API_KEY=sk-...
   ```
4. Generate the prompt template file (only needs to be run once):
   ```
   python prompt_generator.py
   ```
5. Launch the Streamlit app:
   ```
   streamlit run ui.py
   ```
