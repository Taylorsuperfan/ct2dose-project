from pathlib import Path
import os
import sqlite3
import statistics

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from gitsource import GithubRepositoryDataReader
from minsearch import Index

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)


COMMIT = "8c1834d"
QUERY = "How does the agentic loop keep calling the model until it stops?"

INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
""".strip()

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()


def load_env_files():
    """
    Load .env from the current folder or any parent folder.
    This is useful when your API key is stored at the repo root.
    """
    paths = [Path.cwd(), *Path.cwd().parents]

    # Load parent env first, then closer env files can override.
    for base in reversed(paths):
        env_path = base / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)
            print(f"Loaded env from: {env_path}")


def make_llm_client():
    """
    Supports:
    - Groq through OpenAI-compatible Chat Completions
    - OpenAI through Responses API
    """
    provider = os.getenv("LLM_PROVIDER")

    if provider is None:
        if os.getenv("GROQ_API_KEY"):
            provider = "groq"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        else:
            raise RuntimeError("No GROQ_API_KEY or OPENAI_API_KEY found.")

    provider = provider.lower().strip()

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("LLM_PROVIDER=groq but GROQ_API_KEY is missing.")

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        return provider, client, model

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is missing.")

        client = OpenAI(api_key=api_key)
        model = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
        return provider, client, model

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def get_usage_value(usage, *names):
    """
    Read token usage from either:
    - OpenAI Responses API usage: input_tokens/output_tokens
    - OpenAI-compatible Chat Completions usage: prompt_tokens/completion_tokens
    """
    if usage is None:
        return None

    for name in names:
        value = getattr(usage, name, None)
        if value is not None:
            return int(value)

    if isinstance(usage, dict):
        for name in names:
            value = usage.get(name)
            if value is not None:
                return int(value)

    return None


def choose_closest(value, options):
    return min(options, key=lambda option: abs(float(value) - float(option)))


def classify_duration_ms(ms):
    if ms < 100:
        return "Under 100ms"
    if ms < 500:
        return "100-500ms"
    if ms < 2000:
        return "500-2000ms"
    return "Over 2000ms"


def classify_token_variation(tokens):
    tokens = [int(t) for t in tokens if pd.notna(t)]

    if len(set(tokens)) == 1:
        return "They are identical"

    min_tokens = min(tokens)
    max_tokens = max(tokens)

    if min_tokens == 0:
        return "They vary more than 50%"

    relative_range = (max_tokens - min_tokens) / min_tokens

    if relative_range <= 0.10:
        return "Within 10% of each other"
    if relative_range <= 0.50:
        return "Within 50% of each other"
    return "They vary more than 50%"


def build_index():
    print("Loading course lesson pages...")

    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id=COMMIT,
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )

    documents = [file.parse() for file in reader.read()]

    print(f"Loaded {len(documents)} lesson pages.")

    index = Index(
        text_fields=["content"],
        keyword_fields=["filename"],
    )
    index.fit(documents)

    return index, documents


class RAGTraced:
    def __init__(
        self,
        index,
        llm_client,
        tracer,
        provider,
        model,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
    ):
        self.index = index
        self.llm_client = llm_client
        self.tracer = tracer
        self.provider = provider
        self.model = model
        self.instructions = instructions
        self.prompt_template = prompt_template

    def search(self, query, num_results=5):
        with self.tracer.start_as_current_span("search") as span:
            span.set_attribute("query", query)
            span.set_attribute("num_results", num_results)

            results = self.index.search(
                query,
                num_results=num_results,
            )

            span.set_attribute("result_count", len(results))

            if results:
                span.set_attribute("first_result_filename", results[0]["filename"])

            return results

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc["filename"])
            lines.append(doc["content"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)

        return self.prompt_template.format(
            question=query,
            context=context,
        )

    def llm(self, prompt):
        with self.tracer.start_as_current_span("llm") as span:
            span.set_attribute("provider", self.provider)
            span.set_attribute("model", self.model)

            if self.provider == "groq":
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.instructions},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=500,
                )

                answer = response.choices[0].message.content
                usage = response.usage

                input_tokens = get_usage_value(
                    usage,
                    "input_tokens",
                    "prompt_tokens",
                )
                output_tokens = get_usage_value(
                    usage,
                    "output_tokens",
                    "completion_tokens",
                )

            elif self.provider == "openai":
                response = self.llm_client.responses.create(
                    model=self.model,
                    input=[
                        {"role": "developer", "content": self.instructions},
                        {"role": "user", "content": prompt},
                    ],
                )

                answer = response.output_text
                usage = response.usage

                input_tokens = get_usage_value(
                    usage,
                    "input_tokens",
                    "prompt_tokens",
                )
                output_tokens = get_usage_value(
                    usage,
                    "output_tokens",
                    "completion_tokens",
                )

            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

            if input_tokens is not None:
                span.set_attribute("input_tokens", input_tokens)

            if output_tokens is not None:
                span.set_attribute("output_tokens", output_tokens)

            # Cost is optional for this homework.
            # We store 0.0 so the SQLite schema always has a value.
            span.set_attribute("cost", 0.0)

            return answer

    def rag(self, query):
        with self.tracer.start_as_current_span("rag") as span:
            span.set_attribute("query", query)

            search_results = self.search(query)
            prompt = self.build_prompt(query, search_results)
            answer = self.llm(prompt)

            span.set_attribute("answer_length", len(answer or ""))

            return answer


class SQLiteSpanExporter(SpanExporter):
    def __init__(self, db_path="traces.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS spans (
                name TEXT,
                start_time INTEGER,
                end_time INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost REAL
            )
        """)
        self.conn.commit()

    def export(self, spans):
        for span in spans:
            attrs = dict(span.attributes or {})

            self.conn.execute(
                "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?)",
                (
                    span.name,
                    span.start_time,
                    span.end_time,
                    attrs.get("input_tokens"),
                    attrs.get("output_tokens"),
                    attrs.get("cost"),
                ),
            )

        self.conn.commit()
        return SpanExportResult.SUCCESS

    def shutdown(self):
        self.conn.close()

    def force_flush(self, timeout_millis=30000):
        return True


def run_console_trace(index, llm_client, provider_name, model):
    print("\n" + "=" * 80)
    print("Q1-Q3: ConsoleSpanExporter run")
    print("=" * 80)

    trace_provider = TracerProvider()
    trace_provider.add_span_processor(
        SimpleSpanProcessor(ConsoleSpanExporter())
    )

    tracer = trace_provider.get_tracer("llm-zoomcamp-console")

    rag = RAGTraced(
        index=index,
        llm_client=llm_client,
        tracer=tracer,
        provider=provider_name,
        model=model,
    )

    answer = rag.rag(QUERY)

    print("\nAnswer:")
    print(answer)

    # The class instruments rag(), search(), and llm().
    q1_span_count = 3

    return q1_span_count


def run_sqlite_traces(index, llm_client, provider_name, model):
    print("\n" + "=" * 80)
    print("Q4-Q6: SQLiteSpanExporter run")
    print("=" * 80)

    db_path = Path("traces.db")

    if db_path.exists():
        db_path.unlink()
        print("Removed old traces.db")

    trace_provider = TracerProvider()
    trace_provider.add_span_processor(
        SimpleSpanProcessor(SQLiteSpanExporter(str(db_path)))
    )

    tracer = trace_provider.get_tracer("llm-zoomcamp-sqlite")

    rag = RAGTraced(
        index=index,
        llm_client=llm_client,
        tracer=tracer,
        provider=provider_name,
        model=model,
    )

    print("Running the same query 4 times...")

    for i in range(4):
        print(f"Run {i + 1}/4")
        _ = rag.rag(QUERY)

    # SimpleSpanProcessor writes synchronously, but force flush just in case.
    trace_provider.force_flush()

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM spans", conn)
    conn.close()

    df["duration_ms"] = (df["end_time"] - df["start_time"]) / 1_000_000

    return df


def summarize_answers(q1_span_count, df):
    print("\n" + "=" * 80)
    print("Trace data")
    print("=" * 80)

    print(df)

    span_names = sorted(df["name"].unique().tolist())

    print("\nSpan names:")
    print(span_names)

    print("\nDuration summary by span:")
    duration_by_name = (
        df.groupby("name")["duration_ms"]
        .agg(["count", "sum", "mean", "median"])
        .sort_values("sum", ascending=False)
    )
    print(duration_by_name)

    llm_df = df[df["name"] == "llm"].copy()

    input_tokens = llm_df["input_tokens"].dropna().astype(int).tolist()

    print("\nLLM input tokens across runs:")
    print(input_tokens)

    llm_durations = llm_df["duration_ms"].tolist()

    # Ignore cold start if possible.
    if len(llm_durations) >= 2:
        typical_llm_duration_ms = statistics.median(llm_durations[1:])
    else:
        typical_llm_duration_ms = statistics.median(llm_durations)

    q2_raw = input_tokens[0]
    q2_answer = choose_closest(q2_raw, [700, 7000, 70000, 700000])

    q3_answer = classify_duration_ms(typical_llm_duration_ms)

    q4_answer = ", ".join(span_names)

    non_rag_duration = (
        df[df["name"] != "rag"]
        .groupby("name")["duration_ms"]
        .sum()
        .sort_values(ascending=False)
    )

    q5_answer = non_rag_duration.index[0]

    q6_answer = classify_token_variation(input_tokens[:4])

    answers = {
        "Q1 - spans in one trace": q1_span_count,
        "Q2 - closest input tokens": q2_answer,
        "Q3 - typical LLM duration": q3_answer,
        "Q4 - span names": q4_answer,
        "Q5 - slowest non-rag span": q5_answer,
        "Q6 - input token variation": q6_answer,
    }

    raw_values = {
        "Q2 raw first input_tokens": q2_raw,
        "Q3 raw typical llm duration ms": typical_llm_duration_ms,
        "Q5 duration by non-rag span ms": non_rag_duration.to_dict(),
        "Q6 raw input tokens": input_tokens[:4],
    }

    print("\n" + "=" * 80)
    print("FORM ANSWERS")
    print("=" * 80)

    for key, value in answers.items():
        print(f"{key}: {value}")

    print("\nRaw values:")
    for key, value in raw_values.items():
        print(f"{key}: {value}")

    return answers, raw_values, duration_by_name


def write_readme(answers, raw_values, duration_by_name):
    readme = f"""# Homework 5: Monitoring

## Setup

- Course: DataTalksClub LLM Zoomcamp 2026
- Module: 05 Monitoring
- Instrumentation: OpenTelemetry
- Trace persistence: SQLite
- RAG knowledge base: 72 course lesson pages from commit `8c1834d`
- Search engine: minsearch text index

## Answers

| Question | Raw observation | Selected answer |
|---|---|---|
| Q1 | The traced RAG wraps `rag`, `search`, and `llm`, producing 3 spans per RAG call. | {answers["Q1 - spans in one trace"]} |
| Q2 | First `llm` span input tokens: {raw_values["Q2 raw first input_tokens"]} | {answers["Q2 - closest input tokens"]} |
| Q3 | Typical `llm` duration: {raw_values["Q3 raw typical llm duration ms"]:.2f} ms | {answers["Q3 - typical LLM duration"]} |
| Q4 | Span names in SQLite: {answers["Q4 - span names"]} | rag, search, and llm |
| Q5 | Total non-rag duration by span: {raw_values["Q5 duration by non-rag span ms"]} | {answers["Q5 - slowest non-rag span"]} |
| Q6 | Input tokens across 4 runs: {raw_values["Q6 raw input tokens"]} | {answers["Q6 - input token variation"]} |

## Duration summary


## Notes
The first LLM call can be slower because of cold start. For Q3, I used the median duration after the first run when multiple runs were available.
“””
Path("README.md").write_text(readme)
print("\nWrote README.md")
def main():
load_env_files()
provider_name, llm_client, model = make_llm_client()

print("Provider:", provider_name)
print("Model:", model)

index, documents = build_index()

q1_span_count = run_console_trace(
    index=index,
    llm_client=llm_client,
    provider_name=provider_name,
    model=model,
)

df = run_sqlite_traces(
    index=index,
    llm_client=llm_client,
    provider_name=provider_name,
    model=model,
)

answers, raw_values, duration_by_name = summarize_answers(
    q1_span_count=q1_span_count,
    df=df,
)

write_readme(
    answers=answers,
    raw_values=raw_values,
    duration_by_name=duration_by_name,
)
if name == “main”:
main()
