# comedy-factory

[![CI](https://github.com/hankehly/comedy-factory/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/hankehly/comedy-factory/actions/workflows/ci.yml)

## Joke Generator Workflow

```mermaid
flowchart TD
    classDef agent fill:#f3e8ff,stroke:#9333ea,color:#581c87
    classDef llm fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef system fill:#dcfce7,stroke:#16a34a,color:#14532d

    s1(["Scan news for topics"]):::agent
    s2(["Generate subtext"]):::llm
    s3{"Grade subtext"}:::llm
    s4(["Generate Joke"]):::llm
    s5{"Grade Joke"}:::llm
    s6(["Write image prompt"]):::llm
    s7(["Generate image (no text)"]):::llm
    s8[["Render caption on image"]]:::system
    s9(["Evaluate joke holistically"]):::llm
    s10[["Save asset bundle"]]:::system

    s1 --> s2 --> s3
    s3 -- criteria met --> s4
    s3 -. "not met: re-run with feedback" .-> s2
    s4 --> s5
    s5 -- criteria met --> s6
    s5 -. "not met: re-run with feedback" .-> s4
    s6 --> s7 --> s8 --> s9 --> s10
```

Legend: purple nodes are **Agent** steps; blue nodes are **Augmented LLM** steps (diamond = evaluation gate); green nodes are **System** steps. Dotted edges are feedback loops.

Other considerations:
* Alter image/font style (e.g., line sketch, realistic, anime, etc.)
* Include a "post to social" step (human)
