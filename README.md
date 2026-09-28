```mermaid
---
config:
  layout: elk
---

flowchart TD

    A["Input Token IDs"] --> B["Token Embedding"]
    B --> C["Position Dropout"]
    C --> TB1

    subgraph TB1["Transformer Block 1"]
        direction TB

        TB1_IN["Input X"] --> TB1_N1["RMSNorm"]
        TB1_N1 --> ATTN1

        subgraph ATTN1["Qwen3 Attention"]
            direction TB

            A1["Normalized X"]
            A1 --> Q1["Q_proj"]
            A1 --> K1["K_proj"]
            A1 --> V1["V_proj"]

            Q1 --> QR1["Reshape Q"]
            K1 --> KR1["Reshape K"]
            V1 --> VR1["Reshape V"]

            QR1 --> QN1["Q_norm"]
            KR1 --> KN1["K_norm"]

            QN1 --> RQ1["RoPE Q"]
            KN1 --> RK1["RoPE K"]

            VR1 --> KV1["Repeat KV Heads"]
            RK1 --> KV1

            RQ1 --> SDPA1["Scaled Dot-Product Attention"]
            KV1 --> SDPA1

            SDPA1 --> AO1["Attention Output"]
            AO1 --> RO1["Reshape"]
            RO1 --> WO1["W_o"]
            WO1 --> ATTNO1["Attention Output"]
        end

        ATTNO1 --> D1["Dropout"]
        TB1_IN --> ADD1["+"]
        D1 --> ADD1

        ADD1 --> TB1_N2["RMSNorm"]
        TB1_N2 --> FF1

        subgraph FF1["SwiGLU FeedForward"]
            direction TB

            FFI1["Normalized X"]
            FFI1 --> GP1["Gate_proj"]
            FFI1 --> UP1["Up_proj"]

            GP1 --> SILU1["SiLU"]
            SILU1 --> MUL1["Element-wise Multiply"]
            UP1 --> MUL1

            MUL1 --> DO1["Dropout"]
            DO1 --> DP1["Down_proj"]
            DP1 --> FFO1["FFN Output"]
        end

        FFO1 --> ADD2["+"]
        ADD1 --> ADD2
        ADD2 --> TB1_OUT["Output X"]
    end

    TB1 --> MORE["⋮"]
    MORE --> TBN["Transformer Block N"]

    TBN --> FINAL_NORM["Final RMSNorm"]
    FINAL_NORM --> OUTPUT_DROPOUT["Output Dropout"]
    OUTPUT_DROPOUT --> LM_HEAD["LM Head"]
    LM_HEAD --> LOGITS["Logits"]

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style TB1_OUT fill:#f9f,stroke:#333,stroke-width:2px
    style LOGITS fill:#f9f,stroke:#333,stroke-width:2px
```