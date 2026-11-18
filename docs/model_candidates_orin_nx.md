# Candidate VLMs for on-robot deployment (Jetson Orin NX)

Model shortlist from a ChatGPT search (2026-09-07), for a later phase: which models to compare
against the fine-tuned Qwen3.5-2B, and at which precision, once the rover runs on the Orin NX.

| Model                                    | Role in your comparison                                                    | Precision to evaluate on the Orin NX              |
| ---------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------- |
| **Qwen3.5-2B** — your existing fine-tune | Baseline: establish whether larger models actually improve path prediction | Current 16-bit; then quantized                    |
| **Qwen3.5-4B**                           | First larger model to test, keeping your existing training approach        | 16-bit if runtime memory permits; 8-bit and 4-bit |
| **Qwen3-VL-2B-Instruct**                 | Compare a different vision architecture at a similar size                  | 16-bit                                            |
| **Qwen3-VL-4B-Instruct**                 | Main alternative to Qwen3.5-4B for spatial prediction                      | 16-bit if memory permits; 8-bit and 4-bit         |
| **LFM2.5-VL-1.6B**                       | Smaller alternative worth profiling when latency is critical               | 16-bit initially                                  |
