# Vendored AgentDojo

This directory vendors AgentDojo 0.1.35 for release reproducibility.

Local patches:

- Register and run the IPI-Aware long-horizon attack variants from the release
  code without relying on a private AgentDojo checkout.
- Use OpenAI `system` messages for local/vLLM models.
- Thread local vLLM sampling and timeout settings through AgentDojo's pipeline
  config, including `temperature`, `top_p`, `max_completion_tokens`, and client
  timeout.
- Allow `vllm_parsed` to use the caller-provided local model id while preserving
  AgentDojo's OpenAI-compatible tool-calling path.
- Request vLLM token ids and pass Qwen chat-template kwargs through the normal
  OpenAI-compatible request body so tool calls are parsed by vLLM, not by
  AgentDojo-side string recovery.

The upstream package metadata and MIT license from the installed
`agentdojo==0.1.35` wheel are kept under `dist-info/`.
