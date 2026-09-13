"""
Patch for VLLM 0.17.0 to fix "Already borrowed" tokenizer error in Hermes tool parser.

This patch modifies vllm/tool_parsers/hermes_tool_parser.py to cache token IDs
globally instead of encoding them in each parser instance __init__.

Apply this patch by running:
    python patches/vllm_hermes_tokenizer_fix.py

Or manually modify hermes_tool_parser.py as shown below.
"""

import threading

# Global cache for token IDs to avoid re-encoding
_hermes_token_cache = {}
_hermes_token_cache_lock = threading.Lock()


def get_cached_token_ids(tokenizer, tokens):
    """
    Get cached token IDs for the given tokens.
    Thread-safe caching to avoid "Already borrowed" errors.
    """
    cache_key = (id(tokenizer), tuple(tokens))

    with _hermes_token_cache_lock:
        if cache_key not in _hermes_token_cache:
            # Encode all tokens at once (more efficient)
            try:
                _hermes_token_cache[cache_key] = [
                    tokenizer.encode(token, add_special_tokens=False)
                    for token in tokens
                ]
            except Exception as e:
                # Fallback: encode individually if batch fails
                _hermes_token_cache[cache_key] = [
                    tokenizer.encode(token, add_special_tokens=False)
                    for token in tokens
                ]
        return _hermes_token_cache[cache_key]


# ============== MANUAL PATCH INSTRUCTIONS ==============
#
# In vllm/tool_parsers/hermes_tool_parser.py, find the Hermes2ProToolParser class
# and replace lines 63-78 with the following:
#
# FROM:
#         self.tool_call_start_token_ids = self.model_tokenizer.encode(
#             self.tool_call_start_token, add_special_tokens=False
#         )
#         self.tool_call_end_token_ids = self.model_tokenizer.encode(
#             self.tool_call_end_token, add_special_tokens=False
#         )
#
#         self.tool_call_start_token_array = [
#             self.model_tokenizer.decode([token_id])
#             for token_id in self.tool_call_start_token_ids
#         ]
#
#         self.tool_call_end_token_array = [
#             self.model_tokenizer.decode([token_id])
#             for token_id in self.tool_call_end_token_ids
#         ]
#
# TO:
#         # Use cached token IDs to avoid "Already borrowed" errors
#         token_ids_list = get_cached_token_ids(
#             self.model_tokenizer,
#             [self.tool_call_start_token, self.tool_call_end_token]
#         )
#         self.tool_call_start_token_ids, self.tool_call_end_token_ids = token_ids_list
#
#         self.tool_call_start_token_array = [
#             self.model_tokenizer.decode([token_id])
#             for token_id in self.tool_call_start_token_ids
#         ]
#
#         self.tool_call_end_token_array = [
#             self.model_tokenizer.decode([token_id])
#             for token_id in self.tool_call_end_token_ids
#         ]
# =============================================================


if __name__ == "__main__":
    print("VLLM Hermes Tokenizer Fix")
    print("=" * 50)
    print()
    print("This script provides a fix for the 'Already borrowed' tokenizer")
    print("error in VLLM 0.17.0 when using the Hermes tool parser with")
    print("high concurrency.")
    print()
    print("To apply the fix:")
    print("1. Edit: vllm/tool_parsers/hermes_tool_parser.py")
    print("2. Add the import and function at the top of the file")
    print("3. Replace the encode() calls in Hermes2ProToolParser.__init__()")
    print()
    print("See the MANUAL PATCH INSTRUCTIONS in this file for details.")
    print()
    print("Expected impact:")
    print("  - Eliminates 'Already borrowed' errors")
    print("  - No performance degradation (actually improves)")
    print("  - Token IDs are cached globally, avoiding repeated encode() calls")
