#!/usr/bin/env python
"""Run our activation-steering defense inside AgentDojo's real agent loop.

WHY AGENTDOJO. Everything in this repo is measured on one benchmark family
(Nemotron) plus a parameter-abuse corpus we wrote ourselves. InjecAgent was tried as a second
corpus and does not fire on gpt-oss-20b at all -- 0/24 undefended, with the model naming the
injection as malicious in its own reasoning (FINDINGS.md section 4b). AgentDojo is the right
target instead, for a reason that matters: **this exact model has a published undefended
baseline there** -- 0.491 (template) to 0.731 (PAIR) on AgentDojo, and 0.483-0.726 on
Mind2Web (IntentGuard, arXiv:2512.00966, Tables 1-2). So there is both a working attack and a
number to compare against.

It is also a genuinely different setting from everything measured here so far: a MULTI-TURN
agent loop with real tool execution against a stateful environment, where the injection is
placed by AgentDojo into tool output that the agent then reads. Our defense sees each tool
result as it comes back, which is the deployment-realistic case.

WHICH AGENTDOJO PATH THIS MIRRORS. AgentDojo ships two agent styles: `OpenAILLM` (NATIVE
function calling) and `LocalLLM` (a text-prompted agent whose tools live in the system prompt
and whose calls are emitted as `<function=NAME>{...}</function>` text). We mirror `OpenAILLM`.

That is a deliberate choice, not convenience. gpt-oss is a harmony function-calling model, and
its chat template REFUSES to render a tool-role message that is not preceded by an assistant
message carrying structured `tool_calls` -- the text-prompted path cannot produce one. Forcing
it would mean putting tool output somewhere other than a tool turn, which is precisely the
variable this project measures. Native function calling keeps the injected content inside a
real tool turn, exactly as the Nemotron harness does.

WHERE THE STEERING IS APPLIED. Every tool-role message's content, located by SENTINEL DIFF
against the same rendered conversation (never by string search), then mapped to token indices.
On each turn the whole history is re-rendered, so every tool result seen so far is steered --
including ones from earlier turns, which is correct: their KV is rebuilt each turn.

WIRE FORMATS. Five, dispatched on the tokenizer's own vocab (`fmt_of`), never on a model name:

  harmony  gpt-oss. Channels; tool calls as `to=functions.NAME ... <|call|>`; the
           user-visible answer is the `final` channel; reasoning is the `analysis` channel.
  chatml   Qwen3 et al. Hermes-style tool calling: the chat template renders tool results as
           `<tool_response>` blocks inside a user turn, the model emits
           `<tool_call>{"name":...,"arguments":{...}}</tool_call>` blocks, and the THINKING
           variants open every completion INSIDE `<think>` (the generation prompt itself ends
           with `<think>\n`), so the completion's reasoning is everything before the first
           `</think>` and the user-visible answer is what follows it.
  phi3     Phi-3 instruct. NO native tool format and NO reasoning region: tools are offered
           as the LLMail few-shot JSON convention in the first user turn, calls are emitted
           as bare one-line JSON objects, tool results come back as PHI3_TOOL_RESULT user
           turns, and the episode is rendered multi-turn by src.templates.phi3_render_agent
           (apply_template's `phi3_agent=True`, passed only here so single-turn artifacts
           keep their bytes). A finished turn ends at <|end|> / <|endoftext|>.
  glm45    GLM-4.5 family. Its own chat template accepts this bridge's assistant
           `tool_calls` dicts and `tool` role AS-IS (tool results render as
           `<|observation|>\n<tool_response>...</tool_response>`), so rendering needs no
           port at all. The model SELF-OPENS `<think>...</think>` (the generation prompt
           ends `<|assistant|>` with nothing pre-opened -- in_think False), then emits
           `<tool_call>NAME\n<arg_key>K</arg_key>\n<arg_value>V</arg_value></tool_call>`
           blocks, which src.scoring.parse_tool_calls already reads. Parsing shares the
           chatml branches (reasoning strip = the self-opened <think> blocks); a finished
           turn carries one of the generation_config eos markers
           <|endoftext|>/<|user|>/<|observation|> (ids 151329/151336/151338, verified
           against the cached tokenizer_config, 2026-09-04).
  gemma4   Gemma-4 family. Its own chat template accepts this bridge's assistant
           `tool_calls` dicts and `tool` role AS-IS: tool results are rendered INLINE in
           the same model turn as `<|tool_response>response:NAME{value:<|"|>...<|"|>}
           <tool_response|>` (forward scan from the assistant message carrying the calls),
           and after a call with no result yet the generation prompt ends with a bare
           `<|tool_response>`. generation_config eos = [<eos>, <turn|>, <|tool_response>]
           (ids 1/106/50), so the model STOPS ITSELF exactly where the executor injects
           the tool result -- no custom stopping criteria. The model self-opens (and the
           non-thinking generation prompt pre-closes) a `<|channel>thought...<channel|>`
           reasoning region -- in_think is always False; the strip is _GEMMA_THINK in
           src/scoring.reasoning_free. Calls are `<|tool_call>call:NAME{k:<|"|>v<|"|>}
           <tool_call|>` blocks, which src.scoring.parse_tool_calls already reads.
           HAZARD, measured (FINDINGS 23y): Gemma's markers are PAIRED half-pipe tags
           (`<|turn>`...`<turn|>`), so the old `_HARMONY_TOKEN` regex `<\\|[^|]*\\|>`
           swallowed legitimate content between them; the scrubs below use
           _SPECIAL_TOKEN, which cannot cross a `<`/`>` boundary (Gemma AgentDojo port,
           2026-09-04).

The span logic is shared: sentinel diff is exact under any template. Only rendering-free
parsing (which calls were EMITTED, what the user-visible answer is, whether the generation
finished) differs per format, and each of those has one dispatch point below.

Usage (see agentdojo_smoke.py for the driver):
    from agentdojo_bridge import SteeredLLM
    pipe = AgentPipeline([SystemMessage(...), InitQuery(), SteeredLLM(...), ToolsExecutionLoop([...])])
"""
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _probe_eval as E  # noqa: E402

X = E.X

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement  # noqa: E402
from agentdojo.functions_runtime import EmptyEnv, FunctionCall, FunctionsRuntime  # noqa: E402
from agentdojo.types import ChatAssistantMessage, text_content_block_from_string  # noqa: E402

SENTINEL = "ZQXADSENTINELXQZ"


def fmt_of(tok):
    """Bridge-side wrapper over the shared discriminator (src/scoring.fmt_of, moved there
    2026-08-31 so the adaptive harnesses share it): this bridge implements exactly six
    formats (harmony, chatml, phi3, glm45, gemma4, llama31), so anything else refuses here
    rather than mis-parsing -- a wrong parser executes quoted-not-emitted tool calls (see
    `_executable_calls`)."""
    fmt = X.fmt_of(tok)
    if fmt == "other":
        raise SystemExit("unsupported tokenizer: none of harmony (<|channel|>), ChatML "
                         "(<|im_start|>), Phi-3 (<|user|>/<|end|>), GLM-4.5 "
                         "(<|observation|>/<arg_key>), Gemma-4 (<|turn>/<|tool_call>) or "
                         "Llama-3.1 (<|start_header_id|>/<|python_tag|>) markers in the "
                         "vocab -- add a format branch before running this model")
    return fmt


def _tools_for_template(runtime):
    """AgentDojo Function objects -> the tool schema list our renderer expects."""
    return [{"type": "function",
             "function": {"name": f.name,
                          "description": f.description,
                          "parameters": f.parameters.model_json_schema()}}
            for f in runtime.functions.values()]


def _to_template_messages(messages, tool_content_override=None):
    """AgentDojo ChatMessages -> chat-template messages.

    `tool_content_override` replaces the i-th tool message's content, which is how the sentinel
    diff locates each tool span without ever searching for the content as a substring.
    """
    out, tool_i = [], 0
    for m in messages:
        role = m["role"]
        content = m.get("content")
        if isinstance(content, list):
            content = "".join(c.get("content", "") for c in content if isinstance(c, dict))
        content = content or ""
        if role == "assistant":
            calls = m.get("tool_calls") or []
            # defensive: a single stray harmony/ChatML/Gemma token in history kills the
            # whole run, and a leaked <think>/<|channel>thought block would re-enter the
            # next prompt as assistant prose. _SPECIAL_TOKEN, not _HARMONY_TOKEN: the old
            # regex swallowed legitimate content between paired Gemma markers (measured,
            # FINDINGS 23y); on the other four formats the two are behaviourally identical
            # over every stored completion (regression render diff, Gemma port 2026-09-04).
            msg = {"role": "assistant",
                   "content": _SPECIAL_TOKEN.sub(
                       "", X._GEMMA_THINK.sub("", _THINK_BLOCK.sub("", content)))}
            if calls:
                msg["tool_calls"] = [
                    {"type": "function",
                     "function": {"name": c.function,
                                  "arguments": dict(c.args)}} for c in calls]
            out.append(msg)
        elif role == "tool":
            if m.get("error"):
                body = json.dumps({"error": m["error"]})
            else:
                body = content if isinstance(content, str) else json.dumps(content)
            if tool_content_override is not None and tool_i == tool_content_override[0]:
                body = tool_content_override[1]
            tool_i += 1
            name = (m.get("tool_call") or {}).function if m.get("tool_call") else None
            out.append({"role": "tool", "name": name or "tool", "content": body})
        else:
            out.append({"role": role, "content": content})
    return out, tool_i


def _tool_spans(tok, messages, tools, no_think=False, names_out=None):
    """Char ranges of every tool message's content in the rendered conversation.

    SENTINEL DIFF, per tool message: render once normally, once with that message's content
    replaced by a sentinel, and take the differing range. String search would silently pick the
    wrong occurrence when a tool result is echoed in an assistant turn, which is exactly what
    an injection makes the model do.

    `names_out`: optional list; when given, the producing TOOL's name is appended for every
    span KEPT (same order/length as the returned spans). The content-leaf selector
    (span_select != "full") keys its whole-payload-content rule on the tool name; the default
    full-span path never passes it, so the pre-selector call signature is unchanged.
    """
    base_msgs, n_tools = _to_template_messages(messages)
    _tool_names = [m.get("name") for m in base_msgs if m.get("role") == "tool"]
    # phi3_agent selects the multi-turn Phi-3 renderer (src.templates.phi3_render_agent);
    # it is read ONLY inside apply_template's phi3 branch, a no-op for harmony/chatml.
    text = X.apply_template(tok, base_msgs, tools, gen_prompt=True, no_think=no_think,
                            phi3_agent=True)
    spans = []
    for i in range(n_tools):
        alt_msgs, _ = _to_template_messages(messages, tool_content_override=(i, SENTINEL))
        alt = X.apply_template(tok, alt_msgs, tools, gen_prompt=True, no_think=no_think,
                               phi3_agent=True)
        a = 0
        while a < min(len(text), len(alt)) and text[a] == alt[a]:
            a += 1
        b = 0
        while b < min(len(text), len(alt)) - a and text[-1 - b] == alt[-1 - b]:
            b += 1
        lo, hi = a, len(text) - b
        # THE SANITY GUARD src/templates.sentinel_span HAS AND THIS COPY DROPPED.
        # The gpt-oss template embeds strftime_now() in the system prompt, evaluated
        # INDEPENDENTLY in each of the two renders. A date rollover between them collapses the
        # common prefix to the date digits and the "span" balloons to ~80% of the prompt --
        # measured: 659 of 819 chars, 126 tokens steered instead of 14, starting inside
        # `Reasoning: medium`. Silent, and doubly damaging because tool_outputs_from() builds
        # payload_clean from these same spans, so a blown span swallows the whole prompt into
        # payload_clean, empties attack_evidence, and zeroes our tier-1 metric.
        body_len = hi - lo
        # ref_len on the STRIPPED body: ChatML templates apply |trim to tool content, so a
        # WHITESPACE-ONLY tool result renders to a 0-char span while the raw body is >0
        # chars -- the ratio guard then raised on a body with nothing to steer (Qwen3.8
        # port review, 2026-08-31; same class as the empty-body abort below).
        ref_len = len(_nth_tool_body(messages, i).strip())
        if ref_len == 0 and body_len == 0:
            # An EMPTY tool result (slack's read tools return "" routinely) renders as an
            # empty span: nothing to steer, and nothing for the ratio guard to check. This
            # was a raise until 2026-08-30 and aborted 5 of 180 Qwen cells (all slack, the
            # attacked arm's third turn). Skip the message, keep the guard for real bodies.
            continue
        ratio = body_len / max(1, ref_len)
        if not (0.5 < ratio < 3.0):
            raise ValueError(
                f"tool span {i} is {body_len} chars for {ref_len} chars of content "
                f"(ratio {ratio:.1f}) -- the two renders disagree outside the content region "
                f"(date rollover in the template?). Refusing to steer a span this wrong.")
        if hi > lo:
            spans.append((lo, hi))
            if names_out is not None:
                names_out.append(_tool_names[i])
    return text, spans


def _nth_tool_body(messages, n):
    """The n-th tool message's rendered body, for the span sanity check."""
    msgs, _ = _to_template_messages(messages)
    bodies = [m["content"] for m in msgs if m.get("role") == "tool"]
    return bodies[n] if n < len(bodies) else ""


# ── CONTENT-LEAF SPAN SELECTOR (FINDINGS §26.12, owner charter 2026-09-08) ──────────────
#
# Schema-aware selection of the FREE-TEXT CONTENT inside each tool-result span, so steering
# can be applied to the injectable prose (email bodies, descriptions, comments, reviews,
# issue prose, file contents, web text) while leaving the YAML/JSON structure (keys,
# booleans, timestamps, IDs, addresses, URLs, numerics) unedited. FAIL-CLOSED: any parse or
# classification uncertainty makes the WHOLE span content -- silent uncovered text is the
# measured hazard (§26.12: the span diagnostic's ut9xit3 cell lost the defense to a missed
# carrier), so every uncertain branch falls back to full-span steering and is logged.
#
# Vocabulary grounded in the replayed §25e corpus (tmp/leafsel/build_corpus.py: 6,380 real
# tool payloads from all 351 gpt-oss AgentDyn cells): injections in STRUCTURED payloads land
# only under content-named keys (`body`, `description` on that grid); the file/web/git tools
# return unstructured prose/HTML where the whole payload is content.

SPAN_SELECTS = ("full", "leaf", "struct", "random", "energy", "contract")

# Tools whose whole payload IS content (file contents / web text / invoice text / git
# conflict messages that embed file contents). Never YAML-parsed: a .txt file that happens
# to parse as a YAML mapping would otherwise be classified key-by-key -- the measured
# read_file-dict hazard (55/584 replayed read_file payloads parse as mappings).
WHOLE_CONTENT_TOOLS = frozenset((
    "read_file", "browse_webpage", "get_webpage", "download_file_through_id",
    "download_file_through_url", "get_invoice", "git_pull", "git_push", "git_clone",
))

def _norm_key(k):
    return re.sub(r"[^a-z]+", "", str(k).lower())

# Keys whose values are free-text CONTENT (steered, including all string descendants).
# `error` is deliberate: tool error strings are model-facing prose that can echo
# attacker-derived arguments. `subject`/`title` are attacker-writable free text (not on the
# owner's exclusion list; included by the same reasoning as body).
_CONTENT_KEYS = frozenset((
    "body", "description", "subject", "title", "content", "text", "message",
    "comments", "comment", "reviews", "review", "issues", "issue", "notes", "note",
    "feedback", "webcontent", "error", "summary", "bio", "readme", "prose",
))
# Keys whose values are STRUCTURE (excluded). Owner exclusion list (§26.12) plus the full
# structural vocabulary observed in the replayed corpus.
_STRUCT_KEYS = frozenset((
    "id", "type", "status", "read", "sender", "recipient", "recipients", "cc", "bcc",
    "attachments", "timestamp", "date", "time", "starttime", "endtime", "allday",
    "participants", "location", "name", "email", "username", "owner", "ownerusername",
    "password", "url", "weburl", "phone", "phonenumber", "address", "iban", "amount",
    "balance", "price", "rating", "quantity", "productid", "orderid", "deliverydate",
    "orderdate", "brand", "category", "productname", "collaborators",
    "pendingcollaborators", "stars", "repository", "starredrepository",
    "repositoryfiletree", "permissions", "execute", "write", "repos", "repositories",
    "sshkeys", "firstname", "lastname", "idnumber", "passportnumber",
    "bankaccountnumber", "bankaccountpassword", "code", "discount", "filename",
    "filetype", "size", "lastmodified", "currentday", "handler", "recurring", "user",
    "author", "path", "priceperday", "priceperperson", "pricemin", "pricemax",
))
_STRUCT_KEY_SUFFIXES = ("id", "date", "time", "email", "username", "password", "number",
                        "account", "url")

# Shape rules for UNKNOWN keys' string values: these forms cannot carry an imperative
# instruction and are excluded as structure (owner list: timestamps, IDs, addresses, URLs,
# numerics). Anything else with prose shape is UNCERTAIN -> whole-span fallback.
_SHAPE_STRUCT = re.compile(
    r"^(?:[\w.+-]+@[\w.-]+"                       # email address
    r"|(?:https?://|www\.)\S+"                    # URL
    r"|\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?"  # date / timestamp
    r"|[+-]?\d[\d ,.:/-]*"                        # numeric / account-ish
    r"|\S+"                                       # single token, no whitespace
    r")$")

_JSON_UNESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f",
                   '"': '"', "\\": "\\", "/": "/"}


def _unescape_with_map(s):
    """Decode the chat template's JSON-string escaping (measured: gpt-oss renders tool
    content as json.dumps(..., ensure_ascii=False)[1:-1]); returns (decoded, offmap) where
    offmap[i] = start offset in `s` of decoded char i (offmap[len] = len(s)).
    Raises ValueError on any escape it does not recognise (caller falls back full-span)."""
    out, om, i, n = [], [], 0, len(s)
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            d = s[i + 1]
            if d in _JSON_UNESCAPES:
                out.append(_JSON_UNESCAPES[d]); om.append(i); i += 2; continue
            if d == "u" and i + 6 <= n:
                out.append(chr(int(s[i + 2:i + 6], 16))); om.append(i); i += 6; continue
            raise ValueError(f"unknown escape \\{d} at {i}")
        out.append(c); om.append(i); i += 1
    om.append(n)
    return "".join(out), om


_HTML_HINT = re.compile(
    r"<(?:!doctype|html|head|body|div|p|h[1-6]|form|a |a>|span|ul|ol|li|table|tr|td|label|"
    r"input|button|strong|em|br|img|title|section|header|footer)\b", re.I)


def _html_text_regions(txt):
    """Char regions of VISIBLE TEXT NODES in an HTML payload (tags/attrs/script/style
    excluded -- the §26.12 HTML rule). Offsets are exact: html.parser reports (line, col)
    per event and feeds data verbatim with convert_charrefs=False. Raises on any parser
    anomaly (caller falls back to the whole payload -- the fail-closed direction)."""
    from html.parser import HTMLParser
    line_off = [0]
    for m in re.finditer("\n", txt):
        line_off.append(m.end())

    class P(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=False)
            self.regs, self.skip = [], 0
        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style"):
                self.skip += 1
        def handle_endtag(self, tag):
            if tag in ("script", "style") and self.skip:
                self.skip -= 1
        def handle_data(self, data):
            if self.skip or not data.strip():
                return
            ln, col = self.getpos()
            start = line_off[ln - 1] + col
            assert txt[start:start + len(data)] == data, "html offset drift"
            self.regs.append((start, start + len(data)))
    p = P()
    p.feed(txt)
    p.close()
    return p.regs


def _yaml_leaf_regions(ytext):
    """(content_regions, info) over a decoded YAML/JSON payload, via yaml.compose marks
    (exact char offsets per scalar). Raises on parse failure or any UNCERTAIN
    classification -- unknown key with a prose-shaped string value -- so the caller falls
    back to whole-span steering (fail-closed)."""
    import yaml
    node = yaml.compose(ytext, Loader=yaml.SafeLoader)
    if node is None:
        raise ValueError("empty yaml")
    regs, info = [], {"content_keys": set(), "struct_keys": set(), "shape_struct": 0,
                      "shape_struct_keys": set()}

    def scalar_str(n):
        return n.tag == "tag:yaml.org,2002:str"

    def add(n):
        regs.append((n.start_mark.index, n.end_mark.index))

    def walk(n, in_content):
        if n.id == "mapping":
            for k, v in n.value:
                if k.id != "scalar":
                    raise ValueError("non-scalar mapping key")
                nk = _norm_key(k.value)
                if nk in _CONTENT_KEYS:
                    info["content_keys"].add(nk)
                    walk_content(v)
                elif nk in _STRUCT_KEYS or nk.endswith(_STRUCT_KEY_SUFFIXES):
                    info["struct_keys"].add(nk)
                    if v.id != "scalar":
                        walk(v, in_content)   # containers under a struct key still recurse
                elif v.id != "scalar":
                    walk(v, in_content)
                elif not scalar_str(v):
                    info["struct_keys"].add(nk)          # bool/int/float/timestamp/null
                elif in_content:
                    add(v)
                elif _SHAPE_STRUCT.match(v.value.strip() or "-"):
                    # deliberate shape exclusion -- COUNTED AND NAMED in the telemetry
                    # (review1 F2: an unnamed shape exclusion is an unlogged non-coverage
                    # class, and the single-token arm CAN carry a snake_case imperative)
                    info["shape_struct"] += 1
                    info["shape_struct_keys"].add(nk)
                else:
                    raise ValueError(f"uncertain key {nk!r} with prose value")
        elif n.id == "sequence":
            for item in n.value:
                walk(item, in_content)
        else:  # top-level / sequence-item scalar with no key context
            if in_content and scalar_str(n):
                add(n)
            elif scalar_str(n) and not _SHAPE_STRUCT.match(n.value.strip() or "-"):
                # keyless prose (a tool returning a bare string returns a MESSAGE, e.g.
                # send_email/checkout confirmations): content, deliberately -- not an
                # uncertainty fallback. Structural shapes (IDs, dates, URLs, numbers,
                # single tokens such as list_directory paths) stay excluded above.
                add(n)
            elif scalar_str(n):
                # keyless shape exclusion: counted and named too (review1 F2)
                info["shape_struct"] += 1
                info["shape_struct_keys"].add("(keyless)")

    def walk_content(n):
        if n.id == "scalar":
            if scalar_str(n):
                add(n)
        elif n.id == "sequence":
            for item in n.value:
                walk_content(item)
        else:   # mapping under a content key: struct-named keys stay excluded,
                # everything else (incl. unknown prose) is content
            for k, v in n.value:
                nk = _norm_key(k.value) if k.id == "scalar" else ""
                if nk in _STRUCT_KEYS or nk.endswith(_STRUCT_KEY_SUFFIXES):
                    info["struct_keys"].add(nk)
                else:
                    walk_content(v)

    walk(node, False)
    return regs, info


# ── v2: THE PROVENANCE-CONTRACT SELECTOR (owner decision §26.17, 2026-09-08) ───────────
#
# v1 (span_select="leaf") SELECTS content by a key-name allowlist plus value-shape rules;
# the §26.13 post-launch audit showed that trusts strings by NAME and SHAPE, leaving
# adaptive surfaces (prose under struct-named keys, single-token imperatives, HTML
# comments/attributes). v2 ("contract") INVERTS the default: every string the model sees
# is steered UNLESS its field is DECLARED platform-generated in the contract below.
# Closed by construction: unknown keys -> steered; data-as-key text (filenames, product
# names, confirmation prose that yaml-parses into the KEY position) -> the key text
# itself is steered; single-token strings under untrusted keys -> steered (no shape
# rule exists in v2); HTML payloads -> the WHOLE payload is steered (tags, attributes,
# comments, scripts included -- no HTML parsing at all, so no parser surface). Non-string
# YAML-typed scalars (bool/int/float/timestamp/null) are excluded -- a typed literal
# cannot carry words; any OTHER tag (custom/binary) is uncertainty -> whole-span
# fallback, logged (fail-closed).
#
# THE CONTRACT: which fields are TRUSTED (platform-assigned, never third-party-written),
# declared per the AgentDyn suite schemas (default_suites/v1/{dailylife,github,shopping}
# tool return models + environment yamls), not inferred at runtime. Everything else --
# bodies, subjects, names, addresses, usernames, locations, participants, file paths,
# reviews, issues, error echoes -- is externally sourced and steered.
_CONTRACT_TRUSTED = frozenset((
    "id", "status", "read", "timestamp", "lastmodified", "starttime", "endtime",
    "allday", "recurring", "type", "execute", "write", "currentday",
    "orderdate", "deliverydate", "orderid", "productid", "fileid",
))
# NOTE: no trusted-label rule. "Order 1"/"Item 2" list labels normalize to "order"/
# "item" (digits stripped by _norm_key), are not schema keys, and steer as DATA KEYS --
# the fail-closed direction, ~2 tokens each. A label regex was dead code (offline check
# 2026-09-08: it required digits the normalizer removes) and was removed.
# The suites' schema key vocabulary (normalized): keys in this set are TEMPLATE text
# (the renderer's field names), so the KEY text itself is not steered; their VALUES are
# governed by _CONTRACT_TRUSTED. A key OUTSIDE this vocabulary is DATA in key position
# (filename, product name, accidental-mapping prose) and its text IS steered.
_CONTRACT_SCHEMA_KEYS = frozenset((
    "attachments", "bcc", "body", "cc", "read", "recipients", "sender", "status",
    "subject", "timestamp", "id", "allday", "description", "endtime", "location",
    "participants", "starttime", "title", "brand", "category", "price", "productid",
    "productname", "quantity", "rating", "reviews", "deliverydate", "orderdate",
    "orderid", "owner", "collaborators", "pendingcollaborators", "stars", "issues",
    "repositoryfiletree", "type", "children", "execute", "write", "permissions",
    "error", "name", "email", "username", "firstname", "lastname", "address",
    "phonenumber", "passportnumber", "bankaccountnumber", "bankaccountpassword",
    "password", "idnumber", "balance", "amount", "date", "recipient", "currentday",
    "lastmodified", "webcontent", "weburl", "fileid", "filename", "filetype", "size",
    "content", "currentaccount", "iban", "discount", "code", "starredrepository",
    "repository", "accountemail", "ownerusername", "originalprice", "paymentamount",
    "problems", "instructions", "submission", "invoicefororderid", "useexactformat",
))
_YAML_TYPED = frozenset(("tag:yaml.org,2002:bool", "tag:yaml.org,2002:int",
                         "tag:yaml.org,2002:float", "tag:yaml.org,2002:timestamp",
                         "tag:yaml.org,2002:null"))


def _contract_yaml_regions(ytext):
    """(steered_regions, info) under the v2 contract; raises on any uncertainty."""
    import yaml
    node = yaml.compose(ytext, Loader=yaml.SafeLoader)
    if node is None:
        raise ValueError("empty yaml")
    regs = []
    info = {"trusted_keys": set(), "steered_keys": set(), "data_keys": set()}

    def add(n):
        regs.append((n.start_mark.index, n.end_mark.index))

    def scalar_kind(n):
        if n.tag == "tag:yaml.org,2002:str":
            return "str"
        if n.tag in _YAML_TYPED:
            return "typed"
        raise ValueError(f"non-core yaml tag {n.tag!r}")   # custom/binary -> fallback

    def walk(n):
        if n.id == "mapping":
            for k, v in n.value:
                if k.id != "scalar":
                    raise ValueError("non-scalar mapping key")
                nk = _norm_key(k.value)
                known = nk in _CONTRACT_SCHEMA_KEYS or nk in _CONTRACT_TRUSTED
                if not known:
                    # DATA in key position: the key text is external content
                    info["data_keys"].add(nk[:40])
                    if scalar_kind(k) == "str":
                        add(k)
                trusted = nk in _CONTRACT_TRUSTED
                if v.id == "scalar":
                    if scalar_kind(v) == "str":
                        if trusted:
                            info["trusted_keys"].add(nk)
                        else:
                            info["steered_keys"].add(nk)
                            add(v)
                    # typed scalars excluded regardless of key
                else:
                    walk(v)   # containers always recurse; trust is per-leaf
        elif n.id == "sequence":
            for item in n.value:
                walk(item)
        else:
            if scalar_kind(n) == "str":   # keyless string (bare return, list item)
                add(n)

    walk(node)
    return regs, info


def contract_regions(seg, tool_name=None):
    """v2 steered char regions within one rendered tool span (offsets relative to seg).

    Same return convention as content_leaf_regions: (regions, info) with
    info["fallback"] set to a reason string when the whole span was taken (fail-closed).
    """
    info = {"tool": tool_name, "fallback": None, "html": False,
            "trusted_keys": [], "steered_keys": [], "data_keys": []}

    def whole():
        return [(0, len(seg))], info

    if "\n" not in seg and "\\" in seg:
        try:
            txt, om = _unescape_with_map(seg)
        except ValueError:
            info["fallback"] = "escape-decode-failed"
            return whole()
    else:
        txt, om = seg, None
    # whole-payload tools (file/web/git): ALL bytes are external content. HTML included
    # whole -- the v1 visible-text-only rule left comments/attrs unsteered (§26.13
    # audit surface); v2 closes it by not parsing HTML at all.
    if tool_name in WHOLE_CONTENT_TOOLS:
        info["html"] = bool(_HTML_HINT.search(txt))
        return whole()
    try:
        regs, winfo = _contract_yaml_regions(txt)
    except Exception as e:
        info["fallback"] = f"yaml-uncertain: {e}"[:120]
        return whole()
    info["trusted_keys"] = sorted(winfo["trusted_keys"])
    info["steered_keys"] = sorted(winfo["steered_keys"])
    info["data_keys"] = sorted(winfo["data_keys"])
    if om is not None:
        regs = [(om[a], om[b]) for a, b in regs]
    regs = sorted(r for r in regs if r[1] > r[0])
    merged = []
    for a, b in regs:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged, info


def content_leaf_regions(seg, tool_name=None):
    """CONTENT char regions within one rendered tool span (offsets relative to `seg`).

    Returns (regions, info). info["fallback"] is a string reason when the whole span was
    taken as content because of parse/classification uncertainty (the §26.12 fail-closed
    rule), else None. regions are sorted, merged, non-empty only where real content exists;
    a genuinely all-structure payload legitimately yields [].
    """
    info = {"tool": tool_name, "fallback": None, "html": False,
            "content_keys": [], "struct_keys": [], "shape_struct": 0,
            "shape_struct_keys": []}

    def whole():
        return [(0, len(seg))], info

    # 1) decode the template's JSON-string escaping (identity when nothing is escaped);
    #    real newlines present means the render was NOT escaped (non-harmony formats).
    if "\n" not in seg and "\\" in seg:
        try:
            txt, om = _unescape_with_map(seg)
        except ValueError:
            info["fallback"] = "escape-decode-failed"
            return whole()
    else:
        txt, om = seg, None

    def back(regions):
        if om is None:
            return regions
        return [(om[a], om[b]) for a, b in regions]

    def merged(regions):
        regions = sorted(r for r in regions if r[1] > r[0])
        out = []
        for a, b in regions:
            if out and a <= out[-1][1]:
                out[-1] = (out[-1][0], max(b, out[-1][1]))
            else:
                out.append((a, b))
        return out

    # 2) whole-payload-content tools: file contents / web text / git conflict messages.
    #    HTML payloads keep visible text nodes only (owner rule); parser anomaly -> whole.
    if tool_name in WHOLE_CONTENT_TOOLS:
        if _HTML_HINT.search(txt):
            info["html"] = True
            try:
                return merged(back(_html_text_regions(txt))), info
            except Exception:
                info["fallback"] = "html-parse-failed"
                return whole()
        return whole()   # plain prose payload: all content, and deliberately so
    # 3) structured payload: YAML/JSON parse with exact scalar marks, classify per key.
    try:
        regs, winfo = _yaml_leaf_regions(txt)
    except Exception as e:
        info["fallback"] = f"yaml-uncertain: {e}"[:120]
        return whole()
    info["content_keys"] = sorted(winfo["content_keys"])
    info["struct_keys"] = sorted(winfo["struct_keys"])
    info["shape_struct"] = winfo["shape_struct"]
    info["shape_struct_keys"] = sorted(winfo["shape_struct_keys"])
    return merged(back(regs)), info


def _select_span_positions(span_select, seg, tool_name, span_tok_idx, offsets, span_lo):
    """Token positions to steer within ONE span, per selector arm.

    span_tok_idx: the span's FULL-span token positions (the deployed rule: token fully
    inside the span). offsets: the encoding's (char_a, char_b) per token. span_lo: the
    span's absolute char start. Returns (positions, info).

      leaf     tokens fully inside a content region (same fully-inside rule as spans)
      contract tokens fully inside a v2 provenance-contract region (§26.17) -- same
               mechanics as leaf with contract_regions supplying the regions
      struct   the exact complement within the span: full minus leaf -- so leaf|struct
               partition the deployed edit set per span, the §26.12 controls' semantics
      random   len(leaf) positions sampled uniformly from the span's full set, seeded from
               the span TEXT (stable across the episode's re-renders and across processes)
      energy   handled by the caller (full positions, alpha scaled); returns full here
    """
    regions, info = (contract_regions(seg, tool_name) if span_select == "contract"
                     else content_leaf_regions(seg, tool_name))
    abs_regions = [(span_lo + a, span_lo + b) for a, b in regions]
    leaf = [i for i in span_tok_idx
            if any(offsets[i][0] >= a and offsets[i][1] <= b for a, b in abs_regions)]
    info["n_full"], info["n_leaf"] = len(span_tok_idx), len(leaf)
    if span_select in ("leaf", "contract"):
        return leaf, info
    if span_select == "struct":
        leaf_set = set(leaf)
        return [i for i in span_tok_idx if i not in leaf_set], info
    if span_select == "random":
        import random as _random
        import zlib
        rng = _random.Random(zlib.crc32(seg.encode("utf-8", "replace")))
        k = len(leaf)
        return sorted(rng.sample(span_tok_idx, k)) if k < len(span_tok_idx) \
            else list(span_tok_idx), info
    return list(span_tok_idx), info   # "energy": full positions, dose scaled by caller


def _executable_calls(completion, fmt, in_think=False):
    """Tool calls the model EMITTED -- never ones it merely QUOTED while refusing.

    `parse_tool_calls` scans for `to=functions.NAME` anywhere in the raw text, and the bridge
    turns whatever it finds into real FunctionCall objects that AgentDojo EXECUTES against the
    live environment. gpt-oss quotes injected instructions in its analysis channel constantly --
    we have traces of it writing out the attacker's exact call while explaining that it will
    not comply. Feeding the whole completion to the parser therefore executes the attack on
    behalf of a model that refused it, and AgentDojo's `security` checker reads the mutated
    environment: A REFUSAL SCORES AS A SUCCESSFUL ATTACK.

    Worse, it is arm-asymmetric in an unknown direction -- steering changes how much injected
    text the model quotes, which is itself one of this project's measured effects.

    So: strip the reasoning region first, then parse. Per format that region is
      harmony  every `analysis` channel message
      chatml   with `in_think` (the PROMPT ended inside `<think>`, which is how the Thinking
               templates render the generation header) everything before the first
               `</think>`; a completion truncated mid-reasoning has emitted nothing
               executable at all. Without `in_think` (a non-thinking chatml model), any
               self-opened `<think>...</think>` blocks -- requiring `</think>` there would
               parse zero calls from every completion, forever.
      glm45    same as chatml-without-in_think: GLM-4.5 SELF-OPENS `<think>...</think>`
               (its generation prompt ends `<|assistant|>`, nothing pre-opened -- measured
               on the stored runs/glm45-air completions, 2026-09-04), and it quotes
               attacker text while deliberating exactly as Qwen does, so the raw text
               must never reach the parser.
      gemma4   the self-opened `<|channel>thought...<channel|>` region, stripped by
               reasoning_free's _GEMMA_THINK branch (in_think always False -- the
               non-thinking generation prompt pre-closes the channel). parse_tool_calls
               already reads the `<|tool_call>call:NAME{...}` blocks.
    AgentDojo's own OpenAI path has no equivalent exposure because it reads structured
    `tool_calls` from the API.
    """
    if fmt in ("chatml", "glm45", "gemma4"):
        return X.parse_tool_calls(reasoning_free(completion, fmt, in_think))
    if fmt == "phi3":
        # No reasoning region exists in this format (no CoT convention at all), so the raw
        # completion is the emitted text; the LLMail line parser inside parse_tool_calls
        # is already strict (call on its own line, starting `{`), so quoted examples
        # prefixed with prose do not parse.
        return X.parse_tool_calls(completion)
    if fmt == "llama31":
        # No reasoning region (reasoning_free is identity for this format). The shared
        # parser's Llama-3.1 branch is line-anchored AND requires the model's own turn
        # terminator (<|eom_id|>/<|eot_id|>) on the call line, so a call QUOTED in prose
        # (no terminator mid-turn) does not parse as executable (Llama AgentDojo port,
        # 2026-09-09; parser strictness reviewed same day, src/scoring.py).
        return X.parse_tool_calls(completion)
    stripped = re.sub(r"<\|channel\|>analysis<\|message\|>.*?(?:<\|end\|>|(?=<\|start\|>)|$)",
                      "", completion, flags=re.S)
    return X.parse_tool_calls(stripped)


# reasoning_free moved to src/scoring.py (2026-08-31) so the adaptive harnesses share it;
# re-exported here because agentdojo_rescore.py and older callers import it from the bridge.
reasoning_free = X.reasoning_free


_ANNOUNCED = set()      # one sigma-provenance line per unique steering config, not per arm

# ONE definition of the valid schedules, imported by the drivers' --steer-schedule choices
# (the DOSE_FREE_MODES lesson, FINDINGS §23e: hand-copied vocabularies drift).
SCHEDULES = ("fixed", "energy-norm")


def sched_step(schedule, alpha, n_steered, n0):
    """Per-forward steering-dose schedule (§26.5 item 2, owner program 2026-09-08).

    Returns ``(alpha_eff, sched_scale, n0)`` for ONE forward that steers ``n_steered``
    token positions; ``n0`` is the schedule's episode anchor (threaded back in by the
    caller, ``None`` until the episode's first steered forward).

    ENERGY DEFINITION. One forward applies, at each of k steered layers, the additive
    step ``step_l = (alpha/sqrt(k)) * sigma_l`` to every one of the ``n_t = n_steered``
    steered positions (Steer: mode=add, scale=sigma, step_rule=fixed -- the deployed
    configuration this bridge constructs). Define the forward's NOMINAL edit energy as
    the squared step summed over edited positions and layers:

        E_t = n_t * sum_l step_l^2 = n_t * s_t^2 * sum_l ((alpha/sqrt(k)) * sigma_l)^2

    where ``s_t`` is this schedule's scale on alpha.

    schedule="fixed"        s_t = 1 (today's behavior): E_t grows LINEARLY with the
                            steered-token count as tool spans accumulate over turns,
                            because every historical tool result is re-steered on each
                            turn's full re-render.
    schedule="energy-norm"  hold E_t ~constant at its value on the episode's FIRST
                            steered forward: s_t = sqrt(n_0 / n_t) with n_0 = the steered
                            count at that first steered forward, CAPPED at 1.0
                            (s_t = min(1, sqrt(n_0/n_t))), so E_t = min(n_t, n_0) * E_0/n_0.

    WHY THE CAP. Without it, a forward with n_t < n_0 would be dosed ABOVE the fixed
    schedule's per-token step (alpha_eff > alpha) -- amplification would confound the
    controlled comparison, because a difference between the arms could then come from
    exceeding the certified dose rather than from normalizing accumulated exposure. Early
    turns are therefore NEVER dosed above fixed; the two schedules differ only where
    exposure has grown past the first steered forward.

    NOMINAL, not exact: Steer's norm-preserving rescale follows each additive step, so
    the realized displacement per token differs from the nominal step. The schedule
    controls the injected (pre-rescale) energy -- the same quantity alpha itself controls.

    schedule="fixed" returns ``alpha`` UNCHANGED (the same object, no arithmetic), which
    is what keeps the default path byte-identical to the pre-schedule bridge.
    """
    if schedule == "fixed":
        return alpha, 1.0, n0
    if n0 is None:
        # floor the anchor (review8 D2): anchoring n0=0 on a hypothetical zero-token
        # first call would freeze the scale at 0 for the whole episode. Unreachable via
        # query() (guarded by non-empty idx), but the pure function must not carry the
        # landmine.
        n0 = max(1, n_steered)
    s = min(1.0, (n0 / max(1, n_steered)) ** 0.5)
    return alpha * s, s, n0


def generation_truncated(generated_ids, eos_token_id):
    """A generated turn is truncated iff it contains none of the model's EOS tokens."""
    eos = eos_token_id if isinstance(eos_token_id, (list, tuple, set)) else [eos_token_id]
    eos = {int(token_id) for token_id in eos if token_id is not None}
    if torch.is_tensor(generated_ids):
        if not eos:
            return True
        eos_tensor = torch.tensor(sorted(eos), device=generated_ids.device,
                                  dtype=generated_ids.dtype)
        return not bool(torch.isin(generated_ids, eos_tensor).any().item())
    return not eos or not any(int(token_id) in eos for token_id in generated_ids)


class SteeredLLM(BasePipelineElement):
    """A local (harmony or ChatML) agent for AgentDojo, with the steering hook over tool
    output.

    `direction=None` runs the model completely unmodified -- that is the undefended arm, and it
    shares every other code path with the defended arm so the two differ in exactly the edit.
    """

    def __init__(self, model, tok, probe_dir=None, direction=None, layers=(12, 16, 20),
                 alpha=8.0, scale="sigma", match_sigma_to="dim_no_override",
                 max_new=512, no_think=False, kv_mask=None, schedule="fixed",
                 span_select="full"):
        self.model, self.tok = model, tok
        # WHICH tokens inside each tool span are steered (§26.12 content-leaf program):
        # "full" (default) = every span token, the deployed behavior, byte-identical path
        # (no selector code executes); "leaf" = content-leaf selection (fail-closed to
        # full-span per uncertain span); "struct" = the exact per-span complement of leaf;
        # "random" = per span, leaf-COUNT-matched positions sampled from the span's full
        # set (seeded from the span text: stable across re-renders and processes);
        # "energy" = full-span positions with alpha scaled per forward by
        # sqrt(leaf_tokens/full_tokens) so TOTAL edit energy matches the leaf arm.
        if span_select not in SPAN_SELECTS:
            raise SystemExit(f"unknown span_select {span_select!r}; have {list(SPAN_SELECTS)}")
        if span_select != "full" and not direction:
            raise SystemExit(
                f"span_select {span_select!r} on an arm with no steering direction -- the "
                f"selector would be a label on an edit that never runs (the FINDINGS §23e "
                f"no-op-wearing-a-label class). Pass a direction or leave span_select='full'.")
        if span_select != "full" and schedule != "fixed":
            raise SystemExit("span_select and a non-fixed steer schedule are mutually "
                             "exclusive -- one dose manipulation per arm")
        if span_select != "full" and kv_mask:
            raise SystemExit("span_select applies to steering; kv_mask arms are unsteered")
        self.span_select = span_select
        self.span_telemetry = None  # set to a list to record per-forward selector stats
        # Dose SCHEDULE over turns (§26.5 item 2): "fixed" = today's behavior, alpha
        # untouched every forward; "energy-norm" = scale alpha per forward so TOTAL
        # per-forward edit energy stays ~constant as tool spans accumulate (see
        # sched_step's docstring for the exact definition and the amplification cap).
        if schedule not in SCHEDULES:
            raise SystemExit(f"unknown steer schedule {schedule!r}; have {list(SCHEDULES)}")
        if schedule != "fixed" and not direction:
            raise SystemExit(
                f"steer schedule {schedule!r} on an arm with no steering direction -- the "
                f"schedule would be a label on an edit that never runs (the FINDINGS §23e "
                f"no-op-wearing-a-label class). Pass a direction or leave schedule='fixed'.")
        self.schedule = schedule
        self._sched_n0 = None       # steered-token count at the episode's FIRST steered forward
        self._sched_last_ntool = 0  # tool msgs in the last query; a DROP = pipeline reused
                                    # for a fresh episode, so the anchor must reset
        # CachePrune (arXiv:2504.21228): `kv_mask` is the path to a mask JSON from
        # tools/controls/build_cacheprune_mask.py. Applied per turn to every tool-result
        # span via a PrunedKVCache -- the same spans steering would edit -- and mutually
        # exclusive with `direction` (two defenses in one arm is not an arm).
        self.kv_spec = None
        if kv_mask:
            if direction:
                raise SystemExit("kv_mask and direction are mutually exclusive")
            self.kv_spec = X.load_kv_mask(kv_mask, model.config)
        self.fmt = fmt_of(tok)
        self.layers, self.alpha, self.scale = list(layers), alpha, scale
        self.max_new, self.no_think = max_new, no_think
        # glm45 included: GLM-4.5 is a thinking model too (self-opened <think>), so the
        # small-max_new hazard below applies to it exactly as to the ChatML Thinking family
        if self.fmt in ("chatml", "glm45", "gemma4") and max_new < 2048 \
                and ("maxnew", max_new) not in _ANNOUNCED:
            _ANNOUNCED.add(("maxnew", max_new))
            # A Thinking model spends 1-4k tokens INSIDE <think> before it can emit anything
            # executable; a turn truncated there produces no tool call and no answer, so a
            # small max_new silently zeroes utility while every other number looks sane.
            # Measured: the alpha-14 dose row ran at the gpt-oss default of 768 and clean
            # utility collapsed 0.875 -> 0.411 with 1-2 llm calls/episode -- voided 2026-08-31.
            print(f"  ** WARNING: max_new={max_new} with a ChatML/Thinking template -- "
                  f"thinking alone routinely exceeds this, truncated turns act as NO-OPs, "
                  f"and utility collapses for reasons that have nothing to do with the "
                  f"defense. The Qwen battery + AgentDojo runs used --max-new 4096. **",
                  flush=True)
        self.direction = direction
        self.dirs = self.sigmas = self.ablate = None
        if direction:
            # `match_sigma_to` picks WHOSE stored sigma sets the step (step = alpha*sigma).
            # None/"" = the steered direction's OWN sigma. The default matches the locked-in
            # gpt-oss cells; for other probe dirs the caller must choose EXPLICITLY, because
            # build_dirs falls back to the direction's own sigma when the requested name has
            # no stored sigma -- silently, which is a unit change, not an error. The print
            # below exists so a fallback can never pass unnoticed in a log.
            self.dirs, self.sigmas, self.ablate = X.build_dirs(
                probe_dir, self.layers, direction, model.device,
                match_sigma_to=(match_sigma_to or None))
            key = (probe_dir, direction, tuple(self.layers), match_sigma_to or "", alpha,
                   schedule, span_select)
            if key not in _ANNOUNCED:
                _ANNOUNCED.add(key)
                per = alpha / (len(self.layers) ** 0.5)
                print(f"[steer] {direction} L={self.layers} alpha={alpha} "
                      f"schedule={schedule} "
                      + (f"span_select={span_select} " if span_select != "full" else "")
                      + f"match_sigma_to={match_sigma_to or '<own>'} "
                      f"sigma={[round(s, 4) for s in self.sigmas]} "
                      f"step(alpha/sqrt(k)*sigma)={[round(per * s, 4) for s in self.sigmas]}",
                      flush=True)
        self.n_calls = 0
        self.n_steered_tokens = 0
        self.n_truncated = 0
        # Set to a list to record (prompt, completion, n_steered, spans) per turn. Off by
        # default -- these prompts are ~4-20k chars each and holding every one of them for a
        # whole benchmark run is a memory leak, not a feature.
        self.transcript = None

    def query(self, query, runtime: FunctionsRuntime, env=EmptyEnv(), messages=(), extra_args=None):
        extra_args = dict(extra_args or {})
        tools = _tools_for_template(runtime)
        # span_select="full" passes names_out=None: the pre-selector call signature, and no
        # selector code executes anywhere below -- the byte-identity requirement (§26.12).
        span_names = None if self.span_select == "full" else []
        text, spans = _tool_spans(self.tok, messages, tools, self.no_think,
                                  names_out=span_names)
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids, om = enc["input_ids"], enc["offset_mapping"]
        idx = [i for i, (a, b) in enumerate(om)
               if b > a and any(a >= lo and b <= hi for lo, hi in spans)]

        # ── content-leaf selection (§26.12): restrict/replace the span positions ──────────
        idx_s, sel_scale, sel_stats = idx, None, None
        if self.span_select != "full" and idx:
            # assign tokens to spans in one pass (spans are sorted and disjoint; offsets of
            # a fast tokenizer are non-decreasing), then assert the union matches the
            # deployed full-span rule exactly -- a silent mismatch here is the §26.12
            # fail-silent class and must abort, not steer the wrong set.
            per_span, si = [[] for _ in spans], 0
            for i, (a, b) in enumerate(om):
                if b <= a:
                    continue
                while si < len(spans) and a >= spans[si][1]:
                    si += 1
                if si < len(spans) and a >= spans[si][0] and b <= spans[si][1]:
                    per_span[si].append(i)
            assert sorted(x for s in per_span for x in s) == idx, \
                "per-span token assignment diverged from the deployed span rule"
            sel_idx, n_leaf_total = [], 0
            sel_stats = {"n_spans": len(spans), "n_fallback": 0, "html_spans": 0,
                         "full_tokens": len(idx), "leaf_tokens": 0,
                         "fallbacks": [], "content_keys": set(), "struct_keys": set(),
                         "shape_struct": 0, "shape_struct_keys": set(),
                         # v2 contract telemetry (empty on v1 arms)
                         "trusted_keys": set(), "steered_keys": set(),
                         "data_keys": set()}
            for s_i, (lo, hi) in enumerate(spans):
                stoks = per_span[s_i]
                if not stoks:
                    continue
                pos, info = _select_span_positions(
                    self.span_select, text[lo:hi], span_names[s_i], stoks, om, lo)
                sel_idx += pos
                n_leaf_total += info["n_leaf"]
                if info["fallback"]:
                    sel_stats["n_fallback"] += 1
                    sel_stats["fallbacks"].append(
                        {"tool": info["tool"], "reason": info["fallback"]})
                sel_stats["html_spans"] += int(info["html"])
                # v1 and v2 infos carry different key inventories; aggregate tolerantly
                for fld in ("content_keys", "struct_keys", "shape_struct_keys",
                            "trusted_keys", "steered_keys", "data_keys"):
                    sel_stats[fld].update(info.get(fld) or [])
                sel_stats["shape_struct"] += info.get("shape_struct", 0)
            sel_stats["leaf_tokens"] = n_leaf_total
            for fld in ("content_keys", "struct_keys", "shape_struct_keys",
                        "trusted_keys", "steered_keys", "data_keys"):
                sel_stats[fld] = sorted(sel_stats[fld])
            if self.span_select == "energy":
                # uniform-reduced full-span dose matched to the leaf arm's TOTAL edit
                # energy: n_full * (s*step)^2 = n_leaf * step^2  =>  s = sqrt(n_leaf/n_full)
                if n_leaf_total == 0:
                    idx_s = []          # leaf arm would edit nothing: matched energy is 0
                else:
                    idx_s = sorted(sel_idx)
                    sel_scale = (n_leaf_total / len(idx_s)) ** 0.5
            else:
                idx_s = sorted(set(sel_idx))
            sel_stats["sel_tokens"] = len(idx_s)
            sel_stats["sel_scale"] = sel_scale
            if self.span_telemetry is not None:
                self.span_telemetry.append(sel_stats)

        # Episode boundary for the dose schedule: within one episode the re-rendered
        # history only ever GAINS tool messages, so a DROP in their count means this
        # pipeline object was reused for a fresh episode and the schedule's n_0 anchor
        # must re-anchor on the new episode's first steered forward. Pure bookkeeping --
        # no effect on the fixed schedule's steps.
        n_toolmsgs = sum(1 for m in messages if m.get("role") == "tool")
        if n_toolmsgs < self._sched_last_ntool:
            self._sched_n0 = None
        self._sched_last_ntool = n_toolmsgs

        steer, sched_scale = None, None
        if self.dirs is not None and idx_s:
            # schedule="fixed" returns self.alpha UNCHANGED (same object, no arithmetic),
            # and idx_s IS idx (same object) under span_select="full", so the default path
            # hands Steer byte-identical arguments to the pre-selector bridge.
            alpha_eff, sched_scale, self._sched_n0 = sched_step(
                self.schedule, self.alpha, len(idx_s), self._sched_n0)
            if sel_scale is not None:
                alpha_eff = alpha_eff * sel_scale   # "energy" arm's matched-dose reduction
            steer = X.Steer(self.model, self.layers, self.dirs, alpha_eff, self.scale,
                            self.sigmas, self.ablate, "add", None, 1.0, True, None, 0.0)
        ids_t = torch.tensor([ids], device=self.model.device)
        self.n_calls += 1
        self.n_steered_tokens += len(idx_s) if (steer or self.kv_spec) else 0
        if steer:
            steer.__enter__()
            steer.positions = [idx_s]
        gen_kw = {}
        if self.kv_spec is not None and idx:
            # fresh PrunedKVCache per turn: the whole history is re-rendered each turn, so
            # every tool-result span seen so far is masked, matching the steering arm
            gen_kw["past_key_values"] = self.kv_spec.make_cache([idx])
        try:
            with torch.no_grad():
                g = self.model.generate(input_ids=ids_t, max_new_tokens=self.max_new,
                                        do_sample=False, pad_token_id=self.tok.pad_token_id,
                                        **gen_kw)
        finally:
            if steer:
                steer.positions = None
                steer.__exit__()
        generated_ids = g[0][ids_t.shape[1]:]
        completion = self.tok.decode(generated_ids, skip_special_tokens=False)
        # Did the PROMPT open a <think> block (Thinking templates end the generation header
        # inside one)? That decides whether `</think>` is required before anything counts as
        # emitted -- checked per prompt, not per model name, so a non-thinking chatml model
        # parses correctly too. glm45 is in the check for the same reason: today its
        # template never pre-opens <think> (measured 2026-09-04, so this is always False),
        # but a variant that does must flip the parse, not silently leak reasoning.
        # gemma4 is deliberately NOT in the check: its template opens AND CLOSES the
        # thought channel in the generation prompt (`<|channel>thought\n<channel|>`), so
        # the prompt never ends inside a reasoning region -- in_think is False by
        # construction and the self-opened-region strip lives in reasoning_free.
        in_think = self.fmt in ("chatml", "glm45") and text.rstrip().endswith("<think>")
        self.n_truncated += int(generation_truncated(
            generated_ids, self.model.generation_config.eos_token_id))
        if self.transcript is not None:
            # `fmt`/`in_think` ride along so scorers can strip the reasoning region OFFLINE
            # (reasoning_free) without re-rendering the prompt
            # sched_scale = the per-forward multiplier the dose schedule applied to alpha
            # (None on an unsteered forward; 1.0 always under schedule="fixed") -- the
            # engagement evidence for the §26.5 item 2 controlled experiment.
            rec = {"prompt": text, "completion": completion,
                   "n_steered": len(idx_s) if steer else 0,
                   "fmt": self.fmt, "in_think": in_think,
                   "schedule": self.schedule,
                   "sched_scale": sched_scale if steer else None,
                   "spans": spans, "steered_char_ranges": spans if steer else []}
            if self.span_select != "full":
                # §25h steered-tokens standard for the selector arms: per-forward evidence
                # of what was selected, what fell back, and the leaf/full token counts.
                # Only written on selector arms so default-path artifact schemas are
                # untouched (the sched_scales precedent).
                rec["span_select"] = self.span_select
                rec["sel_stats"] = sel_stats
            self.transcript.append(rec)

        calls = _executable_calls(completion, self.fmt, in_think)
        # keep only calls the runtime actually offers; a hallucinated name would abort the loop
        known = set(runtime.functions)
        tool_calls = [FunctionCall(function=n, args=(a if isinstance(a, dict) else {}))
                      for n, a in calls if n in known]
        out = ChatAssistantMessage(
            role="assistant",
            content=[text_content_block_from_string(_final_text(completion, self.fmt,
                                                                in_think))],
            tool_calls=tool_calls or None)
        return query, runtime, env, [*messages, out], extra_args


_HARMONY_TOKEN = re.compile(r"<\|[^|]*\|>")     # also matches ChatML's <|im_start|>/<|im_end|>
# _SPECIAL_TOKEN supersedes _HARMONY_TOKEN in the scrubs (Gemma port, 2026-09-04).
# _HARMONY_TOKEN's `[^|]*` happily crosses `<`/`>` boundaries, so on Gemma's PAIRED
# half-pipe markers (`<|turn>model KEEP <turn|>`) it swallowed the content between them
# (measured, FINDINGS 23y). _SPECIAL_TOKEN matches the same singleton forms --
# `<|X|>` (harmony/chatml/phi3/glm45 specials), `<|X>` (Gemma opens incl. <|"|> via the
# first alternative) and `<X|>` (Gemma closes) -- but its content classes exclude `<`,
# `>` and `|`, so it can never span from one marker into another. Behaviourally identical
# to _HARMONY_TOKEN on all stored non-Gemma completions (regression render diff).
# _HARMONY_TOKEN itself is kept: verify_glm_bridge.py asserts against it by name.
_SPECIAL_TOKEN = re.compile(r"<\|[^|<>]*\|>|<\|[^|<>]*>|<[^|<>]*\|>")
_THINK_BLOCK = X._THINK_BLOCK  # shared with src/scoring.reasoning_free
# a whole emitted Gemma call block; unclosed-at-truncation stripped to end-of-string,
# mirroring _TOOL_CALL_BLOCK's policy
_GEMMA_CALL_BLOCK = re.compile(r"<\|tool_call>.*?(?:<tool_call\|>|$)", re.S)
# Also strip BARE <function=...> blocks outside a <tool_call> wrapper: the XML parser
# executes them (deliberately -- the undercount direction fabricates defense wins), so
# _final_text must not leak the same executed call into the answer text that AgentDojo's
# utility checkers substring-match and the next turn's history (Qwen3.8 port review,
# 2026-08-31).
_TOOL_CALL_BLOCK = re.compile(
    r"<tool_call>.*?(?:</tool_call>|$)|<function=[A-Za-z0-9_]+>.*?(?:</function>|$)", re.S)


def _final_text(completion, fmt, in_think=False):
    """The user-visible answer only: harmony's `final` channel / chatml's post-`</think>`
    text, or "" on a tool-call turn.

    MUST NOT fall back to the raw completion. AgentDojo feeds each assistant message back into
    the next turn, and gpt-oss's chat template hard-refuses content containing `<|channel|>`
    tags: "you should pass analysis messages ... in the 'thinking' field". Returning the raw
    string therefore poisoned every multi-turn task -- the FIRST turn succeeded, the second
    died rendering the history, and the whole run surfaced as a TypeError from a None prompt.
    A tool-call turn has no final channel and its content is legitimately empty.

    The chatml equivalent of that hazard is the reasoning leaking into the CHECKER, not the
    template: AgentDojo's utility checkers substring-match the last assistant message, and a
    `<think>` region that quotes the expected answer (or the injection) would satisfy them
    without the model ever answering. So: reasoning stripped (with `in_think`, everything
    before the first `</think>` -- the Thinking template opens the completion inside
    `<think>`, and truncated mid-reasoning => no answer exists => ""; without it, any
    self-opened `<think>` blocks), `<tool_call>` blocks stripped (they are calls, not prose;
    the harmony path likewise never returns the commentary channel), specials stripped.

    gemma4 has its own branch (Gemma AgentDojo port, 2026-09-04): cut at the first
    generation_config terminator, strip the self-opened `<|channel>thought` region and
    whole `<|tool_call>...<tool_call|>` blocks, then scrub with the Gemma-safe
    _SPECIAL_TOKEN -- both measured breaks of the old path (`_final_text` leaking the
    call block; `_HARMONY_TOKEN` swallowing content between paired markers, FINDINGS 23y)
    are covered by named regressions in tmp/dojo_expand/gemma/verify_gemma_bridge.py.
    The fmt-dispatched branches below keep _HARMONY_TOKEN deliberately: they never see
    Gemma content, and keeping them byte-stable is what the regression diff certifies.

    glm45 routes through the chatml branch VERBATIM (GLM AgentDojo port, 2026-09-04),
    per the measured assessment (FINDINGS 23y) re-verified on stored runs/glm45-air
    completions: self-opened <think> stripped by _THINK_BLOCK, the <tool_call>NAME\n
    <arg_key>... blocks stripped whole by _TOOL_CALL_BLOCK (its body match is
    content-agnostic), and the eos markers <|endoftext|>/<|user|>/<|observation|> all
    match _HARMONY_TOKEN. GLM's markers are singletons, not paired open/close tags, so
    the Gemma over-stripping hazard (content swallowed between paired markers) cannot
    occur here -- measured, not assumed, in tmp/dojo_expand/glm/verify_glm_bridge.py.
    """
    if fmt == "gemma4":
        # Cut at the first terminator: <|tool_response> (the model stopping itself where
        # the executor will inject the result), <turn|> (turn close) or <eos>; everything
        # after (right-padding included) is not this turn's answer. Then drop the
        # self-opened thought channel, drop emitted call blocks WHOLE (they are calls,
        # not prose -- _TOOL_CALL_BLOCK does not match Gemma's shape, and _HARMONY_TOKEN
        # left the call bodies behind, the measured `_final_text leaks the call block`
        # break in FINDINGS 23y), and strip residual markers with the Gemma-safe
        # _SPECIAL_TOKEN, never _HARMONY_TOKEN (the over-strip break).
        txt = re.split(r"<\|tool_response>|<turn\|>|<eos>", completion, maxsplit=1)[0]
        txt = X._GEMMA_THINK.sub("", txt)
        txt = _GEMMA_CALL_BLOCK.sub("", txt)
        return _SPECIAL_TOKEN.sub("", txt).strip()
    if fmt in ("chatml", "glm45"):
        if in_think:
            parts = completion.split("</think>", 1)
            if len(parts) == 1:
                return ""
            txt = parts[1]
        else:
            txt = completion
        txt = _TOOL_CALL_BLOCK.sub("", txt)
        return _HARMONY_TOKEN.sub("", _THINK_BLOCK.sub("", txt)).strip()
    if fmt == "phi3":
        # No reasoning region; the answer is the turn's prose. Cut at the terminator
        # (<|end|>/<|endoftext|>), drop LLMail-convention call lines (they are CALLS, not
        # prose -- leaking one into the answer would feed an executed call back into the
        # next turn's history and into AgentDojo's substring-matching utility checkers,
        # the exact hazard _TOOL_CALL_BLOCK guards on chatml), then strip specials.
        txt = re.split(r"<\|end\|>|<\|endoftext\|>", completion, maxsplit=1)[0]
        keep = [ln for ln in txt.splitlines()
                if not (ln.strip().startswith("{") and '"function"' in ln)]
        return _HARMONY_TOKEN.sub("", "\n".join(keep)).strip()
    if fmt == "llama31":
        # No reasoning region; a turn ends at <|eom_id|> (tool call, more turns expected)
        # or <|eot_id|>/<|end_of_text|>. Cut at the first terminator (right-padding after
        # it is not this turn's answer), drop native tool-call JSON lines WHOLE (they are
        # CALLS, not prose -- leaking one feeds an executed call back into the next turn's
        # history and into AgentDojo's substring-matching utility checkers, the same
        # hazard the phi3/chatml branches guard), then strip residual specials with the
        # boundary-safe _SPECIAL_TOKEN (Llama markers are singleton <|X|> forms).
        txt = re.split(r"<\|eom_id\|>|<\|eot_id\|>|<\|end_of_text\|>",
                       completion, maxsplit=1)[0]
        # call lines may carry the <|python_tag|> ipython-mode prefix (same shape the
        # shared parser strips) -- normalize before the drop test or the call leaks
        keep = [ln for ln in txt.splitlines()
                if not ((s := ln.strip().removeprefix("<|python_tag|>").lstrip())
                        .startswith("{") and '"name"' in s
                        and ('"parameters"' in s or '"arguments"' in s))]
        return _SPECIAL_TOKEN.sub("", "\n".join(keep)).strip()
    m = re.findall(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)",
                   completion, re.S)
    return _HARMONY_TOKEN.sub("", m[-1]).strip() if m else ""
