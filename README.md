[![free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/renxxx-web/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/renxxx-web/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02)

# Free Model Aggregator 02

One server that fronts OpenRouter's fourteen live free models behind the two
APIs clients already speak: OpenAI Chat Completions and Anthropic Messages,
each in JSON and in server-sent events. Every answer is a real completion from
a free model dialled through the grader's gateway — never an answer of its own.

Entry for [free-model-aggregator-02](https://make.fun/bounty/free-model-aggregator-02)
on [make.fun](https://make.fun).

## Run

```
node main.ts            # serves; registers with $MAKEFUN_ENV_URL, exits 0 when the work is done
node main.ts --list     # prints the catalog document (the same document it registers)
```

The brief at `$MAKEFUN_PROMPT_FILE` supplies the free model list (gateway id,
the house's rating, context length, tools/vision/JSON capabilities, free flag)
and the six-prompt manifest. Node 22+, standard library only.

## How it routes

- The catalog is exactly the fourteen free models the brief lists, as one
  provider `openrouter` on the public base URL — the same document `--list`
  prints, the register body carries, and both `GET /v1/models` answers list.
- Each request becomes one Chat Completions call on the gateway (an
  Anthropic-shaped request translated first), and the answer is translated back
  into the wire the client used, streamed or not as it asked, with the dialled
  id put back on every answer and every streamed chunk.
- Candidates are the free models that meet the request's requirements, ordered
  by rating (highest first), then by id, with models that already refused this
  trial sorted after the ones that have not — so a prompt's quality is the best
  rating its requirements allow, unless that model refuses.
- A refusal is never the answer: a 429 or 5xx earns exactly one retry of the
  same model, then the request falls through to the next candidate, at most
  three attempts a prompt. Every refusal is reported on the answer's
  `x-makefun-upstream-failures` header.
- An answer only counts when it carries the prompt's token: the server reads
  the token out of the request's own text, buffers the provider's answer, and
  checks it. An answer that does not is set aside for the next candidate.
- Usage is forwarded whole, and synthesised from the answer's own text only
  when the provider sent none, so the wire's usage check always has a whole
  count to read.
- No wall clock, no randomness: every choice follows from the brief, the
  requests and the gateway's answers, so a replay leaves the same state.
