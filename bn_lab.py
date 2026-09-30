"""
AI Laboratory
Theory answers (Q1-Q14) are written as comments next to the relevant code.
"""

import random
from collections import defaultdict, Counter

random.seed(42)  # reproducible results for the lab record

# Q1. Why is the chain-rule decomposition useful for generating text?
#   P(x1..xT) = P(x1) * prod_{t=2..T} P(xt | x1..x_{t-1}).
#   It turns one huge joint distribution (impossible to write down) into a
#   sequence of small "predict the next token" problems. To generate text we
#   sample x1, then x2 given x1, then x3 given x1,x2, and so on.
#
# Q2. Independence assumption of X1 -> X2 -> X3 -> ... -> XT (first order):
#   P(Xt | X1, ..., X_{t-1}) = P(Xt | X_{t-1})
#   i.e. Xt is conditionally independent of X1..X_{t-2} given X_{t-1}
#   (Markov property):  Xt _||_ {X1..X_{t-2}} | X_{t-1}.


START, END = "<START>", "<END>"

RAW = """the cat sat on the mat
the cat sat on the rug
the dog sat on the mat
the dog ran to the park
the cat ran to the park
the dog sat on the rug"""

# Each sentence -> list of tokens, e.g. [<START>, the, cat, ..., <END>]
sentences = [[START] + line.lower().split() + [END] for line in RAW.splitlines()]
vocab = sorted({w for s in sentences for w in s})


# PART IV/V: FIRST-ORDER MODEL  P(Xt | X_{t-1})

class FirstOrderLM:
    def __init__(self):
        # Q4: transition counts are stored here: counts[prev][next] = C(prev, next)
        self.counts = defaultdict(Counter)
        self.probs = {}

    def train(self, sents):
        # Count every consecutive pair (w_i, w_j)
        for s in sents:
            for prev, nxt in zip(s[:-1], s[1:]):
                self.counts[prev][nxt] += 1
        # Q5: P(Xt | X_{t-1}) is computed here:
        #     P(wj | wi) = C(wi, wj) / sum_k C(wi, wk)
        self.probs = {
            prev: {w: c / sum(cnt.values()) for w, c in cnt.items()}
            for prev, cnt in self.counts.items()
        }

    def dist(self, prev):
        # Q7: unseen word -> no row in the table. We return None instead of
        # crashing; the generator then stops the sentence. (A naive LLM
        # version would raise KeyError or divide by zero.) Smoothing would be
        # the proper fix, but is outside this lab.
        return self.probs.get(prev)

    def predict(self, prev):
        # Most probable next token: argmax_w P(w | prev)
        d = self.dist(prev)
        return max(d, key=d.get) if d else None

    def next_token(self, prev, mode="sample"):
        d = self.dist(prev)
        if d is None:
            return None
        if mode == "greedy":
            # Mode A (Part X): always the argmax (ties -> first seen)
            return max(d, key=d.get)
        # Q6: Mode B: SAMPLING from the distribution. Every token with p>0
        # can be chosen, in proportion to its probability. Greedy is
        # deterministic and repeats itself; sampling gives variation.
        return random.choices(list(d.keys()), weights=list(d.values()))[0]

    def generate(self, mode="sample", max_len=20):
        # X1 ~ P(X1 | <START>), X2 ~ P(X2 | X1), ... stop at <END>
        # max_len is needed: greedy can loop forever (the cat sat on the cat ...)
        out, prev = [], START
        for _ in range(max_len):
            nxt = self.next_token(prev, mode)
            if nxt is None or nxt == END:
                break
            out.append(nxt)
            prev = nxt
        return " ".join(out)


# PART XI/XII: SECOND-ORDER MODEL  P(Xt | X_{t-2}, X_{t-1})

class SecondOrderLM:
    def __init__(self):
        self.counts = defaultdict(Counter)  # counts[(w_{t-2}, w_{t-1})][w_t]
        self.probs = {}

    def train(self, sents):
        for s in sents:
            s = [START] + s  # pad so the first real word has a 2-token context
            for a, b, c in zip(s[:-2], s[1:-1], s[2:]):  # observed triples
                self.counts[(a, b)][c] += 1
        self.probs = {
            ctx: {w: c / sum(cnt.values()) for w, c in cnt.items()}
            for ctx, cnt in self.counts.items()
        }

    def dist(self, ctx):
        return self.probs.get(ctx)  # None if context never observed

    def next_token(self, ctx, mode="sample"):
        d = self.dist(ctx)
        if d is None:
            return None
        if mode == "greedy":
            return max(d, key=d.get)
        return random.choices(list(d.keys()), weights=list(d.values()))[0]

    def generate(self, mode="sample", max_len=20):
        out, ctx = [], (START, START)
        for _ in range(max_len):
            nxt = self.next_token(ctx, mode)
            if nxt is None or nxt == END:
                break
            out.append(nxt)
            ctx = (ctx[1], nxt)  # slide the 2-word window
        return " ".join(out)


# Helpers

def show_dist(d):
    return ", ".join(f"{w}: {p:.3f}" for w, p in sorted(d.items(), key=lambda x: -x[1]))


def check_normalisation(model, name):
    # PART VII: for every context, sum_v P(v | context) must equal 1.
    # Q8: a total of 0.87 means the implementation is wrong: e.g. wrong
    #     denominator, dropped transitions (e.g. <END> ignored), or
    #     normalising over the wrong row. Probabilities do not form a valid
    #     distribution, so sampling/argmax results can't be trusted.
    print(f"\n--- Normalisation check ({name}) ---")
    ok = True
    for ctx, d in model.probs.items():
        total = sum(d.values())
        if abs(total - 1.0) > 1e-9:
            ok = False
            print("FAIL", ctx, total)
    print("All rows sum to 1.0" if ok else "Some rows do NOT sum to 1!")


def n_params(model):
    # number of distinct non-zero probabilities stored in the CPT
    return sum(len(d) for d in model.probs.values())



# MAIN EXPERIMENT

if __name__ == "__main__":
    m1, m2 = FirstOrderLM(), SecondOrderLM()
    m1.train(sentences)
    m2.train(sentences)

    # ---- Q3: CPT for selected words + zero-probability transitions 
    print("=== Q3: P(next | current), first-order ===")
    for w in ["the", "cat", "dog", "sat", "ran"]:
        print(f"P(. | {w}) -> {show_dist(m1.probs[w])}")
        zeros = [v for v in vocab if v not in m1.probs[w] and v != START]
        print(f"   zero-probability transitions: {zeros}")
    # Example: P(mat | cat) = 0 because 'cat mat' never occurs in the data.

    # ---- PART VII: normalisation tests 
    check_normalisation(m1, "first-order")
    check_normalisation(m2, "second-order")

    # ---- PART VIII: predictions ---
    print("\n=== Part VIII: next-word distributions and argmax ===")
    for w in ["the", "cat", "dog", "sat", "on", "ran", "to"]:
        print(f"P(. | {w}) = {show_dist(m1.probs[w])}   argmax = {m1.predict(w)}")
    # Q9: Not always the words a human expects. The model knows only the
    #     counts in 6 sentences: e.g. after 'the' it gives cat/dog equal weight
    #     and breaks ties arbitrarily; it has no grammar or meaning (it might
    #     even allow 'the' -> 'park' after 'sat on'). A probability model
    #     reflects the frequencies in its training data, while human
    #     expectations come from world knowledge, grammar and semantics.

    # ---- PART IX: generate 20 sentences and save 
    gen1 = [m1.generate("sample") for _ in range(20)]
    with open("generated_sentences.txt", "w") as f:
        f.write("# First-order (sampling)\n" + "\n".join(gen1) + "\n")
    print("\n=== Part IX: 20 sampled sentences (saved to generated_sentences.txt) ===")
    print("\n".join(gen1))

    # ---- PART X: greedy vs sampling 
    print("\n=== Part X: greedy vs sampling (5 each) ===")
    greedy = [m1.generate("greedy") for _ in range(5)]
    sampled = [m1.generate("sample") for _ in range(5)]
    print("Greedy :", *greedy, sep="\n  ")
    print("Sample :", *sampled, sep="\n  ")
    # Q10: Sampling gives more variation. Greedy always takes argmax, so with
    #      the same model it produces the same sentence every time (and may
    #      loop until max_len). Sampling follows the whole distribution, so
    #      lower-probability continuations are also used.

    # ---- Second-order: CPT and generation 
    print("\n=== Second-order: selected contexts ===")
    for ctx in [(START, START), (START, "the"), ("sat", "on"), ("on", "the"), ("ran", "to")]:
        print(f"P(. | {ctx}) -> {show_dist(m2.probs[ctx])}")
    gen2 = [m2.generate("sample") for _ in range(20)]
    with open("generated_sentences.txt", "a") as f:
        f.write("\n# Second-order (sampling)\n" + "\n".join(gen2) + "\n")
    print("\nSecond-order sampled sentences:", *gen2[:8], sep="\n  ")

    # Q11: first- vs second-order
    #  1. Graph: first order has one parent per node (X_{t-1} -> X_t); second
    #     order has two parents (X_{t-2} -> X_t <- X_{t-1}).
    #  2. CPT: rows indexed by single words (V rows) vs word pairs (up to V^2
    #     rows), each row a distribution over V words.
    #  3. Context: 1 previous word vs 2 previous words, so it can capture
    #     things like 'sat on' -> 'the' rather than only 'on' -> 'the'.
    #  4. Data: much more is needed, as the table grows from V^2 to V^3
    #     entries and most contexts are seen rarely or never.

    # ---- PART XIII: comparison
    V = len(vocab)
    print("\n=== Part XIII: comparison ===")
    print(f"Vocabulary size V = {V}")
    print(f"{'':32}{'1st order':>12}{'2nd order':>12}")
    print(f"{'Non-zero parameters':32}{n_params(m1):>12}{n_params(m2):>12}")
    print(f"{'Observed contexts':32}{len(m1.probs):>12}{len(m2.probs):>12}")
    print(f"{'Possible contexts (V or V^2)':32}{V:>12}{V**2:>12}")
    print(f"{'Zero-prob contexts (unseen)':32}{V-len(m1.probs):>12}{V**2-len(m2.probs):>12}")
    print(f"{'Full table size (V^2 or V^3)':32}{V**2:>12}{V**3:>12}")
    u1 = len(set(m1.generate("sample") for _ in range(200)))
    u2 = len(set(m2.generate("sample") for _ in range(200)))
    print(f"{'Unique sentences in 200 samples':32}{u1:>12}{u2:>12}")
    print(f"{'Unique training sentences':32}{len(set(map(tuple, sentences))):>12}")
    # Typical finding: the 1st-order model invents new sentences, some odd
    # (e.g. 'the dog sat on the park'), giving higher diversity. The 2nd-order
    # model is more coherent but mostly reproduces training sentences (lower
    # diversity), because most 2-word contexts have only one observed
    # continuation.

    # Q12: More context improves prediction because the conditional
    #      distribution can depend on more of the history (sharper, more
    #      accurate predictions). But the CPT grows exponentially with context
    #      length (V^k rows), so with limited data most contexts are rare or
    #      unseen, the estimates are unreliable (overfitting / memorising)
    #      and many probabilities are zero.

    # PART XIV-XV: CONCEPTUAL ANSWERS

    # Q13. Approach B is preferable because:
    #  - Specifying behaviour: it states exactly what P is, how it is
    #    estimated (counts) and how text is generated (sampling), so the LLM
    #    has less room to guess.
    #  - Understanding the representation: I know the model is a CPT of
    #    counts, so I can find where counts/probabilities live in the code.
    #  - Validating the implementation: I can inspect the code against the spec
    #    (e.g. is it sampling or argmax? are unseen words handled?).
    #  - Probabilistic invariants: rows must sum to 1, probabilities in [0,1],
    #    no probability for unobserved transitions; these are testable.
    #  - Implementation vs model: the model is the maths P(Xt | Xt-1); the
    #    Python is just one implementation and may be buggy. Approach A gives
    #    something that "runs" but can't be checked because there is no spec.
    #
    # Q14. What the Bayesian network view added:
    #  - Factorisation of the joint: P(X1..XT) = prod P(Xt | parents(Xt)),
    #    which shows exactly which CPTs are needed.
    #  - Reasoning about independence assumptions: the graph shows what each
    #    word ignores (first order: everything before X_{t-1}).
    #  - Effect of increasing context: adding an edge (second order) adds a
    #    parent, which grows the CPT and the data requirement in a visible way.
    #  - A principled method for generation: ancestral sampling, i.e. sample
    #    parents first, then children given the parents.
    #  - Testing against the specification: each CPT row must be a valid
    #    distribution, which gave the normalisation test.
    
    # PART XIV link: modern LMs still model P(xt | x1..x_{t-1}); only the
    # representation changes (neural network instead of a CPT, learned by
    # gradient descent instead of counting).
