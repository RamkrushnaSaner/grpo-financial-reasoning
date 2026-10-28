"""A 15-question behavioural probe for the financial task, grouped by what we
EXPECT to happen rather than by difficulty:

    A in-distribution      -> IMPROVE    (the hypothesis)
    B near-distribution    -> UNKNOWN    (transfer: the only route to a reasoning claim)
    C out-of-distribution  -> NO CHANGE  (negative control; not scored)
    D adversarial          -> varies     (regression probes, e.g. over-applied format)

Questions with ``answer=None`` are shown but not scored. A probe may set its
own ``max_new_tokens``.
"""
PROBES = [
    dict(id="A1", capability="sum of three values", expect="IMPROVE",
         context="Quarterly revenues: Q1=$2.7B, Q2=$2.4B, Q3=$3.6B.",
         question="What is the total revenue in $B for the first three quarters?",
         answer=8.7, answer_type="dollar"),
    dict(id="A2", capability="1-step ratio", expect="IMPROVE",
         context="The company generated revenue of $11.8B and operating profit of $4.6B.",
         question="What was the operating margin in percent?",
         answer=39.0, answer_type="percentage"),
    dict(id="A3", capability="2-step growth", expect="IMPROVE",
         context="In Q3 2024, revenue was $5.9B, compared to $4.5B in Q3 2023.",
         question="What was the year-over-year revenue growth rate in percent?",
         answer=31.1, answer_type="percentage"),
    dict(id="A4", capability="2-step projection", expect="IMPROVE",
         context="Current annual revenue is $4.8B with a 6% annual growth rate.",
         question="What is the projected revenue in $B for next year?",
         answer=5.09, answer_type="dollar"),
    dict(id="B1", capability="3-step (unseen depth)", expect="UNKNOWN",
         context="Revenue was $8.0B. Costs were 65% of revenue. Tax is 20% of profit.",
         question="What is the after-tax profit in $B?",
         answer=2.24, answer_type="dollar"),
    dict(id="B2", capability="reverse direction", expect="UNKNOWN",
         context="Operating margin was 25% and operating profit was $3.0B.",
         question="What was the revenue in $B?",
         answer=12.0, answer_type="dollar"),
    dict(id="B3", capability="non-financial arithmetic", expect="UNKNOWN",
         context="A train travels 240 km in 3 hours.",
         question="What is its average speed in km/h?",
         answer=80.0, answer_type="integer"),
    dict(id="C1", capability="logical deduction", expect="NO CHANGE",
         context="All analysts at the firm have a CFA. Priya works at the firm.",
         question="Does Priya necessarily have a CFA? Answer yes or no.",
         answer=None, answer_type=None),
    dict(id="C2", capability="open explanation", expect="NO CHANGE",
         context="",
         question="In two sentences, explain what operating margin measures.",
         answer=None, answer_type=None),
    dict(id="D1", capability="unanswerable", expect="BOTH FAIL",
         context="Quarterly revenues: Q1=$2.7B, Q2=$2.4B.",
         question="What is the total revenue in $B for the first three quarters?",
         answer=None, answer_type=None),
    dict(id="D2", capability="concise answer wanted", expect="MAY REGRESS",
         context="Revenue was $10.0B.",
         question="State the revenue in $B. Reply with the number only, nothing else.",
         answer=10.0, answer_type="dollar"),
    dict(id="D3", capability="trivial lookup", expect="MAY REGRESS",
         context="Operating profit was $3.3B.",
         question="What was the operating profit in $B?",
         answer=3.3, answer_type="dollar"),
    dict(id="D4", capability="distractor numbers", expect="UNKNOWN",
         context=("Founded in 1997. Employs 45,000 people across 12 countries. "
                  "Revenue was $6.2B and operating profit was $1.55B."),
         question="What was the operating margin in percent?",
         answer=25.0, answer_type="percentage"),
    dict(id="D5", capability="negative value", expect="IMPROVE",
         context="In Q3 2024, revenue was $3.2B, compared to $4.0B in Q3 2023.",
         question="What was the year-over-year revenue growth rate in percent?",
         answer=-20.0, answer_type="percentage"),
    dict(id="D6", capability="tight token budget", expect="UNKNOWN",
         context="Quarterly revenues: Q1=$1.1B, Q2=$2.2B, Q3=$3.3B.",
         question="What is the total revenue in $B for the first three quarters?",
         answer=6.6, answer_type="dollar", max_new_tokens=48),
]
