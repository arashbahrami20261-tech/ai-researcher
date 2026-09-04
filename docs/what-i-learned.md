# What I learned building an AI research agent

I spent about three days building an agent that searches AI literature, forms
hypotheses, writes code, runs it in a sandbox, and criticises its own
results. It works. All ten milestones on my roadmap are done, and there
are 134 tests.

I should say up front what this is not. I am a few weeks into learning
Python. I built this with Claude's help, and most of the code was
written that way. What I actually did was decide what to build, run it,
read what came back, and figure out why it was wrong.

That last part is where everything in this document came from. Every
lesson below cost me a real bug, and none of them came from the design.
They came from running the thing.

---

## Passing tests proved the wrong thing

I had 31 tests. All green, plus continuous integration. Then I ran the
system for real and asked it: how do transformers handle long context?

It came back with papers on superconductor thermodynamics and vascular
image segmentation.

The search function was sorting arXiv results by date instead of
relevance. arXiv receives hundreds of papers a day, so date-sorting
returns whatever was posted that morning and happened to match a word.

Here is the part that stung. Every one of my tests mocked arXiv's
*reply*. So I had thoroughly tested that I parse a response correctly,
and had never once tested that I send a sensible request. The tests
were not weak. They were pointed at the wrong half of the interaction.

The fix took a line. Understanding what my test suite had actually been
proving took longer.

## A default nobody chose is still a decision

The sorting was controlled by a parameter, `sort_by_newest`, which
defaulted to `True`. It looked reasonable: newer research is usually
better research.

But every caller inherited that without ever choosing it. The research
loop called `search_papers(question, max_results=5)` and got date
sorting it had never asked for.

An optional parameter with a bad default is worse than no parameter.
The caller does not know a choice is being made, so nobody reviews it.

The fix was not to remove the parameter — sometimes you really do want
the newest work. It was to flip the default and make anyone who wants
recency ask for it out loud.

## Some jobs should not go to a model

Later, the system ran three experiments in a chain and the timings did
not add up. Binary search appeared to get faster as the list got
bigger, which is not how binary search works.

My result critic had passed all three, because it reviewed each result
alone and never saw the others. So I built the missing memory: each
cycle records what it tested and what it followed, and walking those
links backwards gives the critic the earlier results.

Then I gave it the history and asked again.

It said the result was fine.

I asked the same question a second time. Same three numbers, nothing
changed. This time it said the result was not fine.

Same input, opposite answers. The model did not know. It was guessing,
and sometimes guessing right. If I had tested once and got the right
answer, I would have shipped something that is wrong half the time and
never found out.

So I stopped trying to prompt my way out of it. Comparing three numbers
is not a judgement call, it is arithmetic. I wrote it as a plain Python
function that checks whether a metric moves the direction it should as
the input grows. Deterministic, testable, free, and identical every
time.

The model critic still runs. It is good at reading code and spotting a
misplaced timer — it found a real one, unprompted, and named the exact
line. It is just no longer responsible for anything a comparison
operator can do.

## A prompt is a hope; code is a constraint

The new check went quiet almost immediately, and it took me a while to
notice why.

The model had renamed its metrics between cycles. `linear_seconds`
became `linear_search_time`. Each series then held a single data point,
and a series of one cannot contradict anything.

The check was not broken. It was starved, and it failed silently.

I added an instruction telling the model to keep the names. It worked
in cycle two and the model renamed them again in cycle three. So I
stopped asking. The names are now fixed by the first cycle that
measures anything, and enforced where the code runs: wrong names,
rejected, retry with an explanation.

This kept happening. Every time the model ignored an instruction, the
answer was the same shape — move the rule from the prompt into a check
the code performs. A prompt is a request. Only code is a constraint.

## The constant I picked to catch a bug was tuned to miss it

My trend check had a tolerance: how far a value could sit from the
trend before being flagged. I set it to 0.5. I did not look at any data
when I chose that number.

A live run then produced `linear_seconds` of 1.85, 13.91, 7.41 while the
list doubled from 100k to 200k elements. Linear search does not get
faster on a longer list. The drop was 46.8%.

The threshold was 50%.

The number I had chosen to catch exactly this class of bug was tuned to
miss it by three percentage points.

I lowered it to 0.25 and added the real series as a regression test.
Lowering it also turned an existing test red, which was correct — that
test's assertion had been an artefact of the loose threshold, so I
split it into two tests that each check one thing.

## Building the part that says "I don't know"

The evaluation engine compares every result against a baseline I state
up front. No baseline, no claim: a bare number is never reported as an
improvement.

Its first real test: binary search beat its baseline by 94%.

It returned INCONCLUSIVE. Only one run.

It was right. One run is not evidence — change the seed and the number
moves. It refuses in three situations: a single run, a change under 1%,
and a difference smaller than the spread across runs.

Building a system that says "I don't know" is harder than building one
that says "success". The easy version reports the number and moves on.
Most demos do exactly that.

## Half your test cases should be clean

The last milestone was a benchmark, so that "it got better" could be a
number rather than a feeling. Fixed tasks with known answers: coding
problems with a correct value, code samples with a planted bug, metric
series with a planted contradiction.

One design choice mattered more than the rest. Half the bug cases are
clean code with no bug in them.

Without that, a reviewer that flags absolutely everything scores 100%.
It catches every bug, because it catches everything. Only counting
false alarms makes the useless version score badly.

The first baseline: 83.3% on bug detection over three runs, zero
variance, zero false alarms. And the failure was specific, which is the
whole point. It caught both bugs where the number itself was absurd. It
missed the one where the number looked normal — accuracy measured
against training labels instead of the test set. 99% accuracy.
Plausible. Wrong. Visible only in the code, and the critic read the code
and said it was fine.

## The improvement that got thrown away

Then I let the system try to fix itself: see its score, see the case it
failed, propose new instructions for itself. Re-run the benchmark. The
number decides.

It proposed adding "ensure the correct metric is being measured" — a
restatement of something the prompt already said in different words.

83.3% before. 83.3% after. Rejected.

If I had only read that proposal, I would have accepted it. It sounds
thoughtful. It sounds like exactly the right fix. I would have shipped a
longer prompt, felt like I had improved something, and improved
nothing.

The finding underneath: some weaknesses are not prompt-shaped. Spotting
data leakage needs understanding what a test set is for. You cannot
reminder your way there.

## Deciding not to build something

The roadmap called for a multi-agent orchestrator — a Research Director
coordinating specialised agents.

I did not build it. The coordination already exists in plain
deterministic code that can be tested, and putting a model in charge of
deciding what runs next would make the system less predictable without
making it more capable.

That is in the README as a decision, with the reasoning, not as a TODO.
An unbuilt component with a written justification is finished work. An
unbuilt component with no explanation is a gap.

---

## What it still cannot do

The roadmap is finished. The system is not.

It has never run against the Anthropic API with a real key — everything
was verified against a local 7B model through the provider abstraction.
The sandbox only has the Python standard library, so it cannot run real
machine learning experiments. The trend check uses cycle number as a
proxy for input size, because input size is not recorded per experiment,
so it can tell that a value fell when it should have risen but not
whether it rose by the right amount. And I have not yet chosen the field
the system should specialise in, which is the largest open question.

## The pattern

Reading these back, most of them are the same mistake wearing different
clothes: I trusted something I had not measured.

The test suite I had not checked the coverage of. The default I had not
reviewed. The model answer I had not run twice. The instruction I had
not verified was followed. The constant I had picked out of the air.

Which is a strange thing to keep doing while building a system whose
entire purpose is refusing to accept unmeasured claims.

The code is at
https://github.com/arashbahrami20261-tech/ai-researcher
