# Submission checklist (Canvas: report PDF + code zip)

Deadline: 3:30pm Friday 27 November 2026; late submissions lose 30 percent.

## Code zip

- [ ] `python -m pytest` passes
- [ ] `scripts/make_submission.sh path/to/signed_sheet.pdf` builds `psoann_submission.zip`
- [ ] Unzip into an empty folder and check `pip install -e ".[dev]"`, `python -m pytest` and `psoann run --runs 3` work
- [ ] The README tells a stranger how to run it (it does: Quick start)
- [ ] `docs/AI_USAGE.md` is complete and true, including what you changed or fixed in AI-written code
- [ ] Signed **Coursework Group Signing Sheet** is in the zip (or embedded in the report). **No marks are issued without it.**
- [ ] No code you did not write is unattributed

## Report PDF (max 6 pages and 3000 words excluding references and appendices; Arial or Calibri, 12 pt or larger)

- [ ] Sections: Implementation, Results and Discussion, Conclusions, References (appendices optional)
- [ ] Implementation: structure of the code, exactly how PSO is coupled to the ANN (vector layout, bounds, fitness = training accuracy), Algorithm 39 mapping, extensions
- [ ] Results: which questions, which values and why (with references), mean and standard deviation over 30 runs, tables and plots, **why** each result happened
- [ ] Compare with previously published banknote results
- [ ] Conclusions: summary and what you would do next
- [ ] Every reference exists, is cited in the text, and is in a standard format (Harvard); none unused
- [ ] The text and the analysis are your own work (see `docs/AI_USAGE.md`)

## Before you submit

- [ ] Each of you can explain both halves (ANN and PSO) in a viva
- [ ] The Algorithm 39 line numbers in the code comments agree with the book page
