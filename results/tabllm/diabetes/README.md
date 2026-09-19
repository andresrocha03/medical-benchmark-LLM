# TabLLM response patterns for the diabetes dataset

This directory contains responses for 821 test instances: 720 instances with
the target `no hypoglycemic event tomorrow` and 101 with the target
`hypoglycemic event tomorrow`. The requested output labels were `negative` and
`positive`, respectively.

The table reports the observed contents of the `output` column. The evaluator
accepts a response when it contains exactly one of the two labels, even when
the response also contains punctuation or explanatory text. A response that
contains neither label is classified as non-relevant.

| Model | Serialization | Observed raw-answer pattern | Evaluator result |
|---|---|---|---|
| BioMistral | JSON | `negative` for all 821 instances | 821 negative |
| BioMistral | LLM | `negative` for all 821 instances | 821 negative |
| BioMistral | Text template | `negative` for all 821 instances | 821 negative |
| Llama 3 | JSON | `negative` for all 821 instances | 821 negative |
| Llama 3 | LLM | `negative` for all 821 instances | 821 negative |
| Llama 3 | Text template | `negative` for all 821 instances | 821 negative |
| Mistral | JSON | `negative.` for all 821 instances | 821 negative |
| Mistral | LLM | `negative.` for 820 instances; one explanatory response containing `positive` | 820 negative, 1 positive |
| Mistral | Text template | `negative.` for all 821 instances | 821 negative |
| Meditron | JSON | Three malformed JSON-like variants, all containing `positive`; one variant occurs 814 times | 821 positive |
| Meditron | LLM | Prompt or patient-record echoes without a label for 805 instances; longer responses containing `positive` for 16 | 805 non-relevant, 16 positive |
| Meditron | Text template | Prompt or patient-record echoes without a label for 808 instances; longer responses containing `positive` for 13 | 808 non-relevant, 13 positive |

## Summary

- BioMistral and Llama 3 returned the same `negative` response for every test
  instance under all three serialization methods.
- Mistral returned the same `negative.` response for every instance under JSON
  and text-template serialization. Under LLM serialization, it returned 820
  negative predictions and one positive prediction.
- Meditron did not produce direct label-only answers. With JSON serialization,
  every response nevertheless contained `positive` and was parsed as positive.
  With LLM and text-template serialization, most responses contained no valid
  label and were counted as non-relevant.
- No response in these files contained both labels, so there were no ambiguous
  responses.
