# medical-benchmark-LLM

The main objective is to build a benchmark for comparing the performance of LLMs, tabular foundation models and classical tabular methods in diverse medical datasets.

### Project structure

- **Data**: Processing data in order to make it adequate for the models.

- **Models**: Implementing, training and evaluating models.

- **Results**: Saving metrics and producing aggregated plots to compare models results.


### Results

We observed that the models struggled to deal with the diabetes dataset that presents unbalance. In this case, LightGBM was the best model, but with a very low recall (0.248), indicating that it mostly predicted negative labels.

For the hepatitis dataset, the foundation model TARTE was the best by a considerable margin, though its execution time ("training"+predicting) was one of the biggest ones.

Finally, for the heart dataset, we observe a great performance from another foundation model, TABICLv2. Which had also the third least execution time.

### Conclusions

Overall, the project brought up important results for me. 
- Unbalaced data in the diabetes dataset was definitevely a problem, regardless of the model. 
- Foundation models show evidences of better results than tree methods in balanced medical datasets. 
- TabICLv2 proposal (being a foundation model with adjustments that guarantee higher-performance) was confirmed. Its execution times were sufficiently close to the tree methods while its results was significantly higher.
- Tree methods are still a more fast and simple way to train and predict models, but their results are being surpassed.
- The zero-shot approach using LLMs for tabular data was disappointing. I believe that using medical datasets may trigger the models' internal filters, given that they are not intended for making diagnoses. Consequently, there were many invalid responses, with the models refusing to answer or predict an outcome. When predictions were made, the results were poor.
- For next steps, it would be interesting to investigate strategies of improving the results of tabular foundation models on unbalanced medical datasets.

### Remarks
The original project was done for a course in Telecom Paris, but due to personal interest, I decided to improve and correct some aspects of it.

Corrections from original project:

- Defining the same evaluation protocol for all models (F1-macro, precision, recall, accuracy, AUC, training time and prediction time).

- TabPFN was being tuned in the test set, leading do test leakage.

- Removed the threshold tuning from ICLv2, that was also leading to test leakage.

- Replaced the custom TARTE reconstruction with small, cluster-runnable modules
  around the authors' `tarte-ai` frozen featurizer.

- Introduced utils.py inside models/ to reuse functions accross model evaluations.

- Reorganized the TabLLM to integrate with the project structure.

- Included the demand for receiving standard binary answers ("positive" or "negative") when evaluating the zero-shot with LLMs.

- Documented functions.

- All models were executed with the same computational resource (NVDIA RTX 3090 24GB).

### References

- [Post explaining TFMs](https://towardsdatascience.com/tabular-foundation-models/)

- [TabPFN](https://arxiv.org/pdf/2511.08667)

- [Tarte](https://arxiv.org/abs/2505.14415)

- [TabICLv2](https://arxiv.org/abs/2602.11139)

- [TabLLM](https://arxiv.org/pdf/2210.10723)

- [TabLLM](https://www.rubrik.com/blog/ai/23/getting-the-best-zero-
shot-performance-on-your-tabular-data-with-llms)

- [Heart Dataset](http://archive.ics.uci.edu/dataset/45/heart+disease)

- [Hepatitis Dataset](https://archive.ics.uci.edu/dataset/46/hepatitis)

- [Diabetes Dataset](https://archive.ics.uci.edu/dataset/34/diabetes)

- [LLaMA3 Instruct](https://huggingface.co/meta-llama)

- [Mistral 7B Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-
v0.2)

- [Meditron](https://huggingface.co/epfl-llm/meditron-7b)

- [BioMistral](https://huggingface.co/BioMistral/BioMistral-7B)
