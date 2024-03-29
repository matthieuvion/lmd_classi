<div align="center">
  <h2>Classification - from LLM fine-tuning to quantized e5 embeddings </h2>
  <p align="center">
    <a href="https://www.kaggle.com/amadevs/code"><img src="https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=Kaggle&logoColor=white" alt="Kaggle Badge"/></a>
    <a href="https://huggingface.co/gentilrenard"><img src="https://img.shields.io/badge/HuggingFace-black?style=flat&logo=huggingface&logoColor=white" alt="HuggingFace Badge"/></a>
  </p>
</div>
<br/>

<div align="center">
<em>End to end benchmark on comments classification with SetFit vs. fine-tuned Mistral-7b vs. multi-e5-base + clf layer.  

Fully reproducible guide w/ online notebooks, model(s), dataset.</em>
</div><br>

> [!NOTE]  
> Following my previous [work](https://github.com/matthieuvion/lmd_viz) on people's engagement with the Ukraine War, I > decided to manually annotate approximately 300 comments (out of 180k) and train a classifier to assess the weight of pro-Russian comments. We initially experimented with a few-shot learning model (SetFit), but then we found ourselves going down the rabbit hole.   

<br>

## Tldr; online notebooks

Should work on Google Colab too, maybe with a few adaptations for fine-tuning with `Unsloth`.

| Notebook | Description | Resource |
|--------------------------------------|-----------------------------------------------------------------------------|----------|
| lmd_setfit_modeling_logistic_head | Baseline model - few shots clf using SetFit | [notebook](https://www.kaggle.com/code/amadevs/lmd-setfit-modeling-logistic-head) |
| lmd_mistral_synthetic_gen_testprompt | Synthetic data gen - prepare dataset - prompts tests Mistral-7B-OpenHermes | [notebook](https://www.kaggle.com/code/amadevs/lmd-mistral-synthetic-gen-testprompt) |
| lmd_mistral_synthetic_gen_run        | Synthetic data gen - run (output : 2k synthetic samples) | [notebook](https://www.kaggle.com/code/amadevs/lmd-mistral-synthetic-gen-run) |
| lmd_mistral_synthetic_fine_tune      | Fine-tuning Mistral-7B-base for classi. (output: json label) w/ synth. data, using Unsloth (Qlora), Alpaca template | [notebook](https://www.kaggle.com/code/amadevs/lmd-mistral-synthetic-fine-tune) |
| lmd_setfit_mistral_evaluation        | Benchmark SetFit / fine-tuned Mistral (several experiments) | [notebook](https://www.kaggle.com/code/amadevs/lmd-setfit-mistral-evaluation) |
| lmd_setfit_mistral_inference         | Augment original dataset - voting ensemble SetFit + f-tuned LLM to infer 20k unlabeled comments| [notebook](https://www.kaggle.com/code/amadevs/lmd-setfit-mistral-inference) |
| lmd_multi-e5_train                   | multi-e5/bge embeddings + nn classifier on augmented dataset (several experiments) | [notebook](https://www.kaggle.com/code/amadevs/lmd-multi-e5-train) |
| e5_onnx_optimization                 | multi-e5 - ONNX conversion & optimization/quantization | [notebook](https://www.kaggle.com/code/amadevs/e5-onnx-optimization) |
| lmd_e5_evaluation                    | Benchmark all models - focus on global, minority class and inference latency | [notebook](https://www.kaggle.com/code/amadevs/lmd-e5-evaluation)


## Detailed guide

### Baseline : few shots learning with Huggingface/SetFit

- Three labels 0. support to Ukraine, 1. (rather) support Russia 2. off topic / no opinion.
- Initial seed (annotated data) was labeled with `label studio` : around 400 samples with 'oversampling' on the minority class to capture enough information :
- Overall (obvious/direct) support for Russia is rare (+- 10%), and Le Monde subscribers love to digress (2 is vast majority).
- We used our Faiss index / vector search previously built to retrieve enough "pro russian" comments among the 180k we scrapped, along with random exploration.
- We tried many optimizations on the few shot model `SetFit` (not shared here): # labels, grid search, different heads.
- Compared to sample size, a very good performance (76% accuracy) & deployablility (5ms latency) but not satisfied with performance on our class of interest (pro-russian comments).

| Notebook | Description | Resource |
|--------------------------------------|-----------------------------------------------------------------------------|----------|
| lmd_setfit_modeling_logistic_head | Baseline model - few shots clf using SetFit | [notebook](https://www.kaggle.com/code/amadevs/lmd-setfit-modeling-logistic-head)


### Synthetic data generation (Mistral-OpenHermes)

- Goal : augment our initial, manually annotated seed, with synthetic data so we can fine-tune a LLM to perform classification.
- Room for improvement : efforts on prompting (context + examples + lot of tests), but still had to throw 40% of comments eventually. But manual (random) review showed good enough results : credible comments with right ssociated label.

| Notebook | Description | Resource |
|--------------------------------------|-----------------------------------------------------------------------------|----------|
| lmd_mistral_synthetic_gen_testprompt | Synthetic data gen - prepare dataset - prompts tests Mistral-7B-OpenHermes | [notebook](https://www.kaggle.com/code/amadevs/lmd-mistral-synthetic-gen-testprompt) |
| lmd_mistral_synthetic_gen_run        | Synthetic data gen - run (output : 2k synthetic samples) | [notebook](https://www.kaggle.com/code/amadevs/lmd-mistral-synthetic-gen-run)


### Mistral-7B-base fine-tuning (using `Unsloth`): outputs json with predicted class label

- We are *not* using LLM embeddings with a classification layer. Instead we fine-tune our model with annotated + synthetic data so it predicts a label {label:0} or {label:1} our {label:2} given a prompt (isntruction + comment).
- Our Alpaca-like template showed good performance with Mistral-7b base v0.1 (no improvement with recently released v0.2).
- LLM as a predictor shows very good accuracy (80%) and most importantly performs well on our minority class.
- We could use it to extend our dataset ? We have nearly 180k unlabeled comments that could be used to train a standard classifier!

| Notebook | Description | Resource |
|--------------------------------------|-----------------------------------------------------------------------------|----------|
| lmd_mistral_synthetic_fine_tune      | Fine-tuning Mistral-7B-base for classi. (output: json label) w/ synth. data, using Unsloth (Qlora), Alpaca template | [notebook](https://www.kaggle.com/code/amadevs/lmd-mistral-synthetic-fine-tune) |
| lmd_setfit_mistral_evaluation        | Benchmark SetFit / fine-tuned Mistral (several experiments) | [notebook](https://www.kaggle.com/code/amadevs/lmd-setfit-mistral-evaluation)

### Multi-e5-base embeddings based clf. ONNX & latency optim.
- End goal : retain enough accuracy while lowering inference time (900ms for fine-tuned LLM).
- Train a classifier on top of multi-e5-embeddins (tested m3-bge as well). Several dataset composition tried (synthetic data and/or llm-predicted labels and/or initial seed).
- Our accuracy is good enough, especially on minority class, considering we trained on 20k synthetic/llm-predicted data.
- Final model is converted to ONNX, quantized & optimized -> 80ms avg latency.


| Notebook | Description | Resource |
|--------------------------------------|-----------------------------------------------------------------------------|----------|
| lmd_setfit_mistral_inference         | Augment original dataset - voting ensemble SetFit + f-tuned LLM to infer 20k unlabeled comments| [notebook](https://www.kaggle.com/code/amadevs/lmd-setfit-mistral-inference) |
| lmd_multi-e5_train                   | multi-e5/bge embeddings + nn classifier on augmented dataset (several experiments) | [notebook](https://www.kaggle.com/code/amadevs/lmd-multi-e5-train) |
| e5_onnx_optimization                 | multi-e5 - ONNX conversion & optimization/quantization | [notebook](https://www.kaggle.com/code/amadevs/e5-onnx-optimization) |
| lmd_e5_evaluation                    | Benchmark all models - focus on global, minority class and inference latency | [notebook](https://www.kaggle.com/code/amadevs/lmd-e5-evaluation)


## Ressources & links

Ressources I found to be particularly useful re. prompting, template, fine-tuning, embeddings performance, quantization etc.

- [MLabonne Repo](https://github.com/mlabonne/llm-course)  
- [Dataset Gen - Kaggle example](https://www.kaggle.com/code/phanisrikanth/generate-synthetic-essays-with-mistral-7b-instruct)  
- [Dataset Gen - blog w/ prompt examples](https://hendrik.works/blog/leveraging-underrepresented-data)  
- [Prepare dataset- /r/LocalLLaMA best practice classi](https://www.reddit.com/r/LocalLLaMA/comments/173o5dv/comment/k448ye1/?utm_source=reddit&utm_medium=web2x&context=3)  
- [Prepare dataset - using gpt3.5](https://medium.com/@kshitiz.sahay26/how-i-created-an-instruction-dataset-using-gpt-3-5-to-fine-tune-llama-2-for-news-classification-ed02fe41c81f) 
- [Prepare dataset - Predibase prompts for diverse fine-tuning tasks](https://predibase.com/lora-land)
- [Fine tune OpenHermes-2.5-Mistral-7B - including prompt template gen](https://towardsdatascience.com/fine-tune-a-mistral-7b-model-with-direct-preference-optimization-708042745aac)  
- [Fine tune - Unsloth colab example](https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharing)
- [Fine tune - w/o unsloth](https://gathnex.medium.com/mistral-7b-fine-tuning-a-step-by-step-guide-52122cdbeca8) or [wandb](https://wandb.ai/vincenttu/finetuning_mistral7b/reports/Fine-tuning-Mistral-7B-with-W-B--Vmlldzo1NTc3MjMy) or [philschmid](https://www.philschmid.de/fine-tune-llms-in-2024-with-trl#6-deploy-the-llm-for-production)
- [Fine tune - impact of parameters S. Raschka](https://lightning.ai/pages/community/lora-insights/)
- [Embeddings - multilingual, latest comparison w/ e5-multi ](https://towardsdatascience.com/openai-vs-open-source-multilingual-embedding-models-e5ccb7c90f05)
- [Philschmid ONNX optim](https://github.com/philschmid/optimum-transformers-optimizations/blob/master/notebook.ipynb)