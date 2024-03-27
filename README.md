<div align="center">
  <h2>Classifier(s) pro-russian comments on Le Monde</h2>
  <p align="center">
    <a href="https://www.kaggle.com/amadevs/code"><img src="https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=Kaggle&logoColor=white" alt="Kaggle Badge"/></a>
    <a href="https://huggingface.co/gentilrenard"><img src="https://img.shields.io/badge/HuggingFace-black?style=flat&logo=huggingface&logoColor=white" alt="HuggingFace Badge"/></a>
  </p>
</div>
<br/>

Following my [previous work](https://github.com/matthieuvion/lmd_viz) on people's engagement about Ukraine War, decided to manually annotate around 300 comments (out of 180k) and train a clf to check pro-russian comments activity level.  

We first experimented w/ a few shots model (SetFit) but there goes the rabbit hole. We ended-up doing synthetic data generation in order to fine-tune a Mistral LLM for classification (json / label generation).  

Our end goal being deployment/performance, we then extend our initial dataset to around 20k samples (LLM as predictor) and train a multi-e5-base clf on our "synthetic data" while retaining a sufficient amount of accuracy with 10x less latency.

## Tldr; give me the notebooks

Everything's runnable on Kaggle T40/P100 and should work on Colab too.

| Notebook | Description | Ressource |
|----------|-------------|----------|
| xx | xx | Open in [Kaggle]()|

## Detailed guide & notes

### Baseline model (SetFit)

| Notebook | Description | Ressource |
|----------|-------------|----------|
| xx | xx | Open in [Kaggle]()|

### Synthetic data generation

| Notebook | Description | Ressource |
|----------|-------------|----------|
| xx | xx | Open in [Kaggle]()|

### Fine-tuning on a classification task

| Notebook | Description | Ressource |
|----------|-------------|----------|
| xx | xx | Open in [Kaggle]()|

### Reduce cost: nn classifier w/ e5-base embeddings, quantize

| Notebook | Description | Ressource |
|----------|-------------|----------|
| xx | xx | Open in [Kaggle]()|

## Ressources & links

Ressources I found to be particularly useful re. prompting, fine-tuning, embeddings, quantization etc.

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