<div align="center">
  <h2>Comments classifier - from LLM fine-tuning to e5-base embeddings </h2>
  <p align="center">
    <a href="https://www.kaggle.com/amadevs/code"><img src="https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=Kaggle&logoColor=white" alt="Kaggle Badge"/></a>
    <a href="https://huggingface.co/gentilrenard"><img src="https://img.shields.io/badge/HuggingFace-black?style=flat&logo=huggingface&logoColor=white" alt="HuggingFace Badge"/></a>
  </p>
</div>
<br/>

Following my [previous work](https://github.com/matthieuvion/lmd_viz) on people's engagement about Ukraine War, decided to manually annotate around 300 comments (out of 180k) and train a clf to assess pro-russian comments opinion weight.
We first experimented w/ a few shots model (SetFit) but there goes the rabbit hole.  
Decided we would share things we learned in the form of runnable (16gb VRAM) notebooks. Models, dataset, notebooks are fully re-usable.

## Tldr; give me the notebooks

Everything's runnable on Kaggle T40/P100 (+-16gb VRAM) and should work on Colab too.

| Notebook | Description | Resource |
|--------------------------------------|-----------------------------------------------------------------------------|----------|
| lmd_setfit_modeling_logistic_head | Baseline model - few shots clf using SetFit | Open in [Kaggle](https://www.kaggle.com/code/amadevs/lmd-setfit-modeling-logistic-head) |
| lmd_mistral_synthetic_gen_testprompt | Synthetic data - prepare dataset - prompts tests Mistral-7B OpenHermes / Alpaca | Open in [Kaggle](https://www.kaggle.com/code/amadevs/lmd-mistral-synthetic-gen-testprompt) |
| lmd_mistral_synthetic_gen_run        | Synthetic data - run (2k samples) | Open in [Kaggle](https://www.kaggle.com/code/amadevs/lmd-mistral-synthetic-gen-run) |
| lmd_mistral_synthetic_fine_tune      | Fine-tuning Mistral-7B-base for classification on synthetic data using Unsloth (Qlora)| Open in [Kaggle](https://www.kaggle.com/code/amadevs/lmd-mistral-synthetic-fine-tune) |
| lmd_setfit_mistral_evaluation        | Benchmark SetFit / fine-tuned Mistral, several experiments | Open in [Kaggle](https://www.kaggle.com/code/amadevs/lmd-setfit-mistral-evaluation) |
| lmd_setfit_mistral_inference         | Augment dataset - voting ensemble SetFit + ft Mistral to predict 20k unlabeled comments| Open in [Kaggle](https://www.kaggle.com/code/amadevs/lmd-setfit-mistral-inference) |
| lmd_multi-e5_train                   | multi-e5/bge embeddings + nn classifier on augmented dataset | Open in [Kaggle](https://www.kaggle.com/code/amadevs/lmd-multi-e5-train) |
| e5_onnx_optimization                 | multi-e5 - ONNX conversion & optimization/quantization | Open in [Kaggle](https://www.kaggle.com/code/amadevs/e5-onnx-optimization) |
| lmd_e5_evaluation                    | Benchmark all models - focus on global, minority class and inference latency | Open in [Kaggle](https://www.kaggle.com/code/amadevs/lmd-e5-evaluation) |

## Detailed notes

### Baseline model (SetFit)

| Notebook | Description | Resource |
|--------------------------------------|-----------------------------------------------------------------------------|----------|
| lmd_setfit_modeling_logistic_head | Baseline model - few shots clf using SetFit | Open in [Kaggle](https://www.kaggle.com/code/amadevs/lmd-setfit-modeling-logistic-head) |


### Synthetic data generation (Mistral / Alpaca)

| Notebook | Description | Resource |
|--------------------------------------|-----------------------------------------------------------------------------|----------|
| lmd_mistral_synthetic_gen_testprompt | Synthetic data - prepare dataset - prompts tests Mistral-7B OpenHermes / Alpaca | Open in [Kaggle](https://www.kaggle.com/code/amadevs/lmd-mistral-synthetic-gen-testprompt) |
| lmd_mistral_synthetic_gen_run        | Synthetic data - run (2k samples) | Open in [Kaggle](https://www.kaggle.com/code/amadevs/lmd-mistral-synthetic-gen-run) |

### LLM fine-tuning on a classification task

| Notebook | Description | Resource |
|--------------------------------------|-----------------------------------------------------------------------------|----------|
| lmd_mistral_synthetic_fine_tune      | Fine-tuning Mistral-7B-base for classification on synthetic data using Unsloth (Qlora)| Open in [Kaggle](https://www.kaggle.com/code/amadevs/lmd-mistral-synthetic-fine-tune) |
| lmd_setfit_mistral_evaluation        | Benchmark SetFit / fine-tuned Mistral, several experiments | Open in [Kaggle](https://www.kaggle.com/code/amadevs/lmd-setfit-mistral-evaluation) |

### Can we train a "standard" classifier on the augmented dataset ?

| Notebook | Description | Resource |
|--------------------------------------|-----------------------------------------------------------------------------|----------|
| lmd_setfit_mistral_inference         | Augment dataset - voting ensemble SetFit + ft Mistral to predict 20k unlabeled comments| Open in [Kaggle](https://www.kaggle.com/code/amadevs/lmd-setfit-mistral-inference) |
| lmd_multi-e5_train                   | multi-e5/bge embeddings + nn classifier on augmented dataset | Open in [Kaggle](https://www.kaggle.com/code/amadevs/lmd-multi-e5-train) |
| e5_onnx_optimization                 | multi-e5 - ONNX conversion & optimization/quantization | Open in [Kaggle](https://www.kaggle.com/code/amadevs/e5-onnx-optimization) |
| lmd_e5_evaluation                    | Benchmark all models - focus on global, minority class and inference latency | Open in [Kaggle](https://www.kaggle.com/code/amadevs/lmd-e5-evaluation) |

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