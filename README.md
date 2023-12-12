# Localized Symbolic Knowledge Distillation for Visual Commonsense Models [Neurips 2023]

Repo for **LSKD**: Distilling **localized** (e.g. bounding boxes), **visual commonsense** knowledge to Visual Language Models with ChatGPT generated data and filtering.

[[paper](https://arxiv.org/abs/2312.04837)] [[dataset](https://storage.googleapis.com/ai2-jamesp-public/msymkd/data/lskd/train_critic_input_all_vcr_vg_region_verbalizations~chatgpt_qar_filtered_threshold_0.8_1M.jsonl)]

![lskd_example](assets/example.jpg)

## The Localized Commonsense Knowledge (LCK) Dataset
Dataset with localized reasoning is provided [here](https://storage.googleapis.com/ai2-jamesp-public/msymkd/data/lskd/train_critic_input_all_vcr_vg_region_verbalizations~chatgpt_qar_filtered_threshold_0.8_1M.jsonl)  
```
>>> pprint(df.iloc[1])
image                                       VG_100K/2348412.jpg
source                             chatgpt_region_any_v4_people
split                                                     train
index                       chatgpt_region_any_v4_people-858297
region                                                      [4]
region_all                                            [0, 2, 4]
references    [{'name': '4', 'boxes': [[379.6391601562, 152....
question      [What, is, the, significance, of, the, gold, l...
answer        [The, gold, lion, on, the, woman's, shirt, in,...
rationale     [Lions, are, often, used, as, symbols, of, str...
prediction                                             0.940065
```

## Installation
```
pip install -e .
```

## Distillation Checkpoints & Results

![lskd_example](assets/results.jpg)

```
```


### Critic Model for Data Filtering
Run the following command to run the finetuned critic model in distriubted setting.
This saves the output json file in `run.output_dir`
```
torchrun --nproc_per_node=4 evaluate.py --cfg-path lavis/projects/blip2/eval/laion/laion_sample_critic_ft_filtering.yaml \
  --options run.output_dir=output/BLIP2/laion_samples/filtering/critic_ft
```

Zero shot model:
Run the following command to run the finetuned model.
```
torchrun --nproc_per_node=4 evaluate.py --cfg-path lavis/projects/blip2/eval/laion/laion_sample_critic_ft_filtering_zs.yaml \
  --options run.output_dir=output/BLIP2/laion_samples/filtering/zs
```

## References
```
@inproceedings{Park2023LocalizedSK,
  title={Localized Symbolic Knowledge Distillation for Visual Commonsense Models},
  author={Jae Sung Park and Jack Hessel and Khyathi Raghavi Chandu and Paul Pu Liang and Ximing Lu and Peter West and Youngjae Yu and Qiuyuan Huang and Jianfeng Gao and Ali Farhadi and Yejin Choi},
  year={2023},
  url={https://api.semanticscholar.org/CorpusID:266149843}
}
```
