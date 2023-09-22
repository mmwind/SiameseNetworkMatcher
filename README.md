# Seamese ANN for text sentences matching

Project uses pre-trained BERT model and triplet loss. 

# Future research plan

Intro
 - Find proper dataset
	 - [Amazon-GoogleProducts](https://dbs.uni-leipzig.de/file/Amazon-GoogleProducts.zip)
	 - [Abt-Buy](https://dbs.uni-leipzig.de/file/Abt-Buy.zip)
		See also
		https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_datasets_for_entity_resolution
		https://paperswithcode.com/datasets?task=entity-resolution
		https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md

Run siamese pretrained BERT-based experiment from this repo (in Collab)
- Model class
- Triplet loss class
- Trainer

Train tokenizer on Aptekar dataset
 - Choose tokenizer model (BPE or anything else)
 - Add special tokens ( It should be units, numbers, Transformers tokens like \<UNK\>)
 - Train tokenizer and choose proper output size
	
Build our own siamese Transformer model in pytorch for Triplet loss
 - Model class
 - Triplet loss class
 - Trainer
 - Check on datasets
