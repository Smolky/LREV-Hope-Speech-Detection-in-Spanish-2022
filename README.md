# Hope Speech Detection in Spanish. *The LGBT case*
## Abstract
In recent years, systems have been developed to monitor online content and 
remove abusive, offensive or hateful content. Comments in online social media 
have been analyzed to find and stop the spread of negativity using methods 
such as hate speech detection, identification of offensive language or detection 
of abusive language. We define hope speech as the type of speech that is able to 
relax a hostile environment and that helps, gives suggestions and inspires for 
good to a number of people when they are in times of illness, stress, loneliness 
or depression. Detecting it automatically, in order to give greater diffusion to 
positive comments, can have a very significant effect when it comes to fighting 
against sexual or racial discrimination or when we intend to foster less bellicose 
environments. In this article we perform a complete study on hope speech in 
Spanish, analyzing existing solutions and available resources. In addition, we 
have generated a quality resource, a new Twitter dataset on LGBT community, and 
we have conducted some experiments that can serve as a baseline for further 
research.


# Dataset
According to Twitter's policy for public distribuition of user data, we have anonymised the dataset

> The best place to get Twitter Content is directly from Twitter. Consequently, we restrict the 
> redistribution of Twitter Content to third parties. If you provide Twitter Content to third parties, 
> including downloadable datasets or via an API, you may only distribute Tweet IDs, Direct Message IDs, 
> and/or User IDs (except as described below). We also grant special permissions to academic researchers 
> sharing Tweet IDs and User IDs for non-commercial research purposes.

The dataset is available in the files:

```
dataset/train.tsv
dataset/dev.tsv
dataset/test.tsv
```

As a result, we are unable to directly share the entire Tweet text. Instead, we realese the dataset with the 
Twitter IDs and the labels.

It is easy to find on the Internet scripts that shows how to extract tweets from the IDs:
https://medium.com/analytics-vidhya/fetch-tweets-using-their-ids-with-tweepy-twitter-api-and-python-ee7a22dcb845


## Installation
We have included the dependencies in the ```requirements.txt```. Please, create a 
virtual environment to replicate the results.

Moreover, we have included the results in the ```assets/hopespeech``` folder and the 
linguistic and negation features. However, the rest of features, the pretrained models,
nor the models trained are not included due to size limitations. 

The pretrained models can be downloaded from: https://fasttext.cc/docs/en/crawl-vectors.html
and placed within the ```embeddings/pretrained/``` folder.


## How to use
To train the models and validate using the validation split, using the feature sets

```
python train.py --dataset=hopespeech --corpus=2021 --model=deep-learning --features=lf
python train.py --dataset=hopespeech --corpus=2021 --model=deep-learning --features=se
python train.py --dataset=hopespeech --corpus=2021 --model=deep-learning --features=we
python train.py --dataset=hopespeech --corpus=2021 --model=deep-learning --features=bf
python train.py --dataset=hopespeech --corpus=2021 --model=deep-learning --features=ne
```

To evaluate a model with the test split
```
python evaluate.py --dataset=hopespeech --corpus=2021 --model=deep-learning --features=lf
python evaluate.py --dataset=hopespeech --corpus=2021 --model=deep-learning --features=se
python evaluate.py --dataset=hopespeech --corpus=2021 --model=deep-learning --features=we
python evaluate.py --dataset=hopespeech --corpus=2021 --model=deep-learning --features=bf
python evaluate.py --dataset=hopespeech --corpus=2021 --model=deep-learning --features=ne
```

To fine tune the transformer model with the Spanish BERT (BETO)
```
python train.py --dataset=hopespeech --corpus=2021 --model=transformers
```


## Citation
@todo