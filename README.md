# LREV-Hope-Speech-Detection-in-Spanish-2022
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

## Installation
We have included the dependencies in the ```requirements.txt```. Please, create a 
virtual environment to replicate the results.

Moreover, we have included the results in the ```assets/hopespeech``` folder and the 
linguistic and negation features. However, the rest of features, the pretrained models,
nor the models trained are not included due to size limitations. 


## How to use
To train the models and validate using the validation split, using the feature sets

```
python train.py --dataset=hotespeech --corpus=2021 --model=deep-learning --features=lf
python train.py --dataset=hotespeech --corpus=2021 --model=deep-learning --features=se
python train.py --dataset=hotespeech --corpus=2021 --model=deep-learning --features=we
python train.py --dataset=hotespeech --corpus=2021 --model=deep-learning --features=bf
python train.py --dataset=hotespeech --corpus=2021 --model=deep-learning --features=ne
```

To evaluate a model with the test split
```
python evaluate.py --dataset=hotespeech --corpus=2021 --model=deep-learning --features=lf
python evaluate.py --dataset=hotespeech --corpus=2021 --model=deep-learning --features=se
python evaluate.py --dataset=hotespeech --corpus=2021 --model=deep-learning --features=we
python evaluate.py --dataset=hotespeech --corpus=2021 --model=deep-learning --features=bf
python evaluate.py --dataset=hotespeech --corpus=2021 --model=deep-learning --features=ne
```

To fine tune the transformer model with the Spanish BERT (BETO)
```
python train.py --dataset=hotespeech --corpus=2021 --model=transformers
```


## Citation
@todo