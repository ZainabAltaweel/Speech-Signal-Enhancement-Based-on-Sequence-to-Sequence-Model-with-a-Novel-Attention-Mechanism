

This repository is the official implementation of Speech Signal Enhancement Based on Sequence to Sequence Model with a Novel Attention Mechanism (not published yet). 

> 📋This model is based on the recent advancement in the attention-based sequence to sequence model in NLP. [This blog](https://towardsdatascience.com/intuitive-understanding-of-attention-mechanism-in-deep-learning-6c9482aecf4f) explains the attention mechanism, and [this tutorial](https://www.tensorflow.org/tutorials/text/nmt_with_attention) implements it for language translation problem. Inspired by this model we created our work that is basically based on [this paper](https://ieeexplore.ieee.org/document/8683169).

> Model Architecture: 

>Figure 1 shows model architecture overview 
![Figure 1](/Figures/ModelSummary.jpg)

>Figure 2: Model architecture: the blue neurons represent a GRU cell which is used in both the encoder and the decoder, while the red neurons are a DNN. The encoder’s output and hidden states are fed to the attention layer, where the context vector is calculated. The context vector determines the relevance between input and output frequency features. Finally, the context vector from the attention layer is concatenated with the hidden states from the encoder. The concatenated vectors are fed to the decoder to predict the final output.
![Figure 2](/Figures/ModelArchitecture.png)
## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

> 📋A virtual python environment was setup to contain all the libraries that are needed for the model to be trained. We used Conda to install and manage packages for the environment. We trained our model on NVIDIA GeForce GTX 1060. 

>Dataset: We used the audio dataset from [the Voice Bank](https://www.semanticscholar.org/paper/The-voice-bank-corpus%3A-Design%2C-collection-and-data-Veaux-Yamagishi/2904e55a65d9441a8becd7111b751beb73c04a6c) corpus that includes a total of 11572 utterances. We divided the data as follows: 10000 utterances for training and validation, 80% and 20% respectively and 1000 utterance for testing. This dataset was corrupted with noise signals from [NOIZEUS IEEE corpus](https://www.researchgate.net/publication/224640875_Subjective_Comparison_of_Speech_Enhancement_Algorithms).
The dataset was preprocessed and saved as h5 files and it can bedownloaded [here](https://drive.google.com/drive/folders/1F8sZgAL8J4Mq_f7hY6fsfvG0mq7mkgaU?usp=sharing).

## Training
To train the model, run train.py

> 📋To train the model, we fed the data to the model in batches of 64. The encoder-decoder model was implemented as explained in figure 2. Different scenarios were tested. The first scenario is to test the baseline model, the second scenario is the encoder-decoder model with GRU cells, the third one is to build the same model with implementing attention mechanism. We conducted the training process for each of the scenarios for 100 epochs with Mean Square Error MSE as the loss function and ADAM optimizer. The number of neurons was changed for multiple runs of each scenario. 


## Evaluation

To evaluate the model, run evaluate.py

> 📋The model was tested by feeding a noisy input sequence to the trained model, the output of our model is a NumPy array representing the magnitude of the spectrogram. It is not possible to reconstruct the audio file from the magnitude component only, because it is missing another important component for reconstruction which is the phase component. For this paper We adopted Griffin and Lim's algorithm.

## Results

training the model with the three different scenarios and the two datasets gives the following results: 

Table 1 PESQ and STOI measures of the proposed model (att-enc-dec-GRU) and the baseline models vs the PESQ and STOI measures of the original noise speech for dataset-1.

![Table 2](/Figures/Result1.jpg)

Table 2 PESQ and STOI measures of the proposed model (att-enc-dec-GRU) and the baseline models vs the PESQ and STOI measures of the original noise speech for dataset-2.

![Table 2](/Figures/Result2.jpg)

In order to further facilitate the model’s performance, a speech spectrogram is shown in figure 3 before and after being processed.

Figure 3: A comparison between the original noisy signal, clean signal and the enhanced signal after
being processed by our training models with and without attention.
![Figure 3](/Figures/ComparisionGraphs.png)

> Sample audio signals can be found in [this folder](/AudioSamples)


