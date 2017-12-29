# LSTM-Sentence-Similarity

Theano Implementation of "Siamese Recurrent Architectures for Learning Sentence Similarity".

Mueller, J and Thyagarajan, A.  Siamese Recurrent Architectures for Learning Sentence Similarity.  Proceedings of the 30th AAAI Conference on Artificial Intelligence (AAAI 2016).
 http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12195

## Getting Started:
Download the word2vec model from https://code.google.com/archive/p/word2vec/  and download the file: GoogleNews-vectors-negative300.bin.gz and insert into data/word_embeddings

Use train_lstm.py in order to train a new lstm, use train_entailment.py to train an entailment classifier based on the already trained lstm mode. Other training methods exist in alternative_trains/



