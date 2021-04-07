# Product Categorization by Title Using Deep Neural Networks as Feature Extractor

Leonardo S. Paulucio, Thiago M. Paixão, Rodrigo F. Berriel, Alberto F. De Souza, Claudine Badue and Thiago Oliveira-Santos

Paper published in  2020 International Joint Conference on Neural Networks ([IJCNN](https://www.ijcnn.org/))

DOI: [10.1109/IJCNN48605.2020.9207093](https://doi.org/10.1109/IJCNN48605.2020.9207093).

### Abstract
Natural Language Processing (NLP) has been receiving increasing attention in the past few years. In part, this is related to the huge flow of data being made available everyday on the internet, which increased the need for automatic tools capable of analyzing and extracting relevant information, especially from the text. In this context, text classification became one of the most studied tasks on the NLP domain. The objective is to assign predefined categories or labels to text or sentences. Important applications include sentence classification, sentiment analysis, spam detection, among many others. This work proposes an automatic system for product categorization using only their titles. The proposed system employs a state-of-the-art deep neural network as a tool to extract features from the titles to be used as input in different machine learning models. The system is evaluated in the large-scale Mercado Libre dataset, which has the common characteristics of real-world problems such as imbalanced classes, unreliable labels, besides having a large number of samples: 20,000,000 in total. The results showed that the proposed system was able to correctly categorize the products with a balanced accuracy of 86.57% on the local test split of the Mercado Libre dataset. It also surpassed the fourth place on the public rank of the MeLi Data Challenge with 91.19% of balanced accuracy, which represents less than 1% of the difference to the winner.

[![Overview](https://github.com/lspaulucio/product-categorization-ijcnn-2020/blob/master/images/thumbnail.jpeg)](https://ieeexplore.ieee.org/document/9207093)

### Dataset

- Mercado Libre Data Challenge: [Link](https://ml-challenge.mercadolibre.com/2019/downloads)

- Model weights: [Download](https://drive.google.com/drive/folders/1taz_iIwJQ-AunaF5EzFZ9d39HvriKe6Q?usp=sharing)

- Local splits: 
Splits created from the original Mercado Libre train set to perform the model fine-tuning: [Download](https://drive.google.com/drive/folders/1YY8I9o-R7BvU8WGRLaUKhoOsvI1gqe-o?usp=sharing)

- Processed splits: 
Splits after the preprocessing stage: [Download](https://drive.google.com/drive/folders/11ey3nY4UhA_A-aqp9Xo7maTr-3IJT0Mu?usp=sharing)

**Folds**: 
- Folds created from both train and validation local splits: [Download](https://drive.google.com/drive/folders/1Uq4BvADvm93CWdXuXjfEcRdSjpaZGrSP?usp=sharing) 

- Folds created from all data: [Download](https://drive.google.com/drive/folders/1249DBijRF06JGLi7FQQluCKIR9Uht2WI?usp=sharing)

### BibTex

If you find this useful, consider citing:
    
    @INPROCEEDINGS{paulucio2020ijcnn,
      author    = {L. S. {Paulucio} 
                  and T. M. {Paixão} 
                  and R. F. {Berriel} 
                  and A. F. {De Souza} 
                  and C. {Badue} 
                  and T. {Oliveira-Santos}},
      booktitle={2020 International Joint Conference on Neural Networks (IJCNN)}, 
      title={Product Categorization by Title Using Deep Neural Networks as Feature Extractor}, 
      year={2020},
      pages={1-7},
    }
