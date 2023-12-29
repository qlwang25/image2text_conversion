## Image-to-Text Conversion and Aspect-Oriented Filtration for Multimodal Aspect-Based Sentiment Analysis


---
### Abstract
Multimodal aspect-based sentiment analysis (MABSA) aims to determine the sentiment polarity of each aspect mentioned in the text based on multimodal content. Various approaches have been proposed to model multimodal sentiment features for each aspect via modal interactions. However, most existing approaches have two shortcomings: 
* The representation gap between textual and visual modalities may increase the risk of misalignment in modal interactions;
* In some examples where the image is not related to the text, the visual information may not enrich the textual modality when learning aspect-based sentiment features. In such cases, blindly leveraging information from visual modal may introduce noises in reasoning the aspect-based sentiment expressions.

To tackle the shortcomings mentioned above, we propose **an end-to-end MABSA framework with image conversion and noise filtration**. Specifically, 
1. to bridge the representation gap in different modalities, we attempt to translate images into the input space of a pre-trained language model (PLM). To this end, we develop an image-to-text conversion module that can convert an image to an implicit sequence of token embedding.
2. Moreover, an aspect-oriented filtration module is devised to alleviate the noise in the implicit token embeddings, which consists of two attention operations. The former aims to create an enhanced aspect embedding as a query, and the latter seeks to use this query to retrieve relevant auxiliary information from the implicit token embeddings to supplement the textual content.
3. After filtering the noise, we leverage a PLM to encode the text, aspect, and image prompt derived from filtered implicit token embeddings as sentiment features to perform aspect-based sentiment prediction.

Experimental results on two MABSA datasets show that our framework achieves state-of-the-art performance. Furthermore, extensive experimental analysis demonstrates the proposed framework has superior robustness and efficiency.


###  Overview
<p align="center">
  <img src="./images/model.png" alt=" Overview of the proposed model.">
</p>

Figure shows the overview of our designed framework, which consists of three modules: 
1) image-to-text conversion module, which contains an image feature extractor and a transformer-based encoder-decoder architecture. It treats an image as input and outputs an implicit sequence of token embedding hatched from the visual modality.
2) aspectoriented filtration module, which performs aspect2context attention and aspect2im-token attention. Here, aspect2context attention intends to enhance the native aspect representation using contextual information of textual data. The aspect2imtoken attention aims to filter noise in the implicit token embeddings and extract auxiliary information for predicting the sentiment of the aspect.
3) prediction module, which adopts a PLM to model the interaction between text, aspect, and image prompt derived from pooling filtered implicit token embeddings. Then, the hidden state corresponding to the [CLS] token is considered as the sentiment representation of the aspect, followed by a softmax layer for sentiment prediction.


### Main Results
<p align="center">
  <img src="./images/main_results.png" alt="results" width="70%">
</p>


### Case Study
To better intuitively understand the superiority of the proposed framework, we show the differences between predictions on four test examples in Figure 6. Here, the compared models are the framework with auxiliary cross-modal relation detection (denoted by JML [85]), the framework with vision-language post-training (denoted by VLP [57]), and the proposed framework.
<p align="center">
  <img src="./images/case_study.png" alt="results">
</p>

---
### Follow the steps below to run the code:
1. download Catr model `checkpoint.pth` from [BaiduYunPan](https://pan.baidu.com/s/1ZYDTkFoeXaQBExU-DjgqzA) [
code：2023], and put it in `./catr` directory
2. download [Bertweet-base](https://arxiv.org/abs/2005.10200), and put it in `./bertweet-base` directory
3. download [dataset](https://www.ijcai.org/proceedings/2019/751)
4. install packages (see `requirements.txt`)
5. run `bash scripts/*.sh`

---
### Cite
```
@ARTICLE{10319094,
  author={Wang, Qianlong and Xu, Hongling and Wen, Zhiyuan and Liang, Bin and Yang, Min and Qin, Bing and Xu, Ruifeng},
  journal={IEEE Transactions on Affective Computing}, 
  title={Image-to-Text Conversion and Aspect-Oriented Filtration for Multimodal Aspect-Based Sentiment Analysis}, 
  year={2023},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TAFFC.2023.3333200}}
}

```

