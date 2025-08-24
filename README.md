## Wavelet Transform
The Wavelet Transform is a powerful mathematical tool for analyzing signals, images, and time series data across multiple scales. Unlike the Fourier Transform, which only provides frequency information, the Wavelet Transform provides both time (localization) and frequency (scale) information.

The Jupyter Notebook Wavelet_Transform.ipynb provides an end-to-end implementation and visualization of wavelet transforms, showcasing their power in signal processing and data analysis.

## Multi Scale Input 
This repository provides an implementation of **Multi-Scale Input for Transformer-based models**, a technique used to enhance sequence modeling by feeding inputs at different temporal or spatial resolutions.  

The notebook `Multi_Scale_Input_to_the_Transformer.ipynb` demonstrates how multi-scale features can be integrated into the Transformer architecture, improving its ability to capture both **global (long-range)** and **local (short-term)** dependencies.  

---

## 🔎 Overview  

Transformers are powerful for sequential data but have challenges with:  
- Long sequences (computationally heavy)  
- Missing fine-grained details  

**Multi-scale input solves this** by processing data at multiple resolutions and combining them in the Transformer.  

This helps the model learn:  
- **Short-term dependencies** → fine-scale input  
- **Long-term patterns** → coarse-scale input  

---

## 📘 What is Multi-Scale Input?  

Multi-scale means providing **different granularities** of the same data.  

Example:  
- Time series → raw OHLCV vs. weekly averages  
- NLP → word embeddings + sentence embeddings  
- Vision → images at multiple resolutions  

---

## ✨ Notebook Features  

- Multi-scale preprocessing pipeline  
- Custom Transformer with multi-scale input  
- Visualization of attention maps across scales  
- Training and evaluation loop  
- Example dataset demonstration  
