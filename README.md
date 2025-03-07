# Multimodal-Depression-Detection
MDD-Net: Multimodal Depression Detection through Mutual Transformer
----

**python implementation**

<!-- ```python
Version :   0.0.1  
Author  :   Md Rezwanul Haque
Email   :   mr3haque@uwaterloo.ca 
``` -->
---
### **Related resources**:

**LOCAL ENVIRONMENT**  
```python
OS          :   Ubuntu 24.04.1 LTS       
Memory      :   128.0 GiB
Processor   :   Intel® Xeon® w5-3425 × 24  
Graphics    :   NVIDIA RTX A6000
CPU(s)      :   24
Gnome       :   46.0 
```
---

**python requirements**
* **pip requirements**: ```pip install -r requirements.txt``` 

### Execution (Depression Detection)
- ```$ conda activate your_env``` --> ```i.e., $ conda activate rezwan``` 

- To train and validate:

    ```$ python mainkfold.py```

- To inference:
    ```$ python infer_mainkfold.py```

### 📖 Citation:
- If you find this project useful for your research, please cite this paper.

```bibtex
@inproceedings{mdd-net,
    title={MDD-Net: Multimodal Depression Detection through Mutual Transformer}},
    author={Haque, Md Rezwanul and Islam, Md. Milon and Raju, S M Taslim Uddin and Altaheri, Hamdi and Nassar, Lobna and Karray, Fakhri},
    booktitle={2025 International Joint Conference on Neural Networks (IJCNN)},
    pages={--},
    year={2025 [Submitted]},
    organization={IEEE}
}
```

### 🙌🏻 Acknowledgement

- We acknowledge the wonderful work of [GSA-Network](https://openreview.net/forum?id=KiFeuZu24k) and [HAT-Net](https://arxiv.org/abs/2106.03180). 
- The training pipelines are adapted from [depression-detection](https://github.com/AllenYolk/depression-detection).