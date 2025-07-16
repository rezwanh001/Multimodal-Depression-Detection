# Multimodal-Depression-Detection
Paper Title: **MDD-Net: Multimodal Depression Detection through Mutual Transformer**

#### ***Proceedings of the 2025 IEEE International Conference on Systems, Man, and Cybernetics (SMC), Vienna, Austria. Copyright 2025 by the author(s).***
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
- ```$ conda activate your_env```

- To train and validate:

    ```$ python mainkfold.py```

- To inference:
    ```$ python infer_mainkfold.py```

### 📖 Citation:
- If you find this project useful for your research, please cite [this paper](https://arxiv.org/abs/****.*****)

```bibtex
@inproceedings{haque2025mdd,
    title={MDD-Net: Multimodal Depression Detection through Mutual Transformer},
    author={Haque, Md Rezwanul and Islam, Md. Milon and Raju, S M Taslim Uddin and Altaheri, Hamdi and Nassar, Lobna and Karray, Fakhri},
    journal = {arXiv preprint arXiv:****.*****},
    year = {2025}
}
```

### 🙌🏻 Acknowledgement

- We acknowledge the wonderful work of [GSA-Network](https://openreview.net/forum?id=KiFeuZu24k) and [HAT-Net](https://arxiv.org/abs/2106.03180). 
- The training pipelines are adapted from [depression-detection](https://github.com/AllenYolk/depression-detection).

## License

This project is licensed under the [MIT License](LICENSE).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
