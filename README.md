# Multimodal-Depression-Detection
Multimodal Depression Detection through Mutual Transformer
----

**python implementation**

```python
Version :   0.0.1  
Author  :   Md Rezwanul Haque
Email   :   mr3haque@uwaterloo.ca 
```
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

    ```$ python mainkfold.py``

- To inference:
    ```$ python infer_mainkfold.py```

### Dataset Paper:

    ```
    @inproceedings{yoon2022d,
        title={D-vlog: Multimodal vlog dataset for depression detection},
        author={Yoon, Jeewoo and Kang, Chaewon and Kim, Seungbae and Han, Jinyoung},
        booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
        volume={36},
        number={11},
        pages={12226--12234},
        year={2022}
    }
    ```