# PLMR
## pytorch implementation of the KDD2025 paper: Boosting Explainability through Selective Rationalization in Pre-trained Language Models.
## Environments

- **Ubuntu 18.04.6**  
- **Python 3.9**

### Requirements

- **torch==2.0.0**  
- **tqdm==4.66.4**
- **pandas==2.2.2**  
- **numpy==1.26.4**

## Datasets

### Beer Reviews
The dataset is available [here](https://people.csail.mit.edu/taolei/beer/).  
After downloading, move it to the `./data/beer` directory.

### Hotel Reviews
You can download it from [here](https://people.csail.mit.edu/yujia/files/r2a/data.zip).  
Then, copy the following files from `data/oracle` and place them in the `./data/hotel` directory:  
- `hotel_Location.train`
- `hotel_Location.dev`
- `hotel_Service.train`
- `hotel_Service.dev`
- `hotel_Cleanliness.train`
- `hotel_Cleanliness.dev`

Additionally, transfer these files from `data/target` into the `./data/hotel/annotations` directory:  
- `hotel_Location.train`
- `hotel_Service.train`
- `hotel_Cleanliness.train`

### Word Embedding
Download the **GloVe** word embedding file: `glove.6B.100d.txt`.  
Move it to the `./data/hotel/embeddings` directory.

## Running Example

### Beer

#### Appearance:
```bash
python run.py --aspect 0 --dim_reduction_start 5 --dim_reduction_end 7 --lr_trans 0.000005 --lr_mlp 0.00002 --epochs 10 --sparsity_percentage 0.2



