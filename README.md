
# 📚 Discovering Meaningful Co-Reading Patterns in Comics & Graphic Novels

## 🔍 Overview

Analyzing user reading behavior in the **Goodreads Comics & Graphic Novels dataset**, which includes over **542,000 reviews, 59,000 users, and 89,000 books**. The focus is on discovering meaningful co-reading patterns using **frequent itemset mining (FP-Growth)** and association rule mining. These insights help reveal how readers engage with comics and graphic novels, and demonstrate how such patterns can be leveraged for recommendation systems and personalization.

---

## 📓 Main Notebook

👉 The main deliverable is:

**`main_notebook.ipynb`**

This notebook contains:
- Data preprocessing pipeline  
- Frequent itemset mining (FP-Growth)  
- Association rule generation  
- Visualizations and interpretation  

---

## ❓ Research Questions

- **RQ1:** What frequent co-occurring book sets emerge under varying support thresholds, and how can support, confidence, and lift be used to identify meaningful associations? 
- **RQ2(not implemented):** How effectively can TF-IDF representations capture semantic similarity between reviews, and can clustering reveal coherent groups of user preferences?
- **RQ3(not implemented):** Can topic modeling (LDA) uncover latent themes in user reviews that are not captured by surface-level TF-IDF representations?

---

## 🎥 Project Video

👉 **[INSERT YOUR YOUTUBE LINK HERE]**

---

## 📊 Data

### Dataset Used
- [Goodreads Reviews (Comics & Graphic Novels)](https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz)

- [Goodreads Books Metadata (Comics & Graphic Novels)](https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_books_comics_graphic.json.gz)

### Preprocessing Steps

The preprocessing pipeline was implemented in `preprocessing.py` and extended with book metadata processing. The following steps were applied:

#### 🔹 Reviews Dataset Processing

- Loaded raw dataset from  
  `goodreads_reviews_comics_graphic.json.gz`

- Cleaned and standardized data:
  - Ensured correct data types for `user_id`, `book_id`, and `rating`
  - Removed rows with missing or invalid values

- Removed invalid rows:
  - Missing `user_id` or `book_id`
  - Missing or incorrect ratings  

- Checked and handled duplicates:
  - No duplicate rows were found

- Processed text data:
  - Cleaned `review_text`
  - Generated feature `review_len_tokens`

- Transformed dataset into transactional format:
  - Grouped book interactions by `user_id`
  - Converted each user into a basket (list of books)

- Filtered users:
  - Removed users with fewer than 2 book interactions  
  - Reduced dataset from **58,111 users → 32,673 users**

- Filtered books:
  - Removed low-frequency books to reduce sparsity

---

#### 🔹 Books Metadata Processing

- Loaded metadata from  
  `goodreads_books_comics_graphic.json.gz`

- Cleaned and extracted relevant fields:
  - `book_id`
  - `title`
  - Additional metadata where available

#### 🔹 Output Files

- `data/processed/cleaned_data.csv` → cleaned review dataset  
- `data/processed/baskets.pkl` → user basket dataset  
- `data/processed/books.csv` → processed book metadata  


## How to Reproduce my work

## Repo Structure

```text
Goodreads_Reviews/
│
├── data/
│   ├── raw/ --- Add the downloaded .gz files(Can be found in Data Section) here 
│   │   ├── goodreads_reviews_comics_graphic.json.gz
│   │   ├── goodreads_books_comics_graphic.json.gz
│   │
│   ├── processed/ --- generated while running the preprocessing step
│   │   ├── cleaned_data.csv
│   │   ├── baskets.pkl
│   │   ├── books.csv
│
├── src/
│   ├── data_loader.py
│   ├── books_dataloader.py
│   ├── preprocessing.py
│   ├── itemset_mining.py
│   ├── recommender.py
│   ├── plotting.py
│   ├── summary_plots.py
│
├── notebooks/
│   ├── 737002453_ProjectCheckpoint1.ipynb
│   ├── 737002453_ProjectCheckpoint2.ipynb
│
├── outputs/
│   ├── figures/
│   │   ├── summary/
│   │   ├── sup_001_freq_5_len_4/
│   │   ├── sup_002_freq_5_len_4/
│   │   ├── sup_0005_freq_5_len_4/
│   │
│   ├── itemsets/
│   │   ├── itemsets_sup_001.csv
│   │   ├── rules_sup_001.csv
│
├── main_notebook.ipynb
├── requirements.txt
└── README.md
```

## 📦 Key Dependencies

- python 3.11  
- pandas 2.2.0  
- numpy 1.26.0  
- mlxtend 0.23.4  
- scikit-learn 1.4.1  
- matplotlib 3.8.0  
- seaborn 0.13.0  
- tqdm 4.67.3

👉 Full list of dependencies is available in `requirements.txt`.


## 📈 Results Summary

Frequent itemset mining revealed strong co-reading patterns among users in the comics dataset. Lower support thresholds (0.005) produced a larger number of itemsets (779) and rules (2,126), capturing broader reading behavior, while higher thresholds (0.02) produced fewer but more reliable patterns (52 itemsets, 32 rules).  

Association rules with **lift greater than 1** consistently highlighted meaningful relationships beyond popularity bias, showing that certain books are frequently read together due to shared themes or series connections.  

👉 Overall, the key takeaway is that **FP-Growth effectively uncovers hidden co-reading structures**, enabling more accurate and behavior-driven recommendation systems.
