# Goodreads Reviews (Comics & Graphic)

## Project Overview

This project uses the **Goodreads Reviews (Comics & Graphic)** dataset to study reader opinions and behavior in the comics/graphic genre through **user ratings** and **free-text reviews**. The dataset contains large-scale, real-world user-generated content, enabling analysis of sentiment-like signals, recurring themes in reviews, and interaction structure between users and books.

## Project Goals

The goal of this project is to perform data mining analyses on Goodreads review data to uncover patterns in **what readers talk about**, **how they rate books**, and **how books relate to each other through shared readership**.

- **Short-term goal:** apply **course-aligned techniques** (e.g., text mining, clustering, anomaly detection, and basket/co-occurrence analysis) to build strong baselines and extract core patterns.
- **Long-term goal:** extend the analysis using **beyond-course techniques** (e.g., topic modeling and transformer-based embeddings such as SBERT) to capture deeper semantic structure and produce more interpretable insights.

## Dataset
- **Name:** Goodreads Reviews — Comics & Graphic  
- **Source:** https://mcauleylab.ucsd.edu:8443/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz  
- **Scale:** 542,338 reviews (JSON Lines)  

### Key Fields
- Identifiers: `user_id`, `book_id`, `review_id`
- Labels/signals: `rating` (1–5), `n_votes`, `n_comments`
- Text: `review_text`
- Time: `date_added`, `date_updated`, `started_at`, `read_at`


## Setup
### Requirements
- Python 3.9+
- Recommended libraries:
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - (optional) `scikit-learn`
  - (optional, external techniques) `sentence-transformers`, `bertopic`

