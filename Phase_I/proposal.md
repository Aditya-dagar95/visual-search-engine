# One-Page Proposal: Retail Product Visual Search and Recommendation Engine

Name: Aditya Dagar | SAP ID: 590015198 | B.Tech CSE, 2nd Year | UPES Dehradun
Course: CSAI3027P Deep Learning | Individual Hackathon | Date: 18 Oct 2026

## 1. Problem and users
Shoppers often cannot describe a product in words but have a photo of it. Text search fails for visual attributes such as pattern, cut and colour. This project builds an image-based search that returns similar catalogue items. Users: online shoppers, and catalogue teams checking for duplicate or near-duplicate listings.

## 2. Intended use and limits
Input: a product photo. Output: top-K similar items with similarity scores. Not intended for identifying people, surveillance, or any high-stakes decision. No personal or facial data is used.

## 3. Dataset
Fashion Product Images (Small), Kaggle (Myntra catalogue). Labels: articleType, colour, gender and others. Categories with fewer than 50 images are dropped. Known biases: studio backgrounds, a single catalogue, mostly apparel. Licence to be recorded in the source log.

## 4. Split plan (leakage-safe)
- Split A (seen categories): stratified 70/15/15 by articleType. Test images are queries; train and val images form the gallery.
- Split B (unseen categories): about 20% of articleTypes held out entirely, to test generalisation to new classes.
- Seed 42; manifests saved as CSV.

## 5. Baselines
1. HOG features with kNN retrieval (classical).
2. Frozen ImageNet ResNet50, 2048-d pooled embedding, L2-normalised, FAISS inner-product search.

## 6. Innovation hypothesis
Fine-tuning a projection head (2048 to 512 to 256) with batch-hard triplet loss improves Recall@10 over the frozen embedding, especially on unseen categories. In Phase II, fusing text (title/category) embeddings and adding a re-ranker should improve it further. Each step is tested in an ablation table.

## 7. Metrics
Recall@1/5/10, mAP@10, NDCG@10, and p95 query latency (measured, not assumed).

## 8. Experiment plan
Phase I: audit, splits, two baselines, triplet runs 1 and 2, learning curves, architecture diagram.
Phase II: text fusion, re-ranker, ablation, error analysis, FastAPI + Gradio app, model card.

## 9. Risks
Overfitting to catalogue style, look-alike products, out-of-catalogue queries, Kaggle GPU quota. Mitigation: frozen backbone except layer4, unseen-category test, GPU hours tracked.

## 10. Disclosure
Pretrained ResNet50 (torchvision) is used with a stated fine-tuning strategy. All sources and any GenAI use are logged in source_log.md.
