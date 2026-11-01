# Customer Segmentation & Market Basket Analysis

Groups the customers of an online retailer by how they shop, and finds which products tend to be bought together. The goal is marketing that fits each group: cross-selling products that go together and up-selling to the most valuable customers.

Built with Python, pandas, scikit-learn and mlxtend.

## The data

The public **Online Retail** dataset from the UCI Machine Learning Repository: transactions from a UK online gift shop, with invoice number, product description, quantity, unit price, date, customer ID and country. The script downloads it directly, so there is nothing to set up.

## What the script does

**1. Cleaning**
- Drops rows with missing values and duplicates
- Flags outliers in quantity and price with z-scores (more than 3 standard deviations away) and plots them

**2. Customer features**
- Total amount spent, purchase frequency (number of invoices) and recency (days since last purchase), the core of RFM analysis
- Average unit price and average quantity per invoice
- Country as one-hot columns

**3. Segmentation with K-Means**
- Uses the elbow method (k from 1 to 10) to choose the number of segments, settling on **3 customer segments**
- Visualises the segments with PCA in 2D and 3D

**4. Checking the clusters**
- Silhouette score and Davies–Bouldin index
- Cohesion (how tight each cluster is) and separation (how far apart the clusters are)
- Silhouette scores for 2 to 5 clusters on a sample, to sanity-check the choice of k

**5. RFM score**
- Ranks every customer on recency, frequency and monetary value and combines them into one weighted score (15% recency, 28% frequency, 57% monetary)

**6. Market basket analysis**
- Turns each invoice into a basket of products
- Uses the **Apriori** algorithm to find product sets bought together in at least 1% of baskets
- Builds association rules ranked by lift, and plots support, confidence and lift for every rule

## Charts

![Elbow method and segments](https://github.com/user-attachments/assets/59de6e86-2963-454a-a7ef-675adf1e1631)

![Customer segments](https://github.com/user-attachments/assets/0600744e-0e9f-4a13-b356-69650b32589a)

<img width="643" alt="PCA visualisation of the clusters" src="https://github.com/user-attachments/assets/93f09a4a-2caa-48de-b5de-3f25b9001d5e" />

![Association rules](https://github.com/user-attachments/assets/f9029f59-f16a-4d84-8df2-0ed78f495845)

## Run it

```bash
git clone https://github.com/shashank274/Data-Science-Analytics.git
cd Data-Science-Analytics
pip install -r requirements.txt
python customer_segmentation.py
```

The script is laid out in sections (separated by comment lines) that follow the steps above, so it also works well pasted cell by cell into a notebook.

---

More of my work: [portfolio-pied-gamma-88.vercel.app](https://portfolio-pied-gamma-88.vercel.app)
