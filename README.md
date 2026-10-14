# Social Media Usage and Emotional Well-Being

Capstone Project 3 by Himani Verma.

This Streamlit app loads the saved preprocessing and Random Forest pipeline to predict one of six emotion labels. It also displays the notebook's validation comparison and held-out test results.

## Run locally

Use Python 3.12. Install dependencies with `pip install -r requirements.txt`, then run `streamlit run app.py`.

## Deploy

Upload all files in this directory to a GitHub repository. On Streamlit Community Cloud, create an app from that repository, choose the main branch and `app.py` as the entrypoint, and select Python 3.12. The model and CSV files must remain beside app.py.

## Results

The final held-out test accuracy is 60.38%, with Macro F1 of 0.569 on 53 records. Model selection used validation results. Tuning did not improve the selected Random Forest settings.

## Data and limits

Source: https://www.kaggle.com/datasets/emirhanai/social-media-usage-and-emotional-well-being/

After correcting formatting, handling an invalid gender and excluding one uncertain label, exact duplicates were removed. The 265 retained records were resplit into 159 training, 53 validation and 53 test records. This is an educational classification demonstration, not a diagnostic tool or evidence that social media causes an emotion. Profiles entered in the app are not saved.
