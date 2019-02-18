# Dataset files explained

- **output_no_clean.csv** : output after extracting all reviews, used to avoid to do it again (because it's slow), on the original dataset.
- **output_not_original_no_clean.csv** : Same but with the already preprocessed dataset.

These two files are used in input to the cleanseData function from "import.py", after this, we have:

- **output_X.csv** : output after cleaning (remove less frequent words) with a threshold of X, on the original dataset.
- **output_not_original_X.csv** : Same but on the already preprocessed dataset.
