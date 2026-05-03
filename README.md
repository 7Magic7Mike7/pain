Various scripts and notebooks used to prepare data for the PPP Map.

# Data to (texture) Map transformation

0. The dataset needs to be country-based (i.e., there needs to be a country column)
1. Normalize data to get a pain-value between 0 and 1 (both inclusive)
2. Normalize countries to be ISO A3 codes
3. Use the generate map function to create a texture that colors countries based on the pain-value and passed cmap

Explanatory step by step code can be found in `/scripts/my-steps.ipynb`
