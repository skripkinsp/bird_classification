# Kaggle API Setup for Bird Classification

## Quick Setup

1. **Install Kaggle CLI:**
   ```bash
   pip install kaggle
   ```

2. **Get API Token:**
   - Go to https://www.kaggle.com/account
   - Scroll down to "API" section
   - Click "Create New API Token"
   - Download `kaggle.json`

3. **Place Token:**
   - **Linux/Mac:** Place in `~/.kaggle/kaggle.json` and run `chmod 600 ~/.kaggle/kaggle.json`
   - **Windows:** Place in `C:\Users\<username>\.kaggle\kaggle.json`

4. **Test Setup:**
   ```bash
   kaggle datasets list
   ```

## Troubleshooting

- **Permission denied:** Make sure `kaggle.json` has correct permissions (600 on Linux/Mac)
- **API error:** Check if your account has accepted the competition rules if downloading competition data
- **Rate limits:** Kaggle has download limits; the code will fallback to DVC if available

## Alternative: Manual Download

If Kaggle API doesn't work, you can:
1. Download the dataset manually from https://www.kaggle.com/datasets/antoniozarauzmoreno/birds400
2. Extract it to `data/birds400`
3. Run training as usual
