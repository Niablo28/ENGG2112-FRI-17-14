#!/usr/bin/env bash
set -euo pipefail

SUBDIR=submission

echo "Creating submission bundle in $SUBDIR/..."

mkdir -p "$SUBDIR"

# Copy final report & essentials
if [ -f "submission/ENGG2112_Final_Report.pdf" ]; then
    cp submission/ENGG2112_Final_Report.pdf "$SUBDIR"/
    echo "✓ Copied final report"
else
    echo "⚠️  Warning: ENGG2112_Final_Report.pdf not found"
fi

# Copy figures
if [ -d "reports/figures" ]; then
    cp -r reports/figures "$SUBDIR"/figures
    echo "✓ Copied figures"
fi

# Copy metrics and model card
if [ -f "reports/METRICS.md" ]; then
    cp reports/METRICS.md "$SUBDIR"/
    echo "✓ Copied METRICS.md"
fi

if [ -f "models/model_card.md" ]; then
    cp models/model_card.md "$SUBDIR"/
    echo "✓ Copied model_card.md"
fi

# Copy final model (timestamped or latest)
MODEL_FILE=$(ls -t models/model_*.joblib 2>/dev/null | head -1 || echo "")
if [ -n "$MODEL_FILE" ]; then
    cp "$MODEL_FILE" "$SUBDIR"/
    echo "✓ Copied model: $(basename "$MODEL_FILE")"
else
    echo "⚠️  Warning: No timestamped model found"
fi

# Pack a lean source bundle (exclude heavy folders via .gitattributes)
if command -v git &> /dev/null && [ -d ".git" ]; then
    git archive --format=zip --output "$SUBDIR/source.zip" HEAD
    echo "✓ Created source.zip"
else
    echo "⚠️  Warning: Git not available, skipping source.zip"
fi

# Create packing list
cat > "$SUBDIR/PACKING_LIST.md" << 'EOF'
# Submission Packing List

## Contents

- `ENGG2112_Final_Report.pdf` - Main project report
- `figures/` - Generated visualizations (threshold sweep, calibration, fairness)
- `METRICS.md` - Performance metrics summary
- `model_card.md` - Model documentation
- `model_*.joblib` - Trained model artifact
- `source.zip` - Source code archive (excludes deprecated scripts, notebooks, caches)

## How to Run

1. Extract `source.zip`
2. Install dependencies: `pip install -r requirements.txt`
3. Train model: `make train`
4. Generate figures: `make figures`
5. Run UI: `make ui`

## Reproducibility

- All random seeds fixed (`random_state=42`)
- Requirements pinned in `requirements.txt`
- See `README.md` for full setup instructions
EOF

echo "✓ Created PACKING_LIST.md"
echo ""
echo "Submission bundle ready in $SUBDIR/"
echo "Contents:"
ls -lh "$SUBDIR"/

