EXPORT REPORT FIGURES
====================

To include figures in the LaTeX report, run the full notebook (csgo_ml_workflow.ipynb)
and then run the cell titled "Export Report Figures" at the end of Section 1.4.
That cell saves the following files into this folder:

- fig_class_distribution.png  : Class distribution (winner: t1 vs t2)
- fig_correlation_heatmap.png : Correlation heatmap of key features
- fig_confusion_matrix_lr.png : Confusion matrix for Logistic Regression

Alternatively, you can manually save the notebook output figures:
1. Run the notebook up to and including the model evaluation section
2. Right-click each figure and "Save image as..." with the filenames above
3. Save them into the figures/ folder

The report will compile with pdflatex; if figures are missing, you may see
"File not found" errors. Ensure all three PNG files are present before building.
