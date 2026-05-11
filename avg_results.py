import pandas as pd

print("📊 Analyzing 5-Fold Cross Validation Results...\n")

folds_data = []
for i in range(1, 6):
    df = pd.read_csv(f'result/lossAndAcc_Fold_{i}.csv')
    
    # Find the row with the absolute best Multi-Omics Test AUC
    best_idx = df['auc_geo'].idxmax()
    best_row = df.loc[best_idx]
    
    folds_data.append({
        'Fold': f"Fold {i}",
        'Best Epoch': best_row['epoch'],
        'Train AUC': best_row['auc_train'],  # <--- ADDED THIS!
        'Test AUC': best_row['auc_geo'],
        'Test Acc': best_row['acc_geo'],
        'RNA-Only AUC': best_row['auc_temp'],
        'F1 Score': best_row['f1_geo']
    })

summary_df = pd.DataFrame(folds_data)
summary_df.set_index('Fold', inplace=True)

print("--- INDIVIDUAL FOLD PERFORMANCES ---")
print(summary_df.round(3)) 

print("\n======================================")
print("🏆 FINAL AVERAGED PROJECT SCORES")
print("======================================")
print(summary_df.mean().round(3))