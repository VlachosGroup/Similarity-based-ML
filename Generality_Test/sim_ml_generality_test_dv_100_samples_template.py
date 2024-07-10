from utils_sim_ml_generality import *

with open("gdb13_clustering_model.pkl", "rb") as f:
    kmeans_model_gdb13_equal = pickle.load(f)

with open("gdb13_pca.pkl", "rb") as f:
    pca_gdb13_equal = pickle.load(f)

# we retrieve the scaler used for NN
with open(f"scaler_Cv_tl.pkl", "rb") as f:
    scaler_Cv = pickle.load(f)

def faltten_ls(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]
Cv_feats = ['NumAtoms', 'TIC1', 'NumRings', 'ATSC1Z', 'BCUTd-1l','BCUTdv-1l', 'exactmw', 'BalabanJ', 'ATS3Z', 'chi1v','n7FRing', 'VE1_A', 'Xch-3d', 'GATS1c', 'ATSC7d', 'ATSC6d', 'ATSC1dv', 'BertzCT', 'kappa3', 'n5Ring']
with open(f"../NN_Pretraining/result_Cv_nn_optimization_overall.pkl", "rb") as f: 
    nn_pretrain_results = pickle.load(f)

min_index = np.where(nn_pretrain_results['best_test_losses'] == np.min(nn_pretrain_results['best_test_losses']))[0][0]
best_nn = nn_pretrain_results['models'][min_index]

data = pd.read_csv('Data/modularity_training_data.csv')
train_data = data[data['set'] == 'train'].reset_index(drop=True)
test_data = data[data['set'] == 'test'].reset_index(drop=True)
X_train = pd.read_csv('Data/modularity_train_data_rdkit_decriptors.csv')
X_test = pd.read_csv('Data/modularity_test_data_rdkit_decriptors.csv')

fp_train = generate_fingerprints(train_data['Smiles'].to_list(), "topological_fingerprint")
fp_test = generate_fingerprints(test_data['Smiles'].to_list(), "topological_fingerprint")
fp_train_red = pca_gdb13_equal.transform(fp_train)
labels_kmeans_train = kmeans_model_gdb13_equal.predict(fp_train_red)

fp_test_red = pca_gdb13_equal.transform(fp_test)
labels_kmeans_test = kmeans_model_gdb13_equal.predict(fp_test_red)

# we remove group 2 molecules as there are way too few of them for a model to be trained on and tested with
group2_train_idxs = np.where(labels_kmeans_train == 1)[0]
group2_test_idxs = np.where(labels_kmeans_test == 1)[0]
train_data = train_data.drop(group2_train_idxs)
test_data = test_data.drop(group2_test_idxs)
X_train = pd.read_csv('Data/modularity_train_data_rdkit_decriptors.csv').drop(group2_train_idxs)
X_test = pd.read_csv('Data/modularity_test_data_rdkit_decriptors.csv').drop(group2_test_idxs)
data = pd.concat([train_data, test_data]).reset_index(drop=True)
X = pd.concat([X_train, X_test]).reset_index(drop=True).iloc[:, 1:]

def clean_df(X):
    X = drop_columns_with_nans(X)
    X = drop_columns_with_mordred_missing(X)
    X = drop_duplicated_columns(X)
    X = drop_non_num_cols(X)
    return X

X = clean_df(X)

def parallelize_benchmarking(seed1, seed2,  data, X, target, best_nn, scaler, tl_feats):
    # Initialize lists to store results
    results = {
        'X_train_baseline': [],
        'X_train_group1': [],
        'X_train_group3': [],
        'X_test_baseline': [],
        'X_test_group1': [],
        'X_test_group3': [],
        'y_train_baseline': [],
        'y_train_group1': [],
        'y_train_group3': [],
        'y_test_baseline': [],
        'y_test_group1': [],
        'y_test_group3': [],
        'y_test_pred_baseline_group1': [],
        'y_test_pred_baseline_group3': [],
        'y_test_pred_baseline_overall': [],
        'labels_train_baseline': [],
        'labels_test_baseline': [],
        'top_features_baseline': [],
        'n_features_baseline': [],
        'rmse_cv_baseline': [],
        'rmse_baseline_group1': [],
        'rmse_baseline_group3': [],
        'rmse_baseline_overall': [],
        'top_features_sim_ml_group1': [],
        'top_features_sim_ml_group3': [],
        'n_features_sim_ml_group1': [],
        'n_features_sim_ml_group3': [],
        'y_test_pred_sim_ml_group1': [],
        'y_test_pred_sim_ml_group3': [],
        'y_test_pred_sim_ml_overall': [],
        'rmse_cv_sim_ml_group1': [],
        'rmse_cv_sim_ml_group3': [],
        'rmse_sim_ml_group1': [],
        'rmse_sim_ml_group3': [],
        'rmse_sim_ml_overall': [],
        'top_features_tl': [],
        'n_features_tl': [],
        'y_test_pred_tl_group1': [],
        'y_test_pred_tl_group3': [],
        'y_test_pred_tl_overall': [],
        'rmse_cv_tl': [],
        'rmse_tl_group1': [],
        'rmse_tl_group3': [],
        'rmse_tl_overall': []
        }
    
    # Initialize variables with default values
    rmse_cv_sim_ml_group1 = rmse_cv_sim_ml_group3 = 'NA'
    y_test_pred_sim_ml_group1 = y_test_pred_sim_ml_group3 = ['NA']
    n_features_sim_ml_group1 = n_features_sim_ml_group3 = 'NA'
    top_features_selected_sim_ml_group1 = top_features_selected_sim_ml_group3 = ['NA']
    rmse_sim_ml_group1 = rmse_sim_ml_group3 = rmse_sim_ml_overall = 'NA'
    y_test_pred_sim_ml_overall = np.array([])
    
    # Shuffle data
    data_shuffled = data.sample(frac=1, random_state=seed1).reset_index(drop=True).iloc[:100].reset_index(drop=True)
    X_shuffled = X.sample(frac=1, random_state=seed1).reset_index(drop=True).iloc[:100].reset_index(drop=True)
    data_shuffled = data_shuffled.sample(frac=1, random_state=seed2).reset_index(drop=True).iloc[:100].reset_index(drop=True)
    X_shuffled = X_shuffled.sample(frac=1, random_state=seed2).reset_index(drop=True).iloc[:100].reset_index(drop=True)

    # Split data
    train = data_shuffled.iloc[:80].reset_index(drop=True)
    test = data_shuffled.iloc[80:100].reset_index(drop=True)
    X_train = X_shuffled.iloc[:80].reset_index(drop=True)
    X_test = X_shuffled.iloc[80:100].reset_index(drop=True)

    # Generate fingerprints
    fp_train = generate_fingerprints(train['Smiles'].to_list(), "topological_fingerprint")
    fp_test = generate_fingerprints(test['Smiles'].to_list(), "topological_fingerprint")
    y_train = train[target]
    y_test = test[target]

    # PCA and KMeans
    fp_train_red = pca_gdb13_equal.transform(fp_train)
    labels_train = kmeans_model_gdb13_equal.predict(fp_train_red)
    fp_test_red = pca_gdb13_equal.transform(fp_test)
    labels_test = kmeans_model_gdb13_equal.predict(fp_test_red)

    # Split data based on KMeans labels
    X_train_group1 = X_train.iloc[np.where(labels_train == 0)[0]]
    X_train_group3 = X_train.iloc[np.where(labels_train == 2)[0]]
    y_train_group1 = y_train[np.where(labels_train == 0)[0]]
    y_train_group3 = y_train[np.where(labels_train == 2)[0]]
    X_test_group1 = X_test.iloc[np.where(labels_test == 0)[0]]
    X_test_group3 = X_test.iloc[np.where(labels_test == 2)[0]]
    y_test_group1 = y_test[np.where(labels_test == 0)[0]]
    y_test_group3 = y_test[np.where(labels_test == 2)[0]]

    # Feature selection and training/testing with Random Forest
    X_train_scaled = scaler_Cv.transform(X_train[tl_feats])
    X_test_scaled = scaler_Cv.transform(X_test[tl_feats])
    X_train_tl = extract_features_nth_layer(best_nn, torch.tensor(X_train_scaled, dtype=torch.float32), 2)
    X_test_tl = extract_features_nth_layer(best_nn, torch.tensor(X_test_scaled, dtype=torch.float32), 2)
    X_test_tl_group1 = X_test_tl.iloc[np.where(labels_test == 0)[0]]
    X_test_tl_group3 = X_test_tl.iloc[np.where(labels_test == 2)[0]]

    # Clean and process data
    X_train_uncorr = remove_corr_features(X_train, corr_cutoff=0.9)

    # Check for label availability and process accordingly
    if len(np.where(labels_test == 0)[0]) == 0:
        top_features, top_features_selected, n_features, rmse_cv = feat_importance_and_RFE(X_train_uncorr, y_train, 'MDI', 50)
        _, y_test_pred_overall = train_and_test_rf(X_train_uncorr, y_train, X_test, y_test, top_features_selected['Topfeats'])
        _, y_test_pred_group3 = train_and_test_rf(X_train_uncorr, y_train, X_test_group3, y_test_group3, top_features_selected['Topfeats'])
        y_test_pred_group1 = ['NA']
        rmse_overall = np.sqrt(mean_squared_error(y_test, y_test_pred_overall))
        rmse_group1 = 'NA'
        rmse_group3 = np.sqrt(mean_squared_error(y_test_group3, y_test_pred_group3))

        X_train_group3_cleaned = clean_df(X_train_group3)
        X_train_group3_uncorr = remove_corr_features(X_train_group3_cleaned, corr_cutoff=0.9)
        top_features_sim_ml_group3, top_features_selected_sim_ml_group3, n_features_sim_ml_group3, rmse_cv_sim_ml_group3 = feat_importance_and_RFE(X_train_group3_uncorr, y_train_group3, 'MDI', 50)
        top_features_selected_sim_ml_group1 = ['NA']
        n_features_sim_ml_group1 = 'NA'
        _, y_test_pred_sim_ml_group3 = train_and_test_rf(X_train_group3_uncorr, y_train_group3, X_test_group3, y_test_group3, top_features_selected_sim_ml_group3['Topfeats'])
        y_test_pred_sim_ml_overall = np.array(y_test_pred_sim_ml_group3.tolist())
        y_test_pred_sim_ml_group1 = ['NA']
        rmse_sim_ml_overall = np.sqrt(mean_squared_error(y_test_group3.tolist(), y_test_pred_sim_ml_overall))
        rmse_sim_ml_group1 = 'NA'
        rmse_sim_ml_group3 = np.sqrt(mean_squared_error(y_test_group3, y_test_pred_sim_ml_group3))
  
        top_features_tl, top_features_selected_tl, n_features_tl, rmse_cv_tl = feat_importance_and_RFE(X_train_tl, y_train, 'MDI', 32)
        _, y_test_pred_tl_overall = train_and_test_rf(X_train_tl, y_train, X_test_tl, y_test, top_features_selected_tl['Topfeats'])
        y_test_pred_tl_group1 = ['NA']
        _, y_test_pred_tl_group3 = train_and_test_rf(X_train_tl, y_train, X_test_tl_group3, y_test_group3, top_features_selected_tl['Topfeats'])
        rmse_tl_overall = np.sqrt(mean_squared_error(y_test, y_test_pred_tl_overall))
        rmse_tl_group1 = 'NA'
        rmse_tl_group3 = np.sqrt(mean_squared_error(y_test_group3, y_test_pred_tl_group3))

    elif len(np.where(labels_test == 2)[0]) == 0:
        top_features, top_features_selected, n_features, rmse_cv = feat_importance_and_RFE(X_train_uncorr, y_train, 'MDI', 50)
        _, y_test_pred_overall = train_and_test_rf(X_train_uncorr, y_train, X_test, y_test, top_features_selected['Topfeats'])
        _, y_test_pred_group1 = train_and_test_rf(X_train_uncorr, y_train, X_test_group1, y_test_group1, top_features_selected['Topfeats'])
        y_test_pred_group3 = ['NA']
        rmse_group1 = np.sqrt(mean_squared_error(y_test_group1, y_test_pred_group1))
        rmse_overall = np.sqrt(mean_squared_error(y_test, y_test_pred_overall))
        rmse_group3 = 'NA'

        X_train_group1_cleaned = clean_df(X_train_group1)
        X_train_group1_uncorr = remove_corr_features(X_train_group1_cleaned, corr_cutoff=0.9)
        top_features_sim_ml_group1, top_features_selected_sim_ml_group1, n_features_sim_ml_group1, rmse_cv_sim_ml_group1 = feat_importance_and_RFE(X_train_group1_uncorr, y_train_group1, 'MDI', 50)
        top_features_selected_sim_ml_group3 = ['NA']
        n_features_sim_ml_group3 = 'NA'
        _, y_test_pred_sim_ml_group1 = train_and_test_rf(X_train_group1_uncorr, y_train_group1, X_test_group1, y_test_group1, top_features_selected_sim_ml_group1['Topfeats'])
        y_test_pred_sim_ml_overall = np.array(y_test_pred_sim_ml_group1.tolist())
        y_test_pred_sim_ml_group3 = ['NA']
        rmse_sim_ml_overall = np.sqrt(mean_squared_error(y_test_group1.tolist(), y_test_pred_sim_ml_overall))
        rmse_sim_ml_group1 = np.sqrt(mean_squared_error(y_test_group1, y_test_pred_sim_ml_group1))
        rmse_sim_ml_group3 = 'NA'

        top_features_tl, top_features_selected_tl, n_features_tl, rmse_cv_tl = feat_importance_and_RFE(X_train_tl, y_train, 'MDI', 32)
        _, y_test_pred_tl_overall = train_and_test_rf(X_train_tl, y_train, X_test_tl, y_test, top_features_selected_tl['Topfeats'])
        _, y_test_pred_tl_group1 = train_and_test_rf(X_train_tl, y_train, X_test_tl_group1, y_test_group1, top_features_selected_tl['Topfeats'])
        y_test_pred_tl_group3 = ['NA']
        rmse_tl_overall = np.sqrt(mean_squared_error(y_test, y_test_pred_tl_overall))
        rmse_tl_group1 = np.sqrt(mean_squared_error(y_test_group1, y_test_pred_tl_group1))
        rmse_tl_group3 = 'NA'

    else:
        top_features, top_features_selected, n_features, rmse_cv = feat_importance_and_RFE(X_train_uncorr, y_train, 'MDI', 50)
        _, y_test_pred_overall = train_and_test_rf(X_train_uncorr, y_train, X_test, y_test, top_features_selected['Topfeats'])
        _, y_test_pred_group1 = train_and_test_rf(X_train_uncorr, y_train, X_test_group1, y_test_group1, top_features_selected['Topfeats'])
        _, y_test_pred_group3 = train_and_test_rf(X_train_uncorr, y_train, X_test_group3, y_test_group3, top_features_selected['Topfeats'])
        rmse_overall = np.sqrt(mean_squared_error(y_test, y_test_pred_overall))
        rmse_group1 = np.sqrt(mean_squared_error(y_test_group1, y_test_pred_group1))
        rmse_group3 = np.sqrt(mean_squared_error(y_test_group3, y_test_pred_group3))

        X_train_group1_cleaned = clean_df(X_train_group1)
        X_train_group1_uncorr = remove_corr_features(X_train_group1_cleaned, corr_cutoff=0.9)
        X_train_group3_cleaned = clean_df(X_train_group3)
        X_train_group3_uncorr = remove_corr_features(X_train_group3_cleaned, corr_cutoff=0.9)
        top_features_sim_ml_group1, top_features_selected_sim_ml_group1, n_features_sim_ml_group1, rmse_cv_sim_ml_group1 = feat_importance_and_RFE(X_train_group1_uncorr, y_train_group1, 'MDI', 50)
        top_features_sim_ml_group3, top_features_selected_sim_ml_group3, n_features_sim_ml_group3, rmse_cv_sim_ml_group3 = feat_importance_and_RFE(X_train_group3_uncorr, y_train_group3, 'MDI', 50)
        _, y_test_pred_sim_ml_group1 = train_and_test_rf(X_train_group1_uncorr, y_train_group1, X_test_group1, y_test_group1, top_features_selected_sim_ml_group1['Topfeats'])
        _, y_test_pred_sim_ml_group3 = train_and_test_rf(X_train_group3_uncorr, y_train_group3, X_test_group3, y_test_group3, top_features_selected_sim_ml_group3['Topfeats'])
        y_test_pred_sim_ml_overall = np.array(y_test_pred_sim_ml_group1.tolist() + y_test_pred_sim_ml_group3.tolist())
        rmse_sim_ml_overall = np.sqrt(mean_squared_error(y_test_group1.tolist() + y_test_group3.tolist(), y_test_pred_sim_ml_overall))
        rmse_sim_ml_group1 = np.sqrt(mean_squared_error(y_test_group1, y_test_pred_sim_ml_group1))
        rmse_sim_ml_group3 = np.sqrt(mean_squared_error(y_test_group3, y_test_pred_sim_ml_group3))

        top_features_tl, top_features_selected_tl, n_features_tl, rmse_cv_tl = feat_importance_and_RFE(X_train_tl, y_train, 'MDI', 32)
        _, y_test_pred_tl_overall = train_and_test_rf(X_train_tl, y_train, X_test_tl, y_test, top_features_selected_tl['Topfeats'])
        _, y_test_pred_tl_group1 = train_and_test_rf(X_train_tl, y_train, X_test_tl_group1, y_test_group1, top_features_selected_tl['Topfeats'])
        _, y_test_pred_tl_group3 = train_and_test_rf(X_train_tl, y_train, X_test_tl_group3, y_test_group3, top_features_selected_tl['Topfeats'])
        rmse_tl_overall = np.sqrt(mean_squared_error(y_test, y_test_pred_tl_overall))
        rmse_tl_group1 = np.sqrt(mean_squared_error(y_test_group1, y_test_pred_tl_group1))
        rmse_tl_group3 = np.sqrt(mean_squared_error(y_test_group3, y_test_pred_tl_group3))

    # Append results to lists
    results['X_train_baseline'].append(X_train)
    results['X_test_baseline'].append(X_test)
    results['y_train_baseline'].append(y_train)
    results['y_test_baseline'].append(y_test)
    results['labels_train_baseline'].append(labels_train)
    results['labels_test_baseline'].append(labels_test)
    results['X_train_group1'].append(X_train_group1)
    results['X_train_group3'].append(X_train_group3)
    results['y_train_group1'].append(y_train_group1)
    results['y_train_group3'].append(y_train_group3)
    results['X_test_group1'].append(X_test_group1)
    results['X_test_group3'].append(X_test_group3)
    results['y_test_group1'].append(y_test_group1)
    results['y_test_group3'].append(y_test_group3)
    results['top_features_baseline'].append(top_features_selected)
    results['rmse_cv_baseline'].append(rmse_cv)
    results['n_features_baseline'].append(n_features)
    results['y_test_pred_baseline_overall'].append(y_test_pred_overall)
    results['y_test_pred_baseline_group1'].append(y_test_pred_group1)
    results['y_test_pred_baseline_group3'].append(y_test_pred_group3)
    results['rmse_baseline_overall'].append(rmse_overall)
    results['rmse_baseline_group1'].append(rmse_group1)
    results['rmse_baseline_group3'].append(rmse_group3)
    results['top_features_sim_ml_group1'].append(top_features_selected_sim_ml_group1)
    results['top_features_sim_ml_group3'].append(top_features_selected_sim_ml_group3)
    results['n_features_sim_ml_group1'].append(n_features_sim_ml_group1)
    results['n_features_sim_ml_group3'].append(n_features_sim_ml_group3)
    results['rmse_cv_sim_ml_group1'].append(rmse_cv_sim_ml_group1)
    results['rmse_cv_sim_ml_group3'].append(rmse_cv_sim_ml_group3)
    results['y_test_pred_sim_ml_group1'].append(y_test_pred_sim_ml_group1)
    results['y_test_pred_sim_ml_group3'].append(y_test_pred_sim_ml_group3)
    results['y_test_pred_sim_ml_overall'].append(y_test_pred_sim_ml_overall)
    results['rmse_sim_ml_overall'].append(rmse_sim_ml_overall)
    results['rmse_sim_ml_group1'].append(rmse_sim_ml_group1)
    results['rmse_sim_ml_group3'].append(rmse_sim_ml_group3)
    results['top_features_tl'].append(top_features_selected_tl)
    results['rmse_cv_tl'].append(rmse_cv_tl)
    results['n_features_tl'].append(n_features_tl)
    results['y_test_pred_tl_overall'].append(y_test_pred_tl_overall)
    results['y_test_pred_tl_group1'].append(y_test_pred_tl_group1)
    results['y_test_pred_tl_group3'].append(y_test_pred_tl_group3)
    results['rmse_tl_overall'].append(rmse_tl_overall)
    results['rmse_tl_group1'].append(rmse_tl_group1)
    results['rmse_tl_group3'].append(rmse_tl_group3)

    # Save results to a file
    with open(f"sim_ml_generality_results/result_dv_sample_{seed1 + 1}_{seed2 + 1}.pkl", "wb") as f:
        pickle.dump(results, f)

# Define your seeds and partition them; you can partition it however you want.
n_partitions = 10
partition_number = 1

seeds1 = np.arange(100)
seeds2 = np.arange(100)
seeds1_partitions = np.array_split(seeds1, n_partitions)[partition_number - 1]

# we compare the performance of stadard rf, similarity-based ml, and tl for 10000 different subsets of the dataset
Parallel(n_jobs=-1, verbose=5)(
    delayed(parallelize_benchmarking)(seed1, seed2, data, X, 'log10visc experimental value', best_nn, scaler_Cv, Cv_feats) 
    for seed1 in seeds1_partitions
    for seed2 in seeds2
)
