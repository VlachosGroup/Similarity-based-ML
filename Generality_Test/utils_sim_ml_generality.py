import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from scipy.spatial import Voronoi, voronoi_plot_2d
import dill
from matplotlib.ticker import MaxNLocator
import matplotlib
from joblib import Parallel, delayed
from aimsim.chemical_datastructures.molecule import Molecule
from rdkit import DataStructs
import concurrent.futures
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error

def generate_fingerprint(smiles,fp_type):
    """
    Input for fp_type should be:
    1) "morgan_fingerprint"
    2) "topological_fingerprint"
    3) "atom-pair_fingerprint"
    4) "torsion_fingerprint"
    5) "daylight_fingerprint"

    """
    mol = Molecule(mol_smiles=smiles)
    mol.set_descriptor(fingerprint_type=fp_type)
    fp = mol.get_descriptor_val().tolist()
    return fp

def generate_fingerprints(smiles_list, fp_type):
    fp_matrix = []
    for smiles in smiles_list:
        fp = generate_fingerprint(smiles, fp_type)
        fp_matrix.append(fp)
    fp_matrix = np.array(fp_matrix)

    return fp_matrix

def map_2D(fpmat):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(fpmat)

    return pca, reduced_data

def remove_corr_features(Xdata,corr_cutoff = 0.75):
    """
    This function will drop highly correlated features
    Output: a pd.Dataframe 
    """
    cor_matrix=Xdata.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool_))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_cutoff)]
    print(f"Dropped %d features with correlation coeff. > %0.2f" %(len(to_drop),corr_cutoff))

    Xdata=Xdata.drop(columns=to_drop,axis=1)
    print(f"Remaining features %d" %(Xdata.shape[1]))
    return Xdata

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn_extra.cluster import KMedoids 

def cluster(fingerprint_mat, n_clusters, method, **kwargs):

    """
    Methods are the following:
    kmeans
    kmedoids
    ward
    complete_linkage
    """
    if method == "kmeans":
        model = KMeans(
            n_clusters=n_clusters,
            max_iter=1000,
            random_state=42    
        )

        model.fit(fingerprint_mat)
        labels = model.labels_

        return model, labels
    if method == "kmedoids":
        model = KMedoids(
            n_clusters=n_clusters,
            metric="euclidean",
            random_state=42, 
            **kwargs
        )

        model.fit(fingerprint_mat)
        labels = model.labels_

        return model, labels
    
    elif method in ["complete_linkage", "complete"]:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="euclidean",
            linkage="complete", 
            **kwargs
        )

        model.fit(fingerprint_mat)
        labels = model.labels_

        return model, labels
    

    elif method in ["average", "average_linkage"]:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="euclidean",
            linkage="average", 
            **kwargs
        )

        model.fit(fingerprint_mat)
        labels = model.labels_

        return model, labels
    
    elif method in ["single", "single_linkage"]:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="euclidean",
            linkage="single",
            **kwargs
        )

        model.fit(fingerprint_mat)
        labels = model.labels_

        return model, labels
            
    elif method == "ward":
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="euclidean",
            linkage="ward",
            **kwargs
        )

        model.fit(fingerprint_mat)
        labels = model.labels_

        return model, labels

from sklearn.metrics import silhouette_score
def evaluate_clusters(fp_matrix, n_clusters, method, **kwargs):
    # Compute the sum of squared distances for each number of clusters
    silhouette = {}
    for k in range(2, n_clusters+1):  # This range can be changed based on your requirements        
        # Silhouette score is only defined for more than one cluster
        _, labels = cluster(fp_matrix, k, method, **kwargs)
        silhouette[k] = silhouette_score(fp_matrix, labels)

    # Plotting the Silhouette Method
    plt.figure(figsize = (8,8))
    plt.plot(list(silhouette.keys()), list(silhouette.values()), 'o-', markersize=13)
    plt.xlabel("Number of clusters", fontsize=25)
    plt.ylabel("Silhouette Score", fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    # Getting the current axes
    ax = plt.gca()

    # Setting the number of ticks on the x and y axes
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5)) # For x-axis, aiming for 5 ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5)) # For y-axis, aiming for 6 ticks 

    plt.show()
    print(f"The optimanl number of clusters is {np.where(list(silhouette.values()) == np.max(list(silhouette.values())))[0][0] + 2} with silhouette scorere of {np.max(list(silhouette.values()))} ")

    return list(silhouette.values())

def drop_columns_with_nans(df):
    """
    Drops columns of the DataFrame that contain NaN values.
    
    Args:
    - df (pd.DataFrame): The input DataFrame.
    
    Returns:
    - pd.DataFrame: A new DataFrame with columns containing NaN values dropped.
    """
    return df.dropna(axis=1, how='any')


import mordred
def drop_columns_with_mordred_missing(df):
    mask1 = (df.applymap(type) == mordred.error.Error).any()
    df = df.drop(columns=df.columns[mask1]) 
    mask2 = (df.applymap(type) == mordred.error.Missing).any()
    df = df.drop(columns=df.columns[mask2]) 

    return df 

def drop_non_num_cols(df):
    """
    Drops columns of a pandas DataFrame that contain non-integer and non-float values.

    Parameters:
    - df (pd.DataFrame): The input DataFrame

    Returns:
    - pd.DataFrame: A DataFrame with columns containing only integer and float values
    """
    # Identify columns that have any non-integer and non-float values
    cols_to_keep = df.columns[df.applymap(lambda x: isinstance(x, (int, float))).all()]

    # Filter the dataframe to keep only the desired columns
    return df[cols_to_keep]
    

def plot_parity_plot_test(y_test, y_pred_test):

    # Calculate R^2 and RMSE
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    max1 = np.max(y_test)
    max2 = np.max(y_pred_test)

    maximum = np.min([max1, max2])

    min1 = np.min(y_test)
    min2 = np.min(y_pred_test)
    
    minimum = np.min([min1, min2])

    # Print R^2 and RMSE

    print(f'R2 (Test): {r2_test}')
    print(f'RMSE (Test): {rmse_test}')
    
    # Plot predictions vs actual values
    plt.figure(figsize = [8,8])
    plt.scatter(y_test, y_pred_test, color = 'b', label = 'test')
    plt.plot([minimum - 10 , maximum + 10], [minimum - 10 ,maximum + 10], color='black')
    plt.xlim([minimum - 10 , maximum + 10])
    plt.ylim([minimum - 10, maximum + 10])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)        
    plt.show()


def RF(X, y, k, plot = True):
    # Assume X and y are your data and target values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Initialize the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model
    model.fit(X_train, y_train)

    # Predict the test set results
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate R^2 and RMSE

    r2_train = r2_score(y_train , y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Create a scorer for RMSE
    rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)

    # Using KFold with shuffling
    kf_shuffle = KFold(n_splits=k, shuffle=True, random_state=42)

    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=kf_shuffle, scoring=rmse_scorer)
    
    # Since we are using RMSE and we set greater_is_better=False, the scores will be negative. We negate them to get positive RMSE scores.
    scores = -scores
    print(scores)
    
    max1 = np.max(y_test)
    max2 = np.max(y_pred_test)
    max3 = np.max(y_train)
    max4 = np.max(y_pred_train)

    maximum = np.max([max1, max2, max3 ,max4])

    min1 = np.min(y_test)
    min2 = np.min(y_pred_test)
    min3 = np.min(y_train)
    min4 = np.min(y_pred_train)
    
    minimum = np.min([min1, min2, min3, min4])

    if plot == True:
        # Print R^2 and RMSE
        print(f'R2 (Train): {r2_train}')
        print(f'RMSE (Train): {rmse_train}')
        print(f'R2 (Test): {r2_test}')
        print(f'RMSE (Test): {rmse_test}')
        print(f'RMSE (CV) : {np.average(scores)}')
        
        # Plot predictions vs actual values
        plt.figure(figsize = [6,6])
        plt.scatter(y_train, y_pred_train, color = 'r', label = 'train')
        plt.scatter(y_test, y_pred_test, color = 'b', label = 'test')

        plt.plot([minimum-3, maximum+3], [minimum-3, maximum+3], color='black')
        plt.xlim([minimum-3, maximum+3])
        plt.ylim([minimum-3, maximum+3])  
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)       
        plt.legend(fontsize = 'x-large')
        plt.show()
    
    # return r2_train, r2_test, rmse_train, rmse_test, np.average(scores)
    return y_train,y_pred_train, y_test, y_pred_test

from  math import floor,ceil
def plot_histogram(data):
    Q1 = np.percentile(data, 25)
    Q2 = np.percentile(data, 50)  # This is also the median
    Q3 = np.percentile(data, 75)

    data_min = np.min(data)
    data_max = np.max(data)
    
    IQR = Q3 - Q1
    h = 2*IQR/122**(1/3)
    k = data_max - data_min


    bins = np.arange(floor(data_min)-0.5, ceil(data_max)+0.5, h) 
    plt.figure(figsize=(8,8))
    # data,bins,_ = plt.hist(error, bins=range(floor(error_min)-0.1, ceil(error_max)+0.1, 0.1))
    data,bins,_ = plt.hist(data, bins=bins)
    plt.show()

    return data, bins

def provide_stat(data):
    avg = np.mean(data)
    med = np.median(data)
    std = np.std(data)

    print(f"The average is {avg}")
    print(f"The median is {med}")
    print(f"The standard deviation is {std}")

    return avg, med, std

def get_non_na_values(df, col):
    indices = df[pd.notnull(df[col])].index
    df_non_na = df.iloc[indices] 

    return df_non_na, indices

def get_centroids(fp_mat_reduced, labels, unique_labels):
    centroids = []

    for label in unique_labels:
        centroid = np.average(fp_mat_reduced[np.where(labels == label)[0]], axis = 0).tolist()
        centroids.append(centroid)

    return np.array(centroids)

def plot_voronoi(cood, cluster, labels, cluster_type):
    color_map = {
    0: 'red', # group 1
    1: 'green', # GROUP 2
    2: 'blue', # group3
    3: 'purple', # GROUP 4
    4: 'yellow', #group 5
    5: 'brown', #group 6
    6: 'pink', #group 7
    7: 'orange', #group 8
    8: 'olive', #group 9
    9: 'cyan', #group 10

    }

    if cluster_type == "kmeans":
        colors= [color_map[label] for label in labels]
        centroids = cluster.cluster_centers_

        # Generate Voronoi regions
        vor = Voronoi(centroids)

        # Plot
        voronoi_plot_2d(vor, show_vertices=False, show_points=False)
        plt.scatter(cood[:, 0], cood[:, 1], c=colors)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, c='red')
        plt.show()

    else:
        colors= [color_map[label] for label in labels]
        centroids = get_centroids(cood, labels, list(set(labels)))

        # Generate Voronoi regions
        vor = Voronoi(centroids)

        # Plot
        voronoi_plot_2d(vor, show_vertices=False, show_points=False)
        plt.scatter(cood[:, 0], cood[:, 1], c=colors)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, c='red')
        plt.show()

    return centroids

def apply_voronoi(cood, cluster, labels, centroids):
    color_map = {
    0: 'red', # group 1
    1: 'green', # GROUP 2
    2: 'blue', # group3
    3: 'purple', # GROUP 4
    4: 'yellow', #group 5
    5: 'brown', #group 6
    6: 'pink', #group 7
    7: 'orange', #group 8
    8: 'olive', #group 9
    9: 'cyan', #group 10

    }

    colors= [color_map[label] for label in labels]
    
    # Generate Voronoi regions
    vor = Voronoi(centroids)

    # Plot
    voronoi_plot_2d(vor, show_vertices=False, show_points=False)
    plt.scatter(cood[:, 0], cood[:, 1], c=colors)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, c='red')
    plt.show()

def train_and_test_rf(X_train, y_train, X_test, y_test, descriptors, n_estimators = 100):
    try:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train[descriptors],y_train)
        ypred_test = model.predict(X_test[descriptors])
        plot_parity_plot_test(y_test , ypred_test)
    except:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        ypred_test = model.predict(X_test)
        plot_parity_plot_test(y_test , ypred_test)


    return model, ypred_test

from rdkit import Chem
from rdkit.Chem import Draw
from IPython.display import display

# Make sure to run this line to render RDKit drawings inline in the notebook
from rdkit.Chem.Draw import IPythonConsole

def draw_molecule_from_smiles(smiles_str):
    # Convert the SMILES string to a molecule object
    mol = Chem.MolFromSmiles(smiles_str)
    
    # If the conversion was successful, draw the molecule
    if mol:
        display(Draw.MolToImage(mol))
    else:
        print("Error: Couldn't convert SMILES string to molecule.")

def drop_duplicated_columns(df):
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def train_and_test_rf(X_train, y_train, X_test, y_test, descriptors, n_estimators = 100):
    try:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train[descriptors],y_train)
        ypred_test = model.predict(X_test[descriptors])
        # plot_parity_plot_test(y_test , ypred_test)
    except:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        ypred_test = model.predict(X_test)
        # plot_parity_plot_test(y_test , ypred_test)


    return model, ypred_test


def RF_(X, y, k, plot = True):
    # Assume X and y are your data and target values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Initialize the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model
    model.fit(X_train, y_train)

    # Predict the test set results
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate R^2 and RMSE

    r2_train = r2_score(y_train , y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Create a scorer for RMSE
    rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)

    # Using KFold with shuffling
    kf_shuffle = KFold(n_splits=k, shuffle=True, random_state=42)

    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=kf_shuffle, scoring=rmse_scorer)
    
    # Since we are using RMSE and we set greater_is_better=False, the scores will be negative. We negate them to get positive RMSE scores.
    scores = -scores
    # print(scores)
    
    max1 = np.max(y_test)
    max2 = np.max(y_pred_test)
    max3 = np.max(y_train)
    max4 = np.max(y_pred_train)

    maximum = np.max([max1, max2, max3 ,max4])

    min1 = np.min(y_test)
    min2 = np.min(y_pred_test)
    min3 = np.min(y_train)
    min4 = np.min(y_pred_train)
    
    minimum = np.min([min1, min2, min3, min4])

    if plot == True:
        # Print R^2 and RMSE
        print(f'R2 (Train): {r2_train}')
        print(f'RMSE (Train): {rmse_train}')
        print(f'R2 (Test): {r2_test}')
        print(f'RMSE (Test): {rmse_test}')
        print(f'RMSE (CV) : {np.average(scores)}')
        
        # Plot predictions vs actual values
        plt.figure(figsize = [6,6])
        plt.scatter(y_train, y_pred_train, color = 'r', label = 'train')
        plt.scatter(y_test, y_pred_test, color = 'b', label = 'test')

        plt.plot([minimum-5, maximum+5], [minimum-5, maximum+5], color='black')
        plt.xlim([minimum-5, maximum+5])
        plt.ylim([minimum-5, maximum+5])        
        plt.legend(fontsize = 'x-large')
        plt.show()
    
    return r2_train, r2_test, rmse_train, rmse_test, np.average(scores)


def getTopnFeatsRF_MDI(X,y,n):
    rf_model1 = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model1.fit(X, y)

    # Calculate Gini importance
    importance1 = rf_model1.feature_importances_

    # Normalize importance scores
    importance1 /= np.sum(importance1)

    # Sort the feature importance scores in descending order
    sorted_indices1 = np.argsort(importance1)[::-1]

    # Get the sorted feature names and their importance scores
    sorted_features = [(X.columns[i], importance1[i]) for i in sorted_indices1]
    
    topfeats = []
    importances = []
    # Print the top 5 feature names and their importance scores in descending order of importance
    for feature, importance in sorted_features[:n]:  # Limit to top 5
        # print(f"Feature: {feature}, Importance: {importance}")
        topfeats.append(feature)
        importances.append(importance)

    result = pd.DataFrame({'Topfeats':topfeats, 'MDI':importances})
    return result

def RF_multi(X, y, k, topfeats, numbers, plot = True, leg_log = 'center'):
  
    # initialize the lists
    R2_train = []
    RMSE_train = []
    R2_test = []
    RMSE_test = []
    RMSE_CV = []

    for i,number in enumerate(numbers):

        r2_train, r2_test, rmse_train, rmse_test, rmse_cv = RF_(X[topfeats['Topfeats'].iloc[:number]], y,k, plot)
        R2_train.append(r2_train)
        RMSE_train.append(rmse_train)
        R2_test.append(r2_test)
        RMSE_test.append(rmse_test)
        RMSE_CV.append(np.average(rmse_cv))
    
    if plot == True:
        matplotlib.rcParams['font.family'] = 'Helvetica'
        plt.figure(18, figsize = [8,8])
        plt.plot(numbers, RMSE_CV, '-o', label = 'RMSE (CV)')
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        ax = plt.gca()

        # Setting the number of ticks on the x and y axes
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5)) # For x-axis, aiming for 5 ticks
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5)) # For y-axis, aiming for 6 ticks
        plt.show()

    return R2_train, RMSE_train, R2_test, RMSE_test, RMSE_CV


def feat_importance_and_RFE(X_train, y_train, label, n):
    
    numbers_ar = np.arange(1,n)
    numbers = []
    for numb in numbers_ar:
        numbers.append(int(numb))
    
    topfeats = getTopnFeatsRF_MDI(X_train, y_train, X_train.shape[1])
    R2_train, RMSE_train, R2_test, RMSE_test, RMSE_CV = RF_multi(X_train, y_train, 10, topfeats, numbers, plot = False)
    n_features = np.where(RMSE_CV == np.min(RMSE_CV))[0][0] + 1
    topfeats = topfeats[topfeats[label] != 0]
    topfeats_selected = topfeats.iloc[:n_features]


    return topfeats, topfeats_selected, n_features, RMSE_CV


# Calculate R^2 and RMSE
def plot_parity_custom(y_test1, y_pred_test1, y_test3, y_pred_test3, start, end):
    matplotlib.rcParams['font.family'] = 'Helvetica'
    r2_test = r2_score(y_test1 + y_test3, y_pred_test1 + y_pred_test3)
    rmse_test = np.sqrt(mean_squared_error(y_test1 + y_test3, y_pred_test1 + y_pred_test3))
    
    # Print R^2 and RMSE

    print(f'R2 (Test): {r2_test}')
    print(f'RMSE (Test): {rmse_test}')
    
    # Plot predictions vs actual values
    plt.figure(figsize = [8,8])
    plt.scatter(y_test1, y_pred_test1, s = 70,  color = 'r', label = 'Group 1')
    plt.scatter(y_test3, y_pred_test3, s = 70, color = 'b', label = 'Group 3')
    plt.plot([start, end], [start, end], color='black')
    plt.xlim([start, end])
    plt.ylim([start, end])
    plt.xlabel("Log10(Experimental Viscosity)", fontsize = 25)
    plt.ylabel("Log10(Predicted Viscosity)", fontsize = 25)
    # plt.legend()
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)     

        # Getting the current axes
    ax = plt.gca()

    # Setting the number of ticks on the x and y axes
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5)) # For x-axis, aiming for 5 ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5)) # For y-axis, aiming for 6 ticks     
    plt.show()

def extract_features_nth_layer(model, data, n_th_layer):
    with torch.no_grad():
        features = data
        layer_count = 0  # Initialize a counter to track the number of linear layers processed

        # Process data through the model up to and including the nth "effective" layer
        for i, layer in enumerate(model):
            if isinstance(layer, torch.nn.Linear):
                layer_count += 1  # Increment layer counter for every Linear layer
                features = layer(features)  # Apply the linear layer

                # Check if the next layer is ReLU and we're not at the nth layer
                if layer_count < n_th_layer and (i+1 < len(model) and isinstance(model[i+1], torch.nn.ReLU)):
                    features = torch.relu(features)

                # Stop after the nth linear layer, skipping any ReLU that might follow
                if layer_count == n_th_layer:
                    break

        # Create feature names and build a DataFrame
        names = [f'feature {i+1}' for i in range(features.shape[1])]
        features_df = pd.DataFrame(features.numpy(), columns=names)  # Convert to DataFrame

        return features_df
