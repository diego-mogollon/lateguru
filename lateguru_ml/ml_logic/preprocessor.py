#This file focuses on scaling, feature concatenation, and PCA.

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, csr_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


'''

Below is the new pipeline approach, which needs to be tested

'''

#Preprocessing pipeline for cat, num and bi features
def create_preprocessing_pipeline(categorical_features, numeric_features, binary_features, apply_pca=False, n_components=10):
    #Scaling numeric features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    #One-hot encoding for categorical features
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    #Handling binary features as-is
    binary_transformer = 'passthrough'
    
    #Combine preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('bin', binary_transformer, binary_features)
        ]
    )
    
    if apply_pca:
        #Enable PCA if apply_pca=True
        preprocessor = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('pca', PCA(n_components=n_components))
        ])
    return preprocessor

#Apply pipeline
def preprocess_features(X, categorical_features, numeric_features, binary_features, apply_pca=False, n_components=10):
    preprocessor = create_preprocessing_pipeline(
        categorical_features, numeric_features, binary_features, apply_pca, n_components
    )
    
    # Fit and transform the data
    X_preprocessed = preprocessor.fit_transform(X)
    
    return X_preprocessed


'''

OLD CODE WITHOUT PIPELINE APPROACH HERE BELOW. WILL REMOVE ONCE TESTING THE PIPELINE APPROACH WORKS AS EXPECTED.

'''

# #Scale numeric features
# def scale_numeric_features(X_train, X_test, numeric_features):
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train[numeric_features])
#     X_test_scaled = scaler.transform(X_test[numeric_features])

#     return X_train_scaled, X_test_scaled

# #Combine categorical, scaled numeric, and binary features into single array
# def concatenate_features(X_train_encoded, X_train_scaled, X_train_binary, X_test_encoded, X_test_scaled, X_test_binary):
    
#     X_train_scaled_sparse = csr_matrix(X_train_scaled)
#     X_train_binary_sparse = csr_matrix(X_train_binary)
    
#     X_test_scaled_sparse = csr_matrix(X_test_scaled)
#     X_test_binary_sparse = csr_matrix(X_test_binary)
    
#     X_train_preprocessed = hstack([csr_matrix(X_train_encoded), X_train_scaled_sparse, X_train_binary_sparse])
#     X_test_preprocessed = hstack([csr_matrix(X_test_encoded), X_test_scaled_sparse, X_test_binary_sparse])
    
#     return X_train_preprocessed, X_test_preprocessed

# #Apply PCA to scaled numeric features
# def apply_pca(X_train_scaled, X_test_scaled, n_components=10):
#     pca = PCA(n_components=n_components)
#     X_train_pca = pca.fit_transform(X_train_scaled)    
#     X_test_pca = pca.transform(X_test_scaled)   

#     return X_train_pca, X_test_pca

# def preprocess_features(df):

#     #instantiating a standard scaler
#     scaler = StandardScaler()
#     #instantiating a PCA function with 10 principal components
#     pca = PCA(n_components=10)
#     #instantiating a OneHotEncoder, check on sparse output
#     ohe = OneHotEncoder(sparse_output=False, drop='if_binary')

#     #Defining subsets of X features based on data type
#     X_num = df.select_dtypes(include='number')
#     X_cat = df.select_dtypes(include='object')
#     X_binary = df.select_dtypes(include='bool')

#     #scaling numerical features and encoding categorical features
#     X_num_scaled = scaler.transform(X_num)
#     X_cat_encoded = ohe.transform(X_cat)
    
#     # Convert to sparse matrices
#     X_num_scaled_sparse = csr_matrix(X_num_scaled)
#     X_cat_encoded_sparse = csr_matrix(X_cat_encoded)
#     X_binary_sparse = csr_matrix(X_binary)

#     # Concatenate all preprocessed features
#     X_combined = hstack([X_cat_encoded_sparse, X_num_scaled_sparse, X_binary_sparse])

#     # Transform the data with PCA
#     X_pred = pca.fit_transform(X_combined.toarray())

#     #returning fully preprocessed features, output as a numpy array
#     return X_pred
