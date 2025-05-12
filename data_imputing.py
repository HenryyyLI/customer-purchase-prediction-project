from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier as knn_classifier
from sklearn.neighbors import KNeighborsRegressor as knn_re
from sklearn.metrics import root_mean_squared_error  as rmse
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np


class BaseKNNImputer:
    def __init__(self, n_max_neighbors=10):
        self.n_max_neighbors = n_max_neighbors

    def transform_columns(self, df_fix, visited_num_columns, unvisited_num_columns, visited_cat_columns, unvisited_cat_columns):  
        transformers = []

        # Add pipeline for lists which are not empty
        if visited_num_columns:
            transformers.append(
                ('num_vis', Pipeline(steps=[
                    ('scaler', StandardScaler()),
                ]), visited_num_columns)
            )
        
        if unvisited_num_columns:
            transformers.append(
                ('num_unvis', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                ]), unvisited_num_columns)
            )
        
        if visited_cat_columns:
            transformers.append(
                ('cat_vis', Pipeline(steps=[
                    ('encoding', OneHotEncoder(sparse_output=False, handle_unknown='error')),
                    ('scaler', StandardScaler()), 
                ]), visited_cat_columns)
            )
        
        if unvisited_cat_columns:
            transformers.append(
                ('cat_unvis', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoding', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
                    ('scaler', StandardScaler()),  
                ]), unvisited_cat_columns)
            )
        
        # Create the ColumnTransformer with only non-empty pipelines
        ct = ColumnTransformer(transformers=transformers)
        
        res = ct.fit_transform(df_fix) 
        return pd.DataFrame(res)

    def train_model(self, X_train, y_train, n_neighbors):
        raise NotImplementedError("This method should be implemented in subclasses")

    def select_best_model(self, X_train, X_test, y_train, y_test):
        raise NotImplementedError("This method should be implemented in subclasses")

    def impute(self, column_to_impute, df_mask, visited_num_columns, unvisited_num_columns, visited_cat_columns, unvisited_cat_columns, missing_indices):
        
        # for initial imputing
        if column_to_impute in unvisited_num_columns:
            unvisited_num_columns = [x for x in unvisited_num_columns if x != column_to_impute]
        elif column_to_impute in unvisited_cat_columns:
            unvisited_cat_columns = [x for x in unvisited_cat_columns if x != column_to_impute]
        
        # for following iterative reimputing when no missing data 
        elif column_to_impute in visited_num_columns:
            visited_num_columns = [x for x in visited_num_columns if x != column_to_impute]
        elif column_to_impute in visited_cat_columns:
            visited_cat_columns = [x for x in visited_cat_columns if x != column_to_impute]

        df_imputed_knn = df_mask.copy()

        # Split data into target and feature sets
        # Use precomputed missing indices
        target_indices = missing_indices[column_to_impute]
        df_target = df_imputed_knn.loc[target_indices]
        X_target = df_target.drop(columns=[column_to_impute])

        df_test = df_imputed_knn.drop(target_indices)
        y = df_test[column_to_impute]
        X = df_test.drop(columns=[column_to_impute])

        # Transform columns for both train and test sets
        X_target = self.transform_columns(X_target, visited_num_columns, unvisited_num_columns, visited_cat_columns, unvisited_cat_columns)
        X = self.transform_columns(X, visited_num_columns, unvisited_num_columns, visited_cat_columns, unvisited_cat_columns)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y if isinstance(self, KNNClassifierImputer) else None, shuffle=True)

        # Select the best model
        best_n = self.select_best_model(X_train, X_test, y_train, y_test)

        # Train on the full dataset
        model = self.train_model(X, y, n_neighbors=best_n)
        y_knn = model.predict(X_target)

        # Impute data
        df_imputed_knn.loc[target_indices, column_to_impute] = y_knn

        #print('SUCCESS!')
        return df_imputed_knn

class KNNClassifierImputer(BaseKNNImputer):
    def train_model(self, X_train, y_train, n_neighbors):
        return knn_classifier(n_neighbors=n_neighbors, weights='distance').fit(X_train, y_train)

    def select_best_model(self, X_train, X_test, y_train, y_test):
        f1_scores = {}
        for n in range(10, self.n_max_neighbors, 2):
            model = self.train_model(X_train, y_train, n_neighbors=n)
            y_pred_knn = model.predict(X_test)
            f1_scores[f'{n}'] = f1_score(y_test, y_pred_knn, average='macro')
        best_n = int(max(f1_scores, key=f1_scores.get))
        return best_n

class KNNRegressorImputer(BaseKNNImputer):
    def train_model(self, X_train, y_train, n_neighbors):
        return knn_re(n_neighbors=n_neighbors, weights='distance').fit(X_train, y_train)

    def select_best_model(self, X_train, X_test, y_train, y_test):
        rmse_scores = {}
        for n in range(10, self.n_max_neighbors, 2):
            model = self.train_model(X_train, y_train, n_neighbors=n)
            y_pred_knn = model.predict(X_test)
            rmse_scores[f'{n}'] = rmse(y_test, y_pred_knn)
        best_n = int(min(rmse_scores, key=rmse_scores.get))
        return best_n

class IterativeKNNImputer:
    def __init__(self, n_max_neighbors=10, epochs=2):
        self.n_max_neighbors = n_max_neighbors
        self.epochs = epochs

    def impute(self, df_mask, numerical_cols, categorical_cols):
        all_columns = numerical_cols + categorical_cols
        df_imputed = df_mask.copy()
        unvisited_num_columns = [i for i in numerical_cols]
        unvisited_cat_columns = [i for i in categorical_cols]

        visited_num_columns = []
        visited_cat_columns = []

        # Store missing indices
        missing_indices = {col: df_mask[df_mask[col].isnull()].index for col in all_columns}

        print(f"Epoch 1 started")

        # Fully impute numerical columns
        while len(unvisited_num_columns) > 0:
            column_to_impute = unvisited_num_columns[0]
            #print(f"Imputing numerical column: {column_to_impute}")
            knn_imputer = KNNRegressorImputer(n_max_neighbors=self.n_max_neighbors)
            df_imputed = knn_imputer.impute(column_to_impute, df_imputed, visited_num_columns, unvisited_num_columns, visited_cat_columns, unvisited_cat_columns, missing_indices)

            visited_num_columns.append(column_to_impute)
            unvisited_num_columns = [col for col in unvisited_num_columns if col != column_to_impute]

        # Fully impute categorical columns
        while len(unvisited_cat_columns) > 0:
            column_to_impute = unvisited_cat_columns[0]
            #print(f"Imputing categorical column: {column_to_impute}")
            knn_imputer = KNNClassifierImputer(n_max_neighbors=self.n_max_neighbors)
            df_imputed = knn_imputer.impute(column_to_impute, df_imputed, visited_num_columns, unvisited_num_columns, visited_cat_columns, unvisited_cat_columns, missing_indices)

            visited_cat_columns.append(column_to_impute)
            unvisited_cat_columns = [col for col in unvisited_cat_columns if col != column_to_impute]
        print(f"Epoch 1 finished")

        # Iterate further to recycle
        for epoch in range(1, self.epochs):
            print(f"Epoch {epoch+1} started")
            while len(all_columns) > 0:
                column_to_impute = all_columns[0]
                #print(f"Recycling column: {column_to_impute}")

                if column_to_impute in numerical_cols:
                    knn_imputer = KNNRegressorImputer(n_max_neighbors=self.n_max_neighbors)
                elif column_to_impute in categorical_cols:
                    knn_imputer = KNNClassifierImputer(n_max_neighbors=self.n_max_neighbors)

                # Update df and remove column from the list
                df_imputed = knn_imputer.impute(column_to_impute, df_imputed, visited_num_columns, unvisited_num_columns, visited_cat_columns, unvisited_cat_columns, missing_indices)
                all_columns = [x for x in all_columns if x != column_to_impute]

            print(f"Epoch {epoch+1} finished")
            all_columns = numerical_cols + categorical_cols

        return df_imputed