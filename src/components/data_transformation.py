import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass # decorator used to create data class easily without __init__ , __repr__, __eq__
class DataTransformationConfig:
    #give a file name and create a file path for the preprocessor in artifacts directory
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        #got the data_tr.... to easily access the file path later on
        self.data_tranformation_config = DataTransformationConfig()


    def get_data_transformer_object(self):
        ''' 
        This function is responsible for data trasformation

        '''

        try:
            #the raw data is never useful directly, trying to transform it

            #splitting into numerical and catergorical columns
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            #numerical pipeline creation
            num_pipeline = Pipeline(  #pipeline will stitch the steps together (output of 1 as input to 2)
                steps = [
                ("imputer", SimpleImputer(strategy="median")), #to replace missing values with median values
                ("scaler", StandardScaler()) # scaling reduces the skewedness to we can make better models 
                ]
            )

            '''
            caterogical pipeline to 
            replace missing values with most frequent ones
            one hot encoder to make the several caterogical values into binary ones
            Scaler to scale but it seems useless in terms of categorical features
            '''
            cat_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                
                #scaling happens by removing mean and dividing by variance, with_mean=False instructs
                #StandardScaler to not center the data by removing the mean
                ]
            )

            #logging to check whether we have done our part with these two tpyes of columns, this will help in debugging
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            #to return preprocessor(all these changes on data) as object I will construct an obj
            preprocessor = ColumnTransformer(    #preprocessor object is to handle the two pipelines at once using ColumnTransformer
                [
                ("num_pipeline", num_pipeline, numerical_columns),  #we will pass the pipeline give it a name and the columns on which it need to act
                ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initiate_data_transformation(self, train_path, test_path):

        try:
            # reading the csv files at the test/train path
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            #logging for debugging purpose
            logging.info("read train and test data completed")

            logging.info("Obtaining preporcessing objects")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"] #I think this is not needed

            #to train data I will remove the target column seperately both  in test and train 
            input_feature_train_df = train_df.drop(columns = [target_column_name], axis=1) 
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = [target_column_name], axis=1) 
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe"
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            #now to combine the output columns back .c_ will stack the columns together

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path = self.data_tranformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                test_arr, train_arr,
                self.data_tranformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)