# recommender.py
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.losses import Huber

class ExerciseRecommender:
    def __init__(self):
        self.model = load_model('model/exercise_model.h5', custom_objects={'huber': Huber()})
        print("Model input shape:", self.model.input_shape) 
        
        with open('data/tokenizer.pkl', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
            print("Tokenizer loaded successfully with NumPy version:", np.__version__)
        
        with open('data/scaler.pkl', 'rb') as handle:
            self.scaler = pickle.load(handle)
            print("Scaler loaded successfully with NumPy version:", np.__version__)
        
        self.df = pd.read_excel('data/exercises.xlsx')
    
    def preprocess_input(self, muscle_group, equipment):
        muscle_seq = self.tokenizer.texts_to_sequences([muscle_group])[0]
        equip_seq = self.tokenizer.texts_to_sequences([equipment])[0]
        print(f"Tokenized muscle_group ({muscle_group}):", muscle_seq)
        print(f"Tokenized equipment ({equipment}):", equip_seq)
        max_len = 5
        muscle_seq = pad_sequences([muscle_seq], maxlen=max_len, padding='post')
        equip_seq = pad_sequences([equip_seq], maxlen=max_len, padding='post')
        input_data = np.hstack((muscle_seq, equip_seq))
        return input_data
    
    def recommend(self, muscle_group, equipment):
        input_data = self.preprocess_input(muscle_group, equipment)
        print("Shape of input_data:", input_data.shape)  
        predicted_rating = self.model.predict(input_data)
        predicted_rating = self.scaler.inverse_transform(predicted_rating)[0][0]
        
        temp_df = self.df.copy()
        temp_inputs_list = [self.preprocess_input(muscle_group, equipment) for _ in range(len(temp_df))]
        temp_inputs = np.vstack(temp_inputs_list) 
        print("Shape of temp_inputs:", temp_inputs.shape)  
        temp_ratings = self.model.predict(temp_inputs)
        temp_ratings = self.scaler.inverse_transform(temp_ratings)
        temp_df['Predicted_Rating'] = temp_ratings
        
        temp_df = temp_df[(temp_df['muscle_gp'] == muscle_group) & (temp_df['Equipment'] == equipment)]
        
        if not temp_df.empty:
            recommended_exercise = temp_df.loc[temp_df['Predicted_Rating'].idxmax()]
            actual_rating = recommended_exercise['Rating']
            mse_loss = Huber()
            mse_value = mse_loss([actual_rating], [recommended_exercise['Predicted_Rating']]).numpy()
            print(f"MSE (Huber) between predicted ({recommended_exercise['Predicted_Rating']}) and actual ({actual_rating}) rating: {mse_value}")
            return recommended_exercise['Exercise_Name'], recommended_exercise['Predicted_Rating']
        else:
            return "No matching exercise found", 0

if __name__ == "__main__":
    recommender = ExerciseRecommender()
    exercise, rating = recommender.recommend("chest", "dumbbell")
    print(f"Recommended exercise: {exercise} with predicted rating: {rating}")