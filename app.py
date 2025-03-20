import streamlit as st
from recommender import ExerciseRecommender

st.title("Exercise Recommendation System")

recommender = ExerciseRecommender()

muscle_groups = recommender.df['muscle_gp'].unique().tolist()
equipment_list = recommender.df['Equipment'].unique().tolist()

muscle_group = st.selectbox("Select Muscle Group", muscle_groups)
equipment = st.selectbox("Select Equipment", equipment_list)

if st.button("Get Recommendation"):
    recommended_exercise, predicted_rating = recommender.recommend(muscle_group, equipment)
    if recommended_exercise != "No matching exercise found":
        st.write(f"Recommended Exercise: {recommended_exercise}")
        st.write(f"Predicted Rating: {predicted_rating:.2f}")
    else:
        st.write("No matching exercise found for the selected muscle group and equipment.")