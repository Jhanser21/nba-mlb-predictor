
import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("NBA & MLB Game Predictor")

uploaded_file = st.file_uploader("Upload your game stats CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview:", df.head())

    if 'home_win' in df.columns:
        X = df[['Points', 'Rebounds', 'Assists']]

      y = df["home_win"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))
        st.success(f"Model trained. Accuracy: {acc:.2%}")

        st.subheader("Predict New Game")
        input_data = {}
        for col in X.columns:
            input_data[col] = st.number_input(col, step=0.1)

        if st.button("Predict"):
            new_df = pd.DataFrame([input_data])
            pred = model.predict(new_df)[0]
            prob = model.predict_proba(new_df)[0]
            st.write(f"Prediction: {'Home Win' if pred == 1 else 'Away Win'}")
            st.write(f"Probabilities: Home {prob[1]:.2%} | Away {prob[0]:.2%}")
    else:
        st.error("CSV must contain a 'home_win' column.")
