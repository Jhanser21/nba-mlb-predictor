
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

    # ✅ Validate column structure
    required_cols = {'Points', 'Rebounds', 'Assists', 'home_win'}
    if not required_cols.issubset(df.columns):
        st.error(f"CSV must contain only these columns: {required_cols}")
        st.stop()

    # ✅ Make sure home_win is numeric
    if df['home_win'].dtype == object:
        df['home_win'] = df['home_win'].map({"Yes": 1, "No": 0})

    # ✅ Select numeric features only
    X = df[['Points', 'Rebounds', 'Assists']]
    y = df['home_win']

    # ✅ Split and train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    st.success(f"Model trained. Accuracy: {acc:.2%}")

    # ✅ New Prediction Input
    st.subheader("Predict New Game Outcome")
    input_data = {}
    for col in X.columns:
        input_data[col] = st.number_input(f"{col}", min_value=0.0, step=0.1)

    if st.button("Predict"):
        new_df = pd.DataFrame([input_data])
        pred = model.predict(new_df)[0]
        prob = model.predict_proba(new_df)[0]
        result = 'Home Win' if pred == 1 else 'Away Win'
        st.write(f"🔮 **Prediction:** {result}")
        st.write(f"📊 **Probabilities:** Home: {prob[1]:.2%}, Away: {prob[0]:.2%}")


        
