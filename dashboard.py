import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from streamlit_option_menu import option_menu

# Load data
data = pd.read_csv(r"C:\Users\91876\Documents\dsa\Even sem project\daibetes detector\archive\diabetes.csv")

# Streamlit app
st.title('Diabetes Data Analysis')

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title='Main Menu',
        options=['Home', 'Outcome Count Plot', 'Box Plots', 'Pair Plot', 'Histograms', 'Correlation Heatmap', 'Model Performance'],
        icons=['house', 'bar-chart', 'box', 'scatter', 'bar-chart-line', 'heatmap', 'line-chart'],
        menu_icon='cast',
        default_index=0,
    )

if selected == 'Home':
    st.write("## Welcome to the Diabetes Data Analysis App")
    st.write("This app provides visualizations and analysis of the diabetes dataset.")
    st.write(data.head())

if selected == 'Outcome Count Plot':
    st.write("## Outcome Count Plot")
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Outcome', data=data)
    plt.title("Outcome Count Plot")
    st.pyplot(plt)

if selected == 'Box Plots':
    st.write("## Box Plots")
    plt.figure(figsize=(12, 12))
    for i, col in enumerate(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']):
        plt.subplot(3, 3, i + 1)
        sns.boxplot(x=col, data=data)
    st.pyplot(plt)

if selected == 'Pair Plot':
    st.write("## Pair Plot")
    sns.pairplot(data, hue='Outcome')
    st.pyplot(plt)

if selected == 'Histograms':
    st.write("## Histograms")
    plt.figure(figsize=(12, 12))
    for i, col in enumerate(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']):
        plt.subplot(3, 3, i + 1)
        sns.histplot(x=col, data=data, kde=True)
    st.pyplot(plt)

if selected == 'Correlation Heatmap':
    st.write("## Correlation Heatmap")
    plt.figure(figsize=(12, 12))
    sns.heatmap(data.corr(), vmin=-1.0, center=0, cmap='RdBu_r', annot=True)
    st.pyplot(plt)

if selected == 'Model Performance':
    st.write("## Model Performance")

    # Standard scaling
    sc_x = StandardScaler()
    X = pd.DataFrame(sc_x.fit_transform(data.drop(["Outcome"], axis=1)),
                     columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    y = data['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=True)

    test_scores = []
    train_scores = []

    for i in range(1, 15):
        knn = KNeighborsClassifier(i)
        knn.fit(X_train, y_train)
        train_scores.append(knn.score(X_train, y_train))
        test_scores.append(knn.score(X_test, y_test))

    max_train_score = max(train_scores)
    train_scores_index = [i for i, v in enumerate(train_scores) if v == max_train_score]
    st.write("Max train Score {}% and k={}".format(max_train_score * 100, list(map(lambda x: x + 1, train_scores_index))))

    max_test_score = max(test_scores)
    test_scores_index = [i for i, v in enumerate(test_scores) if v == max_test_score]
    st.write("Max test Score {}% and k={}".format(max_test_score * 100, list(map(lambda x: x + 1, test_scores_index))))

    plt.figure(figsize=(12, 5))
    sns.lineplot(x=range(1, 15), y=train_scores, marker='*', label="Train score")
    sns.lineplot(x=range(1, 15), y=test_scores, marker='o', label="Test score")
    st.pyplot(plt)

    knn = KNeighborsClassifier(13)
    knn.fit(X_train, y_train)
    st.write("Model accuracy on test data: ", knn.score(X_test, y_test))

    y_pred = knn.predict(X_test)
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))

    st.write("Classification Report:")
    st.write(classification_report(y_test, y_pred))
