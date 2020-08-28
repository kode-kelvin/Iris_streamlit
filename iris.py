# lets import the needed

import streamlit as st  # our lib for the app
import pandas as pd  # our package
from PIL import Image, ImageEnhance, ImageFilter
from bokeh.models.widgets import Div


# for graph
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# need for data set
from sklearn.datasets import load_iris

# our model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    # our app --- - ---- -- -----------

    st.title('WEB APP FOR IRIS PREDICTIONS')  # title of app
    st.subheader('Built by KODE-KELVIN')  # sub-header

    st.write('''

        This simple  Web App predicts **Iris flower type**. This was built using Python Streeamlit framework
        ''')

    # load data and all
    iris_dataset = load_iris()

    # specify X,y
    iris_data = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)
    iris_target = pd.DataFrame(iris_dataset.target, columns=['target'])

    X = iris_data
    y = iris_target

    # for the model
    Xi = iris_dataset.data
    yi = iris_dataset.target

    # getting all dataframe
    all_data = pd.concat([X, y], axis=1)

    Lreg = LogisticRegression()  # our model
    # lets train our model with our training set
    Xi_train, Xi_test, yi_train, yi_test = train_test_split(Xi, yi, test_size=0.2, random_state=1, stratify=y)

    Xi_train, Xi_test, yi_train, yi_test = train_test_split(Xi, yi, test_size=0.2, random_state=1, stratify=y)

    Lreg.fit(Xi_train, yi_train)
    x_pred = Lreg.predict(Xi_test)

    # target names -- so we understand what the numbers mean
    species = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}  # dict

    # now lets map it to the data frame
    all_data['target name'] = all_data.target.map(species)

    # load to display all data
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # loading the css

    def load_css(file_name):
        with open(file_name) as f:
            st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

    #..................................

    #.....................................

    # load our css and add css style to our text
    load_css('style.css')

    minitxt = "<div ><span class = 'highlight' > Iris Datasets </span></div>"
    st.markdown(minitxt, unsafe_allow_html=True)

    boldtext = """
    <div style="background-color:teal;padding:10px">
    <h2 style="color:white;text-align:center;">Data Exploration Analysis </h2>
    </div>
    """
    st.markdown(boldtext, unsafe_allow_html=True)

    # explore the meaning of target

    # view all data frame -- EDA
    select = ['View Dataset', 'View Head', 'View Tail']
    Activity = st.selectbox('Data Over-view', select)
    if st.checkbox('Show target meaning'):
        st.text('0 is setosa, 1 is versicolor, 2 is virginica')
        st.write(species)

    if Activity == 'View Dataset':
        st.write(all_data)
    if Activity == 'View Head':
        st.write(all_data.head())
    if Activity == 'View Tail':
        st.write(all_data.tail())
    else:
        pass

    # lets explore more about the dataset --- using multiselect
    explore = ['Columns', 'Shape', 'Show Stats', 'Data Types']
    to_explore = st.multiselect('Explore Dataset', explore)
    if 'Columns' in to_explore:
        st.write(all_data.columns)
    if 'Show Stats' in to_explore:
        st.write(all_data.describe())
    if 'Shape' in to_explore:
        st.write(all_data.shape)
    if 'Data Types' in to_explore:
        st.write(all_data.dtypes)

    # to look at the columns each -- we can do this with if statement but more codes required
    if st.checkbox('Explore Columns'):
        all_colx = list(all_data.columns)
        selected_colx = st.multiselect('Select', all_colx)
        col_df = all_data[selected_colx]
        cho = st.dataframe(col_df)
        if 'target name' in selected_colx:
            st.write(all_data.iloc[:, -1].value_counts())  # this will show value count in target colums

    # display plots

    st.write('''
        # Graphical Presentation
        ''')

    # show correlation
    if st.checkbox('Show correlation'):

        st.write(
            sns.heatmap(all_data.corr(), annot=True, cmap='cubehelix', cbar_kws={'orientation': 'vertical'}, cbar=True, mask=False, linewidths=1, linecolor='k', square=True, vmin=-1, vmax=1));
        st.pyplot()

     # lets look at relationship btw sepalwidth and sepallength
    # using seaborn and displaying various kinds
    xgraph = 'sepal length (cm)'
    ygraph = 'sepal width (cm)'

    rel = ['None', 'Scatter', 'Hex', 'Reg', 'KDE']
    rela = st.radio('Relationship Graph', rel)
    if rela == 'None':
        st.write()
    if rela == 'Scatter':
        st.write(sns.jointplot(xgraph, ygraph, data=all_data)
                 )
        st.pyplot()
    if rela == 'Hex':
        st.write(sns.jointplot(xgraph, ygraph, data=all_data, kind='hex')
                 )
        st.pyplot()
    if rela == 'Reg':
        st.write(sns.jointplot(xgraph, ygraph, data=all_data, kind='reg'))
        st.pyplot()

    if rela == 'KDE':
        st.write(sns.jointplot(xgraph, ygraph, data=all_data, kind='kde'))
        st.pyplot()

    else:
        pass

    # create sidebar and create user input
    st.sidebar.header('ENTER PARAMETERS')

    def user_parameters():
        # the values we are using are values in the iris dataframe
        SepalLength = st.sidebar.slider('Sepal Length (cm)', 4.3, 7.9, 5.4)
        SepalWidth = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.4, 3.4)
        PetalLength = st.sidebar.slider('Petal Length (cm)', 1.0, 6.9, 1.3)
        PetalWidth = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 0.2)
        # create dic to hold the input
        userInput = {'SepalLength': SepalLength,
                     'SepalWidth': SepalWidth,
                     'PetalLength': PetalLength,
                     'PetalWidth': PetalWidth}
        # we store the data into dataframe
        userInputData = pd.DataFrame(userInput, index=[0])
        return userInputData

    df = user_parameters()

    st.subheader('Your Parameters')
    st.write(df)

    predictionDf = Lreg.predict(df)  # to predict the user input
    pred_proba = Lreg.predict_proba(df)

    # lets display what our result means
    st.subheader('Classification and their corresponding number')
    st.write(iris_dataset.target_names)

    # lets print result
    st.subheader('Our Predictions')
    ch = st.write(iris_dataset.target_names[predictionDf])

    # show
    st.subheader('prediction_probability')
    st.write(pred_proba)


# About section -------------
    st.sidebar.subheader('ABOUT APP')
    option_abt = ['About Dataset', 'About Me']

    #select_opt = st.sidebar.checkbox('ABOUT', option_abt)
    if st.sidebar.checkbox(''):
        select_opt = st.sidebar.radio('', option_abt)
        if select_opt == 'About Dataset':
            st.write('''

            **About Iris Dataset**

            **Context**

            The Iris flower data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems. It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species. The data set consists of 50 samples from each of three species of Iris (Iris Setosa, Iris virginica, and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.


            **Content**

            The dataset contains a set of 150 records under 5 attributes - Petal Length, Petal Width, Sepal Length, Sepal width and Class(Species).

            **Acknowledgements**

            This dataset is free and is publicly available at the UCI Machine Learning Repository
            ''')
        else:  # about me and image manipulation
            if st.checkbox('Click To See Me'):
                img = Image.open('IMG_1681.jpg')
                enh = ImageEnhance.Contrast(img)
                img_wth = st.slider('Set Width', 200, 300)
                #heigth_wdth = st.slider('Set Heigth', 100, 300)
                number = st.slider('Enhance Image', 1.0, 5.0)
                st.image(enh.enhance(number), width=img_wth)

            st.write('''

                **About Me**

                My name is Kelvin E. Obed, BSc, MSc
                I am a self UIUX Designer with a passion for User-Centered designs.
                I am fueled by passion to understand how data drives company growth. I consider myself a forever student; eager to both build
                on my academic foundation in Computer Science and Entrepreneurship and stay in tune with the latest strategies and technologies
                that impact company growth.

                ''')
        # create a button to link to other pages

            if st.button('GitHub'):
                js = "window.open('https://github.com/kode-kelvin')"  # New tab or window
                # js = "window.location.href = 'https://www.streamlit.io/'"  # Current tab
                html = '<img src onerror="{}">'.format(js)
                div = Div(text=html)
                st.bokeh_chart(div)

            if st.button('LinkedIn'):
                js = "window.open('https://linkedin.com/in/kelvin-obed-565877171')"  # New tab or window
                # js = "window.location.href = 'https://www.streamlit.io/'"  # Current tab
                html = '<img src onerror="{}">'.format(js)
                div = Div(text=html)
                st.bokeh_chart(div)


if __name__ == '__main__':
    main()
