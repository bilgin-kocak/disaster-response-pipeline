import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet','averaged_perceptron_tagger','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import re
import pickle
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


def load_data(database_filepath):
    '''
    Function to load the Data From SQLite DataBase
    Args:
        database_filepath : path of the database file
    Returns:
        (X,Y,category_names)
        X : features DataFrame
        Y : labels DataFrame
        category_names : List of Column Names which contains different Category
    '''
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('disaster_clean',con=engine)
    #  define feature and target variables X and Y
    X = df['message'].values 
    y = df.loc[:, ~df.columns.isin(['id','message', 'original','genre'])]
    category_names = list(y.columns)
    
    return X, y, category_names


def tokenize(text):
    '''
    Function to tokenizing the taxt into words
    Args:
        text : text string which is needed to be tokenize
    Returns:
        tokens : List of word tokens
    '''
    #Regex to find urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Finds all urls from the provided text
    detected_urls = re.findall(url_regex, text)
    
    #Replaces all urls found with the "urlplaceholder"
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())    
        
    # Extracts the word tokens from the provided text    
    tokens = word_tokenize(text)
      
    # Remove stop words
    stop = stopwords.words("english")
    words = [t for t in tokens if t not in stop]
    
    #Lemmanitizer to remove inflectional and derivationally related forms of a word
    lemmatizer = WordNetLemmatizer()

    # Makes a list of clean tokens
    clean_tokens = []
    for tok in words:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Build the model required for training 
    Args:
        None
    Returns:
        
    '''
    # Creating Machine Learning Pipeline
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer = tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier(min_samples_split=50)) )
    ])


    parameters = {
        'clf__estimator__criterion':['gini','entropy'],
        'clf__estimator__min_samples_split':[10,110],
        'clf__estimator__max_depth':[None,100,500],
        }

    # Creating GridSearch object
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the model based on precision, recall and f1-score
    Args:
        model : object of the model
        X_test : Features for testing
        Y_test : Labels for testing
        category_names : List of Column Names which contains different Category
    '''
    # Predicted value of test data
    Y_test_predicted = model.predict(X_test)

    for i, col in enumerate(category_names):
        print('########################')
        print(col)
        print('########################')
        print(classification_report(Y_test[col], Y_test_predicted.T[i]))
        print('########################')


def save_model(model, model_filepath):
    '''
    Saving the model as pickle file
    Args:
        model : trained classifier object which is needs to be saved
        model_filepath : path of the file where you want to save the model
    '''
    # Dumping the created model to a file
    with open(model_filepath,'wb') as f:
        pickle.dump(model,f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()