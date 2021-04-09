import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Load Data From CSV Files
    
    Args:
        messages_filepath : Path to messages file
        categories_filepath : Path to categories file
    Returns:
        df : DataFrame after merging both messages ans categories file
    '''
   
    #Read csv file and load in the variable as dataframe
    messages = pd.read_csv(messages_filepath) 
    categories = pd.read_csv(categories_filepath)
    
    # merge messages and categories on 'id' column
    df = pd.merge(messages, categories, on=["id"], how='inner')
    
    return df


def clean_data(df):
    '''
    Cleaning the DataFrame
    Args:
        df : merged DataFrame returned by load_data function
    Returns:
        df : 
    '''
    # Creating Different column for different category
    categories = df['categories'].str.split(';',expand=True)
    # Changing the names of the newly created columns
    row = categories.iloc[0]
    category_colnames = [i[:-2] for i in row]
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: int(x[-1]))
    

    # Droping old category column of dataframe
    df.drop(columns=['categories'], inplace=True)

    # concatenating new category dataframe to df
    df = pd.concat([df,categories], axis=1)
    
    # Droping Rows with Duplicates Data
    df.drop_duplicates(inplace=True)
    # drop original column as it is not needed for the ML model
    df.drop(['original'], axis=1, inplace=True)
    # drop rows with NA
    df.dropna(inplace=True)
    
    return df


def save_data(df, database_filename):
    '''
    Save Cleaned Data to a SQLite DataBase
    Args:
        df : DataFrame which is to be Saved
        database_filename : Path to the DataBase File
    '''

    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('disaster_clean', engine, index=False, if_exists='replace')
    


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()