from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
from model.randomforest import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)


def load_data():
    #load the input data
    df = get_input_data()
    return  df

def preprocess_data(df):
    # De-duplicate input data
    df =  de_duplication(df)
    # remove noise in input data
    df = noise_remover(df)
    # translate data to english
    #df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())

    if 'y3' in df.columns:
        df['y2_y3']=df['y2'].astype(str)+'_'+df['y3'].astype(str)

    if 'y4' in df.columns:
        df['y2_y3_y4']=df['y2_y3'].astype(str)+'_'+df['y4'].astype(str)

    return df

def get_embeddings(df:pd.DataFrame):
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df

   

def get_data_object(X: np.ndarray, df: pd.DataFrame):
    data_splits={}
    data_splits['X_train'],data_splits['X_test'],data_splits['y_train'],data_splits['y_test']=train_test_split(X,df['y'],test_size=0.2,random_state=42)

    if 'y2' in df.columns:
        data_splits['y2_train'], data_splits['y2_test'] = train_test_split(df['y2'], test_size=0.2, random_state=42)
    else:
        data_splits['y2_train'], data_splits['y2_test'] = (None, None)

    if 'y2_y3' in df.columns:
        data_splits['y2_y3_train'], data_splits['y2_y3_test'] = train_test_split(df['y2_y3'], test_size=0.2, random_state=42)
    else:
        data_splits['y2_y3_train'], data_splits['y2_y3_test'] = (None, None)

    if 'y2_y3_y4' in df.columns:
        data_splits['y2_y3_y4_train'], data_splits['y2_y3_y4_test'] = train_test_split(df['y2_y3_y4'], test_size=0.2, random_state=42)
    else:
        data_splits['y2_y3_y4_train'], data_splits['y2_y3_y4_test'] = (None, None)

    return data_splits
    
def perform_modelling(data: Data, df: pd.DataFrame,name):
    model,accuracy,predictions=model_predict(data, df)
    return model,accuracy,predictions

if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)
    X, _ = get_embeddings(df)  #Make sure get_embeddings is set up to handle df correctly

    #Ensure the column name used in groupby exists in your DataFrame
    if 'y1' in df.columns:
        grouped_df = df.groupby('y1')
        final_acc = []
        for name, group_df in grouped_df:
            print(f'Processing group: {name}')
            X_group, _ = get_embeddings(group_df)  #Prepare embeddings for each group
            data_splits = get_data_object(X_group, group_df)  #Prepare data splits for this group
            results = model_predict(data_splits)
            for label, accuracy, _ in results:
                if label == "y":
                    continue
                #print(f'Accuracy for {label} in group {name}: {accuracy:.2f}')
                print(f'Accuracy for {label}: {accuracy:.3f}')
            accuracy_values = [accuracy for _, accuracy, _ in results]
            y2acc = accuracy_values[1]
            y3acc = accuracy_values[2]
            y4acc = accuracy_values[3]
            acc = ((100-y2acc)*0.0 + (y2acc-y3acc)*33.3 + (y3acc-y4acc)*67 + (y4acc)*100)/100 #Formula for 3-link chain model accuracy as defined in CA documentation
            print(f"The total accuracy of the chained RandomForest model for group: {name} is: {acc:.3f}")
            final_acc.append(acc)
            print("-------------")#Tidy up output
        final_acc = np.mean(final_acc)
        print(f"Total accuracy of the chained random forest model over both groups: {final_acc:.3f}") 
        
    else:
        print("""Error: 'y1' column not found in DataFrame. 
        DEBUG: TYPE1 Not found in .csv""")