from model.randomforest import RandomForest
from sklearn.metrics import accuracy_score,classification_report


def model_predict(data_splits):
    results = []
    print("RandomForest processing")
    embeddings = data_splits['X_train']  # Assuming embeddings are correctly assigned

    model = RandomForest("RandomForest", embeddings)

    # Iterate over training/testing data
    for label in ['y_train', 'y2_train', 'y2_y3_train', 'y2_y3_y4_train']:
        if label in data_splits and data_splits[label] is not None:
            X_train = data_splits['X_train']
            y_train = data_splits[label]
            X_test = data_splits['X_test']
            y_test = data_splits[label.replace('train', 'test')]

            model.train(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = model.print_results(predictions, y_test)
            results.append((label.replace('_train', ''), accuracy, predictions))
    return results
    

def model_evaluate(model, data):
    model.print_results(data)
