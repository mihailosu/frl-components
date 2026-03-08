import os
import datetime

def persist_model(model, model_name, experiment_name=None):
    '''
    Persist keras model to disk
    '''

    root_path = os.getcwd()

    if experiment_name is not None:
        final_path = os.path.join(root_path, "out", experiment_name)
    else:
        final_path = os.path.join(root_path, "out", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    if not os.path.exists(final_path):
        os.makedirs(final_path)

    path = os.path.join(final_path, f"{model_name}.keras")

    model.save(path)
    print(f"Model saved to {path}")



def persist_memory(memory_df, chosen_df, name, experiment_name=None):
    root_path = os.getcwd()

    if experiment_name is not None:
        final_path = os.path.join(root_path, "out", experiment_name)
    else:
        final_path = os.path.join(root_path, "out", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    if not os.path.exists(final_path):
        os.makedirs(final_path)

    path = os.path.join(final_path, f"{name}.csv")
    path_chosen = os.path.join(final_path, f'chosen_{name}.csv')

    # model.save(path)
    memory_df.to_csv(path)
    chosen_df.to_csv(path_chosen)
    print(f"Memory saved to {path}")
 