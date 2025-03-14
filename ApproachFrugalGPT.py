# pip install azure-ai-inference
import os
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv, dotenv_values
import sys
import pandas as pd
from IPython.display import display
import numpy
from tqdm import tqdm
import shutil
import copy

sys.path.append("src/")
import FrugalGPT

load_dotenv()


supported_LLM = FrugalGPT.getservicename()

#print("supported LLMs:",supported_LLM)



def list_to_dataframe(data_list):
    # The first sublist is the header
    headers = data_list[0]
    # The rest are the data rows
    data = data_list[1:]
    # Create the dataframe
    df = pd.DataFrame(data, columns=headers)
    return df

def convert_and_merge_dataframes(train_df, test_df):
    def extract_last_query_part(query):
        # Split the query by '\n\n' and take the last part
        return query.split('\n\n')[-1]

    def create_converted_df(df, start_query_id=1):
        # Extract the new 'query' and keep 'ref_answer' the same
        df['new_query'] = df['query'].apply(extract_last_query_part)

        # Group by 'new_query' and 'ref_answer' to merge identical queries
        merged_df = df.groupby(['new_query', 'ref_answer'], as_index=False).first()

        # Create a new dataframe with the three columns
        converted_df = pd.DataFrame({
            'query': merged_df['new_query'],
            'ref_answer': merged_df['ref_answer'],
            'query_id': range(start_query_id, start_query_id + len(merged_df))
        })

        return converted_df

    # Convert and merge the train dataframe
    converted_train_df = create_converted_df(train_df)

    # Find the last query_id from the training data
    last_train_query_id = converted_train_df['query_id'].max()

    # Convert and merge the test dataframe, starting query_id from the last training id + 1
    converted_test_df = create_converted_df(test_df, start_query_id=last_train_query_id + 1)

    return converted_train_df, converted_test_df  


train_raw = FrugalGPT.loadcsvdata("data\\AGNEWS\\AGNEWS_train.csv")
test_raw = FrugalGPT.loadcsvdata("data\\AGNEWS\\AGNEWS_test.csv")
train_df = list_to_dataframe(train_raw)
test_df = list_to_dataframe(test_raw)
converted_train, converted_test = convert_and_merge_dataframes(train_df, test_df)
columns_to_save = ['query', 'ref_answer', 'query_id']
converted_train[columns_to_save].to_csv("data\\AGNEWS\\train.csv", index=False, header=False)
converted_test[columns_to_save].to_csv("data\\AGNEWS\\test.csv",index=False, header=False)


def generate_dataframe(service_names, train_data, test_data, genparams,db_path="db\\AGNEWS.sqlite",
                       max_workers=2):
    # Initialize an empty list to store the rows for the DataFrame
    data = []
    MyLLMforAll = FrugalGPT.LLMforAll(
                     db_path=db_path,
                     max_workers=max_workers,

)
    # Dictionary to keep track of markers for each provider
    provider_marker = {}

    # Iterate through the service names
    for name in service_names:
        # Extract provider and method
        provider = name.split('/')[0]
        method = name.split('/')[-1]

        # If the provider is seen for the first time, initialize its marker
        if provider not in provider_marker:
            provider_marker[provider] = 1
        else:
            provider_marker[provider] += 1
        # Get the completion batch for train and test data
        r1_train = MyLLMforAll.get_completion_batch(queries=train_data, genparams=genparams, service_name=name)
        r2_train = FrugalGPT.compute_score(r1_train)
        r1_test = MyLLMforAll.get_completion_batch(queries=test_data, genparams=genparams, service_name=name)
        r2_test = FrugalGPT.compute_score(r1_test)

        # Extract accuracy and cost
        train_acc = r2_train['em']
        train_cost = r2_train['cost']
        test_acc = r2_test['em']
        test_cost = r2_test['cost']

        # Create a row with the schema
        row = {
            "Test_acc": test_acc,
            "Test_cost": test_cost,
            "Test_size": len(test_data),
            "Train_acc": train_acc,
            "Train_cost": train_cost,
            "Train_size": len(train_data),
            "Budget": 10,
            "Method": method,
            "Provider": provider,
            "Marker": provider_marker[provider],
        }

        # Append the row to the data list
        data.append(row)

    # Create the DataFrame from the data list
    df = pd.DataFrame(data)

    return df


dataname = "AGNEWS"
service_names = [
    #'azure_openai/gpt-3.5-turbo',
    #'azure_llama/llama-3-70b-instruct',
    'azure_gpt4o/gpt-4o',
    'azure_mistral_nemo/mistral-nemo',
    'azure_ministral/ministral',
    'azure_gpt4o_mini/gpt-4o-mini',


    #'azure_cohere/command',
    #'azure_ai21/jamba-1.5-large',
    #'google/gemini-1.5-flash',
    #'google/gemini-1.5-pro',
    #'google/gemini-1.5-flash-8b',
    #'openaichat/gpt-4o-2024-08-06',
    #'openaichat/gpt-4o-mini',
    #'openaichat/gpt-4o',
    #'openaichat/gpt-4-turbo',

    #'togetherai/mistralai/Mistral-7B-Instruct-v0.1',
    #'togetherai/Qwen/Qwen2-72B-Instruct',
    #'togetherai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo',
    #'togetherai/meta-llama/Meta-Llama-3-70B-Instruct-Turbo',
    #'togetherai/google/gemma-2-9b-it',
    #'anthropic/claude-3-5-sonnet-20240620',
    #'anthropic/claude-3-haiku-20240307',
    #'ai21/jamba-1.5-mini',
                 ]
genparams=FrugalGPT.GenerationParameter(max_tokens=50, temperature=0.1, stop=['\n'])



test_data = FrugalGPT.loadcsvdata(f"data\\{dataname}\\train.csv")
prefix = open(f'config\\prompt\\{dataname}\\prefix_e8.txt').read()
test_data = FrugalGPT.formatdata(test_data,prefix)

train_data = FrugalGPT.loadcsvdata(f"data\\{dataname}\\test.csv")
prefix = open(f'config\\prompt\\{dataname}\\prefix_e8.txt').read()
train_data = FrugalGPT.formatdata(train_data,prefix)

sample_size = 100
individualmodel_df = generate_dataframe(service_names,
                                        train_data[0:sample_size], test_data[0:sample_size],
                                        genparams,
                                        db_path=f"db\\{dataname}.sqlite",
                                        max_workers=2)
display(individualmodel_df)
#individualmodel_df.to_csv(f"summary_{dataname}_e8_2024.csv")



def compute_tradeoffs(
    train_data,
                      budget_list,
                      name = "test",

                      service_names = [ 'azure_gpt4o/gpt-4o',
                                        #'azure_openai/gpt-3.5-turbo',
                                        #'azure_llama/llama-3-70b-instruct',
                                        'azure_mistral_nemo/mistral-nemo',
                                        'azure_ministral/ministral',
                                        'azure_gpt4o_mini/gpt-4o-mini',

                 ],
                      prefix="",
                      skip=0,
    MyCascade = FrugalGPT.LLMCascade(
          score_noise_injection=False,
  db_path="db\\AGNEWS.sqlite",
  ),

    cascade_depth=3,
  score_test_size=0.55,
                      ):

  for idx,budget in tqdm(enumerate(budget_list)):
    # train the model
    user_budget = budget
    try:
      MyCascade.load(loadpath=f"strategy/{name}/",budget=user_budget)
      print("Already trained. Skipped.")
      continue
    except:
      print("cannot find, start new training")
    if(idx<skip):
      continue
    if(idx==0):
        result = MyCascade.train(train_data[0:100],budget=user_budget,
                                 service_names=service_names,
                                 no_scorer_train=False,
                                 prefix=prefix,
                                 cascade_depth=cascade_depth,
  score_test_size=0.55,

                                 )
    else:
      result = MyCascade.train(train_data[0:100],budget=user_budget,
                               service_names=service_names,
                               no_scorer_train=True,
                               prefix=prefix,
                               cascade_depth=cascade_depth,
  score_test_size=0.55,

                               )
    MyCascade.save(savepath=f"strategy/{name}/")
  return


start_budget = 0.00002 #0.0000001 
end_budget = 0.0050 #0.00065
num_eval = 10

name = f'{dataname}_Model20252602'


budget_list = numpy.linspace(start_budget, end_budget, num_eval)
budget_list[0] = 0.00001
#budget_list = budget_list[::-1]
# load data
dev = FrugalGPT.loadcsvdata(f"data\\{dataname}\\train.csv")
train_data = FrugalGPT.formatdata(dev,prefix)
MyCascade= FrugalGPT.LLMCascade(
          score_noise_injection=False,
  db_path=f"db\\{dataname}.sqlite",
  batch_build=True,
  )

#MyCascade.load(loadpath=f"app/backend/strategy/{name}/",budget=0.0065) #0.00017


service_names_train = [
                        'azure_gpt4o/gpt-4o',
                        #'azure_llama/llama-3-70b-instruct',
                        #'azure_openai/gpt-3.5-turbo',
                        'azure_mistral_nemo/mistral-nemo',
                        'azure_ministral/ministral',
                        'azure_gpt4o_mini/gpt-4o-mini',
                 ]

compute_tradeoffs(train_data=train_data[0:100],
                  budget_list=budget_list,
                  name=name,
                  service_names=service_names_train,
                  prefix=prefix,
                  skip=0, # you can manually skip the first few budgets if they have already been trained.
                  MyCascade=MyCascade,
                  cascade_depth=3,
                  score_test_size=0.55,
                  )


# Specify the folder to zip
#folder_to_zip = f'strategy/{name}'
#output_zip_file = f'{name}.zip'

# Create the zip file
#shutil.make_archive(output_zip_file.replace('.zip', ''), 'zip', folder_to_zip)
#print(f"Folder '{folder_to_zip}' zipped as '{output_zip_file}'.")



def generate_dataframe_from_cascade(MyCascade,budget_list, train_data, test_data, genparams,name):
    # Initialize an empty list to store the rows for the DataFrame
    data = []

    # Iterate through the budget list
    for budget in tqdm(budget_list):
        # Load the strategy for the given budget
        MyCascade.load(loadpath=f"strategy/{name}/", budget=budget)

        # Get the completion batch for train data
        train_result = MyCascade.get_completion_batch(queries=train_data, genparams=genparams)

        # Compute the ACC and cost for train data
        train_acc_cost = FrugalGPT.compute_score(train_result)


        # Get the completion batch for test data
        test_result = MyCascade.get_completion_batch(queries=test_data, genparams=genparams)

        # Compute the ACC and cost for test data
        test_acc_cost = FrugalGPT.compute_score(test_result)

        # Create a row with the schema
        row = {
            "Test_acc": test_acc_cost['em'],
            "Test_cost": test_acc_cost['cost'],
            "Test_size": len(test_data),
            "Train_acc": train_acc_cost['em'],
            "Train_cost": train_acc_cost['cost'],
            "Train_size": len(train_data),
            "Budget": budget,
            "Method": "FrugalGPT",
            "Provider": "FrugalGPT",
            "Marker": 1,  # Marker is always 1 for this function
        }

        # Append the row to the data list
        data.append(row)
        display(row)

    # Create the DataFrame from the data list
    df = pd.DataFrame(data)

    return df

MyCascade_eval = FrugalGPT.LLMCascade()
MyCascade_eval.prefix = prefix
frugalgpt_df = generate_dataframe_from_cascade(MyCascade_eval,
                                               budget_list, train_data[0:100], test_data[0:100], genparams,
                                               name)
display(frugalgpt_df)
#frugalgpt_df.to_csv(f"summary_{dataname}_e8_frugalgpt_2024.csv")


individualmodel_df2 = copy.copy(individualmodel_df)
#individualmodel_df2['Test_cost'] = individualmodel_df2['Test_cost'] * individualmodel_df2['Test_size']
full_pd = pd.concat([frugalgpt_df,individualmodel_df2,])
#full_pd.to_csv(f"summary_{dataname}_e8_full_2024.csv")
display(full_pd)

