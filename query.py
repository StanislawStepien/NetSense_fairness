from google.cloud import bigquery
import pandas as pd
import datetime
from deprecated import deprecated

# 0 records in the DB taking place before S1 date
S1_start = datetime.datetime(2011, 7, 1)  # 2_132_914 rows
S2_start = datetime.datetime(2012, 2, 1)  # 1_604_643 rows
S3_start = datetime.datetime(2012, 7, 1)  # 1_796_436 rows
S4_start = datetime.datetime(2013, 2, 1)  # 812_554 rows
S5_start = datetime.datetime(2013, 7, 1)  # 718_408 rows
S6_start = datetime.datetime(2014, 2, 1)  # 477_751 rows
S7_start = datetime.datetime(2014, 7, 1)  # There is still 33_156 records taking place after S7


def get_records_before_survey_was_completed_for_each_individual():
    """query1 = "SELECT * FROM `netsense-411221.NetSense.SurveyDate`"
    df = run_query_return_df(query1, allow_self_links=True)
    df.to_csv("all_survey_dates.csv", index=False)"""
    output = pd.DataFrame(
        columns=["EgoID", "Completed1", "Completed2", "Completed3", "Completed4", "Completed5", "Completed6"])
    df = pd.read_csv('all_survey_dates.csv')
    output["EgoID"] = df["EgoID"]
    output = output.drop_duplicates(subset='EgoID')
    for i, j in df.iterrows():
        if j['Completed_1'] != 'Nan':
            index = output[output["EgoID"] == j["EgoID"]].index
            a = output["Completed1"][index[0]]
            output = output.replace(a, j["Completed_1"])
        if j['Completed_2'] != 'Nan':
            index = output[output["EgoID"] == j["EgoID"]].index
            a = output["Completed2"][index[0]]
            output = output.replace(a, j["Completed_2"])
    print(output)

@deprecated
def get_queries_for_all_semesters():
    smaller_than_s1 = "SELECT * FROM `netsense-411221.NetSense.BehavioralAll` \
    WHERE DateTime < TIMESTAMP(DATETIME(2011, 7, 1, 0, 0, 0));"
    bigger_than_s1_smaller_than_s2 = "SELECT * FROM `netsense-411221.NetSense.BehavioralAll` \
    WHERE DateTime > TIMESTAMP(DATETIME(2011, 7, 1, 0, 0, 0)) AND DateTime < TIMESTAMP(DATETIME(2012, 2, 1, 0, 0, 0));"
    bigger_than_s2_smaller_than_s3 = "SELECT * FROM `netsense-411221.NetSense.BehavioralAll` \
    WHERE DateTime > TIMESTAMP(DATETIME(2012, 2, 1, 0, 0, 0)) AND DateTime < TIMESTAMP(DATETIME(2012, 7, 1, 0, 0, 0));"
    bigger_than_s3_smaller_than_s4 = "SELECT * FROM `netsense-411221.NetSense.BehavioralAll` \
    WHERE DateTime > TIMESTAMP(DATETIME(2012, 7, 1, 0, 0, 0)) AND DateTime < TIMESTAMP(DATETIME(2013, 2, 1, 0, 0, 0));"
    bigger_than_s4_smaller_than_s5 = "SELECT * FROM `netsense-411221.NetSense.BehavioralAll` \
    WHERE DateTime > TIMESTAMP(DATETIME(2013, 2, 1, 0, 0, 0)) AND DateTime < TIMESTAMP(DATETIME(2013, 7, 1, 0, 0, 0));"
    bigger_than_s5_smaller_than_s6 = "SELECT * FROM `netsense-411221.NetSense.BehavioralAll` \
    WHERE DateTime > TIMESTAMP(DATETIME(2013, 7, 1, 0, 0, 0)) AND DateTime < TIMESTAMP(DATETIME(2014, 2, 1, 0, 0, 0));"
    bigger_than_s6_smaller_than_s7 = "SELECT * FROM `netsense-411221.NetSense.BehavioralAll` \
    WHERE DateTime > TIMESTAMP(DATETIME(2014, 2, 1, 0, 0, 0)) AND DateTime < TIMESTAMP(DATETIME(2014, 7, 1, 0, 0, 0));"
    bigger_than_s7 = "SELECT * FROM `netsense-411221.NetSense.BehavioralAll` \
    WHERE DateTime > TIMESTAMP(DATETIME(2014, 7, 1, 0, 0, 0));"
    queries = [smaller_than_s1, bigger_than_s1_smaller_than_s2, bigger_than_s2_smaller_than_s3,
               bigger_than_s3_smaller_than_s4, bigger_than_s4_smaller_than_s5, bigger_than_s5_smaller_than_s6,
               bigger_than_s6_smaller_than_s7, bigger_than_s7]
    return queries


def run_query_return_df(query, allow_self_links=False) -> pd.DataFrame:
    CREDS = 'netsense-411221-be346442c94a.json'
    job_config = bigquery.LoadJobConfig(create_disposition=bigquery.CreateDisposition.CREATE_NEVER,
                                        write_disposition=bigquery.WriteDisposition.WRITE_EMPTY)
    client = bigquery.Client.from_service_account_json(json_credentials_path=CREDS)
    df = client.query_and_wait(query).to_dataframe()
    if not allow_self_links:  # in the whole dataset there are 4454 self-links
        df.drop(df.loc[df['ReceiverID'] == df['SenderID']].index, inplace=True)
    return df


def save_semesters_df_to_pickle():
    queries = get_queries_for_all_semesters()
    for i, query in enumerate(queries):
        print(f"{i + 1} running query: {query}\n")
        df = run_query_return_df(query)
        print("query run succesfully, saving to: semester_pickles folder\n")
        if i == 0:
            df.to_pickle(f'semester_pickles/before_s{i + 1}.pkl')
        elif i == 7:
            df.to_pickle(f'semester_pickles/after_s{i}.pkl')
        else:
            df.to_pickle(f'semester_pickles/before_s{i + 1}_after_s{i}.pkl')


#get_records_before_survey_was_completed_for_each_individual()
