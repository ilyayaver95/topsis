
import random
import numpy as np               # for linear algebra
import pandas as pd              # for tabular output
import time
from scipy.stats import rankdata # for ranking the candidates

# data = pd.read_csv(r"C:\Users\IlyaY\Desktop\לימודים\תשפג\א\סייבר\project\data\less_data.csv")
# data["years_experience"] = 2022 - data.Grd_yr
# print(data.columns)
# print(data.years_experience)

# # experience frequency
# plt.figure(figsize=(14,6))
# plt.hist(data.years_experience, edgecolor="k",density=True)
# plt.xlabel("years_experience")
# plt.ylabel("Frequency")
# plt.show()

# # experience frequency male\female
# plt.figure(figsize=(14,6))
# plt.hist(data[data["gndr"]=="F"]["years_experience"], edgecolor="k",density=True, alpha=0.7, label = "Female")
# plt.hist(data[data["gndr"]=="M"]["years_experience"], edgecolor="k",density=True, alpha=0.7, label = "Male")
# plt.xlabel("years_experience")
# plt.ylabel("Frequency")
# plt.legend()
# plt.show()

# # Cred by frequency
# data['Cred'].value_counts().plot(kind='bar')
# plt.show()

# # pri_spec by frequency
# data['pri_spec'].value_counts().plot(kind='bar')
# plt.show()

# # Med_sch by frequency
# data = data[data["Med_sch"] != 'OTHER']
# data['Med_sch'].value_counts().plot(kind='bar')
# plt.show()


def smart(raw_data, candidates):
    # Set the criteria and weights for evaluating candidates
    criteria = raw_data.index
    weights = [0.4, 0.1, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05]

    # normalize data
    raw_data = (raw_data - raw_data.mean()) / raw_data.std()
    raw_data['cred_number'] = raw_data['cred_number'].fillna(0)
    # raw_data = raw_data * 10

    benefit_attributes = [1, -1, 1, 1, 1, 1, 1, -1]

    # Initialize a dictionary to store the scores of each candidate
    score_total = {}

    # Iterate over the candidates
    for inx, cand in enumerate(candidates):
        # Initialize the score for this candidate to 0

        for inx, cand in enumerate(candidates):
            score = 0
            for index_of_col, col in enumerate(raw_data.columns):
                score += (raw_data.iloc[inx][col] * weights[index_of_col]) * benefit_attributes[index_of_col]
            score_total[cand] = score


    # Sort the scores dictionary by value in descending order
    sorted_scores = sorted(score_total.items(), key=lambda x: x[1], reverse=True)

    # Return the name of the candidate with the highest score
    return sorted_scores[0][0]

def borda(raw_data, candidates):

    # Define the weights for each attribute
    # (negative attributes should have a negative weight)

    weights = [0.4, 0.1, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05]

    #normalize data
    raw_data = (raw_data - raw_data.mean()) / raw_data.std()
    raw_data['cred_number'] = raw_data['cred_number'].fillna(0)

    benefit_attributes = [1, -1, 1, 1, 1, 1, 1, -1]

    # Calculate the Borda count for each option
    borda_counts = {}  # pd.DataFrame(index=candidates)
    for inx, cand in enumerate(candidates):
        borda_count = 0
        for index_of_col, col in enumerate(raw_data.columns):
            borda_count += (raw_data.iloc[inx][col] * weights[index_of_col]) * benefit_attributes[index_of_col]
        borda_counts[cand] = borda_count

    # Sort the options by Borda count in descending order
    sorted_options = sorted(borda_counts.items(), key=lambda x: x[1], reverse=True)

    # print(f"The best candidate/alternative according to C* is {int(np.max(sorted_options))}")
    return sorted_options[0][0]  # return int(np.max(sorted_options))



def rank_according_to(data, candidates):
    candidates = candidates.to_numpy()
    ranks = rankdata(data).astype(int)
    ranks -= 1
    return candidates[ranks][::-1]

def topsis_gpt(decision_matrix, impact=np.array([1, 1, 1])):
    weights = np.array([0.5, 0.2, 0.3])

    # Normalize the decision matrix
    normalized_matrix = decision_matrix / np.linalg.norm(decision_matrix)

    # Weight the normalized matrix
    weighted_normalized_matrix = normalized_matrix * weights

    # Find the positive and negative ideal solutions
    positive_ideal_solution = np.amax(weighted_normalized_matrix, axis=0)
    negative_ideal_solution = np.amin(weighted_normalized_matrix, axis=0)

    # Calculate the separation between the alternatives and the ideal solutions
    positive_separation = np.sqrt(np.sum((weighted_normalized_matrix - positive_ideal_solution) ** 2, axis=1))
    negative_separation = np.sqrt(np.sum((weighted_normalized_matrix - negative_ideal_solution) ** 2, axis=1))

    # Calculate the relative closeness to the ideal solution
    relative_closeness = positive_separation / (positive_separation + negative_separation)

    # Multiply the relative closeness by the impact to get the final score
    score = relative_closeness * impact

    # Return the index of the alternative with the highest score
    print(f"The best candidate/alternative according to C* is {np.argmax(score)}" )
    # print("The preferences in descending order are " + ", ".join(cs_order) + ".")
    # return np.argmax(score)


def topsis(raw_data, candidates):
    attributes = np.array(["cred_number",
                           "Salary",
                           "years_experience",
                           "patient_review",
                           "is_served_army",
                           "num_of_days_available",
                           "number_of_languages",
                           "is_need_parking"])  # patient_review, is_served_army, num_of_days_available, number_of_languages, is_need_parking
    # candidates = selected_data['NPI']
    raw_data = np.array(raw_data)

    weights = np.array([0.4, 0.1, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05])

    #   from here :
    # The indices of the attributes (zero-based) that are considered beneficial.
    # Those indices not mentioned are assumed to be cost attributes.
    benefit_attributes = set([0, 2, 3, 4, 5, 6])

    # Display the raw data we have
    d = pd.DataFrame(data=raw_data, index=candidates, columns=attributes)
    print(d)
    print("")

    # Step 1 - Normalizing the ratings
    m = len(raw_data)
    n = len(attributes)
    divisors = np.empty(n)
    for j in range(n):
        column = raw_data[:, j]
        divisors[j] = np.sqrt(column @ column)

    raw_data = raw_data / divisors

    columns = ["X_{%d}" % j for j in range(n)]
    dd = pd.DataFrame(data=raw_data, index=candidates, columns=columns)
    print(dd)
    print("")

    # Step 2 - Calculating the Weighted Normalized Ratings

    raw_data *= weights
    ddd = pd.DataFrame(data=raw_data, index=candidates, columns=columns)
    print(ddd)
    print("")

    # Step 3 - Identifying PIS ( A∗ ) and NIS ( A− )
    a_pos = np.zeros(n)
    a_neg = np.zeros(n)
    for j in range(n):
        column = raw_data[:, j]
        max_val = np.max(column)
        min_val = np.min(column)

        # See if we want to maximize benefit or minimize cost (for PIS)
        if j in benefit_attributes:
            a_pos[j] = max_val
            a_neg[j] = min_val
        else:
            a_pos[j] = min_val
            a_neg[j] = max_val

    dddd = pd.DataFrame(data=[a_pos, a_neg], index=["A^*", "A^-"], columns=columns)
    print(dddd)
    print(" ")

    # Step 4 and 5 - Calculating Separation Measures and Similarities to PIS

    sp = np.zeros(m)
    sn = np.zeros(m)
    cs = np.zeros(m)

    for i in range(m):
        diff_pos = raw_data[i] - a_pos
        diff_neg = raw_data[i] - a_neg
        sp[i] = np.sqrt(diff_pos @ diff_pos)  # because it is sequence
        sn[i] = np.sqrt(diff_neg @ diff_neg)  # because it is sequence
        cs[i] = sn[i] / (sp[i] + sn[i])

    ddddd = pd.DataFrame(data=zip(sp, sn, cs), index=candidates, columns=["S^*", "S^-", "C^*"])
    print(ddddd)
    print(" ")

    # Step 6 - Ranking the candidates/alternatives
    cs_order = rank_according_to(cs, candidates)  # here crushed
    sp_order = rank_according_to(sp, candidates)
    sn_order = rank_according_to(sn, candidates)

    dddddd = pd.DataFrame(data=zip(cs_order, sp_order, sn_order), index=range(1, m + 1), columns=["C^*", "S^*", "S^-"])
    print(dddddd)
    print(" ")

    # best_cand = raw_data[index == cs_order[0]]
    return cs_order[0]  # best NPI

    # print(f"The best candidate/alternative according to C* is {best_cand}" )
    #print(f"The preferences in descending order are , {cs_order}")


def prepare_data(path):

    data = pd.read_csv(path)  #, encoding = 'unicode_escape')
    data = data[data['Cred'].notna()]  # remove candidates with no Cred info
    data["years_experience"] = 2022 - data.Grd_yr  # Create column of experience based on graduation year
    new_data = pd.DataFrame()

    for spec in data["pri_spec"].unique():  # create df with only 10 candidates per spec
        temp = data[data["pri_spec"] == spec]
        temp = temp[:10]

        new_data = new_data.append(temp)


    # gender_filter = 'M'  # filter of gender to be applayed in app
    # spec_filter = 'CHIROPRACTIC'  # filter of specification to be applayed in app
    selected_data = new_data

    cred_dict = {'AU': 7, '5': 3, '2': 6, 'CSW': 4,  # Create  dict for credit (ranked by value)
                 'DC': 13, 'DDS': 14, 'DO': 18, 'DPM': 16,
                 'MD': 19, 'MNT': 10, 'NP': 7, 'OD': 14,
                 'PSY': 18, 'PT': 10, 'CP': 7,
                 }

    selected_data["cred_number"] = selected_data["Cred"]  # Create column for cred hyrarchy number
    selected_data["cred_number"] = selected_data["cred_number"].map(cred_dict)

    selected_data["patient_review"] = selected_data["Cred"]  # adding attribute of ranking "patient_review"
    selected_data["is_served_army"]= selected_data["Cred"]  # adding attribute of ranking "is_served_army"
    selected_data["num_of_days_available"] = selected_data["Cred"]  # adding attribute of ranking "num_of_days_available"
    selected_data["number_of_languages"]= selected_data["Cred"]  # adding attribute of ranking "number_of_languages"
    selected_data["is_need_parking"]= selected_data["Cred"]  # adding attribute of ranking "number_of_languages"

    selected_data["Salary"] = selected_data["cred_number"]  # Create salary column -  to be calculated next
    selected_data["Full Name"] = selected_data["frst_nm"]  # Create Full Name column -  to be calculated next

    for index, row in selected_data.iterrows():  # Generate the new columns based on other columns
        selected_data["Salary"][index] = selected_data["Salary"][index] * 500 + 10000 + random.randint(1, 9) * 1000 +  selected_data["years_experience"][index] * 100 # salary calc Todo: make more reasonable calc, and save to csv
        selected_data["Full Name"][index] = selected_data["frst_nm"][index] + "  " + selected_data["lst_nm"][
            index] + "  " + "Ind_PAC_ID " + \
                                            "  " + str(selected_data["Ind_PAC_ID"][index])
        selected_data["patient_review"][index] = random.randint(4, 10)  # adding attribute of ranking "patient_review"
        selected_data["is_served_army"][index] = random.randint(0, 1)  # adding attribute of ranking "is_served_army"
        selected_data["num_of_days_available"][index] = random.randint(4, 7)  # adding attribute of ranking "num_of_days_available"
        selected_data["number_of_languages"][index] = random.randint(1, 5)  # adding attribute of ranking "number_of_languages"
        selected_data["is_need_parking"][index] = random.randint(0, 1)  # adding attribute of ranking "number_of_languages"  # todo: add to the algo

        # selected_data["Salary"][index]
    selected_data.to_csv(r"C:\Users\IlyaY\Desktop\לימודים\תשפג\א\אלגו\project\data\DAC_NationalDownloadableFile_upd_new.csv")  #, encoding = 'unicode_escape')

    #return selected_data[['cred_number', 'Salary', 'years_experience']], selected_data  # The features on which we will check our model

def load_data(path, gender_filter, spec_filter):
    data = pd.read_csv(path)  #, encoding='unicode_escape')
    selected_data = data.iloc[:300, :]  # check of small data_set
    selected_data = selected_data.loc[selected_data["pri_spec"] == spec_filter]  # apply filter
    selected_data = selected_data.loc[selected_data["gndr"] == gender_filter]  # apply filter

    return selected_data[["cred_number",
                           "Salary",
                           "years_experience",
                           "patient_review",
                           "is_served_army",
                           "num_of_days_available",
                           "number_of_languages",
                           "is_need_parking"]], selected_data  # The features on which we will check our model



def calculate_best_choise(gender_filter='F', spec_filter='CHIROPRACTIC'):
    # path = r"C:\Users\IlyaY\Desktop\לימודים\תשפג\א\אלגו\project\data\less_data.csv"
    # path = r"C:\Users\IlyaY\Desktop\לימודים\תשפג\א\אלגו\project\data\DAC_NationalDownloadableFile_upd.csv"
    path = r"C:\Users\IlyaY\Desktop\לימודים\תשפג\א\אלגו\project\data\DAC_NationalDownloadableFile_upd_new.csv"
    raw_data, selected_data = load_data(path, gender_filter, spec_filter)
    # start_time = time.perf_counter()

    # TOPSIS
    best_NPI = topsis(raw_data, selected_data['NPI'])  # candidates = selected_data['NPI']
    # print(f"TOPSIS Best candidate:  {(selected_data[selected_data['NPI'] == best_NPI])['Full Name'].values[0]}")

    # SMART
    # best_NPI = smart(raw_data, selected_data['NPI'])  # candidates = selected_data['NPI']
    # print(f"TOPSIS Best candidate:  {(selected_data[selected_data['NPI'] == best_NPI])['Full Name'].values[0]}")

    # BORDA
    # best_NPI = borda(raw_data, selected_data['NPI'])
    # print(f"BORDA Best candidate:  {(selected_data[selected_data['NPI'] == best_NPI])['Full Name'].values[0]}")
    end_time = time.perf_counter()

    # run_time = end_time - start_time
    # print(f"run time is: {run_time}")
    return (selected_data[selected_data['NPI'] == best_NPI])['Full Name'].values[0]  # topsis


if __name__ == '__main__':
    # path = r"C:\Users\IlyaY\Desktop\לימודים\תשפג\א\אלגו\project\data\less_data.csv"
    path = r"C:\Users\IlyaY\Desktop\לימודים\תשפג\א\אלגו\project\data\DAC_NationalDownloadableFile_upd_new.csv"
    gender_filter = 'M'
    spec_filter = 'CHIROPRACTIC'

    # prepare_data(path)

    print(calculate_best_choise(gender_filter='M', spec_filter='CHIROPRACTIC'))




