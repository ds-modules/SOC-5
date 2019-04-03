from datascience import *
import numpy as np

def clean_table(gss_survey_dat):
    gss_survey_data = gss_survey_dat.where("AGE", are.between('0','89'))
    gss_survey_data = gss_survey_data.where("SEX", are.between_or_equal_to('1','2'))
    gss_survey_data = gss_survey_data.where("EDUC", are.between_or_equal_to('0','96'))
    gss_survey_data = gss_survey_data.where("NATEDUC", are.between_or_equal_to('1','4'))
    gss_survey_data = gss_survey_data.where("NATFARE", are.between_or_equal_to('1','4'))
    gss_survey_data = gss_survey_data.where("NATROAD", are.between_or_equal_to('1','4'))
    gss_survey_data = gss_survey_data.where("NATMASS", are.between_or_equal_to('1','4'))
    gss_survey_data = gss_survey_data.where("NATHEAL", are.between_or_equal_to('1','4'))
    gss_survey_data = gss_survey_data.where("NATENVIR", are.between_or_equal_to('1','4'))
    for label in gss_survey_data.labels:
        gss_survey_data = gss_survey_data.with_column(label, gss_survey_data.column(label).astype(int))
    return gss_survey_data

def generate_3x3_contingency_table(gss_survey_data, attr1, attr2):
    contigTable = gss_survey_data.groups([attr1, attr2]).pivot(attr1, attr2, values="count", collect=np.sum)
    contigTable["total ("+attr1+")"] = [np.sum(list(contigTable.row(n))[1:]) for n in np.arange(0, len(contigTable)-1)]
    sums = ["total"] + [np.sum(contigTable[i]) for i in np.arange(1, 5)]
    contigTable = contigTable.with_row(sums)
    contigTable = contigTable.relabeled(1, "1 ("+attr1+")")
    contigTable = contigTable.relabeled(2, "2 ("+attr1+")")
    contigTable = contigTable.relabeled(3, "3 ("+attr1+")")
    contigTable[attr2] = ["1 ("+attr2+")", "2 ("+attr2+")", "3 ("+attr2+")", "total ("+attr2+")"]
    return contigTable

def find_chi_square(contigTable):
    expected = [[contigTable.column("total (NATMASS)")[row]*(contigTable.column(col + " (NATMASS)")[3]/1535) for col in ["1", "2", "3"]] for row in [0, 1, 2]]
    observed = [[contigTable.column(col + " (NATMASS)")[row] for col in ["1", "2", "3"]] for row in [0, 1, 2]]
    expected = np.concatenate(expected).ravel()
    observed = np.concatenate(observed).ravel()
    return np.sum((expected-observed)**2/expected), 4

def find_expected_dist(contigTable, attr1, attr2):
    expected = contigTable.copy()
    expect = [[contigTable.column("total ("+attr1+")")[row]*(contigTable.column(col + " ("+attr1+")")[3]/1535) for row in [0, 1, 2]] for col in ["1", "2", "3"]]
    expected = expected.select([""+attr2+""])
    for i in [0, 1, 2]:
        expected = expected.with_column(str(i+1) + " ("+attr1+")", expect[i] + [np.sum(expect[i])])
    expected = expected.with_column("total ("+attr1+")", contigTable["total ("+attr1+")"])
    return expected

def generate_means_table(females, columns_of_interest):
    female_data = [females[col] for col in columns_of_interest]
    mean_values = [np.mean(col) for col in female_data]
    std_values = [np.std(col) for col in female_data]
    means_female = Table().with_column("category", columns_of_interest).with_columns("mean", mean_values, "standard deviation", std_values)
    return means_female

def generate_t_value(means_female, means_male, females, males, attr):
    a = means_male.column(0) == attr
    s = 0
    for i in np.arange(len(a)):
        if a[i]:
            s = i
            break
    female_mean_welfare = means_female.row(s)[1]
    male_mean_welfare = means_male.row(s)[1]
    female_sd_welfare = means_female.row(s)[2]
    male_sd_welfare = means_male.row(s)[2]
    s_p = ((females - 1)*(female_sd_welfare)**2 + (males - 1)*(male_sd_welfare)**2)/(males + females - 2)**.5
    t = (female_mean_welfare - male_mean_welfare)/(s_p*(1/females + 1/males)**.5)
    return t