from datascience import *
import numpy as np
array = lambda *args: np.array(args)

import matplotlib.pyplot as plt
import ipywidgets as widgets
from scipy import stats
np.mode = lambda x: stats.mode(x)[0][0]
np.range = lambda x: max(x) - min(x)

### This file contains the Python code that defines many of the functions used in Project 5

# Defining barchart function 
def barchart(categories, heights, x_label, y_label, title, file_name="barchart"):
    plt.figure(figsize=(8,6))
    plt.bar(categories, heights)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig("Output/" + file_name + ".png", dpi=150)

# Defining histogram function
def histogram(array, x_label, y_label, title, file_name="histogram.png"):
    plt.figure(figsize=(8,6))
    plt.hist(array)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig("Output/" + file_name + ".png", dpi=150)
    
    
def filter_values(tbl, column_name, removed):
    filter_func = lambda x: x not in removed
    return tbl.where(column_name, filter_func)
    
def cross_tab(data, col, row):
    # Getting counts
    c_tbl = data.pivot(col, row)
    # Adding row totals
    c_tbl = c_tbl.with_column("total", [np.sum(row[1:]) for row in c_tbl.rows])
    # Adding column totals
    r = ["total"]
    r.extend([np.sum(col) for col in c_tbl.columns[1:]])
    c_tbl = c_tbl.with_row(r)
    # Relabeling columns
    c_tbl = c_tbl.relabeled(label=c_tbl.labels[1:], 
                            new_label = np.array([str(i) + " ({})".format(col) for i in c_tbl.labels[1:]]))
    return c_tbl

def find_proportions(contigTable):
    m_tbl = Table(labels=contigTable.labels)
    
    # Find total of all counts
    total = contigTable.column(contigTable.labels[-1])[-1]
    
    # For each row
    for i in range(0, contigTable.num_rows):
        row = contigTable.rows[i]
        new_row = [row[0]]
        new_row.extend([val/total for val in row[1:]])
        m_tbl = m_tbl.with_row(new_row)
    return m_tbl

def classify(item, endpoints):
    group = 1
    for i in range(0, len(endpoints)-1):
        if i == len(endpoints)-2:
            return str(endpoints[i]) + " - " + str(endpoints[i+1]) + "+"
        elif endpoints[i] <= item < endpoints[i+1]:
            return str(endpoints[i]) + " - " + str(endpoints[i+1]-1)
        else:
            group += 1

def create_categories(tbl, column_name, endpoints):
    new_col_name = column_name + "_group"
    col_groups_names = []
    assert np.count_nonzero(tbl.column(column_name) < endpoints[0]) == 0, "There are values in " + column_name + " that are less than the lowest endpoint!"
    for i in tbl.column(column_name):
        col_groups_names.append(classify(i, endpoints))
    return tbl.with_column(new_col_name, col_groups_names).sort(new_col_name)

def corr_R2(tbl, col_1, col_2):
    x = tbl.column(col_1)
    y = tbl.column(col_2)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    print("Correlation: " + str(r_value))
    print("R-Squared: " + str(r_value**2)) 
    return r_value, r_value**2

def expected_counts(tbl):
    """Given a contingency table, will find the expected counts"""
    grand_total = tbl.row(-1)[-1]
    col_totals = tbl.row(-1)[1:-1]
    row_totals = tbl[tbl.num_columns-1][:-1]
    
    # Add row labels
    expected_tbl = Table().with_column(tbl.labels[0], tbl.column(0))
    for col in range(0, len(col_totals)):
        values = []
        for row in range(0, len(row_totals)):
            value = (col_totals[col]*row_totals[row])/grand_total
            values.append(value)
        values.append(col_totals[col])
        expected_tbl = expected_tbl.with_column(tbl.labels[col+1], values)

    # Adding row totals
    expected_tbl = expected_tbl.with_column(tbl.labels[-1], tbl.column(tbl.labels[-1]))
    return expected_tbl

def compute_chi_square(cross_tbl, expected_tbl):
    """Given a cross tab table and its expected counts table, this function calculates the Chi-Squared statistic"""
    assert cross_tbl.num_rows == expected_tbl.num_rows
    assert cross_tbl.num_columns == expected_tbl.num_columns
    
    n_rows = expected_tbl.num_rows
    n_cols = expected_tbl.num_columns
    vals = []
    for col in range(1, n_cols-1):
        for row in range(0, n_rows-1):
            expected = expected_tbl.column(col)[row]
            observed = cross_tbl.column(col)[row]
            val = ((observed - expected)**2)/expected
            vals.append(val)
    return sum(vals)
