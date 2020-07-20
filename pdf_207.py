import scipy.stats as stats
import csv
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean
from numpy import std
from scipy.stats import norm
from scipy.spatial import distance
import pandas as pd
import seaborn as sns
import scipy
import scipy.stats as stats

#read data into a dataframe
df = pd.read_csv('dataset207_1.csv', usecols=[1])

df2 = pd.read_csv('dataset207_1.csv', usecols=[1,4])

#printing the whole data
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)


#Putting the data from delta_T into an array using np
Δt_array = df.to_numpy()
print(Δt_array)

#Get the frequency of the values in the dataset
df = df.reset_index()
delta_Occur = df['delta_T'].value_counts()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
     print('Frequency of occurence of Δt values: \n', delta_Occur)

#Getting the percentage of occurence of the dataset
delta_percent = df['delta_T'].value_counts(normalize=True) * 100
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print('Percentages of occurence of Δt values: \n', delta_percent)




# Grouping the bins
bins = [-1, 0.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000]

labels = ['-1-0.000000', '0.000001-2.000000', '2.000001-3.000000', '3.000001-4.000000', '4.000001-5.000000', '5.000001-6.000000', '6.000001-7.000000', '7.000001-8.000000', '8.000001-9.000000', '9.000001-10.000000']

bin_spread = pd.cut(df['delta_T'], bins=bins, labels=labels, include_lowest=True)

#draw_data2 = df.groupby(pd.cut(df['delta_T'], [-1, 0.000000, 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000])).size()

#print('Bins: \t\tPercentage', draw_data2, '\t\t', (draw_data2/100))

#Percentage distribution of the bins
delta_val = bin_spread.value_counts(normalize=True)
print(delta_val)

delta_per = bin_spread.value_counts(normalize=True) * 100
print('Bins     \t\t Percentages: \n', delta_per)

############################Calculating the euclidean distance############################################
#Obtaining the mean of the data
delta_average_of_array = mean(Δt_array)
print('Mean = %.3f'% (delta_average_of_array))

#Obtaining the standard deviation of the Δt
std_of_array = std(Δt_array)
print('Standard Deviation = %.3f'
%(std_of_array))

#Calculating the euclidean distance by standardisation
stand_data = ((Δt_array) - (delta_average_of_array)) / (std_of_array)
print('The standardised array of delta_T is: \n', stand_data)

#implementation of the histogram with the original data(array)
fig, ax = plt.subplots()
n, bins, patches = ax.hist(Δt_array, bins, density=1)
# add a 'best fit' line
y = ((1 / (np.sqrt(2 * np.pi) * std_of_array)) *
     np.exp(-0.5 * (1 / std_of_array * (bins - delta_average_of_array))**2))
ax.plot(bins, y, '--')
ax.set_xlabel('delta_T')
ax.set_ylabel('Probability density of packets')
#ax.set_title(r'Histogram of GOOSE data207:  $/mu=delta_average_of_array$, $\sigma=std_of_array$')
title = ' Histogram of dataset207_1: μ' + '=' + str(delta_average_of_array), ' σ = ' + str(std_of_array)
ax.set_title (title)

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()

#Implementation of the histogram with the standardised data
fig, ax = plt.subplots()
n, bins, patches = ax.hist(stand_data, bins, density=1)
# add a 'best fit' line
y = ((1 / (np.sqrt(2 * np.pi) * std_of_array)) *
     np.exp(-0.5 * (1 / std_of_array * (bins - delta_average_of_array))**2))
ax.plot(bins, y, '--')
ax.set_xlabel('delta_T')
ax.set_ylabel('Probability density of packets')
#ax.set_title(r'Histogram of GOOSE data207: ' '$\mu' %(delta_average_of_array), '$\sigma ' %(%std_of_array))
title = 'Histogram of standardised Dataset: μ' + '=' + str(delta_average_of_array), ' σ = ' + str(std_of_array)
ax.set_title(title)

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()



#Obtaining the Euclidean Distance for the two columns - delta_T and GOOSE Length

eucl_d = distance.euclidean(df2['delta_T'], df2['GOOSE LENGTH'])

print('Euclidean Distance:', eucl_d)

std_eucl_dist =14268.901156384632

if std_eucl_dist > eucl_d :
    print('The deviation is by a positive margin of:', (((std_eucl_dist - eucl_d)/std_eucl_dist) * 100),'%')
elif std_eucl_dist == eucl_d:
    print('The data matches 100%, no deviation')
else:
    print('The deviation is by a negative margin of:', (((eucl_d - std_eucl_dist)/std_eucl_dist) * 100),'%' )






