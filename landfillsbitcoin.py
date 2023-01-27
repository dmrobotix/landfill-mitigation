# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 22:33:19 2023

@author: vpaez3
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


## LOAD DATASET ###
lmoc_data = r"C:\Users\vpaez3\Dropbox (GaTech)\bitcoin model\landfillmopdata.csv"
lmoc_df = pd.read_csv(lmoc_data)

waste_in_place = list(lmoc_df['Waste in Place (tons)'])
lfg_generated = list(lmoc_df['LFG Generated (mmscfd)'])
lfg_collected = list(lmoc_df['LFG Collected (mmscfd)'])

## CLEAN UP DATA ###
# remove the commas from the numbers which are strings format instead of 
# number format, then convert the waste in place values to floats.

for el in range(len(waste_in_place)):
    if type(waste_in_place[el]) == str: 
        tmp = waste_in_place[el].replace(',','')
        waste_in_place[el] = tmp

waste_in_place = list(map(float,waste_in_place))

# clean up the dataset so that it only includes landfills that have collection
# systems on site. 

del_ids = []
for el in range(len(waste_in_place)):
    # if row is a nan remove it from all lists
    if np.isnan(waste_in_place[el]):
        del_ids.append(el)
    elif np.isnan(lfg_generated[el]):
        del_ids.append(el)
    elif np.isnan(lfg_collected[el]):
        del_ids.append(el)

waste_in_place = np.delete(np.array(waste_in_place), del_ids)
lfg_generated = np.delete(np.array(lfg_generated), del_ids)
lfg_collected = np.delete(np.array(lfg_collected), del_ids)

### PRINT MEDIAN AND MEAN FOR COMPARIONS ###
print("median lfg collected (mmscfd):", np.median(lfg_collected))
print("median lfg generated (mmscfd):", np.median(lfg_generated))
print("median waste in place (tons)", np.median(waste_in_place))
print("mean lfg collected (mmscfd):", np.mean(lfg_collected))
print("mean lfg generated (mmscfd)", np.mean(lfg_generated))
print("mean waste in place (tons)", np.mean(waste_in_place))

### COMPUTE CONVERSION FACTORS TO MAKE IT CLIMATE IMPACT FORECAST FRIENDLY ###
# convert waste in place so that it is in terms of kilotons and not tons
# https://www.convertunits.com/from/tons/to/metric+tonnes
waste_in_place = waste_in_place/1.1023113109244 # convert to metric tonnes
waste_in_place = waste_in_place/1000 # convert to kilotonnes

lfg_net = lfg_generated - lfg_collected

# calculate per kilotonne
lfg_generated_per_ktonne = lfg_generated/waste_in_place
lfg_collected_per_ktonne = lfg_collected/waste_in_place
lfg_net_per_ktonne = lfg_net/waste_in_place

# what is the lfg not collected when doing the calculation on the medians of 
# the full dataset rather than doing the lfg not collected calculation first 
# and then finding the median of this result
print("---")
print("lfg not collected (mmscfd/kilotonne) -- using median values then computing the result: ", (np.median(lfg_generated)-np.median(lfg_collected))/np.median(waste_in_place))
print("lfg not collected (mmscfd/kilotonne) -- computing result then taking median: ", np.median(lfg_net_per_ktonne))

# convert from mmscfd/kilotonne to (kg/day)/kilotonne
print("---")
# kg of CH4/day
# *1e6 ft^3/mft^3*35.315m^3/ft^3*0.657kg/m^3(ch4 density)*0.5 (ASSUME 50% of LFG is METHANE)
print("CH4 collected ((kg of CH4/day)/kilotonne)", np.median(lfg_collected_per_ktonne)*1e6*35.315*0.657*0.5)
print("CH4 not collected ((kg of CH4/day)/kilotonne)", np.median(lfg_net_per_ktonne)*1e6*35.315*0.657*0.5)
print("CH4 generated ((kg of CH4/day)/kilotonne)", np.median(lfg_generated_per_ktonne)*1e6*35.315*0.657*0.5)

# convert from mmscfd/kilotonne to (kg/year)/kilotonne
# kg of CH4/year
# *1e6 ft^3/mft^3*35.315m^3/ft^3*0.657kg/m^3(ch4 density)*365 days/year
print("---")
print("lfg collected ((kg of CH4/year)/kilotonne)", np.median(lfg_collected_per_ktonne)*1e6*35.315*0.657*365*0.5)
print("lfg not collected ((kg of CH4/year)/kilotonne)", np.median(lfg_net_per_ktonne)*1e6*35.315*0.657*365*0.5)
print("lfg generated ((kg of CH4/year)/kilotonne)", np.median(lfg_generated_per_ktonne)*1e6*35.315*0.657*365*0.5)
print("---")

### ADDITIONAL CALCULATIONS ###
fraction_efficiency = np.median(lfg_collected/lfg_generated)*100 # percent
print("efficiency of LFG collection (%): ", fraction_efficiency)

lfg_generated_per_tonne_per_year = np.median(lfg_generated_per_ktonne)*1e6*35.315*0.657*365*0.5

##########################################
### MOST COMMON LANDFILLS CALCULATIONS ###
# collect the landfills that are less than 13099 kilotonnes in size into a separate
# dataframe, these represent the most common size of landfills in the LMOC 
# database based on the waste_in_place distribution.

# this shows the distribution of waste in place in tons (for reference)
fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(waste_in_place)

ax.set_xlabel('Waste (kilotonnes)')
ax.set_ylabel('Counts')
ax.set_title(r'Histogram of Waste in Place')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()

fig, ax = plt.subplots()
ax.set_xlabel('Waste (kilotonnes)')
ax.set_ylabel('Counts')
ax.set_title(r'Histogram of Waste in Place with most common landfills')
ax.hist(waste_in_place)
#counts, bins = np.histogram(waste_in_place)


wip_idx = np.where(np.array(waste_in_place) <= 13099) # taken from counts
waste_in_place_common = waste_in_place[wip_idx[0]]
lfg_generated_common = lfg_generated[wip_idx[0]]
lfg_collected_common = lfg_collected[wip_idx[0]]

# calculate per kilotonne
lfg_net_common = lfg_generated_common - lfg_collected_common

lfg_generated_common_per_ktonne = lfg_generated_common/waste_in_place_common
lfg_collected_common_per_ktonne = lfg_collected_common/waste_in_place_common
lfg_net_common_per_ktonne = lfg_net_common/waste_in_place_common
# what is the lfg not collected when doing the calculation on the medians of 
# the full dataset rather than doing the lfg not collected calculation first 
# and then finding the median of this result
print("=============")
print("common landfills only!")
print("---")
print("lfg not collected (mmscfd/kilotonne) -- using median values then computing the result: ", (np.median(lfg_generated_common)-np.median(lfg_collected_common))/np.median(waste_in_place_common))
print("lfg not collected (mmscfd/kilotonne) -- computing result then taking median: ", np.median(lfg_net_common_per_ktonne))

# convert from mmscfd/kilotonne to (kg/day)/kilotonne
print("---")
# kg of CH4/day
# *1e6 ft^3/mft^3*35.315m^3/ft^3*0.657kg/m^3(ch4 density)
print("CH4 collected ((kg of CH4/day)/kilotonne)", np.median(lfg_collected_common_per_ktonne)*1e6*35.315*0.657*0.5)
print("CH4 not collected ((kg of CH4/day)/kilotonne)", np.median(lfg_net_common_per_ktonne)*1e6*35.315*0.657*0.5)
print("CH4 generated ((kg of CH4/day)/kilotonne)", np.median(lfg_generated_common_per_ktonne)*1e6*35.315*0.657*0.5)

# convert from mmscfd/kilotonne to (kg/year)/kilotonne
# kg of CH4/year
# *1e6 ft^3/mft^3*35.315m^3/ft^3*0.657kg/m^3(ch4 density)*365 days/year
print("---")
print("CH4 collected ((kg of CH4/year)/kilotonne)", np.median(lfg_collected_common_per_ktonne)*1e6*35.315*0.657*365*0.5)
print("CH4 not collected ((kg of CH4/year)/kilotonne)", np.median(lfg_net_common_per_ktonne)*1e6*35.315*0.657*365*0.5)
print("CH4 generated ((kg of CH4/year)/kilotonne)", np.median(lfg_generated_common_per_ktonne)*1e6*35.315*0.657*365*0.5)
print("---")

### ADDITIONAL CALCULATIONS ###
fraction_efficiency = np.median(lfg_collected_common/lfg_generated_common)*100 # percent
print("efficiency of LFG collection (%): ", fraction_efficiency)

lfg_generated_common_per_ktonne_per_year = np.median(lfg_generated_per_ktonne)*1e6*35.315*0.657*365*0.5
median_waste_in_place_common = np.median(waste_in_place_common) # chosen because
# the distribution is slightly skewe around 4000 kilotonnes

total = lfg_generated_common_per_ktonne_per_year*median_waste_in_place_common*(1-fraction_efficiency/100)
print("---")
print("the unmitigated methane that remains uncaptured (kg of CH4/year): ", total)
ax.hist(waste_in_place_common)

fig, ax = plt.subplots()
ax.set_xlabel('Waste (kilotonnes)')
ax.set_ylabel('Counts')
ax.set_title(r'Histogram of Waste in Place for Common Landfills')
ax.hist(waste_in_place_common)
