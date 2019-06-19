import pandas as pd
import numpy as np
import statsmodels.discrete.discrete_model as sm
import statsmodels.tools.tools as sm2
import datetime
import parameters



pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 999)
pd.set_option('max_colwidth', 999)




###################
### Data Import ###
###################


# Clients will need to enter the file path for the associated files (ETRSource_Source Dataset, ETRInsight_StockReturns, and ETRInsight_FTECReturns Dataset) into the following path strings.


################################
source_file = parameters.source_file
spReturns_file = parameters.spReturns_file
ftecReturns_file = parameters.ftecReturns_file
###################################



source1 = pd.read_csv(source_file)
spReturns = pd.read_csv(spReturns_file)
ftecReturns = pd.read_csv(ftecReturns_file)


# The most recent Survey_ID is set to a macro variable for later use.
survey_max = source1['Survey_ID'].max()



######################################################
### Expected Enterprise Spend + Market Share Theme ###
######################################################


# Spending intentions for each vendor are aggregated to calculate spend metrics:
# Citations, Adoption %, Increase %, Flat %, Decrease %, Replacing %, Net Score.
# Unique number of respondents in each sector is merged on to calculate Market Share.

spend1a = source1.sort_values(by = ['Survey_Description_1', 'Survey_ID', 'Survey_Launch', 'Survey_Close', 'Announcement_Date', 'Respondent_ID', 'Sector_Current', 'Vendor_Current', 'Product_Current', 'Symbol_ID_Current', 'Bloomberg_ID_Current', 'FIGI_ID_Current'])

spend1b = source1[source1.Metric != 'REPLACING'].drop_duplicates(["Survey_Description_1", "Survey_ID", "Survey_Launch", "Survey_Close", "Announcement_Date", "Respondent_ID", "Sector_Current"])

spend2a = spend1a.fillna('---').groupby(['Survey_Description_1', 'Survey_ID', 'Survey_Launch', 'Survey_Close', 'Announcement_Date', 'Sector_Current', 'Vendor_Current', 'Product_Current', 'Symbol_ID_Current', 'Bloomberg_ID_Current', 'FIGI_ID_Current', 'Sector_Historical', 'Vendor_Historical', 'Product_Historical', 'Symbol_ID_Historical', 'Bloomberg_ID_Historical', 'FIGI_ID_Historical', 'Metric']).size().reset_index(name = 'Count') 

spend2b = spend1b.groupby(['Survey_Description_1', 'Survey_ID', 'Survey_Launch', 'Survey_Close', 'Announcement_Date', 'Sector_Current']).size().reset_index(name = 'Count')

spend3 = pd.pivot_table(spend2a, index = ['Survey_Description_1', 'Survey_ID', 'Survey_Launch', 'Survey_Close', 'Announcement_Date', 'Sector_Current', 'Vendor_Current', 'Product_Current', 'Symbol_ID_Current', 'Bloomberg_ID_Current', 'FIGI_ID_Current', 'Sector_Historical', 'Vendor_Historical', 'Product_Historical', 'Symbol_ID_Historical', 'Bloomberg_ID_Historical', 'FIGI_ID_Historical'], columns = 'Metric', values = 'Count').reset_index()

spend4 = spend3.fillna(0)

spend5 = spend4.merge(spend2b, on = ['Survey_Description_1', 'Survey_ID', 'Survey_Launch', 'Survey_Close', 'Announcement_Date', 'Sector_Current'])
spend5['Citations'] = spend5['ADOPTION'] + spend5['INCREASE'] + spend5['FLAT'] + spend5['DECREASE'] + spend5['REPLACING']
spend5['Citations_ExR'] = spend5['ADOPTION'] + spend5['INCREASE'] + spend5['FLAT'] + spend5['DECREASE']
spend5['AdoptionP'] = spend5['ADOPTION'] / spend5['Citations']
spend5['IncreaseP'] = spend5['INCREASE'] / spend5['Citations']
spend5['FlatP'] = spend5['FLAT'] / spend5['Citations']
spend5['DecreaseP'] = spend5['DECREASE'] / spend5['Citations']
spend5['ReplacingP'] = spend5['REPLACING'] / spend5['Citations']
spend5['NetScore'] = (spend5['ADOPTION'] + spend5['INCREASE'] - spend5['DECREASE'] - spend5['REPLACING']) / spend5['Citations']
spend5['MarketShare'] = spend5['Citations_ExR'] / spend5['Count']

spend5 = spend5.drop(['ADOPTION', 'INCREASE', 'FLAT', 'DECREASE', 'REPLACING', 'Citations_ExR', 'Count'], axis = 1)

spend6 = pd.melt(spend5, id_vars= ['Survey_Description_1', 'Survey_ID', 'Survey_Launch', 'Survey_Close', 'Announcement_Date', 'Sector_Current', 'Vendor_Current', 'Product_Current', 'Symbol_ID_Current', 'Bloomberg_ID_Current', 'FIGI_ID_Current', 'Sector_Historical', 'Vendor_Historical', 'Product_Historical', 'Symbol_ID_Historical', 'Bloomberg_ID_Historical', 'FIGI_ID_Historical', 'Citations'], value_vars = ['AdoptionP', 'IncreaseP', 'FlatP', 'DecreaseP', 'ReplacingP', 'NetScore', 'MarketShare']).reset_index()

spend6 = spend6.sort_values(by = ['Survey_Description_1', 'Survey_ID', 'Survey_Launch', 'Survey_Close', 'Announcement_Date', 'Sector_Current', 'Vendor_Current', 'Product_Current', 'Symbol_ID_Current', 'Bloomberg_ID_Current', 'FIGI_ID_Current', 'Sector_Historical', 'Vendor_Historical', 'Product_Historical', 'Symbol_ID_Historical', 'Bloomberg_ID_Historical', 'FIGI_ID_Historical'])


# Survey-over-Survey and Year-over-Year values for each metric are merged on to calculate deltas.    
# Deltas are used to measure recent inflections and longer-term trends.                              


spend7_sos = spend6[['Survey_ID', 'Sector_Current', 'Vendor_Current', 'Product_Current', 'variable', 'value']].rename(columns = {'variable':'Metric', 'Survey_ID':'Survey_ID_sos', 'value':'Value_sos'})
spend7_yoy = spend6[['Survey_ID', 'Sector_Current', 'Vendor_Current', 'Product_Current', 'variable', 'value']].rename(columns = {'variable':'Metric', 'Survey_ID':'Survey_ID_yoy', 'value':'Value_yoy'})

spend7_sos = spend7_sos.sort_values(by = ['Sector_Current', 'Vendor_Current', 'Product_Current', 'Metric', 'Survey_ID_sos'])
spend7_yoy = spend7_yoy.sort_values(by = ['Sector_Current', 'Vendor_Current', 'Product_Current', 'Metric', 'Survey_ID_yoy'])


spend7 = spend6.rename(columns = {'variable':'Metric'})
spend7['Survey_ID_sos'] = spend7['Survey_ID'] - 1
spend7['Survey_ID_yoy'] = np.nan
spend7.loc[spend7.Survey_ID == 3, 'Survey_ID_yoy'] = 1
spend7.loc[spend7.Survey_ID == 4, 'Survey_ID_yoy'] = 2
spend7.loc[spend7.Survey_ID == 5, 'Survey_ID_yoy'] = 2
spend7.loc[(spend7['Survey_ID'] >= 6) & (spend7['Survey_ID'] % 2  == 0), 'Survey_ID_yoy'] = spend7['Survey_ID'] - 3
spend7.loc[(spend7['Survey_ID'] >= 6) & (spend7['Survey_ID'] % 2  == 1), 'Survey_ID_yoy'] = spend7['Survey_ID'] - 4

spend7 = spend7.sort_values(by = ['Sector_Current', 'Vendor_Current', 'Product_Current', 'Metric', 'Survey_ID_sos'])

spend8a = spend7.merge(spend7_sos, how = 'left', on = ['Sector_Current', 'Vendor_Current', 'Product_Current', 'Metric', 'Survey_ID_sos'])

spend8a = spend8a.sort_values(by = ['Sector_Current', 'Vendor_Current', 'Product_Current', 'Metric', 'Survey_ID_yoy'])

spend8b = spend8a.merge(spend7_yoy, how = 'left', on = ['Sector_Current', 'Vendor_Current', 'Product_Current', 'Metric', 'Survey_ID_yoy'])

spend8b['Delta_sos'] = spend8b['value'] - spend8b['Value_sos']
spend8b['Delta_yoy'] = spend8b['value'] - spend8b['Value_yoy']
spend8b = spend8b.drop(['Survey_ID_sos', 'Survey_ID_yoy', 'index'], axis = 1)

spend8b = spend8b.sort_values(by = ['Survey_ID', 'Metric'])






spend8bnonan1 = spend8b[pd.notna(spend8b.Delta_sos)]
spend8bnonan2 = spend8b[pd.notna(spend8b.Delta_yoy)]


# Weighted survey averages for each metric value and delta are calculated to create z-scores.

wm = lambda x: np.average(x, weights = spend8b.loc[x.index, "Citations"])
f = {'Citations': ['sum'], 'value': {'value_wm':wm}, 'Delta_sos': {'Delta_sos_mean':wm}, 'Delta_yoy':{'Delta_yoy_mean':wm}}

surveyavg1 = spend8b.groupby(['Survey_ID', 'Metric']).agg(f)[['value']].reset_index()
mean_agg = spend8bnonan1.groupby(['Survey_ID', 'Metric']).agg(f)[['Delta_sos']].reset_index()
mean_agg2 = spend8bnonan2.groupby(['Survey_ID', 'Metric']).agg(f)[['Delta_yoy']].reset_index()
surveyavg1.columns = surveyavg1.columns.droplevel(1)
mean_agg.columns = mean_agg.columns.droplevel(1)
mean_agg2.columns = mean_agg2.columns.droplevel(1)

surveyavg1 = surveyavg1.merge(mean_agg, how = 'left', on = ['Survey_ID', 'Metric'])
surveyavg1 = surveyavg1.merge(mean_agg2, how = 'left', on = ['Survey_ID', 'Metric'])





wsd = lambda x: np.sqrt(np.cov(x, aweights = spend8b.loc[x.index, "Citations"], ddof = 0))
g = {'Citations': ['sum'], 'value': {'value_wm':wsd}, 'Delta_sos': {'Delta_sos_mean':wsd}, 'Delta_yoy':{'Delta_yoy_mean':wsd}}

sd_agg = spend8b.groupby(['Survey_ID', 'Metric']).agg(g)[['value']].reset_index()
sd_agg1 = spend8bnonan1.groupby(['Survey_ID', 'Metric']).agg(g)[['Delta_sos']].reset_index()
sd_agg2 = spend8bnonan2.groupby(['Survey_ID', 'Metric']).agg(g)[['Delta_yoy']].reset_index()

sd_agg.columns = sd_agg.columns.droplevel(1)
sd_agg1.columns = sd_agg1.columns.droplevel(1)
sd_agg2.columns = sd_agg2.columns.droplevel(1)

surveyavg1 = surveyavg1.merge(sd_agg, how = 'left', on = ['Survey_ID', 'Metric'])
surveyavg1 = surveyavg1.merge(sd_agg1, how = 'left', on = ['Survey_ID', 'Metric'])
surveyavg1 = surveyavg1.merge(sd_agg2, how = 'left', on = ['Survey_ID', 'Metric'])

surveyavg1 = surveyavg1.rename(columns = {"value_x": "Value_SurveyMean", "Delta_sos_x":"Delta_sos_SurveyMean", "Delta_yoy_x":"Delta_yoy_SurveyMean", "value_y":"Value_SurveyStdDev", "Delta_sos_y":"Delta_sos_SurveyStdDev", "Delta_yoy_y":"Delta_yoy_SurveyStdDev"})


spend9 = spend8b.merge(surveyavg1, how = 'left', on = ['Survey_ID', 'Metric']).reset_index().drop(["index"], axis = 1)
spend9["Value_SurveyZ"] = (spend9["value"] - spend9["Value_SurveyMean"]) / spend9["Value_SurveyStdDev"]
spend9["Delta_sos_SurveyZ"] = (spend9["Delta_sos"] - spend9["Delta_sos_SurveyMean"]) / spend9["Delta_sos_SurveyStdDev"]
spend9["Delta_yoy_SurveyZ"] = (spend9["Delta_yoy"] - spend9["Delta_yoy_SurveyMean"]) / spend9["Delta_yoy_SurveyStdDev"]

spend9 = spend9.sort_values(by = ['Survey_Description_1', 'Survey_ID', 'Survey_Launch', 'Survey_Close', 'Announcement_Date', 'Sector_Current', 'Vendor_Current', 'Product_Current', 'Symbol_ID_Current', 'Bloomberg_ID_Current', 'FIGI_ID_Current', 'Sector_Historical', 'Vendor_Historical', 'Product_Historical', 'Symbol_ID_Historical', 'Bloomberg_ID_Historical', 'FIGI_ID_Historical', 'Citations', 'Metric'])
spend9 = spend9.rename(columns = {"value":"Value"})


spend10 = pd.melt(spend9, id_vars = ['Survey_Description_1', 'Survey_ID', 'Survey_Launch', 'Survey_Close', 'Announcement_Date', 'Sector_Current', 'Vendor_Current', 'Product_Current', 'Symbol_ID_Current', 'Bloomberg_ID_Current', 'FIGI_ID_Current', 'Sector_Historical', 'Vendor_Historical', 'Product_Historical', 'Symbol_ID_Historical', 'Bloomberg_ID_Historical', 'FIGI_ID_Historical', 'Citations', 'Metric'], value_vars = ['Value', 'Delta_sos',  'Delta_yoy', 'Value_SurveyZ', 'Delta_sos_SurveyZ', 'Delta_yoy_SurveyZ'])
spend10 = spend10.sort_values(by = ['Survey_Description_1', 'Survey_ID', 'Survey_Launch', 'Survey_Close', 'Announcement_Date', 'Sector_Current', 'Vendor_Current', 'Product_Current', 'Symbol_ID_Current', 'Bloomberg_ID_Current', 'FIGI_ID_Current', 'Sector_Historical', 'Vendor_Historical', 'Product_Historical', 'Symbol_ID_Historical', 'Bloomberg_ID_Historical', 'FIGI_ID_Historical', 'Citations', 'Metric'] )
spend10 = spend10.rename(columns = {"variable":"Metric2", "value":"Value"})



spend11 = spend10[pd.notna(spend10.Value)]
spend11["Metric3"] = spend11["Metric"] + "_" + spend11["Metric2"]


spend12 = pd.pivot_table(spend11, index = ['Survey_Description_1', 'Survey_ID', 'Survey_Launch', 'Survey_Close', 'Announcement_Date', 'Sector_Current', 'Vendor_Current', 'Product_Current', 'Symbol_ID_Current', 'Bloomberg_ID_Current', 'FIGI_ID_Current', 'Sector_Historical', 'Vendor_Historical', 'Product_Historical', 'Symbol_ID_Historical', 'Bloomberg_ID_Historical', 'FIGI_ID_Historical', 'Citations'], columns = ["Metric3"], values = ["Value"]).reset_index()

final = []
for col in spend12.columns.values:
    
    if col[0] == "Value":
        final.append(col[1])
    else:
        final.append(col[0])

spend12.columns = final



# The Adoption_Rating, Increase_Rating, Decrease_Rating, Replacing_Rating, NetScore_Rating and MarketShare_Rating ratings                           
# are assigned based on the following decision tree algorithms.                                                                                     
# See Appendix A1-A5 and Appendix B of the accompanying methodology documentation for graphical representations of the decision tree algorithms.    


vcutoff = parameters.vcutoff
dcutoff = parameters.dcutoff
mincitations = parameters.mincitations


spend13 = spend12

def ratings(x):
    if x["Citations"] >= mincitations:
        #Adoption Rating
        if np.isnan(x["AdoptionP_Delta_sos_SurveyZ"]) == False and np.isnan(x["AdoptionP_Delta_yoy_SurveyZ"]) == False:
            if x["AdoptionP_Value_SurveyZ"] >= vcutoff:
                x["Adoption_Rating"] = "Positive"
            elif x["AdoptionP_Value_SurveyZ"] >= 0 and x["AdoptionP_Delta_sos_SurveyZ"] >= 0 and x["AdoptionP_Delta_yoy_SurveyZ"] >= 0 and (x["AdoptionP_Delta_sos_SurveyZ"] >= dcutoff or x["AdoptionP_Delta_yoy_SurveyZ"] >= dcutoff):
                x["Adoption_Rating"] = "Positive"
            elif x["AdoptionP_Delta_sos_SurveyZ"] <= 0 and x["AdoptionP_Delta_yoy_SurveyZ"] <= 0 and (x["AdoptionP_Delta_sos_SurveyZ"] <= -dcutoff or x["AdoptionP_Delta_yoy_SurveyZ"] <= -dcutoff):
                x["Adoption_Rating"] = "Negative"
        elif np.isnan(x["AdoptionP_Delta_sos_SurveyZ"]) == False and np.isnan(x["AdoptionP_Delta_yoy_SurveyZ"]) == True:
            if x["AdoptionP_Value_SurveyZ"] >= vcutoff:
                x["Adoption_Rating"] = "Positive"
            elif x["AdoptionP_Value_SurveyZ"] >= 0 and x["AdoptionP_Delta_sos_SurveyZ"] >= dcutoff:
                x["Adoption_Rating"] = "Positive"
            elif x["AdoptionP_Delta_sos_SurveyZ"] <= -dcutoff:
                x["Adoption_Rating"] = "Negative"
        elif np.isnan(x["AdoptionP_Delta_sos_SurveyZ"]) == True and np.isnan(x["AdoptionP_Delta_yoy_SurveyZ"]) == True:
            if x["AdoptionP_Value_SurveyZ"] >= vcutoff:
                x["Adoption_Rating"] = "Positive"
        #Increase Rating
        if np.isnan(x["IncreaseP_Delta_sos_SurveyZ"]) == False and np.isnan(x["IncreaseP_Delta_yoy_SurveyZ"]) == False:
            if x["IncreaseP_Value_SurveyZ"] >= vcutoff:
                x["Increase_Rating"] = "Positive"
            elif x["IncreaseP_Value_SurveyZ"] >= 0 and x["IncreaseP_Delta_sos_SurveyZ"] >= 0 and x["IncreaseP_Delta_yoy_SurveyZ"] >= 0 and (x["IncreaseP_Delta_sos_SurveyZ"] >= dcutoff or x["IncreaseP_Delta_yoy_SurveyZ"] >= dcutoff):
                x["Increase_Rating"] = "Positive"
            elif x["IncreaseP_Delta_sos_SurveyZ"] <= 0 and x["IncreaseP_Delta_yoy_SurveyZ"] <= 0 and (x["IncreaseP_Delta_sos_SurveyZ"] <= -dcutoff or x["IncreaseP_Delta_yoy_SurveyZ"] <= -dcutoff):
                x["Increase_Rating"] = "Negative"
        elif np.isnan(x["IncreaseP_Delta_sos_SurveyZ"]) == False and np.isnan(x["IncreaseP_Delta_yoy_SurveyZ"]) == True:
            if x["IncreaseP_Value_SurveyZ"] >= vcutoff:
                x["Increase_Rating"] = "Positive"
            elif x["IncreaseP_Value_SurveyZ"] >= 0 and x["IncreaseP_Delta_sos_SurveyZ"] >= dcutoff:
                x["Increase_Rating"] = "Positive"
            elif x["IncreaseP_Delta_sos_SurveyZ"] <= -dcutoff:
                x["Increase_Rating"] = "Negative"
        elif np.isnan(x["IncreaseP_Delta_sos_SurveyZ"]) == True and np.isnan(x["IncreaseP_Delta_yoy_SurveyZ"]) == True:
            if x["IncreaseP_Value_SurveyZ"] >= vcutoff:
                x["Increase_Rating"] = "Positive"
        #Decrease Rating
        if np.isnan(x["DecreaseP_Delta_sos_SurveyZ"]) == False and np.isnan(x["DecreaseP_Delta_yoy_SurveyZ"]) == False:
            if x["DecreaseP_Value_SurveyZ"] >= vcutoff:
                x["Decrease_Rating"] = "Negative"
            elif x["DecreaseP_Value_SurveyZ"] >= 0 and x["DecreaseP_Delta_sos_SurveyZ"] >= 0 and x["DecreaseP_Delta_yoy_SurveyZ"] >= 0 and (x["DecreaseP_Delta_sos_SurveyZ"] >= dcutoff or x["DecreaseP_Delta_yoy_SurveyZ"] >= dcutoff):
                x["Decrease_Rating"] = "Negative"
            elif x["DecreaseP_Delta_sos_SurveyZ"] <= 0 and x["DecreaseP_Delta_yoy_SurveyZ"] <= 0 and (x["DecreaseP_Delta_sos_SurveyZ"] <= -dcutoff or x["DecreaseP_Delta_yoy_SurveyZ"] <= -dcutoff):
                x["Decrease_Rating"] = "Positive"
        elif np.isnan(x["DecreaseP_Delta_sos_SurveyZ"]) == False and np.isnan(x["DecreaseP_Delta_yoy_SurveyZ"]) == True:
            if x["DecreaseP_Value_SurveyZ"] >= vcutoff:
                x["Decrease_Rating"] = "Negative"
            elif x["DecreaseP_Value_SurveyZ"] >= 0 and x["DecreaseP_Delta_sos_SurveyZ"] >= dcutoff:
                x["Decrease_Rating"] = "Negative"
            elif x["DecreaseP_Delta_sos_SurveyZ"] <= -dcutoff:
                x["Decrease_Rating"] = "Positive"
        elif np.isnan(x["DecreaseP_Delta_sos_SurveyZ"]) == True and np.isnan(x["DecreaseP_Delta_yoy_SurveyZ"]) == True:
            if x["DecreaseP_Value_SurveyZ"] >= vcutoff:
                x["Decrease_Rating"] = "Negative"
        #Replace Rating
        if np.isnan(x["ReplacingP_Delta_sos_SurveyZ"]) == False and np.isnan(x["ReplacingP_Delta_yoy_SurveyZ"]) == False:
            if x["ReplacingP_Value_SurveyZ"] >= vcutoff:
                x["Replacing_Rating"] = "Negative"
            elif x["ReplacingP_Value_SurveyZ"] >= 0 and x["ReplacingP_Delta_sos_SurveyZ"] >= 0 and x["ReplacingP_Delta_yoy_SurveyZ"] >= 0 and (x["ReplacingP_Delta_sos_SurveyZ"] >= dcutoff or x["ReplacingP_Delta_yoy_SurveyZ"] >= dcutoff):
                x["Replacing_Rating"] = "Negative"
            elif x["ReplacingP_Delta_sos_SurveyZ"] <= 0 and x["ReplacingP_Delta_yoy_SurveyZ"] <= 0 and (x["ReplacingP_Delta_sos_SurveyZ"] <= -dcutoff or x["ReplacingP_Delta_yoy_SurveyZ"] <= -dcutoff):
                x["Replacing_Rating"] = "Positive"
        elif np.isnan(x["ReplacingP_Delta_sos_SurveyZ"]) == False and np.isnan(x["ReplacingP_Delta_yoy_SurveyZ"]) == True:
            if x["ReplacingP_Value_SurveyZ"] >= vcutoff:
                x["Replacing_Rating"] = "Negative"
            elif x["ReplacingP_Value_SurveyZ"] >= 0 and x["ReplacingP_Delta_sos_SurveyZ"] >= dcutoff:
                x["Replacing_Rating"] = "Negative"
            elif x["ReplacingP_Delta_sos_SurveyZ"] <= -dcutoff:
                x["Replacing_Rating"] = "Positive"
        elif np.isnan(x["ReplacingP_Delta_sos_SurveyZ"]) == True and np.isnan(x["ReplacingP_Delta_yoy_SurveyZ"]) == True:
            if x["ReplacingP_Value_SurveyZ"] >= vcutoff:
                x["Replacing_Rating"] = "Negative"
        #Net Score Rating
        if np.isnan(x["NetScore_Delta_sos_SurveyZ"]) == False and np.isnan(x["NetScore_Delta_yoy_SurveyZ"]) == False:
            if x["NetScore_Value_SurveyZ"] >= vcutoff:
                x["NetScore_Rating"] = "Positive"
            elif x["NetScore_Value_SurveyZ"] >= 0 and x["NetScore_Delta_sos_SurveyZ"] >= 0 and x["NetScore_Delta_yoy_SurveyZ"] >= 0 and (x["NetScore_Delta_sos_SurveyZ"] >= dcutoff or x["NetScore_Delta_yoy_SurveyZ"] >= dcutoff):
                x["NetScore_Rating"] = "Positive"
            elif x["NetScore_Value_SurveyZ"] <= 0 and x["NetScore_Delta_sos_SurveyZ"] <= 0 and x["NetScore_Delta_yoy_SurveyZ"] <= 0 and (x["NetScore_Delta_sos_SurveyZ"] <= -dcutoff or x["NetScore_Delta_yoy_SurveyZ"] <= -dcutoff):
                x["NetScore_Rating"] = "Negative"
            elif x["NetScore_Value_SurveyZ"] <= -vcutoff:
                x["NetScore_Rating"] = "Negative"
        elif np.isnan(x["NetScore_Delta_sos_SurveyZ"]) == False and np.isnan(x["NetScore_Delta_yoy_SurveyZ"]) == True:
            if x["NetScore_Value_SurveyZ"] >= vcutoff:
                x["NetScore_Rating"] = "Positive"
            elif x["NetScore_Value_SurveyZ"] >= 0 and x["NetScore_Delta_sos_SurveyZ"] >= dcutoff:
                x["NetScore_Rating"] = "Positive"
            elif x["NetScore_Value_SurveyZ"] <= 0 and x["NetScore_Delta_sos_SurveyZ"] <= -dcutoff:
                x["NetScore_Rating"] = "Negative"
            elif x["NetScore_Value_SurveyZ"] <= -vcutoff:
                x["NetScore_Rating"] = "Negative"
        elif np.isnan(x["NetScore_Delta_sos_SurveyZ"]) == True and np.isnan(x["NetScore_Delta_yoy_SurveyZ"]) == True:
            if x["NetScore_Value_SurveyZ"] >= vcutoff:
                x["NetScore_Rating"] = "Positive"
            elif x["NetScore_Value_SurveyZ"] <= -vcutoff:
                x["NetScore_Rating"] = "Negative"
        #Market Share Rating
        if np.isnan(x["MarketShare_Delta_sos_SurveyZ"]) == False and np.isnan(x["MarketShare_Delta_yoy_SurveyZ"]) == False:
            if x["MarketShare_Delta_sos_SurveyZ"] >= 0 and x["MarketShare_Delta_yoy_SurveyZ"] >= 0 and (x["MarketShare_Delta_sos_SurveyZ"] >= dcutoff or x["MarketShare_Delta_yoy_SurveyZ"] >= dcutoff):
                x["MarketShare_Rating"] = "Positive"
            elif x["MarketShare_Delta_sos_SurveyZ"] <= 0 and x["MarketShare_Delta_yoy_SurveyZ"] <= 0 and (x["MarketShare_Delta_sos_SurveyZ"] <= -dcutoff or x["MarketShare_Delta_yoy_SurveyZ"] <= -dcutoff):
                x["MarketShare_Rating"] = "Negative"
        elif np.isnan(x["MarketShare_Delta_sos_SurveyZ"]) == False and np.isnan(x["MarketShare_Delta_yoy_SurveyZ"]) == True:
            if x["MarketShare_Delta_sos_SurveyZ"] >= dcutoff:
                x["MarketShare_Rating"] = "Positive"
            elif x["MarketShare_Delta_sos_SurveyZ"] <= -dcutoff:
                x["MarketShare_Rating"] = "Negative"

    return x




spend13 = spend13.apply(ratings, axis = 1).drop(["MarketShare_Value_SurveyZ"], axis = 1)


spend_final = spend13.sort_values(by = ['Survey_Description_1', 'Survey_ID', 'Survey_Launch', 'Survey_Close', 'Announcement_Date', 'Sector_Current', 'Vendor_Current', 'Product_Current', 'Symbol_ID_Current', 'Bloomberg_ID_Current', 'FIGI_ID_Current', 'Sector_Historical', 'Vendor_Historical', 'Product_Historical', 'Symbol_ID_Historical', 'Bloomberg_ID_Historical', 'FIGI_ID_Historical']).replace({'---': np.nan})


###########################################
## Peer Benchmarking / Competition Theme ##
###########################################


# Pairwise combinations of vendors within the same sector are matched.
# Shared accounts Citations and Net Scores are calculated for each pairwise combination.

peer1a = source1.rename({"Vendor_Current":"Vendor_Filter", "Product_Current":"Product_Filter", "Metric":"Metric_Filter"}, axis = 1)
peer1b = source1[["Survey_ID", "Respondent_ID", "Sector_Current", "Vendor_Current", "Product_Current", "Metric"]].rename({"Vendor_Current":"Vendor_Calc", "Product_Current":"Product_Calc", "Metric":"Metric_Calc"}, axis = 1)



temp = peer1a.merge(peer1b, how = 'inner', on = ['Survey_ID', 'Respondent_ID', 'Sector_Current'])
temp = temp.fillna('.')
temp = temp[(temp['Vendor_Filter'] != temp['Vendor_Calc']) | (temp['Product_Filter'] != temp['Product_Calc'])] 
temp = temp.replace({'.': np.nan})


peer3 = temp
peer3['Metric_Filter_Group'] = np.where(np.logical_or(peer3['Metric_Filter']=='INCREASE', peer3['Metric_Filter'] == 'ADOPTION'), 'Pos', 'Neg')

peer4 = peer3.fillna('---').groupby(["Survey_Description_1", "Survey_ID", "Survey_Launch", "Survey_Close", "Announcement_Date", "Sector_Current", "Vendor_Filter", "Product_Filter", "Symbol_ID_Current", "Bloomberg_ID_Current", "FIGI_ID_Current", "Sector_Historical", "Vendor_Historical", "Product_Historical", "Symbol_ID_Historical", "Bloomberg_ID_Historical", "FIGI_ID_Historical", "Metric_Filter_Group", "Vendor_Calc", "Product_Calc", "Metric_Calc"]).size().reset_index(name = "Count") 




peer5 = pd.pivot_table(peer4, index = ["Survey_Description_1", "Survey_ID", "Survey_Launch", "Survey_Close", "Announcement_Date", "Sector_Current", "Vendor_Filter", "Product_Filter", "Symbol_ID_Current", "Bloomberg_ID_Current", "FIGI_ID_Current", "Sector_Historical", "Vendor_Historical", "Product_Historical", "Symbol_ID_Historical", "Bloomberg_ID_Historical", "FIGI_ID_Historical", "Metric_Filter_Group", "Vendor_Calc", "Product_Calc"], columns = "Metric_Calc", values = "Count").reset_index()



peer6 = peer5.fillna(0).replace({'---': np.nan})

if "ADOPTION" not in peer6.columns:
    peer6['ADOPTION'] = 0.0

# Year-over-Year values for each metric are merged on to calculate deltas.
# Deltas are used to measure longer-term trends.


peer7_yoy = peer6[['Survey_ID', 'Sector_Current', 'Vendor_Filter', 'Product_Filter', "Metric_Filter_Group", "Vendor_Calc", "Product_Calc"]].rename(columns = {'Survey_ID':'Survey_ID_yoy'})
peer7 = peer6

peer7['Peer_Citations'] = peer7['ADOPTION'] + peer7['INCREASE'] + peer7['FLAT'] + peer7['DECREASE'] + peer7['REPLACING']
peer7['Peer_NetScore'] = (peer7['ADOPTION'] + peer7['INCREASE'] - peer7['DECREASE'] - peer7['REPLACING']) / peer7['Peer_Citations']
peer7 = peer7.drop(['ADOPTION', 'INCREASE', 'FLAT', 'DECREASE', 'REPLACING'], axis = 1)

peer7['Survey_ID_yoy'] = np.nan
peer7.loc[peer7.Survey_ID == 3, 'Survey_ID_yoy'] = 1
peer7.loc[peer7.Survey_ID == 4, 'Survey_ID_yoy'] = 2
peer7.loc[peer7.Survey_ID == 5, 'Survey_ID_yoy'] = 2
peer7.loc[(peer7['Survey_ID'] >= 6) & (peer7['Survey_ID'] % 2  == 0), 'Survey_ID_yoy'] = peer7['Survey_ID'] - 3
peer7.loc[(peer7['Survey_ID'] >= 6) & (peer7['Survey_ID'] % 2  == 1), 'Survey_ID_yoy'] = peer7['Survey_ID'] - 4

peer7_yoy = peer7[['Survey_ID', 'Sector_Current', 'Vendor_Filter', 'Product_Filter', "Metric_Filter_Group", "Vendor_Calc", "Product_Calc", "Peer_Citations", "Peer_NetScore"]].rename(columns = {'Survey_ID':'Survey_ID_yoy', 'Peer_Citations':'Peer_Citations_yoy', 'Peer_NetScore' : 'Peer_NetScore_yoy'})

peer7 = peer7.sort_values(by = ['Sector_Current', 'Vendor_Filter', 'Product_Filter', 'Metric_Filter_Group', 'Vendor_Calc', 'Product_Calc', 'Survey_ID_yoy'])
peer7_yoy = peer7_yoy.sort_values(by = ['Sector_Current', 'Vendor_Filter', 'Product_Filter', 'Metric_Filter_Group', 'Vendor_Calc', 'Product_Calc', 'Survey_ID_yoy'])

peer8 = peer7.merge(peer7_yoy, how = 'left', on = ['Sector_Current', 'Vendor_Filter', 'Product_Filter', 'Metric_Filter_Group', 'Vendor_Calc', 'Product_Calc', 'Survey_ID_yoy'])
peer8 = peer8.drop(['Survey_ID_yoy'], axis = 1)
peer8pos = peer8.loc[peer8['Metric_Filter_Group'] == "Pos"].rename(columns = {'Peer_Citations':'PeerPos_Citations', 'Peer_NetScore':'PeerPos_NetScore', 'Peer_Citations_yoy':'PeerPos_Citations_yoy', 'Peer_NetScore_yoy':'PeerPos_NetScore_yoy'})
peer8neg = peer8.loc[peer8['Metric_Filter_Group'] == "Neg"].rename(columns = {'Peer_Citations':'PeerNeg_Citations', 'Peer_NetScore':'PeerNeg_NetScore', 'Peer_Citations_yoy':'PeerNeg_Citations_yoy', 'Peer_NetScore_yoy':'PeerNeg_NetScore_yoy'})


# Each Competitor (Vendor_Calc) is assigned an Accelerating, Decelerating, or None Net Effect within the primary vendor's (Vendor_Filter) accounts.
# See Appendix C of the accompanying methodology documentation for a graphical representation of this decision tree algorithm.

peer9 = peer8pos.merge(peer8neg, how = "outer", on = ["Survey_Description_1", "Survey_ID", "Survey_Launch", "Survey_Close", "Announcement_Date", "Sector_Current", "Vendor_Filter", "Product_Filter", "Symbol_ID_Current", "Bloomberg_ID_Current", "FIGI_ID_Current", "Sector_Historical", "Vendor_Historical", "Product_Historical", "Symbol_ID_Historical", "Bloomberg_ID_Historical", "FIGI_ID_Historical", "Vendor_Calc", "Product_Calc"]) #.drop(['Metric_Filter_Group'], axis = 1)
peer9 = peer9.sort_values(by = ["Survey_Description_1", "Survey_ID", "Survey_Launch", "Survey_Close", "Announcement_Date", "Sector_Current", "Vendor_Filter", "Product_Filter", "Symbol_ID_Current", "Bloomberg_ID_Current", "FIGI_ID_Current", "Sector_Historical", "Vendor_Historical", "Product_Historical", "Symbol_ID_Historical", "Bloomberg_ID_Historical", "FIGI_ID_Historical", "Vendor_Calc", "Product_Calc"] )

peer9['NetEffect'] = np.nan


peermincitations = parameters.peermincitations
deltayoy = parameters.deltayoy


def NetEffect(x):
    if x['PeerPos_Citations'] >= peermincitations and x['PeerPos_Citations_yoy'] >= peermincitations and x['PeerNeg_Citations'] >= peermincitations and x['PeerNeg_Citations_yoy'] >= peermincitations:
        if (x['PeerPos_NetScore'] - x['PeerPos_NetScore_yoy'] > 0) and (x['PeerNeg_NetScore'] - x['PeerNeg_NetScore_yoy'] > 0) and (x['PeerPos_NetScore'] - x['PeerPos_NetScore_yoy'] >= deltayoy or x['PeerNeg_NetScore'] - x['PeerNeg_NetScore_yoy'] >= deltayoy):
            x['NetEffect'] = 'Accelerating'
        if (x['PeerPos_NetScore'] - x['PeerPos_NetScore_yoy'] < 0) and (x['PeerNeg_NetScore'] - x['PeerNeg_NetScore_yoy'] < 0) and (x['PeerPos_NetScore'] - x['PeerPos_NetScore_yoy'] <= -deltayoy or x['PeerNeg_NetScore'] - x['PeerNeg_NetScore_yoy'] <= -deltayoy):
            x['NetEffect'] = 'Decelerating'
    return x

peer9 = peer9.apply(NetEffect, axis = 1).drop(["Metric_Filter_Group_y", "Metric_Filter_Group_x"], axis = 1)

peer10 = peer9.fillna('---').groupby(["Survey_Description_1", "Survey_ID", "Survey_Launch", "Survey_Close", "Announcement_Date", "Sector_Current", "Vendor_Filter", "Product_Filter", "Symbol_ID_Current", "Bloomberg_ID_Current", "FIGI_ID_Current", "Sector_Historical", "Vendor_Historical", "Product_Historical", "Symbol_ID_Historical", "Bloomberg_ID_Historical", "FIGI_ID_Historical", "NetEffect"]).size().reset_index(name = "Count")

peer11 = pd.pivot_table(peer10, index = ["Survey_Description_1", "Survey_ID", "Survey_Launch", "Survey_Close", "Announcement_Date", "Sector_Current", "Vendor_Filter", "Product_Filter", "Symbol_ID_Current", "Bloomberg_ID_Current", "FIGI_ID_Current", "Sector_Historical", "Vendor_Historical", "Product_Historical", "Symbol_ID_Historical", "Bloomberg_ID_Historical", "FIGI_ID_Historical"], columns = "NetEffect", values = "Count").reset_index().drop("---", axis = 1)

peer12 = peer11.fillna(0)

peer13 = peer12.rename(columns = {'Vendor_Filter':'Vendor_Current', 'Product_Filter':'Product_Current', 'Accelerating':'Peer_Accelerating', 'Decelerating':'Peer_Decelerating'})
peer13["Peer_Rating"] = np.nan

# The Peer_Rating rating is assigned based on the following decision tree algorithm.
# See Appendix C of the accompanying methodology documentation for a graphical representation of this decision tree algorithm.

peerdelta = parameters.peerdelta

def peerRating(x):
    if x["Peer_Accelerating"] - x["Peer_Decelerating"] >= peerdelta:
        x["Peer_Rating"] = "Negative"
    elif x["Peer_Decelerating"] - x["Peer_Accelerating"] >= peerdelta:
        x["Peer_Rating"] = "Positive"
    return x

peer13 = peer13.apply(peerRating, axis = 1)

peer_final = peer13.sort_values(by = ["Survey_Description_1", "Survey_ID", "Survey_Launch", "Survey_Close", "Announcement_Date", "Sector_Current", "Vendor_Current", "Product_Current", "Symbol_ID_Current", "Bloomberg_ID_Current", "FIGI_ID_Current", "Sector_Historical", "Vendor_Historical", "Product_Historical", "Symbol_ID_Historical", "Bloomberg_ID_Historical", "FIGI_ID_Historical"]).replace({'---': np.nan})







#####################################################
## Alignment With Major Public Cloud Vendors Theme ##
#####################################################


# Two customer groups are identified: a Cloud Group and a Control Group.
# The Cloud Group consists of customers who are Adopting or Increasing spend with a Public Cloud vendor (AWS, Microsoft, Google),
# while the Control Group consists of all others.
# Each vendor's Net Score and Citations are calculated among each of these customer groups.

cloud1 = source1[source1["Sector_Current"] == "CLOUD COMPUTING / MANAGED HOSTING"]
cloud1 = cloud1[(cloud1["Vendor_Current"] == "AWS") | (cloud1["Vendor_Current"] == "Microsoft") | (cloud1["Vendor_Current"] == "Google")]
cloud1 = cloud1[(cloud1["Metric"] == "ADOPTION") | (cloud1["Metric"] == "INCREASE")]
cloud1 = cloud1.sort_values(by = ["Survey_Description_1", "Survey_ID", "Survey_Launch", "Survey_Close", "Announcement_Date", "Respondent_ID", "Sector_Current", "Vendor_Current", "Product_Current", "Symbol_ID_Current", "Bloomberg_ID_Current", "FIGI_ID_Current"])

cloud2 = cloud1.drop_duplicates(subset = ["Survey_Description_1", "Survey_ID", "Survey_Launch", "Survey_Close", "Announcement_Date", "Respondent_ID"])[["Survey_Description_1", "Survey_ID", "Survey_Launch", "Survey_Close", "Announcement_Date", "Respondent_ID"]]
cloud2["Group"] = "Cloud"
cloud_n = cloud2.groupby(["Survey_Description_1", "Survey_ID", "Survey_Launch", "Survey_Close", "Announcement_Date"]).size().reset_index(name = "Cloud_N")

cloud3 = source1.merge(cloud2, how = 'left', on = ["Survey_Description_1", "Survey_ID", "Survey_Launch", "Survey_Close", "Announcement_Date", "Respondent_ID"])
cloud3["Group"] = cloud3["Group"].fillna("Control")

cloud3 = cloud3.fillna("---")
cloud4 = cloud3.groupby(["Group", "Survey_Description_1", "Survey_ID", "Survey_Launch", "Survey_Close", "Announcement_Date", "Sector_Current", "Vendor_Current", "Product_Current", "Symbol_ID_Current", "Bloomberg_ID_Current", "FIGI_ID_Current", "Sector_Historical", "Vendor_Historical", "Product_Historical", "Symbol_ID_Historical", "Bloomberg_ID_Historical", "FIGI_ID_Historical", "Metric"]).size().reset_index(name = "Count")

cloud5 = pd.pivot_table(cloud4, index = ["Group", "Survey_Description_1", "Survey_ID", "Survey_Launch", "Survey_Close", "Announcement_Date", "Sector_Current", "Vendor_Current", "Product_Current", "Symbol_ID_Current", "Bloomberg_ID_Current", "FIGI_ID_Current", "Sector_Historical", "Vendor_Historical", "Product_Historical", "Symbol_ID_Historical", "Bloomberg_ID_Historical", "FIGI_ID_Historical"], columns = "Metric", values = "Count").reset_index() #.drop("---", axis = 1)

cloud6 = cloud5.fillna(0)

cloud7 = cloud6

cloud7['Citations'] = cloud7['ADOPTION'] + cloud7['INCREASE'] + cloud7['FLAT'] + cloud7['DECREASE'] + cloud7['REPLACING']
cloud7['NetScore'] = (cloud7['ADOPTION'] + cloud7['INCREASE'] - cloud7['DECREASE'] - cloud7['REPLACING']) / cloud7['Citations']

cloud7 = cloud7.drop(['ADOPTION', 'INCREASE', 'FLAT', 'DECREASE', 'REPLACING'], axis = 1)

cloud7a = (cloud7[cloud7['Group'] == "Cloud"]).drop(['Group'], axis = 1)
cloud7b = (cloud7[cloud7['Group'] == "Control"]).drop(['Group'], axis = 1)

cloud7a = cloud7a.rename(columns = {"Citations":"Cloud_Citations", "NetScore":"Cloud_NetScore"})
cloud7b = cloud7b.rename(columns = {"Citations":"Control_Citations", "NetScore":"Control_NetScore"})


cloud8 = cloud7a.merge(cloud7b, how = 'outer', on = ["Survey_Description_1", "Survey_ID", "Survey_Launch", "Survey_Close", "Announcement_Date", "Sector_Current", "Vendor_Current", "Product_Current", "Symbol_ID_Current", "Bloomberg_ID_Current", "FIGI_ID_Current", "Sector_Historical", "Vendor_Historical", "Product_Historical", "Symbol_ID_Historical", "Bloomberg_ID_Historical", "FIGI_ID_Historical"])



# Survey-over-Survey and Year-over-Year values for each metric are merged on to calculate deltas.
# Deltas are used to measure recent inflections and longer-term trends.


cloud9 = cloud8.merge(cloud_n, on = ["Survey_Description_1", "Survey_ID", "Survey_Launch", "Survey_Close", "Announcement_Date"])
cloud9["Cloud_NetScore_Delta_Control"] = cloud9["Cloud_NetScore"] - cloud9["Control_NetScore"]
cloud9.loc[np.isnan(cloud9["Control_Citations"]) == True, "Control_Citations"] = 0
cloud9["Survey_Citations"] = cloud9["Cloud_Citations"] + cloud9["Control_Citations"]
cloud9["Cloud_Overlap"] = cloud9["Cloud_Citations"] / cloud9["Survey_Citations"]
cloud9["Cloud_Share"] = cloud9["Cloud_Citations"] / cloud9["Cloud_N"]

cloud9['Survey_ID_sos'] = cloud9['Survey_ID'] - 1
cloud9['Survey_ID_yoy'] = np.nan
cloud9.loc[cloud9.Survey_ID == 3, 'Survey_ID_yoy'] = 1
cloud9.loc[cloud9.Survey_ID == 4, 'Survey_ID_yoy'] = 2
cloud9.loc[cloud9.Survey_ID == 5, 'Survey_ID_yoy'] = 2
cloud9.loc[(cloud9['Survey_ID'] >= 6) & (cloud9['Survey_ID'] % 2  == 0), 'Survey_ID_yoy'] = cloud9['Survey_ID'] - 3
cloud9.loc[(cloud9['Survey_ID'] >= 6) & (cloud9['Survey_ID'] % 2  == 1), 'Survey_ID_yoy'] = cloud9['Survey_ID'] - 4



cloud9_sos = cloud9[['Survey_ID', 'Sector_Current', 'Vendor_Current', 'Product_Current', 'Symbol_ID_Current', 'Bloomberg_ID_Current', 'FIGI_ID_Current', 'Cloud_Citations', 'Cloud_NetScore', 'Cloud_Share']].rename(columns = {'Survey_ID':'Survey_ID_sos', 'Cloud_Citations':'Cloud_Citations_sos', 'Cloud_NetScore':'Cloud_NetScore_sos', 'Cloud_Share':'Cloud_Share_sos'})
cloud9_yoy = cloud9[['Survey_ID', 'Sector_Current', 'Vendor_Current', 'Product_Current', 'Symbol_ID_Current', 'Bloomberg_ID_Current', 'FIGI_ID_Current', 'Cloud_Citations', 'Cloud_NetScore', 'Cloud_Share']].rename(columns = {'Survey_ID':'Survey_ID_yoy', 'Cloud_Citations':'Cloud_Citations_yoy', 'Cloud_NetScore':'Cloud_NetScore_yoy', 'Cloud_Share':'Cloud_Share_yoy'})

cloud9 = cloud9.sort_values(by = ['Sector_Current', 'Vendor_Current', 'Product_Current', 'Symbol_ID_Current', 'Bloomberg_ID_Current', 'FIGI_ID_Current', 'Survey_ID_sos'])
cloud9_sos = cloud9_sos.sort_values(by = ['Sector_Current', 'Vendor_Current', 'Product_Current', 'Symbol_ID_Current', 'Bloomberg_ID_Current', 'FIGI_ID_Current', 'Survey_ID_sos'])

cloud10a = cloud9.merge(cloud9_sos, how = 'left', on = ['Sector_Current', 'Vendor_Current', 'Product_Current', 'Symbol_ID_Current', 'Bloomberg_ID_Current', 'FIGI_ID_Current', 'Survey_ID_sos'])
cloud10a = cloud10a.sort_values(by = ['Sector_Current', 'Vendor_Current', 'Product_Current', 'Symbol_ID_Current', 'Bloomberg_ID_Current', 'FIGI_ID_Current', 'Survey_ID_yoy'])

cloud9_yoy = cloud9_yoy.sort_values(by = ['Sector_Current', 'Vendor_Current', 'Product_Current', 'Symbol_ID_Current', 'Bloomberg_ID_Current', 'FIGI_ID_Current', 'Survey_ID_yoy'])

cloud10b = cloud10a.merge(cloud9_yoy, how = 'left', on = ['Sector_Current', 'Vendor_Current', 'Product_Current', 'Symbol_ID_Current', 'Bloomberg_ID_Current', 'FIGI_ID_Current', 'Survey_ID_yoy']).drop(["Survey_ID_sos", "Survey_ID_yoy"], axis = 1)
cloud10b["Cloud_NetScore_Delta_sos"] = cloud10b["Cloud_NetScore"] - cloud10b["Cloud_NetScore_sos"]
cloud10b["Cloud_NetScore_Delta_yoy"] = cloud10b["Cloud_NetScore"] - cloud10b["Cloud_NetScore_yoy"]


# The Cloud_Rating rating is assigned based on the following decision tree algorithm.
# See Appendix D of the accompanying methodology documentation for a graphical representation of this decision tree algorithm.

cloud10b["Cloud_Rating"] = np.nan

def cloudratings(x):
    if np.isnan(x["Cloud_NetScore_sos"]) == False and np.isnan(x["Cloud_NetScore_yoy"]) == False:
        if x["Cloud_NetScore"] >= .7:
            x["Cloud_Rating"] = "Positive"
        elif x["Cloud_NetScore"] >= .35 and x["Cloud_NetScore_Delta_sos"] > 0 and x["Cloud_NetScore_Delta_yoy"] > 0 and (x["Cloud_NetScore_Delta_sos"] + x["Cloud_NetScore_Delta_yoy"]) / 2 > .02 and x["Cloud_NetScore_Delta_Control"] > 0:
            x["Cloud_Rating"] = "Positive"
        elif x["Cloud_NetScore"] >= .35 and x["Cloud_NetScore_Delta_sos"] > -.05 and x["Cloud_NetScore_Delta_yoy"] > -.05 and (x["Cloud_NetScore_Delta_sos"] + x["Cloud_NetScore_Delta_yoy"]) / 2 > -.02 and x["Cloud_NetScore_Delta_Control"] > .05 and x["Control_Citations"] >= 5:
            x["Cloud_Rating"] = "Positive"
        elif x["Cloud_NetScore_Delta_sos"] < 0 and x["Cloud_NetScore_Delta_yoy"] < 0 and (x["Cloud_NetScore_Delta_sos"] + x["Cloud_NetScore_Delta_yoy"]) / 2 < -.02 and x["Cloud_NetScore_Delta_Control"] < 0:
            x["Cloud_Rating"] = "Negative"
        elif x["Cloud_NetScore_Delta_sos"] < .05 and x["Cloud_NetScore_Delta_yoy"] < .05 and (x["Cloud_NetScore_Delta_sos"] + x["Cloud_NetScore_Delta_yoy"]) / 2 < .02 and x["Cloud_NetScore_Delta_Control"] < -.05 and x["Control_Citations"] >= 5:
            x["Cloud_Rating"] = "Negative"
        elif x["Cloud_NetScore"] <= .1:
            x["Cloud_Rating"] = "Negative"
        
        if x["Cloud_Rating"] == "Positive" and (x["Cloud_NetScore_sos"] <= 0 or x["Cloud_NetScore_yoy"] <= 0):
            x["Cloud_Rating"] = np.nan
        if x["Cloud_Rating"] == "Positive" and (x["Cloud_Share"] < x["Cloud_Share_sos"] * .5 or x["Cloud_Share"] < x["Cloud_Share_yoy"] * .5):
            x["Cloud_Rating"] = np.nan
        if x["Cloud_Rating"] == "Negative" and (x["Cloud_Share"] > x["Cloud_Share_sos"] * 1.5 or x["Cloud_Share"] > x["Cloud_Share_yoy"] * 1.5):
            x["Cloud_Rating"] = np.nan

        if x["Cloud_NetScore"] <= .1:
            x["Cloud_Rating"] = "Negative"

    if np.isnan(x["Cloud_NetScore_sos"]) == False and np.isnan(x["Cloud_NetScore_yoy"]) == True:
        if x["Cloud_NetScore"] >= .7:
            x["Cloud_Rating"] = "Positive"
        elif x["Cloud_NetScore"] >= .35 and x["Cloud_NetScore_Delta_sos"] > .02 and x["Cloud_NetScore_Delta_Control"] > .02:
            x["Cloud_Rating"] = "Positive"
        elif x["Cloud_NetScore"] >= .35 and x["Cloud_NetScore_Delta_sos"] > -.02 and x["Cloud_NetScore_Delta_Control"] > .05 and x["Control_Citations"] >= .05:
            x["Cloud_Rating"] = "Positive"
        elif x["Cloud_NetScore_Delta_sos"] < -.02 and x["Cloud_NetScore_Delta_Control"] < -.02:
            x["Cloud_Rating"] = "Negative"
        elif x["Cloud_NetScore_Delta_sos"] < .02 and x["Cloud_NetScore_Delta_Control"] < -.05 and x["Control_Citations"] >= .05:
            x["Cloud_Rating"] = "Negative"
        elif x["Cloud_NetScore"] <= .1:
            x["Cloud_Rating"] = "Negative"

        if x["Cloud_Rating"] == "Positive" and x["Cloud_NetScore_sos"] <= 0:
            x["Cloud_Rating"] = np.nan
        if x["Cloud_Rating"] == "Positive" and x["Cloud_Share"] < x["Cloud_Share_sos"] * .5:
            x["Cloud_Rating"] = np.nan
        if x["Cloud_Rating"] == "Negative" and x["Cloud_Share"] > x["Cloud_Share_sos"] * 1.5:
            x["Cloud_Rating"] = np.nan

        if x["Cloud_NetScore"] <= .1:
            x["Cloud_Rating"] = "Negative"

    if np.isnan(x["Cloud_NetScore_sos"]) == True and np.isnan(x["Cloud_NetScore_yoy"]) == True:
        if x["Cloud_NetScore"] >= .7:
            x["Cloud_Rating"] = "Positive"
        elif x["Cloud_NetScore"] <= .1:
            x["Cloud_Rating"] = "Negative"
    
    if x["Survey_Citations"] < 30 or x["Cloud_Citations"] < 5 or x["Cloud_Overlap"] < .4:
        x["Cloud_Rating"] = np.nan

    if x["Sector_Current"] == "CLOUD COMPUTING / MANAGED HOSTING" and (x["Vendor_Current"] == "AWS" or x["Vendor_Current"] == "Microsoft" or x["Vendor_Current"] == "Google"):
        x["Cloud_Rating"] = "Positive"

    return x

cloud11 = cloud10b.apply(cloudratings, axis = 1).drop(["Cloud_Citations_sos", "Cloud_NetScore_sos", "Cloud_Citations_yoy", "Cloud_NetScore_yoy", "Control_Citations"], axis = 1)

cloud_final = cloud11.sort_values(by = ["Survey_Description_1", "Survey_ID", "Survey_Launch", "Survey_Close", "Announcement_Date", "Sector_Current", "Vendor_Current", "Product_Current", "Symbol_ID_Current", "Bloomberg_ID_Current", "FIGI_ID_Current", "Sector_Historical", "Vendor_Historical", "Product_Historical", "Symbol_ID_Historical", "Bloomberg_ID_Historical", "FIGI_ID_Historical"]).replace({'---': np.nan})
spend_final = spend_final.sort_values(by = ["Survey_Description_1", "Survey_ID", "Survey_Launch", "Survey_Close", "Announcement_Date", "Sector_Current", "Vendor_Current", "Product_Current", "Symbol_ID_Current", "Bloomberg_ID_Current", "FIGI_ID_Current", "Sector_Historical", "Vendor_Historical", "Product_Historical", "Symbol_ID_Historical", "Bloomberg_ID_Historical", "FIGI_ID_Historical"])
peer_final = peer_final.sort_values(by = ["Survey_Description_1", "Survey_ID", "Survey_Launch", "Survey_Close", "Announcement_Date", "Sector_Current", "Vendor_Current", "Product_Current", "Symbol_ID_Current", "Bloomberg_ID_Current", "FIGI_ID_Current", "Sector_Historical", "Vendor_Historical", "Product_Historical", "Symbol_ID_Historical", "Bloomberg_ID_Historical", "FIGI_ID_Historical"])



############################################
## Model Creation and Performance Testing ##
############################################


# Metrics from all themes are merged together.
# Stock Price returns are merged on.
# Stock Price return survey averages and z-scores are calculated to determine outperformers and underperformers.
# See Appendix E of the accompanying methodology documentation for a graphical representation of this process.

# Please note, only data from 2015 and on is used for the remainder of this program.
# The time difference between the start of the Source Dataset(2010) and the remainder of Insight Dataset(2015) is primarily due to two factors:
# [1] a shift in technology spend that occurred during that time frame from a largely CapEx model to a mix between CapEx and OpEx
# and
# [2] ETR’s sample of respondents(i.e., the number of CIOs and IT Decision Makers participating in ETR’s ecosystem and taking our surveys) approximately doubled between 2010 and 2015.


cloud_final = cloud_final.drop(["Cloud_N", "Survey_Citations"], axis = 1)

returns1 = spend_final.merge(peer_final, how = 'outer', on = ["Survey_Description_1", "Survey_ID", "Survey_Launch", "Survey_Close", "Announcement_Date", "Sector_Current", "Vendor_Current", "Product_Current", "Symbol_ID_Current", "Bloomberg_ID_Current", "FIGI_ID_Current", "Sector_Historical", "Vendor_Historical", "Product_Historical", "Symbol_ID_Historical", "Bloomberg_ID_Historical", "FIGI_ID_Historical"]).reset_index()
returns1 = returns1.merge(cloud_final, how = 'outer', on = ["Survey_Description_1", "Survey_ID", "Survey_Launch", "Survey_Close", "Announcement_Date", "Sector_Current", "Vendor_Current", "Product_Current", "Symbol_ID_Current", "Bloomberg_ID_Current", "FIGI_ID_Current", "Sector_Historical", "Vendor_Historical", "Product_Historical", "Symbol_ID_Historical", "Bloomberg_ID_Historical", "FIGI_ID_Historical"]).reset_index().drop(["level_0", "index"], axis = 1)
returns1 = returns1[returns1["Survey_ID"] >= 18]
returns1 = returns1[returns1["Citations"] >= mincitations]
returns1 = returns1.sort_values(by = ["Survey_ID", "Bloomberg_ID_Historical"], na_position = 'first').reset_index()
spReturns2 = spReturns.sort_values(by = ["Survey_ID", "Bloomberg_ID_Historical"], na_position = 'first').reset_index()[["Survey_ID", "Bloomberg_ID_Historical", "Return_End"]]


returns1 = returns1[['Survey_Description_1',	'Survey_ID',	'Survey_Launch',	'Survey_Close',	'Announcement_Date',	'Sector_Current',	'Vendor_Current',	'Product_Current',	'Symbol_ID_Current',	'Bloomberg_ID_Current',	'FIGI_ID_Current',	'Sector_Historical',	'Vendor_Historical',	'Product_Historical',	'Symbol_ID_Historical',	'Bloomberg_ID_Historical',	'FIGI_ID_Historical',	'Citations',	'AdoptionP_Value',	'DecreaseP_Value',	'DecreaseP_Value_SurveyZ',	'FlatP_Value',	'FlatP_Value_SurveyZ',	'IncreaseP_Value',	'IncreaseP_Value_SurveyZ',	'MarketShare_Value',	'NetScore_Value',	'NetScore_Value_SurveyZ',	'ReplacingP_Value',	'ReplacingP_Value_SurveyZ',	'AdoptionP_Value_SurveyZ',	'AdoptionP_Delta_sos',	'AdoptionP_Delta_sos_SurveyZ',	'DecreaseP_Delta_sos',	'DecreaseP_Delta_sos_SurveyZ',	'FlatP_Delta_sos',	'FlatP_Delta_sos_SurveyZ',	'IncreaseP_Delta_sos',	'IncreaseP_Delta_sos_SurveyZ',	'MarketShare_Delta_sos',	'MarketShare_Delta_sos_SurveyZ',	'NetScore_Delta_sos',	'NetScore_Delta_sos_SurveyZ',	'ReplacingP_Delta_sos',	'ReplacingP_Delta_sos_SurveyZ',	'AdoptionP_Delta_yoy',	'AdoptionP_Delta_yoy_SurveyZ',	'DecreaseP_Delta_yoy',	'DecreaseP_Delta_yoy_SurveyZ',	'FlatP_Delta_yoy',	'FlatP_Delta_yoy_SurveyZ',	'IncreaseP_Delta_yoy',	'IncreaseP_Delta_yoy_SurveyZ',	'MarketShare_Delta_yoy',	'MarketShare_Delta_yoy_SurveyZ',	'NetScore_Delta_yoy',	'NetScore_Delta_yoy_SurveyZ',	'ReplacingP_Delta_yoy',	'ReplacingP_Delta_yoy_SurveyZ',	'Adoption_Rating',	'Increase_Rating',	'Decrease_Rating',	'Replacing_Rating',	'NetScore_Rating',	'MarketShare_Rating',	'Peer_Accelerating',	'Peer_Decelerating',	'Peer_Rating',	'Cloud_Citations',	'Cloud_NetScore',	'Control_NetScore',	'Cloud_NetScore_Delta_Control',	'Cloud_Overlap',	'Cloud_Share',	'Cloud_Share_sos',	'Cloud_Share_yoy',	'Cloud_NetScore_Delta_sos',	'Cloud_NetScore_Delta_yoy',	'Cloud_Rating']]

returns2 = returns1.merge(spReturns2, how = 'left', on = ["Survey_ID", "Bloomberg_ID_Historical"])

returns2_means = pd.DataFrame()

returns2_means["Return_End_Mean"] = returns2.groupby("Survey_ID")["Return_End"].mean()
returns2_means["Return_End_StdDev"] = returns2.groupby("Survey_ID")["Return_End"].std()
returns2_means = returns2_means.reset_index()

returns3 = returns2.merge(returns2_means, on = "Survey_ID")
returns3["Return_EndZ"] = (returns3["Return_End"] - returns3["Return_End_Mean"]) / returns3["Return_End_StdDev"]
returns3["PosNeg_EndZ"] = np.nan

zcutoff = parameters.zcutoff

returns3.loc[(np.isnan(returns3["Return_EndZ"]) == False) & (returns3["Return_EndZ"] >= zcutoff), "PosNeg_EndZ"] = "Positive"
returns3.loc[(np.isnan(returns3["Return_EndZ"]) == False) & (returns3["Return_EndZ"] <= -zcutoff), "PosNeg_EndZ"] = "Negative"

returns3_orig = returns3

# The logistic regression model is trained to model the probability of outperformance.
# The original dataset is passed through the trained model to determine the model's historical performance.

# Prune unuseful data/last survey data
returns3 = returns3[returns3["Survey_ID"] < survey_max]
returns3 = returns3[pd.notnull(returns3["PosNeg_EndZ"])]

predictors = returns3[["Adoption_Rating", "Increase_Rating", "Decrease_Rating", "Replacing_Rating", "NetScore_Rating", "MarketShare_Rating", "Peer_Rating", "Cloud_Rating"]]
response = returns3["PosNeg_EndZ"]
predictors = predictors.replace({"Positive": 1, np.nan:0, "Negative":-1})
response = response.replace({"Positive": 1, "Negative":0})

# Create Design Matrix for regression

rating_list = ["Adoption_Rating", "Increase_Rating", "Decrease_Rating", "Replacing_Rating", "NetScore_Rating", "MarketShare_Rating", "Peer_Rating", "Cloud_Rating"]
def designMatrix(x):
    x["Intercept"] = 1
    for rating in rating_list:
        x[rating + "Negative"] = np.nan
        x[rating + "Positive"] = np.nan
        if x[rating] == 0:
            x[rating + "Negative"] = -1
            x[rating + "Positive"] = -1
        elif x[rating] == 1:
            x[rating + "Negative"] = 0
            x[rating + "Positive"] = 1
        elif x[rating] == -1:
            x[rating + "Negative"] = 1
            x[rating + "Positive"] = 0
    return x

newdm = predictors.apply(designMatrix, axis = 1).drop(rating_list, axis = 1)

# While loop manually performs step-wise backwards selection
selection_completed = False
while selection_completed == False:
    # Fit model function
    smmodel = sm.Logit(response, newdm)
    result = smmodel.fit(method = 'newton')

    print(result.summary())
    print(result.wald_test_terms())

    # Retrieve Wald test statistics

    wald_results = []
    for rating in rating_list:
        hypotheses = "(" + rating + "Negative = 0), (" + rating + "Positive = 0)"
        wald_results.append(result.wald_test(hypotheses).pvalue)

    pvalues = pd.DataFrame(rating_list, columns = ["Predictor"])
    pvalues["p-value"] = pd.Series(wald_results)


    maxpvalue = pvalues["p-value"].max().item()
    maxp = pvalues[pvalues["p-value"] == maxpvalue]

    if maxpvalue <= .2:
        selection_completed = True
    else:
        drop_column = maxp.iloc[0, 0]
        newdm = newdm.drop([drop_column + "Positive", drop_column + "Negative"], axis = 1)
        rating_list.remove(drop_column)
        print("Dropped Predictor:" + drop_column)




newpred = returns3_orig[rating_list]
newpred = newpred.replace({"Positive": 1, np.nan:0, "Negative":-1})

newpred = newpred.apply(designMatrix, axis = 1).drop(rating_list, axis = 1)

p_Positive = result.predict(newpred)
p_Negative = 1-p_Positive



forecast1 = returns3_orig
forecast1["P_Positive"] = p_Positive
forecast1["P_Negative"] = p_Negative

# Vendors across multiple sectors or products are combined using a citation-weighted average to determine the vendor's overall probability of outperformance.
# Only vendors above a certain threshold are assigned a Positive / Negative forecast.


forecast1 = forecast1.fillna("---")

wm = lambda x: np.average(x, weights = forecast1.loc[x.index, "Citations"]) 
f = {'Citations': ['sum'], 'P_Positive': {'P_Positive_mean':wm}}
weightedmeans = forecast1.groupby(['Survey_Description_1', 'Survey_ID', 'Survey_Launch', 'Survey_Close', 'Announcement_Date', 'Vendor_Historical', 'Symbol_ID_Historical', 'Bloomberg_ID_Historical', 'FIGI_ID_Historical']).agg(f)[['P_Positive']].reset_index()
weightedmeans.columns = weightedmeans.columns.droplevel(1)

spReturns = spReturns.sort_values(by=['Survey_ID', 'Bloomberg_ID_Historical'])

forecast2 = weightedmeans.sort_values(by=['Survey_ID', 'Bloomberg_ID_Historical']).replace({"---":np.nan})
forecast3 = forecast2.merge(spReturns, how = 'left', on = ['Survey_ID', 'Bloomberg_ID_Historical'])

forecast3 = forecast3[pd.isnull(forecast3["P_Positive"]) == False]
forecast3 = forecast3[pd.isnull(forecast3["Bloomberg_ID_Historical"]) == False]

forecast3["Insight_Forecast"] = np.nan

upperpcutoff = parameters.upperpcutoff
lowerpcutoff = parameters.lowerpcutoff

forecast3.loc[forecast3["P_Positive"] >= upperpcutoff, "Insight_Forecast"] = "Positive"
forecast3.loc[forecast3["P_Positive"] <= lowerpcutoff, "Insight_Forecast"] = "Negative"

forecast3 = forecast3[pd.isnull(forecast3["Insight_Forecast"]) == False]

# Historical returns are calculated to measure model performance.



forecast3 = forecast3.sort_values(by = ["Survey_ID", "Insight_Forecast", "Bloomberg_ID_Historical"])

forecast4 = forecast3.groupby(["Survey_Description_1", "Survey_ID", "Survey_Launch", "Survey_Close", "Announcement_Date", "Window_Start", "Window_End", "Insight_Forecast"])["Return_End"].mean().reset_index(name = "Return_End_Mean")
temp = forecast3.groupby(["Survey_Description_1", "Survey_ID", "Survey_Launch", "Survey_Close", "Announcement_Date", "Window_Start", "Window_End", "Insight_Forecast"])["Return_End"].count().reset_index(name = "Vendor_Count")

forecast4["Vendor_Count"] = temp["Vendor_Count"]

forecast5a = pd.pivot_table(forecast4, index = ["Survey_Description_1", "Survey_ID", "Survey_Launch", "Survey_Close", "Announcement_Date", "Window_Start", "Window_End"], columns = "Insight_Forecast", values = "Return_End_Mean").reset_index()
forecast5b = pd.pivot_table(forecast4, index = ["Survey_Description_1", "Survey_ID", "Survey_Launch", "Survey_Close", "Announcement_Date", "Window_Start", "Window_End"], columns = "Insight_Forecast", values = "Vendor_Count").reset_index()
forecast5a = forecast5a.rename(columns = {"Positive":"Positive_Returns", "Negative":"Negative_Returns"})
forecast5b = forecast5b.rename(columns = {"Positive":"Positive_VendorCount", "Negative":"Negative_VendorCount"})

forecast6 = forecast5a.merge(forecast5b, on = ["Survey_Description_1", "Survey_ID", "Survey_Launch", "Survey_Close", "Announcement_Date", "Window_Start", "Window_End"]).sort_values("Survey_ID")

ftecReturns = ftecReturns.sort_values("Survey_ID")[["Survey_ID", "Benchmark_Fidelity_MSCI_IT_ETF"]]

forecast7 = forecast6.merge(ftecReturns, how = 'left', on = "Survey_ID")

forecast7["Index_Type"] = "Single Insight Dataset"
forecast7["Index_Window"] = forecast7["Window_Start"] + "-" + forecast7["Window_End"]
forecast7["Window_Year"] = forecast7["Window_Start"].str.slice(start = -4)
forecast7["Positive_Returns_log"] = np.log(forecast7["Positive_Returns"] + 1)
forecast7["Negative_Returns_log"] = np.log(forecast7["Negative_Returns"] + 1)
forecast7["Benchmark_Return_log"] = np.log(forecast7["Benchmark_Fidelity_MSCI_IT_ETF"] + 1)

forecast7b = forecast7.groupby("Window_Year")[["Positive_Returns_log", "Negative_Returns_log", "Benchmark_Return_log"]].sum().reset_index()
forecast7c = forecast7[["Positive_Returns_log", "Negative_Returns_log", "Benchmark_Return_log"]].sum().reset_index().set_index('index').T


def strtodate(x):
    format_str = '%m/%d/%Y'
    return datetime.datetime.strptime(x, format_str)

forecast7["Window_Start"] = forecast7["Window_Start"].apply(strtodate)
forecast7["Window_End"] = forecast7["Window_End"].apply(strtodate)



mindate = forecast7.groupby("Window_Year")[["Window_Start", "Window_End"]].min().reset_index()
maxdate = forecast7.groupby("Window_Year")[["Window_Start", "Window_End"]].max().reset_index()

window7b = mindate.merge(maxdate, on = "Window_Year")
window7b = window7b.rename(columns = {"Window_Start_x":"Window_Start_Min", "Window_Start_y":"Window_Start_Max", "Window_End_x":"Window_End_Min", "Window_End_y":"Window_End_Max"})

window7c = forecast7[["Window_Start", "Window_End"]].min().reset_index().set_index('index').T
window7c = window7c.rename(columns = {"Window_Start":"Window_Start_Min", "Window_End":"Window_End_Min"})
maxdate = forecast7[["Window_Start", "Window_End"]].max().reset_index().set_index('index').T
maxdate = maxdate.rename(columns = {"Window_Start":"Window_Start_Max", "Window_End":"Window_End_Max"})

window7c["Window_Start_Max"] = maxdate["Window_Start_Max"]
window7c["Window_End_Max"] = maxdate["Window_End_Max"]

def datetostr(x):
    format_str = '%m/%d/%Y'
    return x.strftime(format_str)



forecast8b = forecast7b.merge(window7b, on = "Window_Year")
forecast8b["Index_Type"] = "Four Consecutive Insight Datasets"
forecast8b["Window_Start_Min"] = forecast8b["Window_Start_Min"].apply(datetostr)
forecast8b["Window_Start_Max"] = forecast8b["Window_Start_Max"].apply(datetostr)
forecast8b["Window_End_Min"] = forecast8b["Window_End_Min"].apply(datetostr)
forecast8b["Window_End_Max"] = forecast8b["Window_End_Max"].apply(datetostr)
forecast8b["Index_Window"] = forecast8b["Window_Start_Min"].astype(str) + "-" + forecast8b["Window_End_Max"].astype(str)
forecast8b["Positive_Returns_Consecutive"] = np.exp(forecast8b["Positive_Returns_log"]) - 1
forecast8b["Negative_Returns_Consecutive"] = np.exp(forecast8b["Negative_Returns_log"]) - 1
forecast8b["Benchmark_Returns_Consecutive"] = np.exp(forecast8b["Benchmark_Return_log"]) - 1


forecast8c = forecast7c
forecast8c["Window_Start_Min"] = window7c["Window_Start_Min"]
forecast8c["Window_End_Min"] = window7c["Window_End_Min"]
forecast8c["Window_Start_Max"] = window7c["Window_Start_Max"]
forecast8c["Window_End_Max"] = window7c["Window_End_Max"]
forecast8c["Index_Type"] = "Cumulative Insight Dataset"
forecast8c["Window_Start_Min"] = forecast8c["Window_Start_Min"].apply(datetostr)
forecast8c["Window_Start_Max"] = forecast8c["Window_Start_Max"].apply(datetostr)
forecast8c["Window_End_Min"] = forecast8c["Window_End_Min"].apply(datetostr)
forecast8c["Window_End_Max"] = forecast8c["Window_End_Max"].apply(datetostr)
forecast8c["Index_Window"] = forecast8c["Window_Start_Min"].astype(str) + "-" + forecast8c["Window_End_Max"].astype(str)
forecast8c["Positive_Returns_Consecutive"] = np.exp(forecast8c["Positive_Returns_log"]) - 1
forecast8c["Negative_Returns_Consecutive"] = np.exp(forecast8c["Negative_Returns_log"]) - 1
forecast8c["Benchmark_Returns_Consecutive"] = np.exp(forecast8c["Benchmark_Return_log"]) - 1

forecast9 = forecast7[["Index_Type", "Index_Window", "Positive_Returns", "Negative_Returns", "Benchmark_Fidelity_MSCI_IT_ETF"]]
forecast9 = forecast9.append(forecast8b[["Index_Type", "Index_Window", "Positive_Returns_Consecutive", "Negative_Returns_Consecutive", "Benchmark_Returns_Consecutive"]].rename(columns = {"Positive_Returns_Consecutive":"Positive_Returns", "Negative_Returns_Consecutive":"Negative_Returns", "Benchmark_Returns_Consecutive":"Benchmark_Fidelity_MSCI_IT_ETF"}))
forecast9 = forecast9.append(forecast8c[["Index_Type", "Index_Window", "Positive_Returns_Consecutive", "Negative_Returns_Consecutive", "Benchmark_Returns_Consecutive"]].rename(columns = {"Positive_Returns_Consecutive":"Positive_Returns", "Negative_Returns_Consecutive":"Negative_Returns", "Benchmark_Returns_Consecutive":"Benchmark_Fidelity_MSCI_IT_ETF"}))
forecast9 = forecast9.reset_index().drop("index", axis = 1)

# Final Datasets

InsightData_Final = returns1.sort_values(by = ["Survey_ID", "Sector_Current", "Vendor_Current", "Product_Current"], na_position = 'first').reset_index(drop = True)

InsightData_Final = InsightData_Final[["Survey_Description_1","Survey_ID","Survey_Launch","Survey_Close","Announcement_Date",
                                        "Sector_Current","Vendor_Current","Product_Current","Symbol_ID_Current","Bloomberg_ID_Current",
                                        "FIGI_ID_Current","Sector_Historical","Vendor_Historical","Product_Historical","Symbol_ID_Historical",
                                        "Bloomberg_ID_Historical","FIGI_ID_Historical","Citations","AdoptionP_Value","AdoptionP_Value_SurveyZ",
                                        "AdoptionP_Delta_sos","AdoptionP_Delta_sos_SurveyZ","AdoptionP_Delta_yoy","AdoptionP_Delta_yoy_SurveyZ",
                                        "Adoption_Rating","IncreaseP_Value","IncreaseP_Value_SurveyZ","IncreaseP_Delta_sos",
                                        "IncreaseP_Delta_sos_SurveyZ","IncreaseP_Delta_yoy","IncreaseP_Delta_yoy_SurveyZ","Increase_Rating",
                                        "FlatP_Value","FlatP_Value_SurveyZ","FlatP_Delta_sos","FlatP_Delta_sos_SurveyZ","FlatP_Delta_yoy",
                                        "FlatP_Delta_yoy_SurveyZ","DecreaseP_Value","DecreaseP_Value_SurveyZ","DecreaseP_Delta_sos",
                                        "DecreaseP_Delta_sos_SurveyZ","DecreaseP_Delta_yoy","DecreaseP_Delta_yoy_SurveyZ","Decrease_Rating",
                                        "ReplacingP_Value","ReplacingP_Value_SurveyZ","ReplacingP_Delta_sos","ReplacingP_Delta_sos_SurveyZ",
                                        "ReplacingP_Delta_yoy","ReplacingP_Delta_yoy_SurveyZ","Replacing_Rating","NetScore_Value"
                                        ,"NetScore_Value_SurveyZ","NetScore_Delta_sos","NetScore_Delta_sos_SurveyZ","NetScore_Delta_yoy",
                                        "NetScore_Delta_yoy_SurveyZ","NetScore_Rating","MarketShare_Value","MarketShare_Delta_sos",
                                        "MarketShare_Delta_sos_SurveyZ","MarketShare_Delta_yoy","MarketShare_Delta_yoy_SurveyZ",
                                        "MarketShare_Rating","Peer_Accelerating","Peer_Decelerating","Peer_Rating","Cloud_Citations",
                                        "Cloud_NetScore","Cloud_Share","Cloud_Share_sos","Cloud_NetScore_Delta_sos","Cloud_Share_yoy",
                                        "Cloud_NetScore_Delta_yoy","Control_NetScore","Cloud_Overlap","Cloud_NetScore_Delta_Control","Cloud_Rating"
]]
forecast3 = forecast3.sort_values(by = ["Survey_ID", "Insight_Forecast", "Vendor_Historical"])
InsightForecast_Final = forecast3.drop(["P_Positive", "Return_End", "Window_Start", "Window_End"], axis = 1).reset_index(drop = True)[["Survey_Description_1", "Survey_ID", "Survey_Launch", "Survey_Close", "Announcement_Date", "Insight_Forecast", "Vendor_Historical", "Bloomberg_ID_Historical", "Symbol_ID_Historical", "FIGI_ID_Historical"]]
InsightPerformance_Final = forecast9

# Final Datasets are objects of type pd.DataFrame. Please consult the pandas documentation (Online at http://pandas.pydata.org/pandas-docs/stable/) to export DataFrames to their desired format or output.

print("InsightData_Final:")
print(InsightData_Final)
print("InsightForecast_Final:")
print(InsightForecast_Final)
print("InsightPerformance_Final:")
print(InsightPerformance_Final)






