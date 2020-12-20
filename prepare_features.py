import requests
import os
import zipfile

from args import *
from utils import *


#Создаем две таблицы с количеством смертей и подтвержденных случаев по пути features_path 
#с данных которые сейчас есть
def convert_timeseries_last_date(input_path, features_path):
    s = pd.read_csv(input_path + 'cases_country_latest.csv')
    ccce_names = get_ccce_code_dict()
    arr_confirmed = []
    arr_deaths = []
    for index, row in s.iterrows():
        name, name2, date, cases, deaths = row['Country_Region'], row['ISO3'], row['Last_Update'], row['Confirmed'], row['Deaths']
        date = (date.split(' ')[0]).replace('-', '.')
        if name in ccce_names:
            c = ccce_names[name]
        else:
            c = 'XXX'
        name = name.replace(',', '_')
        name = name.replace(' ', '_')
        name2 = c

        arr_confirmed.append((name2, name, date, cases))
        arr_deaths.append((name2, name, date, deaths))

    out_path = features_path + 'time_table_flat_latest_{}.csv'.format('confirmed')
    out = open(out_path, 'w')
    out.write('name,name2,date,cases\n')
    for i in range(len(arr_confirmed)):
        name, name2, date, cases = arr_confirmed[i]
        out.write("{},{},{},{}\n".format(name, name2, date, cases))
    out.close()

    out_path = features_path + 'time_table_flat_latest_{}.csv'.format('deaths')
    out = open(out_path, 'w')
    out.write('name,name2,date,cases\n')
    for i in range(len(arr_deaths)):
        name, name2, date, cases = arr_deaths[i]
        out.write("{},{},{},{}\n".format(name, name2, date, cases))
    out.close()
    

#актуализация данных на основании COVID-19-master
def convert_timeseries_confirmed(input_path, features_path, data_type, latest_day_flag):
    if data_type == 'confirmed':
        tbl = pd.read_csv(input_path + 'COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    else:
        tbl = pd.read_csv(input_path + 'COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    con = pd.read_csv(input_path + 'countries.csv', dtype={'iso_alpha3': str}, keep_default_na=False)


    not_found = 0
    found = 0
    unique_countries = tbl['Country/Region'].unique()
    for u in unique_countries:
        if u not in list(con['ccse_name'].values):
            not_found += 1
        else:
            found += 1

    dates = list(tbl.columns.values)
    dates.remove('Province/State')
    dates.remove('Country/Region')
    dates.remove('Lat')
    dates.remove('Long')

    ccce_names = get_ccce_code_dict()
    summary = dict()
    for d in dates:
        for index, row in tbl.iterrows():
            cases = row[d]
            cntry = row['Country/Region']
            if (cntry, d) not in summary:
                summary[(cntry, d)] = 0
            summary[(cntry, d)] += cases

    out_path = features_path + 'time_table_flat_{}.csv'.format(data_type)
    out = open(out_path, 'w')
    out.write('name,name2,date,cases\n')
    for el in sorted(list(summary.keys())):
        cntry, date = el
        arr = date.split('/')
        m, d, y = arr
        dt = "20{}.{:02d}.{:02d}".format(y, int(m), int(d))
        if cntry in ccce_names:
            c = ccce_names[cntry]
        else:
            c = 'XXX'
        cntry = cntry.replace(',', '_')
        cntry = cntry.replace(' ', '_')
        out.write("{},{},{},{}\n".format(c, cntry, dt, summary[el]))
    out.close()

    s = pd.read_csv(out_path)
    if latest_day_flag:
        latest = pd.read_csv(features_path + 'time_table_flat_latest_{}.csv'.format(type))
        s = pd.concat((s, latest))
    s.sort_values(['name', 'date'], inplace=True)
    s.to_csv(out_path, index=False)
    
#создание актуальных фичей для обучения по дням
def create_time_features_confirmed(plus_day, data_type, features_path):
    feat_size = 10

    ccce_names = get_ccce_code_dict()
    s = pd.read_csv(features_path + 'time_table_flat_{}.csv'.format(data_type))
    unique_dates = sorted(s['date'].unique())[::-1]
    unique_countries = sorted(s['name2'].unique())

    val_matrix = np.zeros((len(unique_countries), len(unique_dates)), dtype=np.int32)

    out = open(features_path + 'features_predict_{}_day_{}.csv'.format(data_type, plus_day), 'w')
    if plus_day > 0:
        out.write('target,')
    out.write('name1,name2,date')
    for i in range(feat_size):
        out.write(',case_day_minus_{}'.format(i))
    out.write('\n')

    for index, row in s.iterrows():
        name, name2, date, cases = row['name'], row['name2'], row['date'], row['cases']
        i0 = unique_countries.index(name2)
        i1 = unique_dates.index(date)
        val_matrix[i0, i1] = cases

    for i in range(len(unique_countries)):
        name1 = unique_countries[i]
        if name1 in ccce_names:
            name2 = ccce_names[name1]
        else:
            name2 = 'XXX'
        for j in range(plus_day, len(unique_dates) - feat_size):
            if plus_day > 0:
                target = val_matrix[i, j - plus_day]
                out.write('{},'.format(target))
            out.write('{},{},{}'.format(name1, name2, unique_dates[j]))
            for k in range(j, j + feat_size):
                out.write(',{}'.format(val_matrix[i, k]))
            out.write('\n')
    out.close()
    

#Анализ первого дня заражения
#Создание csv с датой первого случая в категориях 'confirmed' и 'deaths'
def get_first_case_date(data_type, input_path, features_path):
    if data_type == 'confirmed':
        tbl = pd.read_csv(input_path + 'COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    else:
        tbl = pd.read_csv(input_path + 'COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    con = pd.read_csv(input_path + 'countries.csv', dtype={'iso_alpha3': str}, keep_default_na=False)

    not_found = 0
    found = 0
    unique_countries = tbl['Country/Region'].unique()
    for u in unique_countries:
        if u not in list(con['ccse_name'].values):
            not_found += 1
        else:
            found += 1

    dates = list(tbl.columns.values)
    dates.remove('Province/State')
    dates.remove('Country/Region')
    dates.remove('Lat')
    dates.remove('Long')

    ccce_names = get_ccce_code_dict()
    summary = dict()
    for d in dates:
        for index, row in tbl.iterrows():
            cases = row[d]
            cntry = row['Country/Region']
            if (cntry, d) not in summary:
                summary[(cntry, d)] = 0
            summary[(cntry, d)] += cases

    summary_sorted = dict()
    dates_sorted = set()
    all_countries = set()
    for el in sorted(list(summary.keys())):
        cntry, date = el
        arr = date.split('/')
        m, d, y = arr
        dt = "20{}.{:02d}.{:02d}".format(y, int(m), int(d))
        if cntry in ccce_names:
            c = ccce_names[cntry]
        else:
            c = 'XXX'
        dates_sorted |= set([dt])
        all_countries |= set([cntry])
        summary_sorted[(cntry, dt)] = summary[el]
    dates_sorted = sorted(list(dates_sorted))
    all_countries = sorted(list(all_countries))


    first_case = dict()
    for a in all_countries:
        first_case[a] = '2020.01.21'
        for i, dt in enumerate(dates_sorted[:-1]):
            el1 = (a, dates_sorted[i])
            el2 = (a, dates_sorted[i+1])
            if summary_sorted[el1] == 0 and summary_sorted[el2] > 0:
                first_case[a] = dates_sorted[i+1]

    out_path = features_path + 'first_date_{}.csv'.format(data_type)
    out = open(out_path, 'w')
    out.write('name,name2,date\n')
    for cntry in sorted(list(first_case.keys())):
        dt = first_case[cntry]
        if cntry in ccce_names:
            c = ccce_names[cntry]
        else:
            c = 'XXX'
        cntry = cntry.replace(',', '_')
        cntry = cntry.replace(' ', '_')
        out.write("{},{},{}\n".format(c, cntry, dt))
    out.close()
