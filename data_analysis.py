import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

dataset_path = 'IMDB-Movie-Data.csv'

data = pd.read_csv(dataset_path)

data_indexed = pd.read_csv(dataset_path, index_col="Title")
data.head()

data.info()

genre = data['Genre']

some_cols = data[['Title', 'Genre', 'Actors', 'Director', 'Rating']]
data.iloc[10:15][['Title', 'Rating', 'Revenue (Millions)']]

data[((data['Year'] >= 2010) & (data['Year'] <= 2015))
     & (data['Rating'] < 6.0)
     & (data['Revenue (Millions)'] > data['Revenue (Millions)'].quantile(0.95))]

data.groupby('Director')[['Rating']].mean().head()

data.groupby('Director')[['Rating']].mean().sort_values(
    ['Rating'], ascending=False).head()

data.isnull().sum()

data.drop('Metascore', axis=1).head()

revenue_mean = data_indexed['Revenue (Millions)'].mean()
print("The mean revenue is: ", revenue_mean)

data_indexed['Revenue (Millions)'].fillna(revenue_mean, inplace=True)


def rating_group(rating):
    if rating >= 7.5:
        return 'Good'
    elif rating >= 6.0:
        return 'Average'
    else:
        return 'Bad'


data['Rating_category'] = data['Rating'].apply(rating_group)
data[['Title', 'Director', 'Rating', 'Rating_category']].head(5)


dataset_path = "opsd_germany_daily.csv"

opsd_daily = pd.read_csv(dataset_path)

print(opsd_daily.shape)
print(opsd_daily.dtypes)
opsd_daily.head(3)

opsd_daily = opsd_daily.set_index('Date')
opsd_daily.head(3)
opsd_daily = pd.read_csv('opsd_germany_daily.csv',
                         index_col=0, parse_dates=True)

opsd_daily['Year'] = opsd_daily.index.year
opsd_daily['Month'] = opsd_daily.index.month
opsd_daily['Weekday Name'] = opsd_daily.index.day_name()

opsd_daily.sample(5, random_state=0)

opsd_daily.loc['2014-01-20':'2014-01-22']

sns.set(rc={'figure.figsize': (11, 4)})
opsd_daily['Consumption'].plot(linewidth=0.5)

cols_plot = ['Consumption', 'Solar', 'Wind']
axes = opsd_daily[cols_plot].plot(
    marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('Daily Totals (GWh)')
plt.show()

fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
for name, ax in zip(['Consumption', 'Solar', 'Wind'], axes):
    sns.boxplot(data=opsd_daily, x='Month', y=name, ax=ax)
    ax.set_ylabel('GWh')
    ax.set_title(name)
    if ax != axes[-1]:
        ax.set_xlabel('')
plt.show()

pd.date_range('1998-03-10', '1998-03-15', freq='D')

time_sample = pd.to_datetime(['2013-02-03', '2013-02-06', '2013-02-08'])

consum_sample = opsd_daily.loc[time_sample, ['Consumption']].copy()
print(consum_sample)

consum_freq = consum_sample.asfreq('D')
consum_freq['Consumption - Forward Fill'] = consum_sample.asfreq(
    'D', method='ffill')

data_columns = ['Consumption', 'Wind', 'Solar', 'Wind+Solar']
opsd_weekly_mean = opsd_daily[data_columns].resample('W').mean()
opsd_weekly_mean.head(3)

print(opsd_daily.shape[0])
print(opsd_weekly_mean.shape[0])
start, end = '2017-01', '2017-06'
fig, ax = plt.subplots()
ax.plot(opsd_daily.loc[start:end, 'Solar'], marker='.',
        linestyle='-', linewidth=0.5, label='Daily')
ax.plot(opsd_weekly_mean.loc[start:end, 'Solar'], marker='o',
        markersize=8, linestyle='-', label='Weekly Mean')
ax.set_ylabel('Solar Production (GWh)')
ax.legend()
plt.show()

opsd_annual = opsd_daily[data_columns].resample('A').sum(min_count=360)
opsd_annual = opsd_annual.set_index(opsd_annual.index.year)
opsd_annual.index.name = 'Year'
opsd_annual['Wind+Solar/Consumption'] = opsd_annual['Wind+Solar'] / \
    opsd_annual['Consumption']
opsd_annual.tail(3)

ax = opsd_annual.loc[2012:, 'Wind+Solar/Consumption'].plot.bar(color='C0')
ax.set_ylabel('Fracton')
ax.set_ylim(0, 0.3)
ax.set_ylim(0, 0.3)
ax.set_title('Wind + Solar Share of Annual Consumption')
plt.xticks(rotation=0)

opsd_7d = opsd_daily[data_columns].rolling(7, center=True).mean()
opsd_7d.head(10)

opsd_365d = opsd_daily[data_columns].rolling(
    window=365, center=True, min_periods=360).mean()
fig, ax = plt.subplots()
ax.plot(opsd_daily['Consumption'], marker='.', markersize=2,
        color='0.6', linestyle='None', label='Daily')
ax.plot(opsd_7d['Consumption'], linewidth=2, label='7 days rolling average')
ax.plot(opsd_365d['Consumption'], color='0.2',
        linewidth=3, label='365 days rolling average')
# Set x-ticks to yearly interval and add legend and labels
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.legend()
ax.set_xlabel('Year')
ax.set_ylabel('Consumption (GWh)')

plt.show()

fig, ax = plt.subplots()

for nm in ['Wind', 'Solar', 'Wind+Solar']:
    ax.plot(opsd_365d[nm], label=nm)
    # Set x- ticks to yearly interval , adjust y- axis limits , add legend and labels
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.set_ylim(0, 400)
    ax.legend()
    ax.set_ylabel('Production (GWh)')
    ax.set_title('Trends in Electricity Production (365 -d Rolling Means )')

plt.show()
