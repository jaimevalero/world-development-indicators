

# Open two CSV files
import pandas as pd


# Cargar los dos CSV en DataFrames
df_life_expectancy = pd.read_csv('files/API_SP.DYN.LE00.IN_DS2_en_csv_v2_5728852.csv')
df_gdp_per_capita = pd.read_csv('files/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_5728786.csv')

# Obtener el último año de cada país (columna más a la derecha)
#last_year_column_name = df_life_expectancy.columns[-2]  # Nombre de la columna del último año
last_year_column_name = "2021"
df_life_expectancy = df_life_expectancy[['Country Name', last_year_column_name]]
df_life_expectancy = df_life_expectancy.rename(columns={last_year_column_name: 'Life expectancy'})

last_year_column_name = df_gdp_per_capita.columns[-2]  # Nombre de la columna del último año
df_gdp_per_capita = df_gdp_per_capita[['Country Name', last_year_column_name]]
df_gdp_per_capita = df_gdp_per_capita.rename(columns={last_year_column_name: 'GDP Per Capita'})

# Combinar los dos DataFrames utilizando el país como clave (merge)
merged_df = pd.merge(df_life_expectancy, df_gdp_per_capita, on='Country Name')

# Imprimir el DataFrame resultante
print(merged_df)

# Remove fieles whit any column with NaN
merged_df = merged_df.dropna()



df = merged_df



import pandas as pd
import plotly.express as px
import numpy as np

# Supposing you already have the DataFrame merged_df with the combined data

# Create the interactive scatter plot without permanent labels
fig = px.scatter(merged_df, x='GDP Per Capita', y='Life expectancy', text='Country Name', labels={'text': ''})

# Calculate the linear regression coefficients
coefficients = np.polyfit(merged_df['GDP Per Capita'], merged_df['Life expectancy'], 1)
slope = coefficients[0]
intercept = coefficients[1]

# Add the regression line to the plot
fig.add_scatter(x=merged_df['GDP Per Capita'], y=slope * merged_df['GDP Per Capita'] + intercept,
                mode='lines', name='Regression Line', line=dict(color='red', width=2))

# Add title and axis labels
fig.update_layout(title='Correlation between GDP Per Capita and Life Expectancy with Regression Line',
                  xaxis_title='GDP Per Capita',
                  yaxis_title='Life expectancy')

# Configure label interaction with the mouse
fig.update_traces(textposition='top center', hoverinfo='text')

# Show the interactive plot
fig.show()




