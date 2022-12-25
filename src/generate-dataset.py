import pandas as pd

from tqdm import tqdm                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              



def get_original_data() -> pd.DataFrame:
    df = pd.read_csv("data/WDIData.csv")
    return df


def get_indicators():
    with open("data/indicators.txt") as file:
        indicators_list = [line.rstrip() for line in file if line != "" ]

    indicators_sorted = sorted(set(indicators_list))
    indicators_sorted.pop(0)
    indicators = indicators_sorted
    return indicators

def generate_trasversed_data() -> pd.DataFrame :
    df = get_original_data()    

    years = [c for c in df.columns if "19" in c or "20" in c ]
    countries = sorted((set(df["Country Code"].values)))

    #indicators = sorted((set(df["Indicator Name"].values)))
    # Get data filtered by indicators
    indicators = get_indicators()


    df = df[df['Indicator Name'].isin(indicators)]

    all_countries_list = []
    for country in tqdm(countries) : 
        dict_country = { }
        df_country=  df.query( " `Country Code` == @country ")
        for indicator in indicators : 
            df_country_indicators_years = df_country.query( " `Indicator Name` == @indicator " )[years]
        #country_ind = df_country_indicators_years.T
        #country_ind.columns=[[indicator]]
            dict_country[ indicator ] =  df_country_indicators_years.iloc[0].tolist() 

        df_country = pd.DataFrame(dict_country)
        df_country.insert (0, "Country Code", country)
        df_country.insert (0, "Year", years)

        all_countries_list.append(df_country)

    df_all = pd.concat(all_countries_list)
    if 'Unnamed: 0' in df_all.columns :
        del df_all["Unnamed: 0"]

    return df_all.to_csv("data/df_all.csv")
    

# regenerate data
df = generate_trasversed_data()

